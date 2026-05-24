import os
import json
import math
from contextlib import nullcontext
import torch
import numpy as np
import torch.distributed as dist
from torch.distributions import LogisticNormal
from copy import deepcopy
from ml_collections import ConfigDict
from random import random
from typing import Optional, Union, Dict
from tqdm import tqdm
from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.cuda.amp import GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
from typing import List, Dict, Union, Tuple

from encoders import EncNormalizer, ESM2EncoderModel
from compressors import HourglassProteinCompressionTransformer, trim_or_pad_batch_first
from model.fm_estimator import FlowEstimatorEMB, FlowEstimatorEMBwithVI
from model.ema_model import ExponentialMovingAverage
from model.text_conditioning import NullConditioning, TextConditionEncoder
from flow_matching_utils.length_sampler import LengthSampler
from flow_matching_utils.reflow_dataset import ReflowDataset
from utils import load_fasta_file, set_seed, gather_texts, dict_to_cuda, reduce_tensor, make_mask_wo_SEP_CLS, masked_mean, masked_std, TgAmpJsonlDataset
try:
    from evaluation import calculate_fid_for_files
except Exception:
    calculate_fid_for_files = None

import ot as pot
try:
    import torchdyn  # type: ignore
    from torchdyn.core import NeuralODE  # type: ignore
except Exception:
    torchdyn = None
    NeuralODE = None

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from torchcfm.optimal_transport import OTPlanSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


writer = SummaryWriter('') if SummaryWriter is not None else _NullSummaryWriter()

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def amp_jsonl_collate_fn(batch):
    output = {}
    for key in batch[0].keys():
        output[key] = [item[key] for item in batch]
    return output



AMP_CONDITION_KEYS = (
    "label",
    "length_bin_id",
    "charge_bin_id",
    "kr_bin_id",
    "hydrophobicity_bin_id",
    "cys_bin_id",
)

AMP_ATTRIBUTE_FIELDS = (
    ("length_bin_id", "length_embedding"),
    ("charge_bin_id", "charge_embedding"),
    ("kr_bin_id", "kr_embedding"),
    ("hydrophobicity_bin_id", "hydrophobicity_embedding"),
    ("cys_bin_id", "cys_embedding"),
)

def _dist_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class _TorchWrapper(torch.nn.Module):
    def __init__(self, model, attention_mask, model_type, use_compress, hidden_size):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask

    def forward(self, t, x, *args, **kwargs):
        if t.dim() == 0:
            t = t.expand(x.size(0)).to(x)
        elif t.dim() == 1 and t.size(0) == 1 and x.size(0) != 1:
            t = t.expand(x.size(0)).to(x)
        else:
            t = t.to(x)
        return self.model(
            x_t=x,
            time_t=t,
            attention_mask=self.attention_mask,
            cls=None,
            token_condition=None,
            token_condition_mask=None,
            global_condition=None,
        )


def torch_wrapper(model, attention_mask, model_type, use_compress, hidden_size):
    return _TorchWrapper(model, attention_mask, model_type, use_compress, hidden_size)

class FlowMatchingRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            latent_mode: str = "embeddings"
    ):
        # Basic info
        self.config = config
        self.device = torch.device(self.config.device)
        self.latent_mode = latent_mode
        self.eval = eval
        self.class_type = config.class_type
        self.use_text_conditioning = bool(getattr(self.config, "text", None) and self.config.text.enabled)
        self.is_amp_finetune = bool(getattr(self.config, "task", None) and self.config.task.stage == "amp_finetune")

        self.config.bert_config.use_cross_attention = bool(self.use_text_conditioning and self.config.text.use_cross_attention)
        self.config.bert_config.use_adaptive_norm = bool(self.use_text_conditioning and self.config.text.use_adaptive_norm)

        self.checkpoints_folder = config.training.checkpoints_folder
        self.suffix = (
            self.config.checkpoints_prefix
            or self.config.project_name
            or "flow_matching"
        )
    
        self.enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_mean,
            enc_std_path=self.config.data.enc_std,
            enc_max_path=self.config.data.enc_max,
            enc_min_path=self.config.data.enc_min,
        ).to(self.device)
        self.encoder_decoder = ESM2EncoderModel(
            self.config.model.hg_name,
            device=self.config.device,
            enc_normalizer=self.enc_normalizer,
            decoder_path=self.config.decoder_path,
            max_seq_len=self.config.data.max_sequence_len,
        )
        self.compresser = HourglassProteinCompressionTransformer(
            dim=self.config.model.hidden_size, 
            depth=self.config.compress.depth, 
            downproj_factor=self.config.compress.downproj_factor, 
            shorten_factor=self.config.compress.shorten_factor, 
            attn_resampling=self.config.compress.attn_resampling, 
            updown_sample_type=self.config.compress.updown_sample_type, 
            heads=self.config.compress.heads, 
            dim_head=self.config.compress.dim_head, 
            causal=self.config.compress.causal, 
            norm_out=self.config.compress.norm_out,
            use_quantizer=self.config.compress.use_quantizer, 
            n_e=self.config.compress.n_e, 
            e_dim=self.config.compress.e_dim, 
            vq_beta=self.config.compress.vq_beta, 
            enforce_single_codebook_per_position=self.config.compress.enforce_single_codebook_per_position, 
            fsq_levels=self.config.compress.fsq_levels
        ).to(self.device)
        self.compresser.from_pretrained(self.config.compress.checkpoint)

        self.optimizer = None
        self.scheduler = None
        self.sampler = None
        self.flow_matcher = None
        self.step = 0
        self.best_valid_total_loss = float("inf")
        self.best_valid_step = -1
        self.stale_validation_evals = 0

        if self.config.use_compress:
            self.model = FlowEstimatorEMBwithVI(
                input_size=self.config.model.compressed_hidden_size,
                config=config.bert_config
            ).to(self.device).train()
        else:
            self.model = FlowEstimatorEMB(
                input_size=self.config.model.hidden_size,
                config=config.bert_config
            ).to(self.device).train()

        self.ddp_model = self.model
        if self.config.ddp:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )
        self.total_number_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config.model.total_number_params = self.total_number_params
        self.device = next(self.model.parameters()).device

        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.model.eval()
        
        self.train_dataset = None
        self.valid_dataset = None
        self.length_sampler = LengthSampler(path=self.config.data.train_dataset_path, max_len=self.config.data.max_sequence_len - 2)
        self.null_conditioning = NullConditioning(self.config.model.hidden_size).to(self.device)
        self.text_encoder = None
        if self.use_text_conditioning and self.config.text.encoder_name:
            self.text_encoder = TextConditionEncoder(
                encoder_name=self.config.text.encoder_name,
                condition_dim=self.config.text.condition_dim,
                max_length=self.config.text.max_length,
                dropout=self.config.text.dropout_prob,
            ).to(self.device)
            if getattr(self.config.text, "freeze_encoder", False):
                for param in self.text_encoder.encoder.parameters():
                    param.requires_grad = False
        
        
        if self.config.ddp and dist.get_rank() == 0 and self.config.wandb and wandb is not None and not eval:
            wandb.init(
                project=self.config.project_name,
                name=self.config.checkpoints_prefix,
                config=dict(self.config),
                mode="online"
            )

        self.logistic_normal_dist = LogisticNormal(loc=torch.tensor(self.config.fm.m), scale=torch.tensor(self.config.fm.s))

    def _get_amp_lengths(self, path):
        dataset = TgAmpJsonlDataset(path, min_len=self.config.amp.min_len, max_len=self.config.amp.max_len)
        return [len(item["sequence"]) for item in dataset.records]

    def _amp_batch_to_model_inputs(self, batch):
        sequences = batch["sequence"]
        texts = batch["text"]
        with torch.no_grad():
            clean_X, tokenized_X = self.encoder_decoder.batch_encode(sequences)
            if self.config.use_compress:
                tokens = tokenized_X["input_ids"]
                mask = tokenized_X["attention_mask"]
                clean_X = trim_or_pad_batch_first(clean_X, pad_to=self.config.data.max_sequence_len, pad_idx=0)
                if mask.shape[1] != clean_X.shape[1]:
                    mask = trim_or_pad_batch_first(mask, clean_X.shape[1], pad_idx=0)
                    tokens = trim_or_pad_batch_first(tokens, clean_X.shape[1], pad_idx=1)
                clean_X = clean_X.to(self.device)
                mask = mask.to(self.device)
                tokens = tokens.to(self.device)
                clean_X = self.enc_normalizer.minmax_scaling(clean_X)
                z_q, downsampled_mask = self.compresser.encode(x=clean_X, mask=mask.bool(), verbose=self.config.compress.verbose)
                clean_X = z_q
                tokenized_X = {"input_ids": tokens, "attention_mask": downsampled_mask}
            tokenized_X["text"] = texts
            for key in AMP_CONDITION_KEYS:
                if key in batch:
                    tokenized_X[key] = torch.as_tensor(batch[key], device=self.device, dtype=torch.long)
            if "label" in batch:
                label = batch["label"]
                if not torch.is_tensor(label):
                    label = torch.tensor(label)
                tokenized_X["label"] = label.to(device=self.device, dtype=torch.float32)
        return clean_X, tokenized_X

    def _get_model_module(self):
        return self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

    def _get_extended_attention_mask(self, attention_mask, dtype):
        if attention_mask is None:
            return None
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask.to(dtype=dtype)) * torch.finfo(dtype).min
        return extended_attention_mask

    def _field_to_long_tensor(self, batch, key, batch_size, default=None):
        if not isinstance(batch, dict) or key not in batch:
            if default is None:
                return None
            return torch.full((batch_size,), int(default), device=self.device, dtype=torch.long)
        value = batch[key]
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.long).view(-1)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.long).view(-1)
        if tensor.numel() == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size)
        if tensor.numel() != batch_size:
            raise ValueError(f"Condition field `{key}` has {tensor.numel()} values for batch_size={batch_size}")
        return tensor

    def _make_attribute_cls(self, batch, batch_size, seq_len, dtype):
        if not bool(getattr(self.config.text, "use_attribute_condition", False)):
            return None
        model_module = self._get_model_module()
        pieces = []
        for key, embedding_name in AMP_ATTRIBUTE_FIELDS:
            if not hasattr(model_module, embedding_name):
                continue
            ids = self._field_to_long_tensor(batch, key, batch_size)
            if ids is None:
                continue
            embedding = getattr(model_module, embedding_name)
            ids = ids.clamp(0, embedding.num_embeddings - 1)
            pieces.append(embedding(ids))
        if not pieces:
            return None
        cls = torch.stack(pieces, dim=0).sum(dim=0)
        if hasattr(model_module, "direct_condition_layer_norm"):
            cls = model_module.direct_condition_layer_norm(cls)
        cls = cls.to(dtype=dtype)
        return cls[:, None, :].expand(batch_size, seq_len, -1)

    def _prepare_conditioning(self, batch, batch_size, seq_len, dtype):
        if not self.use_text_conditioning:
            null_conditioning = self.null_conditioning(batch_size, seq_len, self.device, dtype)
            return {
                "token_condition": null_conditioning.token_condition,
                "global_condition": null_conditioning.global_condition,
                "attention_mask": null_conditioning.attention_mask,
            }, False

        if isinstance(batch, dict) and ("texts" in batch or "text" in batch):
            if self.text_encoder is None:
                null_conditioning = self.null_conditioning(batch_size, seq_len, self.device, dtype)
                return {
                    "token_condition": null_conditioning.token_condition,
                    "global_condition": null_conditioning.global_condition,
                    "attention_mask": null_conditioning.attention_mask,
                }, False
            conditioning = self.text_encoder(batch, device=self.device)
        elif isinstance(batch, dict) and "token_condition" in batch and "global_condition" in batch:
            token_condition = batch["token_condition"].to(device=self.device, dtype=dtype)
            global_condition = batch["global_condition"].to(device=self.device, dtype=dtype)
            attention_mask = batch.get("text_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=self.device)
            conditioning = {
                "token_condition": token_condition,
                "global_condition": global_condition,
                "attention_mask": attention_mask,
            }
        else:
            null_conditioning = self.null_conditioning(batch_size, seq_len, self.device, dtype)
            return {
                "token_condition": null_conditioning.token_condition,
                "global_condition": null_conditioning.global_condition,
                "attention_mask": null_conditioning.attention_mask,
            }, False

        if isinstance(conditioning, dict):
            token_condition = conditioning["token_condition"]
            global_condition = conditioning["global_condition"]
            attention_mask = conditioning.get("attention_mask")
        else:
            token_condition = conditioning.token_condition
            global_condition = conditioning.global_condition
            attention_mask = conditioning.attention_mask

        if token_condition.shape[1] != seq_len:
            if token_condition.shape[1] > seq_len:
                token_condition = token_condition[:, :seq_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :seq_len]
            else:
                pad_len = seq_len - token_condition.shape[1]
                token_condition = torch.nn.functional.pad(token_condition, (0, 0, 0, pad_len))
                if attention_mask is not None:
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        if attention_mask is None:
            attention_mask = torch.ones(token_condition.shape[:2], device=self.device, dtype=torch.long)

        return {
            "token_condition": token_condition,
            "global_condition": global_condition,
            "attention_mask": attention_mask,
        }, True

    def _apply_cfg_dropout(self, conditioning, is_conditional):
        if not is_conditional:
            return conditioning
        dropout_prob = float(getattr(self.config.text, "cfg_dropout_prob", 0.0))
        batch_size = conditioning["global_condition"].shape[0]
        keep_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)

        if dropout_prob > 0 and self.ddp_model.training:
            keep_mask = torch.rand(batch_size, device=self.device) > dropout_prob
            null_conditioning = self.null_conditioning(
                batch_size,
                conditioning["token_condition"].shape[1],
                self.device,
                conditioning["token_condition"].dtype,
            )
            keep_global = keep_mask[:, None]
            keep_token = keep_mask[:, None, None]
            conditioning["global_condition"] = torch.where(
                keep_global,
                conditioning["global_condition"],
                null_conditioning.global_condition,
            )
            conditioning["token_condition"] = torch.where(
                keep_token,
                conditioning["token_condition"],
                null_conditioning.token_condition,
            )
            conditioning["attention_mask"] = torch.where(
                keep_mask[:, None],
                conditioning["attention_mask"],
                null_conditioning.attention_mask,
            )

        conditioning["condition_keep_mask"] = keep_mask
        return conditioning

    def _compute_condition_consistency_loss(self, predicted_clean, conditioning, mask):
        if conditioning is None:
            return predicted_clean.new_tensor(0.0)
        if self.config.loss.cc_coef <= 0:
            return predicted_clean.new_tensor(0.0)

        model_module = self._get_model_module()
        if mask is None:
            pooled_latent = predicted_clean.mean(dim=1)
        else:
            mask = mask.float()
            pooled_latent = (predicted_clean * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        predicted_condition = model_module.latent_condition_head(pooled_latent)
        target_condition = conditioning["global_condition"]
        keep_mask = conditioning.get("condition_keep_mask")
        if keep_mask is not None:
            keep_mask = keep_mask.bool()
            if keep_mask.sum() == 0:
                return predicted_clean.new_tensor(0.0)
            predicted_condition = predicted_condition[keep_mask]
            target_condition = target_condition[keep_mask]

        mse = torch.mean((predicted_condition - target_condition) ** 2)
        cosine = 1 - torch.nn.functional.cosine_similarity(predicted_condition, target_condition, dim=-1).mean()
        return mse + cosine
    
    def _pool_latent(self, latent, mask):
        if mask is None:
            return latent.mean(dim=1)
        mask = mask.to(device=latent.device, dtype=latent.dtype)
        return (latent * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _make_label_cls(self, batch, batch_size, seq_len, dtype):
        if not bool(getattr(self.config.text, "use_label_condition", False)):
            return None
        model_module = self._get_model_module()
        if not hasattr(model_module, "label_embedding"):
            return None
        labels = self._field_to_long_tensor(batch, "label", batch_size)
        if labels is None:
            return None
        labels = labels.clamp(0, model_module.label_embedding.num_embeddings - 1)
        cls = model_module.label_embedding(labels).to(dtype=dtype)
        return cls[:, None, :].expand(batch_size, seq_len, -1)

    def _compute_activity_loss(self, predicted_clean, batch, mask):
        if float(getattr(self.config.loss, "label_coef", 0.0)) <= 0:
            return predicted_clean.new_tensor(0.0)
        if not isinstance(batch, dict) or "label" not in batch:
            return predicted_clean.new_tensor(0.0)

        labels = batch["label"]
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=self.device)
        labels = labels.to(device=self.device, dtype=predicted_clean.dtype).view(-1)

        pooled = self._pool_latent(predicted_clean, mask)
        logits = self._get_model_module().activity_head(pooled).squeeze(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def _compute_condition_contrast_loss(self, x_t, time_t, target_velocity, v_t, conditioning, batch, mask, cls):
        if float(getattr(self.config.loss, "contrast_coef", 0.0)) <= 0:
            return x_t.new_tensor(0.0)
        if not self.ddp_model.training:
            return x_t.new_tensor(0.0)
        if conditioning is None or not isinstance(batch, dict) or "label" not in batch:
            return x_t.new_tensor(0.0)

        labels = batch["label"]
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=self.device)
        labels = labels.to(device=self.device).long().view(-1)

        if torch.unique(labels).numel() < 2:
            return x_t.new_tensor(0.0)

        wrong_indices = []
        for i in range(labels.shape[0]):
            candidates = torch.nonzero(labels != labels[i], as_tuple=False).flatten()
            if candidates.numel() == 0:
                return x_t.new_tensor(0.0)
            wrong_indices.append(candidates[(i + int(self.step)) % candidates.numel()])
        wrong_indices = torch.stack(wrong_indices).to(device=self.device)

        wrong_conditioning = {}
        for key, value in conditioning.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == x_t.shape[0]:
                wrong_conditioning[key] = value.index_select(0, wrong_indices)
            else:
                wrong_conditioning[key] = value

        wrong_token_mask = self._get_extended_attention_mask(wrong_conditioning["attention_mask"], x_t.dtype)

        v_wrong = self.ddp_model(
            x_t=x_t,
            time_t=time_t,
            attention_mask=mask,
            cls=cls,
            token_condition=wrong_conditioning["token_condition"],
            token_condition_mask=wrong_token_mask,
            global_condition=wrong_conditioning["global_condition"],
        )

        pos = (v_t - target_velocity).pow(2).flatten(1).mean(dim=1)
        neg = (v_wrong - target_velocity).pow(2).flatten(1).mean(dim=1)
        margin = float(getattr(self.config.loss, "contrast_margin", 0.05))
        return torch.nn.functional.relu(margin + pos - neg).mean()
        
        
    # Tool functions to set parameters, ema, optimizer, etc
    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix
        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        model = self.model
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    
    def switch_back_from_ema(self) -> None:
        ema = self.ema
        model = self.model
        ema.restore(model.parameters())

    def set_optimizer(self) -> None:
        trainable_parameters = list(self.model.parameters())
        if self.use_text_conditioning and self.text_encoder is not None:
            trainable_parameters += list(self.text_encoder.parameters())
            trainable_parameters += list(self.null_conditioning.parameters())
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer
    
    def set_scheduler(self) -> None:
        if self.config.scheduler.type == "cosine":
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=self.config.training.training_iters,
                lr_min=self.config.optim.min_lr,
                warmup_lr_init=self.config.optim.warmup_lr,
                warmup_t=self.config.optim.linear_warmup,
                cycle_limit=1,
                t_in_epochs=False,
            )
        elif self.config.scheduler.type == "anneal":
            lambda_scheduler = lambda i: (self.config.optim.max_lr-self.config.optim.warmup_lr)*i / self.config.optim.linear_warmup + self.config.optim.warmup_lr \
                                    if i < self.config.optim.linear_warmup else \
                                    (self.config.optim.min_lr+0.5*(self.config.optim.max_lr-self.config.optim.min_lr)* \
                                    (1.0+math.cos((i-self.config.optim.linear_warmup)/(self.config.training.training_iters \
                                    -self.config.optim.linear_warmup)*math.pi)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_scheduler)

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def set_sampler(self) -> None:
        self.sampler = OTPlanSampler(method=self.config.fm.ot_sampler_mode)

    def set_flow_matcher(self) -> None:
        self.flow_matcher = ConditionalFlowMatcher(sigma=self.config.fm.sigma)

    # Dataloaders
    def set_train_data_generator(self) -> None:
        if self.train_dataset is None:
            if self.is_amp_finetune:
                self.train_dataset = TgAmpJsonlDataset(
                    self.config.amp.train_jsonl,
                    min_len=self.config.amp.min_len,
                    max_len=self.config.amp.max_len,
                )
            else:
                self.train_dataset = load_fasta_file(self.config.data.train_dataset_path)
        print("Train dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=getattr(self.config.training, "num_workers", 4),
            pin_memory=False,
            collate_fn=amp_jsonl_collate_fn if self.is_amp_finetune else None,
        )

    def set_valid_data_generator(self) -> None:
        if self.valid_dataset is None:
            if self.is_amp_finetune:
                self.valid_dataset = TgAmpJsonlDataset(
                    self.config.amp.valid_jsonl,
                    min_len=self.config.amp.min_len,
                    max_len=self.config.amp.max_len,
                )
            else:
                self.valid_dataset = load_fasta_file(self.config.data.test_dataset_path)
        print("Valid dataset length:", len(self.valid_dataset))

        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=max(1, self.config.validation.batch_size // _dist_world_size()),
            num_workers=getattr(self.config.training, "num_workers", 4),
            pin_memory=False,
            collate_fn=amp_jsonl_collate_fn if self.is_amp_finetune else None,
        )

    def set_train_reflow_data_generator(self) -> None:
        if self.train_dataset is None:
            self.train_dataset = ReflowDataset("train", self.config.fm.reflow_datapath, self.config.fm.reflow_datanum)
        print("Train dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=getattr(self.config.training, "num_workers", 4),
            pin_memory=False,
        )

    def set_valid_reflow_data_generator(self) -> None:
        if self.valid_dataset is None:
            self.valid_dataset = ReflowDataset("valid", self.config.fm.reflow_datapath, self.config.fm.reflow_datanum)
        print("Valid dataset length:", len(self.valid_dataset))

        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=max(1, self.config.validation.batch_size // _dist_world_size()),
            num_workers=getattr(self.config.training, "num_workers", 4),
            pin_memory=False,
        )


    # logger
    def log_metric(self, metric_name: str, loader_name: str, value):
        if wandb is None:
            return
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)



    # optimizer_step
    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        self.grad_scaler.unscale_(self.optimizer)

        grad_tensors = [t.grad for t in self.model.parameters() if t.grad is not None]
        zero_norm = torch.tensor(0.0, device=self.device)

        if self.config.model.model_type != "cnn":
            if grad_tensors:
                grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grad_tensors))
            else:
                grad_norm = zero_norm

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm
                )
            clipped_grad_tensors = [t.grad for t in self.model.parameters() if t.grad is not None]
            if clipped_grad_tensors:
                clipped_grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in clipped_grad_tensors))
            else:
                clipped_grad_norm = zero_norm
        else:
            grad_norm = zero_norm
            clipped_grad_norm = zero_norm
        if _dist_rank() == 0:
            writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.step)
        if self.config.wandb and _dist_rank() == 0:
            self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.model.parameters())
        if self.config.scheduler.type == "cosine":
            self.scheduler.step_update(self.step)
        elif self.config.scheduler.type == "anneal":
            self.scheduler.step()
        if self.config.model.model_type != "cnn":
            return grad_norm, clipped_grad_norm
        else:
            return 0
    


    # training code
    def train(
            self,
            project_name: str = 'flow_matching',
            experiment_name: str = 'emb'
    ) -> None:
        self.step = 0
        self.best_valid_total_loss = float("inf")
        self.best_valid_step = -1
        self.stale_validation_evals = 0
        self.set_optimizer()
        self.set_scheduler()
        self.set_grad_scaler()
        self.set_sampler()
        self.set_flow_matcher()
        if self.config.fm.reflow:
            self.set_valid_reflow_data_generator()
        else:
            self.set_valid_data_generator()
        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)

        if self.config.refresh.true:
            self.refresh_checkpoint()
            if getattr(self.config.training, "eval_on_start", True):
                self.estimation()
                valid_loss = self.validate()
                self._update_early_stopping(valid_loss)

        if self.config.fm.reflow:
            load = torch.load(f'{self.config.fm.reflow_ckpt}', map_location="cpu")

            self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
            self.ema.load_state_dict(load["ema"])
            self.ema.to(self.device)
            self.switch_to_ema()

            print(f"Checkpoint refreshed {self.config.refresh.prefix}")

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            if self.config.fm.reflow:
                self.set_train_reflow_data_generator()
            else:
                self.set_train_data_generator()
            self.ddp_model.train()
            should_stop_early = self.train_epoch()

            if should_stop_early or self.step >= self.config.training.training_iters:
                break

        self.model.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()
        writer.close()

    def train_epoch(self):
            for _, X in enumerate(self.train_loader):
                if self.step >= self.config.training.training_iters:
                    return False
                _ = next(self.train_range_iter)

                loss_dict, stat_dict = self.train_step(X)

                if self.step % self.config.training.generate_freq == 0 and _dist_rank() == 0:
                    print("Example Sequences: ", self.generate_text(5))

                if self.step % self.config.training.checkpoint_freq == 0:
                    self.save_checkpoint()

                if self.step % self.config.training.eval_freq == 0:
                    torch.cuda.empty_cache()
                    self.estimation()
                    valid_loss = self.validate()
                    if self._update_early_stopping(valid_loss):
                        if _dist_rank() == 0:
                            print(
                                f"Early stopping triggered at step {self.step}. "
                                f"Best valid total_loss: {self.best_valid_total_loss:0.6f} "
                                f"at step {self.best_valid_step}."
                            )
                        return True

                self.train_range.set_description(
                    f"total: {loss_dict['total_loss'].item():0.4f}, "
                    f"fm: {loss_dict['loss'].item():0.4f}, "
                    f"cc: {loss_dict['cc_loss'].item():0.4f}, "
                    f"act: {loss_dict.get('activity_loss', loss_dict['loss'].new_tensor(0.0)).item():0.4f}, "
                    f"ctr: {loss_dict.get('contrast_loss', loss_dict['loss'].new_tensor(0.0)).item():0.4f}, "
                    f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                )
            return False
    
    def train_step(self, X):
        self.step += 1
        if self.config.fm.reflow:
            z0, z1 = X
            loss_dict, stat_dict = self.calc_loss(clean_x=z0.to(self.device), X=z1.to(self.device))
        elif self.is_amp_finetune:
            clean_X, tokenized_X = self._amp_batch_to_model_inputs(X)
            loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, X=tokenized_X)
        else:
            X = dict_to_cuda(X)
            with torch.no_grad():
                clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)
                if self.config.use_compress:
                    tokens = tokenized_X["input_ids"]
                    mask = tokenized_X["attention_mask"]
                    clean_X = trim_or_pad_batch_first(clean_X, pad_to=self.config.data.max_sequence_len, pad_idx=0)
                    if mask.shape[1] != clean_X.shape[1]:
                        # pad with False
                        mask = trim_or_pad_batch_first(mask, clean_X.shape[1], pad_idx=0)
                        tokens = trim_or_pad_batch_first(tokens, clean_X.shape[1], pad_idx=1)
                    clean_X = clean_X.to(self.device)
                    mask = mask.to(self.device)
                    tokens = tokens.to(self.device)
                    clean_X = self.enc_normalizer.minmax_scaling(clean_X)
                    z_q, downsampled_mask = self.compresser.encode(x = clean_X, mask = mask.bool(), verbose=self.config.compress.verbose)
                    clean_X = z_q
                    tokenized_X = {"input_ids":tokens, "attention_mask":downsampled_mask}
            loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, X=tokenized_X)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])

        if _dist_rank() == 0:
            if self.step % 10 == 0:
                stat_dict["weight_norm"] = torch.sqrt(
                    sum([torch.sum(t.data ** 2) for t in self.model.parameters()]))
                if self.config.wandb:
                    for k, v in loss_dict.items():
                        self.log_metric(k, 'train', v.item())

                    for k, v in stat_dict.items():
                        self.log_metric(k, 'train', v.item())

        return loss_dict, stat_dict
    
    def calc_loss(
            self,
            clean_x, # clean_X: batch_size x seq_length x hidden_dim 128,50,320
            X=None, # tokenized_X: batch_size x seq_length 128,50
            eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        if self.config.fm.reflow:
            mask = None
        elif self.config.fm.use_mask:
            mask = X["attention_mask"] # batch_size x seq_length 128,50
        else:
            mask = None

        batch_size = clean_x.size(0)
        seq_len = clean_x.size(1)

        if self.config.fm.reflow:
            x_0 = clean_x
            x_1 = X
        else:
            x_0 = torch.randn_like(clean_x)
            x_1 = clean_x

        if self.config.use_class:
            if self.config.use_compress:
                cls = torch.zeros((clean_x.shape[0], clean_x.shape[1], self.config.model.hidden_size), dtype=clean_x.dtype, device=self.device)
            else:
                cls = torch.zeros_like(clean_x, dtype=clean_x.dtype, device=self.device)
        else:
            cls = None

        label_cls = self._make_label_cls(X, batch_size, seq_len, clean_x.dtype)
        if label_cls is not None:
            cls = label_cls if cls is None else cls + label_cls
        attribute_cls = self._make_attribute_cls(X, batch_size, seq_len, clean_x.dtype)
        if attribute_cls is not None:
            cls = attribute_cls if cls is None else cls + attribute_cls

        t = torch.rand(batch_size).type_as(x_1)
        x_t = t.unsqueeze(1).unsqueeze(2)*x_1 + (1-t).unsqueeze(1).unsqueeze(2)*x_0
        u_t = x_1 - x_0
        conditioning, is_conditional = self._prepare_conditioning(X, batch_size=batch_size, seq_len=seq_len, dtype=clean_x.dtype)
        conditioning = self._apply_cfg_dropout(conditioning, is_conditional)
        token_condition_mask = self._get_extended_attention_mask(conditioning["attention_mask"], clean_x.dtype)

        label_cls = self._make_label_cls(
            X.get("label") if isinstance(X, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=clean_x.dtype,
        )
        if label_cls is not None:
            cls = label_cls if cls is None else cls + label_cls

        autocast_context = torch.autocast(device_type='cuda', dtype=torch.float32) if self.device.type == "cuda" else nullcontext()
        with autocast_context:
            v_t = self.ddp_model(
                    x_t=x_t, time_t=t,
                    attention_mask=mask,
                    cls=cls,
                    token_condition=conditioning["token_condition"],
                    token_condition_mask=token_condition_mask,
                    global_condition=conditioning["global_condition"],
                )

        weights = torch.ones_like(t, dtype=t.dtype)
        loss = torch.mean(weights.unsqueeze(1).unsqueeze(2)*(v_t - u_t) ** 2)
        predicted_clean = x_0 + v_t
        cc_loss = self._compute_condition_consistency_loss(predicted_clean, conditioning if is_conditional else None, mask)
        activity_loss = self._compute_activity_loss(predicted_clean, X, mask)
        contrast_loss = self._compute_condition_contrast_loss(
            x_t=x_t,
            time_t=t,
            target_velocity=u_t,
            v_t=v_t,
            conditioning=conditioning if is_conditional else None,
            batch=X,
            mask=mask,
            cls=cls,
        )
        total_loss = (
            loss
            + float(getattr(self.config.loss, "cc_coef", 0.0)) * cc_loss
            + float(getattr(self.config.loss, "label_coef", 0.0)) * activity_loss
            + float(getattr(self.config.loss, "contrast_coef", 0.0)) * contrast_loss
        )
        loss_dict = {
            'total_loss': total_loss,
            'loss': loss,
            'cc_loss': cc_loss,
            'activity_loss': activity_loss,
            'contrast_loss': contrast_loss,
        }        
        clean_x_mean, clean_x_std, clean_x_norm = self.get_stat(clean_x, mask)
        x_0_mean, x_0_std, x_0_norm = self.get_stat(x_0, mask)
        stat_dict = {
            "clean_x_mean": clean_x_mean,
            "clean_x_std": clean_x_std,
            "clean_x_norm": clean_x_norm,
            "x_0_mean": x_0_mean,
            "x_0_std": x_0_std,
            "x_0_norm": x_0_norm,
        }
        return loss_dict, stat_dict

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.empty(batch_size, device=self.device).uniform_() * (1 - eps) + eps
    
    def get_stat(self, z, mask):
        if mask is None:
            mask = torch.ones(
                (z.shape[0], z.shape[1]),
                device=self.device,
                requires_grad=False,
                dtype=torch.int64,
            )
        mask_SEP_CLS = make_mask_wo_SEP_CLS(mask)
        mean = masked_mean(z, mask_SEP_CLS)
        std = masked_std(z, mask_SEP_CLS)
        norm = torch.sum(torch.norm(z, dim=2) * mask_SEP_CLS) / torch.sum(mask_SEP_CLS)
        return torch.mean(mean), torch.mean(std), norm


    # validation code
    def validate(self) -> Dict[str, torch.Tensor]:
        prev_mode = self.ddp_model.training

        self.ddp_model.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for X in self.valid_loader:
                if self.config.fm.reflow:
                    z0, z1 = X
                    loss_dict, _ = self.calc_loss(clean_x=z0.to(self.device), X=z1.to(self.device))
                    for k, v in loss_dict.items():
                        if k in valid_loss:
                            valid_loss[k] += v.item() * z0.size(0)
                        else:
                            valid_loss[k] = torch.Tensor([v.item() * z0.size(0)])
                    valid_count += z0.size(0)
                else:
                    X = dict_to_cuda(X)
                    if self.is_amp_finetune:
                        clean_X, tokenized_X = self._amp_batch_to_model_inputs(X)
                    else:
                        clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)
                        if self.config.use_compress:
                            tokens = tokenized_X["input_ids"]
                            mask = tokenized_X["attention_mask"]
                            clean_X = trim_or_pad_batch_first(clean_X, pad_to=self.config.data.max_sequence_len, pad_idx=0)
                            if mask.shape[1] != clean_X.shape[1]:
                                # pad with False
                                mask = trim_or_pad_batch_first(mask, clean_X.shape[1], pad_idx=0)
                                tokens = trim_or_pad_batch_first(tokens, clean_X.shape[1], pad_idx=1)
                            clean_X = clean_X.to(self.device)
                            mask = mask.to(self.device)
                            tokens = tokens.to(self.device)
                            clean_X = self.enc_normalizer.minmax_scaling(clean_X)
                            z_q, downsampled_mask = self.compresser.encode(x = clean_X, mask = mask.bool(), verbose=self.config.compress.verbose)
                            clean_X = z_q
                            tokenized_X = {"input_ids":tokens, "attention_mask":downsampled_mask}

                    loss_dict, _ = self.calc_loss(clean_x=clean_X, X=tokenized_X)
                    for k, v in loss_dict.items():
                        if k in valid_loss:
                            valid_loss[k] += v.item() * clean_X.size(0)
                        else:
                            valid_loss[k] = torch.Tensor([v.item() * clean_X.size(0)])
                    valid_count += clean_X.size(0)

        valid_count = reduce_tensor(valid_count.to(self.device)) if self.config.ddp else valid_count.to(self.device)
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].to(self.device)) if self.config.ddp else valid_loss[k].to(self.device)

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        if self.config.wandb and _dist_rank() == 0:
            for k, v in valid_loss.items():
                self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_model.train(prev_mode)
        return valid_loss

    def _update_early_stopping(self, valid_loss: Dict[str, torch.Tensor]) -> bool:
        patience = int(getattr(self.config.training, "early_stopping_patience", 0) or 0)
        min_delta = float(getattr(self.config.training, "early_stopping_min_delta", 0.0) or 0.0)
        if patience <= 0:
            return False

        metric_key = "total_loss" if "total_loss" in valid_loss else "loss"
        metric_value = valid_loss[metric_key]
        if isinstance(metric_value, torch.Tensor):
            metric_value = float(metric_value.item())
        else:
            metric_value = float(metric_value)

        improved = metric_value < (self.best_valid_total_loss - min_delta)
        if improved:
            self.best_valid_total_loss = metric_value
            self.best_valid_step = self.step
            self.stale_validation_evals = 0
        else:
            self.stale_validation_evals += 1

        if _dist_rank() == 0:
            status = "<- best" if improved else (
                f"stale={self.stale_validation_evals}/{patience}"
            )
            print(
                f"[Valid @ step {self.step}] {metric_key}={metric_value:0.6f} {status}"
            )
            writer.add_scalar("Valid/early_stop_metric", metric_value, self.step)
            writer.add_scalar("Valid/stale_evals", self.stale_validation_evals, self.step)

        return self.stale_validation_evals >= patience



    def save_checkpoint(self, last: bool = False) -> None:
        if _dist_rank() == 0:
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)

            prefix = ''
            if self.config.checkpoints_prefix:
                prefix = self.config.checkpoints_prefix + '_'
            if last:
                prefix = prefix + 'last_'
            else:
                prefix = prefix + str(self.step) + '_'

            torch.save(
                {   
                    "model": self.model.state_dict(),
                    "text_encoder": None if self.text_encoder is None else self.text_encoder.state_dict(),
                    "null_conditioning": self.null_conditioning.state_dict(),
                    #"decoder": self.decoder.state_dict(), #no self.decoder exists
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                    "best_valid_total_loss": self.best_valid_total_loss,
                    "best_valid_step": self.best_valid_step,
                    "stale_validation_evals": self.stale_validation_evals,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save model to: {os.path.join(self.checkpoints_folder, prefix + '.pth')}")



    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        if self.config.refresh.use_pretrain:
            model_state = load.get("model")
            if model_state is None:
                raise KeyError(f"Pretrain checkpoint has no `model` state: {self.config.refresh.prefix}")
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"Pretrain init missing keys: {len(missing)}")
                print(missing[:20])
            if unexpected:
                print(f"Pretrain init unexpected keys: {len(unexpected)}")
                print(unexpected[:20])
            self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
            self.ema.to(self.device)
        else:
            if load.get("model") is not None:
                self.model.load_state_dict(load["model"], strict=False)
            self.ema.load_state_dict(load["ema"])
            self.ema.to(self.device)

        if self.text_encoder is not None and load.get("text_encoder") is not None:
            self.text_encoder.load_state_dict(load["text_encoder"])
        if load.get("null_conditioning") is not None:
            self.null_conditioning.load_state_dict(load["null_conditioning"])

        if not self.config.refresh.use_pretrain:
            self.optimizer.load_state_dict(load["optimizer"])
            self.scheduler.load_state_dict(load["scheduler"])
            self.step = load["step"]
            self.best_valid_total_loss = load.get("best_valid_total_loss", self.best_valid_total_loss)
            self.best_valid_step = load.get("best_valid_step", self.best_valid_step)
            self.stale_validation_evals = load.get("stale_validation_evals", self.stale_validation_evals)
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")


    def generate_text(self, batch_size, conditioning=None, guidance_scale=None):
        lens = self.length_sampler.sample(batch_size)
        attention_mask = torch.zeros((batch_size, self.config.data.max_sequence_len))
        for i in range(batch_size):
            for j in range(lens[i]):
                attention_mask[i, j] = 1

        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            ### self.pred_embeddings Need to write with the new model
            if self.config.fm.use_mask:
                pred_embeddings = self.pred_embeddings(batch_size, attention_mask, conditioning=conditioning, guidance_scale=guidance_scale)
            else:
                pred_embeddings = self.pred_embeddings(batch_size, conditioning=conditioning, guidance_scale=guidance_scale)
            
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                pred_embedding_z0, pred_embedding_z1 = pred_embeddings
                #print(pred_embeddings)
                output = pred_embeddings#(pred_embedding_z0, self.pred_logits(pred_embedding_z1, attention_mask))
                #print(output)
            else:
                output = self.pred_logits(pred_embeddings, attention_mask)
        return output

    def pred_logits(self, pred_embeddings, attention_mask):
        if self.config.use_compress:
            x_recons = self.compresser.decode(pred_embeddings, attention_mask, self.config.compress.verbose)
            x_recons_unscaled = self.enc_normalizer.undo_minmax_scaling(x_recons)
            output = self.encoder_decoder.batch_decode(x_recons_unscaled, attention_mask=attention_mask)
        else:
            output = self.encoder_decoder.batch_decode(pred_embeddings, attention_mask=attention_mask)
        return output
    
    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            attention_mask=None,
            conditioning=None,
            guidance_scale=None,
    ) -> torch.Tensor:
        if self.config.use_compress:
            shape = (batch_size, self.config.data.max_sequence_len, self.config.model.compressed_hidden_size)
        else:
            shape = (batch_size, self.config.data.max_sequence_len, self.config.model.hidden_size)

        with torch.no_grad():
            noise = (torch.randn(shape, device=self.device) * torch.tensor(self.config.fm.sample_std, device=self.device)).to(self.device)
            use_cfg = conditioning is not None and self.use_text_conditioning
            guidance_scale = self.config.text.guidance_scale if guidance_scale is None else guidance_scale

            if use_cfg:
                x = noise
                t_span = torch.linspace(0, 1, self.config.sampling_step + 1, device=self.device)
                null_conditioning = self.null_conditioning(batch_size, x.shape[1], self.device, x.dtype)
                null_mask = self._get_extended_attention_mask(null_conditioning.attention_mask, x.dtype)
                cond_mask = self._get_extended_attention_mask(conditioning["attention_mask"], x.dtype)
                label_cls = self._make_label_cls(conditioning, batch_size, x.shape[1], x.dtype)
                attribute_cls = self._make_attribute_cls(conditioning, batch_size, x.shape[1], x.dtype)
                direct_cls = label_cls
                if attribute_cls is not None:
                    direct_cls = attribute_cls if direct_cls is None else direct_cls + attribute_cls
                for step_id in range(self.config.sampling_step):
                    t_now = torch.full((batch_size,), t_span[step_id], device=self.device, dtype=x.dtype)
                    dt = t_span[step_id + 1] - t_span[step_id]
                    v_uncond = self.ddp_model(
                        x_t=x, time_t=t_now, attention_mask=attention_mask, cls=direct_cls,
                        token_condition=null_conditioning.token_condition,
                        token_condition_mask=null_mask,
                        global_condition=null_conditioning.global_condition,
                    )
                    v_cond = self.ddp_model(
                        x_t=x, time_t=t_now, attention_mask=attention_mask, cls=direct_cls,
                        token_condition=conditioning["token_condition"],
                        token_condition_mask=cond_mask,
                        global_condition=conditioning["global_condition"],
                    )
                    x = x + dt * (v_uncond + guidance_scale * (v_cond - v_uncond))
                pred_embeddings = x
            else:
                t_span = torch.linspace(0, 1, self.config.sampling_step + 1, device=self.device)
                x = noise
                traj = [x]
                for step_id in range(self.config.sampling_step):
                    t_now = torch.full((batch_size,), t_span[step_id], device=self.device, dtype=x.dtype)
                    dt = t_span[step_id + 1] - t_span[step_id]
                    v = self.ddp_model(x_t=x, time_t=t_now, attention_mask=attention_mask, cls=None,
                                       token_condition=None, token_condition_mask=None, global_condition=None)
                    x = x + dt * v
                    traj.append(x)
                pred_embeddings = (traj[0], traj[-1]) if self.config.fm.reflow and self.config.fm.reflow_generate_data else traj[-1]
        return pred_embeddings

    @torch.no_grad()
    def estimation(self) -> None:
        self.model.eval()
        self.switch_to_ema()
        
        num_texts = int(self.config.validation.num_gen_texts / _dist_world_size())
        if _dist_rank() < self.config.validation.num_gen_texts % _dist_world_size():
            num_texts += 1

        seed = self.config.seed + _dist_rank()
        set_seed(seed)
        conditioning = None
        prompts = []
        if self.is_amp_finetune and self.use_text_conditioning:
            sample_records = [self.valid_dataset[i] for i in range(min(num_texts, len(self.valid_dataset)))]
            prompts = [item["text"] for item in sample_records]
            conditioning = {"text": prompts}
            num_texts = len(sample_records)
            conditioning, _ = self._prepare_conditioning(
                conditioning,
                batch_size=num_texts,
                seq_len=self.config.data.max_sequence_len,
                dtype=next(self.model.parameters()).dtype,
            )
            # Copy direct condition fields into validation generation.
            for key in AMP_CONDITION_KEYS:
                if sample_records and key in sample_records[0]:
                    conditioning[key] = torch.as_tensor([item[key] for item in sample_records], device=self.device, dtype=torch.long)
        output = self.generate_text(batch_size=num_texts, conditioning=conditioning)

        result = [{"protein": p} for p in output]
        if conditioning is not None:
            for item, prompt in zip(result, prompts):
                item["prompt"] = prompt
        if self.config.ddp:
            result = gather_texts(result)

        if not self.config.ddp or _dist_rank() == 0:
            texts_path = "./generated_seqs/" + self.suffix
            os.makedirs(texts_path, exist_ok=True)

            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
            json.dump(result, open(file_name, "w"), indent=4)
            print(file_name)

            fid_value = None
            if not self.is_amp_finetune and calculate_fid_for_files is not None:
                fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
                print(f"FID: {fid_value:0.5f}")
                with open("FID.txt","a") as f:
                    f.write(f"FID: {fid_value:0.5f}"+"\n")

        if fid_value is not None and self.config.wandb and self.config.ddp and _dist_rank() == 0:
            self.log_metric(metric_name="FID", loader_name="", value=fid_value)
        if fid_value is not None and _dist_rank() == 0:
            writer.add_scalar('Valid/FID', fid_value, self.step)
            
        self.switch_back_from_ema()
        self.model.train()

    @torch.no_grad()
    def test(self, ckpt, test_num=None) -> None:

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        load = torch.load(ckpt, map_location="cpu")
        ema_ckpt = load["ema"]
        self.ema.load_state_dict(ema_ckpt)
        self.ema.to(self.device)
        self.switch_to_ema()
        if self.text_encoder is not None and load.get("text_encoder") is not None:
            self.text_encoder.load_state_dict(load["text_encoder"])
        if load.get("null_conditioning") is not None:
            self.null_conditioning.load_state_dict(load["null_conditioning"])
        self.model.eval()
        print("Load model checkpoints DONE.")

        num_texts = int(self.config.validation.num_gen_texts / _dist_world_size())
        if _dist_rank() < self.config.validation.num_gen_texts % _dist_world_size():
            num_texts += 1

        seed = self.config.seed + _dist_rank()
        set_seed(seed)

        print("Start generating proteins...")
        torch.cuda.empty_cache()
        batch = 1024
        filtered_output = []
        z0_train = []
        z0_valid= []
        z1_train = []
        z1_valid= []
        cnt = 0
        total_batches = math.ceil(num_texts / batch)
        for i in range(total_batches):
            batch_size = min(batch, num_texts - i * batch)
            print("batch", i + 1, "/", total_batches)
            output = self.generate_text(batch_size)
            torch.cuda.empty_cache()
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                z0s, z1s = output
                for z0, z1 in zip(z0s, z1s):
                    if cnt < 5270:
                        z0_valid.append(z0.cpu())
                        z1_valid.append(z1.cpu())
                    else:
                        z0_train.append(z0.cpu())
                        z1_train.append(z1.cpu())
                    cnt += 1
            else:
                for p in output:
                    if len(p) < 2: #4
                        continue
                    else:
                        new_p = ""
                        for aa in p:
                            if aa in "AFCUDNEQGHLIKOMPRSTVWY":
                                new_p += aa

                        filtered_output.append(new_p)
        result = [{"protein": p} for p in filtered_output]

        if self.config.ddp:
            result = gather_texts(result)
            if self.config.fm.reflow:
                z0_train = gather_texts(z0_train)
                z1_train = gather_texts(z1_train)
                z0_valid = gather_texts(z0_valid)
                z1_valid = gather_texts(z1_valid)
        
        if not self.config.ddp or _dist_rank() == 0:
            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                texts_path = f"./generated_seqs/{self.suffix}/reflow"
            else:
                texts_path = f"./generated_seqs/{self.suffix}"
            os.makedirs(texts_path, exist_ok=True)

            if self.config.fm.reflow and self.config.fm.reflow_generate_data:
                z0_train_file_name = f"{texts_path}/z0_train"+str(test_num)+".npy"
                z1_train_file_name = f"{texts_path}/z1_train"+str(test_num)+".npy"
                z0_valid_file_name = f"{texts_path}/z0_valid.npy"
                z1_valid_file_name = f"{texts_path}/z1_valid.npy"
                print("Saving to files ...")
                np.save(z0_valid_file_name, np.array(z0_valid))
                np.save(z1_valid_file_name, np.array(z1_valid))
                print("Files saved.")
            else:
                file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
                json.dump(result, open(file_name, "w"), indent=4)
                print(file_name)

                fasta_file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.fasta"
                with open(fasta_file_name, "w") as f:
                    cnt = 0
                    for i in result:
                        #if cnt < 1000: #1000
                            #print(i)
                        f.write(">Seq"+str(cnt)+"\n")
                        f.write(i["protein"]+"\n")
                        #else:
                        #    break
                        cnt += 1

                torch.cuda.empty_cache()
                if calculate_fid_for_files is not None:
                    fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
                    print(f"FID: {fid_value:0.5f}")

    @torch.no_grad()
    def test_encoder_decoder(self) -> None:
        self.set_valid_data_generator()
        gathered_output = []
        for X in self.valid_loader:
            X = dict_to_cuda(X)
            pred_embeddings, tokenized_X = self.encoder_decoder.batch_encode(X)
            attention_mask = tokenized_X["attention_mask"]
            output = self.pred_logits(pred_embeddings, attention_mask)
            for seq in output:
                gathered_output.append(seq)

        result = [{"protein": p} for p in gathered_output]

        if self.config.ddp:
            result = gather_texts(result)

        if not self.config.ddp or _dist_rank() == 0:
            texts_path = "./generated_seqs/test"
            os.makedirs(texts_path, exist_ok=True)

            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{len(result)}.json"
            json.dump(result, open(file_name, "w"), indent=4)
            if calculate_fid_for_files is not None:
                fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
                print(f"FID: {fid_value:0.5f}")
