import os

from config import create_config


def create_tc_srf_amp_config():
    config = create_config()
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    default_phase_root = os.path.dirname(workspace_root)

    config.project_name = "TC-SRF-AMP"
    config.checkpoints_prefix = "tc_srf_amp"

    config.data.dataset = "peptides_pretrain"
    config.data.train_dataset_path = "./data/peptides_pretrain/train.fasta"
    config.data.test_dataset_path = "./data/peptides_pretrain/valid.fasta"

    config.data.enc_mean = f"./data/{config.data.dataset}/encodings-{config.model.hg_name_hash}-mean.pt"
    config.data.enc_std = f"./data/{config.data.dataset}/encodings-{config.model.hg_name_hash}-std.pt"
    config.data.enc_max = f"./data/{config.data.dataset}/encodings-{config.model.hg_name_hash}-max.pt"
    config.data.enc_min = f"./data/{config.data.dataset}/encodings-{config.model.hg_name_hash}-min.pt"

    config.training.checkpoints_folder = os.environ.get(
        "TC_SRF_CHECKPOINT_DIR",
        os.path.join(workspace_root, "checkpoints", "tc_srf_amp"),
    )
    config.training.batch_size = int(os.environ.get("TC_SRF_BATCH_SIZE", config.training.batch_size))
    config.training.training_iters = int(os.environ.get("TC_SRF_TRAINING_ITERS", config.training.training_iters))
    config.training.checkpoint_freq = int(os.environ.get("TC_SRF_CHECKPOINT_FREQ", config.training.checkpoint_freq))
    config.training.generate_freq = int(os.environ.get("TC_SRF_GENERATE_FREQ", config.training.generate_freq))
    config.training.eval_freq = int(os.environ.get("TC_SRF_EVAL_FREQ", config.training.eval_freq))
    config.training.eval_on_start = os.environ.get("TC_SRF_EVAL_ON_START", "0") == "1"
    config.training.num_workers = int(os.environ.get("TC_SRF_NUM_WORKERS", 4))
    config.optim.lr = float(os.environ.get("TC_SRF_LR", config.optim.lr))
    config.optim.max_lr = config.optim.lr
    config.optim.min_lr = float(os.environ.get("TC_SRF_MIN_LR", config.optim.min_lr))
    config.optim.linear_warmup = int(os.environ.get("TC_SRF_LINEAR_WARMUP", config.optim.linear_warmup))
    config.validation.batch_size = int(os.environ.get("TC_SRF_VALIDATION_BATCH_SIZE", config.validation.batch_size))
    config.validation.validation_iters = int(os.environ.get("TC_SRF_VALIDATION_ITERS", config.validation.validation_iters))
    config.validation.num_gen_texts = int(os.environ.get("TC_SRF_NUM_GEN_TEXTS", config.validation.num_gen_texts))

    config.text = type(config)()
    config.text.encoder_name = os.path.join(default_phase_root, "scibert_scivocab_uncased")
    config.text.enabled = True
    config.text.max_length = 128
    config.text.dropout_prob = 0.1
    config.text.condition_dim = config.model.hidden_size
    config.text.use_cross_attention = True
    config.text.use_adaptive_norm = True
    config.text.cfg_dropout_prob = float(os.environ.get("TC_SRF_CFG_DROPOUT", 0.1))
    config.text.guidance_scale = float(os.environ.get("TC_SRF_GUIDANCE_SCALE", 2.0))
    config.text.freeze_encoder = os.environ.get("TC_SRF_FREEZE_TEXT_ENCODER", "0") == "1"
    config.text.use_label_condition = os.environ.get("TC_SRF_USE_LABEL_CONDITION", "0") == "1"
    config.text.use_attribute_condition = os.environ.get("TC_SRF_USE_ATTRIBUTE_CONDITION", "0") == "1"
    config.text.guidance_scale = float(os.environ.get("TC_SRF_GUIDANCE_SCALE", 2.0))

    config.loss.cond_coef = float(os.environ.get("TC_SRF_COND_COEF", 0.2))
    config.loss.cc_coef = float(os.environ.get("TC_SRF_CC_COEF", 0.3))
    config.loss.latent_rec_coef = float(os.environ.get("TC_SRF_LATENT_REC_COEF", 0.1))
    config.loss.label_coef = float(os.environ.get("TC_SRF_LABEL_COEF", 0.0))
    config.loss.contrast_coef = float(os.environ.get("TC_SRF_CONTRAST_COEF", 0.0))
    config.loss.contrast_margin = float(os.environ.get("TC_SRF_CONTRAST_MARGIN", 0.05))

    config.task = type(config)()
    config.task.name = "tc_srf_amp"
    config.task.stage = "pretrain"
    config.task.finetune_data_format = "tg_jsonl"

    config.amp = type(config)()
    config.amp.train_jsonl = os.environ.get("TC_SRF_AMP_TRAIN_JSONL", os.path.join(default_phase_root, "ensemble_train.jsonl"))
    config.amp.valid_jsonl = os.environ.get("TC_SRF_AMP_VALID_JSONL", os.path.join(default_phase_root, "ensemble_test.jsonl"))
    config.amp.min_len = 5
    config.amp.max_len = 50

    config.refresh.true = os.environ.get("TC_SRF_REFRESH", "1") == "1"
    config.refresh.use_pretrain = os.environ.get("TC_SRF_USE_PRETRAIN", "1") == "1"
    config.refresh.prefix = os.environ.get(
        "TC_SRF_PRETRAIN_CKPT",
        os.path.join(workspace_root, "checkpoints", "flow_matching", "flow_peptides_pretrain_last_.pth"),
    )

    config.paths = type(config)()
    config.paths.workspace_root = workspace_root
    config.paths.phase_root = default_phase_root
    config.paths.model_name_or_path = os.environ.get("TC_SRF_MODEL_NAME_OR_PATH", config.model.hg_name)
    config.paths.text_encoder_name_or_path = os.environ.get("TC_SRF_TEXT_ENCODER_NAME_OR_PATH", config.text.encoder_name)
    config.paths.decoder_checkpoint = os.environ.get("TC_SRF_DECODER_CKPT", config.decoder_path)
    config.paths.compressor_checkpoint = os.environ.get("TC_SRF_COMPRESSOR_CKPT", config.compress.checkpoint)

    config.model.hg_name = config.paths.model_name_or_path
    config.text.encoder_name = config.paths.text_encoder_name_or_path
    config.decoder_path = config.paths.decoder_checkpoint
    config.compress.checkpoint = config.paths.compressor_checkpoint

    return config


def create_tc_srf_amp_smoke_config():
    config = create_tc_srf_amp_config()
    config.project_name = "TC-SRF-AMP-Smoke"
    config.checkpoints_prefix = "tc_srf_amp_smoke"
    config.training.training_iters = 10
    config.training.checkpoint_freq = 5
    config.training.generate_freq = 5
    config.training.eval_freq = 5
    config.training.batch_size = 2
    config.validation.batch_size = 2
    config.validation.validation_iters = 2
    config.validation.num_gen_texts = 4
    config.sampling_step = 4
    config.text.max_length = 64

    smoke_root = os.path.join(config.paths.workspace_root, "data", "smoke")
    config.data.dataset = "smoke_pretrain"
    config.data.train_dataset_path = os.path.join(smoke_root, "pretrain_train.fasta")
    config.data.test_dataset_path = os.path.join(smoke_root, "pretrain_valid.fasta")
    config.data.enc_mean = os.path.join(smoke_root, f"encodings-{config.model.hg_name_hash}-mean.pt")
    config.data.enc_std = os.path.join(smoke_root, f"encodings-{config.model.hg_name_hash}-std.pt")
    config.data.enc_max = os.path.join(smoke_root, f"encodings-{config.model.hg_name_hash}-max.pt")
    config.data.enc_min = os.path.join(smoke_root, f"encodings-{config.model.hg_name_hash}-min.pt")

    config.amp.train_jsonl = os.path.join(smoke_root, "amp_train.jsonl")
    config.amp.valid_jsonl = os.path.join(smoke_root, "amp_valid.jsonl")
    config.training.checkpoints_folder = os.path.join(config.paths.workspace_root, "checkpoints", "tc_srf_amp_smoke")
    return config
