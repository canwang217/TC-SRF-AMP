import argparse
import logging
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import create_config
from compressors.hourglass import HourglassProteinCompressionTransformer, trim_or_pad_batch_first
from encoders import EncNormalizer, ESM2EncoderModel
from utils import load_fasta_file, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProtFlow compressor with early stopping.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--log-file", type=str, default="train_compressor.log")
    parser.add_argument("--disable-tensorboard", action="store_true")
    return parser.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(config, batch_size, num_workers):
    train_dataset = load_fasta_file(config.data.train_dataset_path)
    valid_dataset = load_fasta_file(config.data.test_dataset_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def build_checkpoint_path(config, checkpoint_dir):
    root = checkpoint_dir or os.path.join(config.paths.workspace_root, "checkpoints", "compressor")
    os.makedirs(root, exist_ok=True)
    filename = f"compressor-{config.model.hg_name_hash}-{config.data.dataset}.pth"
    return root, os.path.join(root, filename)


def build_log_paths(config, checkpoint_dir, log_dir, log_file):
    root = log_dir or os.path.join(checkpoint_dir or os.path.join(config.paths.workspace_root, "checkpoints", "compressor"), "logs")
    os.makedirs(root, exist_ok=True)
    return root, os.path.join(root, log_file)


def setup_logger(log_file):
    logger = logging.getLogger("train_compressor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_writer(log_dir, disabled):
    if disabled or SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=log_dir)


def compressor_state_dict(model):
    state = {
        "encoder": model.enc.state_dict(),
        "decoder": model.dec.state_dict(),
    }
    if model.quantize_scheme in {"vq", "fsq"} and model.quantizer is not None:
        state["quantizer"] = (
            model.quantizer.state_dict(),
            model.fsq_levels,
            getattr(model, "implicit_codebook", None),
        )
        state["pre_quant_proj"] = None if model.pre_quant_proj is None else model.pre_quant_proj.state_dict()
        state["post_quant_proj"] = None if model.post_quant_proj is None else model.post_quant_proj.state_dict()
    return state


def save_checkpoint(model, optimizer, epoch, valid_loss, checkpoint_path, best_valid_loss=None, best_epoch=None, stale_epochs=None):
    payload = compressor_state_dict(model)
    payload["optimizer"] = optimizer.state_dict()
    payload["epoch"] = epoch
    payload["valid_loss"] = valid_loss
    if best_valid_loss is not None:
        payload["best_valid_loss"] = best_valid_loss
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if stale_epochs is not None:
        payload["stale_epochs"] = stale_epochs
    torch.save(payload, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.enc.load_state_dict(checkpoint["encoder"])
    model.dec.load_state_dict(checkpoint["decoder"])

    if model.quantize_scheme in {"vq", "fsq"} and "quantizer" in checkpoint and checkpoint["quantizer"] is not None:
        model.fsq_levels = checkpoint["quantizer"][1]
        model.implicit_codebook = checkpoint["quantizer"][2]
        model.set_up_quantizer(
            model.dim,
            model.downproj_factor,
            model.n_e,
            model.e_dim,
            model.vq_beta,
            model.enforce_single_codebook_per_position,
            model.fsq_levels,
            model.implicit_codebook,
        )
        model.quantizer.load_state_dict(checkpoint["quantizer"][0])
        if model.pre_quant_proj is not None and checkpoint.get("pre_quant_proj") is not None:
            model.pre_quant_proj.load_state_dict(checkpoint["pre_quant_proj"])
        if model.post_quant_proj is not None and checkpoint.get("post_quant_proj") is not None:
            model.post_quant_proj.load_state_dict(checkpoint["post_quant_proj"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def prepare_batch(sequences, encoder, normalizer, compressor, config, device):
    with torch.no_grad():
        embeddings, tokenized = encoder.batch_encode(sequences)

    tokens = tokenized["input_ids"]
    mask = tokenized["attention_mask"]
    embeddings = trim_or_pad_batch_first(embeddings, pad_to=config.data.max_sequence_len, pad_idx=0)
    if mask.shape[1] != embeddings.shape[1]:
        mask = trim_or_pad_batch_first(mask, embeddings.shape[1], pad_idx=0)
        tokens = trim_or_pad_batch_first(tokens, embeddings.shape[1], pad_idx=1)

    embeddings = embeddings.to(device)
    mask = mask.to(device).bool()
    tokens = tokens.to(device)

    embeddings = normalizer.minmax_scaling(embeddings)
    z_q, downsampled_mask = compressor.encode(x=embeddings, mask=mask, verbose=False)
    reconstructed = compressor.decode(z_q, downsampled_mask, verbose=False)
    reconstructed = trim_or_pad_batch_first(reconstructed, pad_to=embeddings.shape[1], pad_idx=0)
    if downsampled_mask.shape[1] != z_q.shape[1]:
        downsampled_mask = trim_or_pad_batch_first(downsampled_mask, z_q.shape[1], pad_idx=0)

    return embeddings, reconstructed, mask, tokens


def reconstruction_loss(target, prediction, mask):
    mask = mask.unsqueeze(-1).to(prediction.dtype)
    squared_error = (prediction - target) ** 2
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def run_epoch(loader, encoder, normalizer, compressor, optimizer, config, device, train_mode, accum_steps=1):
    compressor.train(train_mode)
    losses = []
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    if train_mode:
        optimizer.zero_grad()

    for batch_idx, sequences in enumerate(tqdm(loader, leave=False), start=1):
        with autocast_context:
            target, reconstructed, mask, _ = prepare_batch(
                sequences=sequences,
                encoder=encoder,
                normalizer=normalizer,
                compressor=compressor,
                config=config,
                device=device,
            )
            loss = reconstruction_loss(target, reconstructed, mask)
            raw_loss = loss.detach().item()
            if train_mode:
                loss = loss / accum_steps

        if train_mode:
            loss.backward()
            if batch_idx % accum_steps == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(raw_loss)
        else:
            losses.append(raw_loss)

    return sum(losses) / max(len(losses), 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = create_config()
    device = get_device()

    checkpoint_dir, checkpoint_path = build_checkpoint_path(config, args.checkpoint_dir)
    latest_path = os.path.join(checkpoint_dir, "latest-compressor.pth")
    tb_log_dir, log_file = build_log_paths(config, args.checkpoint_dir, args.log_dir, args.log_file)
    logger = setup_logger(log_file)
    writer = setup_writer(tb_log_dir, args.disable_tensorboard)

    logger.info(f"Using device: {device}")
    logger.info(f"Saving checkpoints to: {checkpoint_dir}")
    logger.info(f"Logging to file: {log_file}")
    logger.info(
        f"Batch size: {args.batch_size} | Accum steps: {args.accum_steps} | "
        f"Effective batch size: {args.batch_size * args.accum_steps}"
    )
    if writer is not None:
        logger.info(f"TensorBoard log dir: {tb_log_dir}")

    normalizer = EncNormalizer(
        enc_mean_path=config.data.enc_mean,
        enc_std_path=config.data.enc_std,
        enc_max_path=config.data.enc_max,
        enc_min_path=config.data.enc_min,
    ).to(device)

    encoder = ESM2EncoderModel(
        config.model.hg_name,
        device=str(device),
        decoder_path=None,
        max_seq_len=config.data.max_sequence_len,
        enc_normalizer=normalizer,
    )

    compressor = HourglassProteinCompressionTransformer(
        dim=config.model.hidden_size,
        depth=config.compress.depth,
        downproj_factor=config.compress.downproj_factor,
        shorten_factor=config.compress.shorten_factor,
        attn_resampling=config.compress.attn_resampling,
        updown_sample_type=config.compress.updown_sample_type,
        heads=config.compress.heads,
        dim_head=config.compress.dim_head,
        causal=config.compress.causal,
        norm_out=config.compress.norm_out,
        use_quantizer=config.compress.use_quantizer,
        n_e=config.compress.n_e,
        e_dim=config.compress.e_dim,
        vq_beta=config.compress.vq_beta,
        enforce_single_codebook_per_position=config.compress.enforce_single_codebook_per_position,
        fsq_levels=config.compress.fsq_levels,
        device=str(device),
    ).to(device)

    optimizer = torch.optim.AdamW(
        compressor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    train_loader, valid_loader = get_loaders(config, args.batch_size, args.num_workers)

    best_valid_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 1

    resume_path = args.resume_path or latest_path
    if args.resume and os.path.exists(resume_path):
        checkpoint = load_checkpoint(compressor, optimizer, resume_path, device)
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", checkpoint.get("valid_loss", float("inf")))
        best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", -1))
        stale_epochs = checkpoint.get("stale_epochs", 0)
        logger.info(
            f"Resumed from: {resume_path} | start_epoch={start_epoch}, "
            f"best_valid_loss={best_valid_loss:.6f}, best_epoch={best_epoch}, stale_epochs={stale_epochs}"
        )
    elif args.resume:
        logger.warning(f"Resume requested but checkpoint not found: {resume_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            loader=train_loader,
            encoder=encoder,
            normalizer=normalizer,
            compressor=compressor,
            optimizer=optimizer,
            config=config,
            device=device,
            train_mode=True,
            accum_steps=args.accum_steps,
        )
        valid_loss = run_epoch(
            loader=valid_loader,
            encoder=encoder,
            normalizer=normalizer,
            compressor=compressor,
            optimizer=optimizer,
            config=config,
            device=device,
            train_mode=False,
            accum_steps=1,
        )

        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        improved = valid_loss < (best_valid_loss - args.min_delta)
        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                compressor,
                optimizer,
                epoch,
                valid_loss,
                checkpoint_path,
                best_valid_loss=best_valid_loss,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
            )
            logger.info(f"[Epoch {epoch}] lr={current_lr:.2e} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} <- best")
        else:
            stale_epochs += 1
            logger.info(f"[Epoch {epoch}] lr={current_lr:.2e} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} stale={stale_epochs}/{args.patience}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/valid", valid_loss, epoch)
            writer.add_scalar("optim/lr", current_lr, epoch)
            writer.add_scalar("train/stale_epochs", stale_epochs, epoch)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                compressor,
                optimizer,
                epoch,
                valid_loss,
                latest_path,
                best_valid_loss=best_valid_loss,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
            )

        if stale_epochs >= args.patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best validation loss: {best_valid_loss:.6f} at epoch {best_epoch}."
            )
            break

    logger.info(f"Best checkpoint: {checkpoint_path}")
    logger.info(f"Best validation loss: {best_valid_loss:.6f} at epoch {best_epoch}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
