from __future__ import annotations

import json
from pathlib import Path

import torch


REQUIRED_MODEL_PATHS = [
    "esm2_t12_35M_UR50D/config.json",
    "esm2_t12_35M_UR50D/pytorch_model.bin",
    "scibert_scivocab_uncased/config.json",
    "scibert_scivocab_uncased/pytorch_model.bin",
    "prot_flow/data/peptides_pretrain/valid.fasta",
    "prot_flow/data/peptides_pretrain/encodings-esm2-35M-mean.pt",
    "prot_flow/data/peptides_pretrain/encodings-esm2-35M-std.pt",
    "prot_flow/data/peptides_pretrain/encodings-esm2-35M-min.pt",
    "prot_flow/data/peptides_pretrain/encodings-esm2-35M-max.pt",
    "prot_flow/checkpoints/decoder-esm2-35M-peptides_pretrain.pth",
    "prot_flow/checkpoints/compressor/compressor-esm2-35M-peptides_pretrain.pth",
    "prot_flow/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth",
    "prot_flow/checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth",
]

OPTIONAL_TRAINING_DATA_PATHS = [
    "ensemble_train.jsonl",
    "ensemble_test.jsonl",
    "ensemble_train_structured.jsonl",
    "ensemble_test_structured.jsonl",
    "prot_flow/data/peptides_pretrain/train.fasta",
]


def count_jsonl(path: Path) -> tuple[int, list[str]]:
    count = 0
    keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not keys:
                keys = sorted(obj.keys())
            count += 1
    return count, keys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    missing = [rel for rel in REQUIRED_MODEL_PATHS if not (root / rel).exists()]
    if missing:
        print("Missing required model files. Run `python prot_flow/scripts/download_models.py` from the repository root:")
        for rel in missing:
            print(f"- {rel}")
        raise SystemExit(1)

    optional_missing = [rel for rel in OPTIONAL_TRAINING_DATA_PATHS if not (root / rel).exists()]
    if optional_missing:
        print("Optional full-training data files are not present:")
        for rel in optional_missing:
            print(f"- {rel}")

    for name in ["ensemble_train.jsonl", "ensemble_test.jsonl"]:
        if not (root / name).exists():
            continue
        count, keys = count_jsonl(root / name)
        print(f"{name}: {count} records, keys={keys}")

    ckpt_path = root / "prot_flow/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth"
    ckpt = torch.load(ckpt_path, map_location="meta")
    print(f"base_flow_step={ckpt.get('step')}")
    print(f"base_flow_best_step={ckpt.get('best_valid_step')}")
    print(f"base_flow_best_loss={ckpt.get('best_valid_total_loss')}")
    print(f"base_flow_text_encoder={ckpt.get('text_encoder') is not None}")
    print("assets_ok")


if __name__ == "__main__":
    main()
