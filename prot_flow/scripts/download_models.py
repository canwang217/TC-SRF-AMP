from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Canana040217/tc-srf-amp-models"
DEFAULT_ALLOW_PATTERNS = [
    "ensemble_train.jsonl",
    "ensemble_test.jsonl",
    "ensemble_train_structured.jsonl",
    "ensemble_test_structured.jsonl",
    "structured_amp_summary.json",
    "prot_flow/checkpoints/**",
    "prot_flow/data/peptides_pretrain/train.fasta",
    "esm2_t12_35M_UR50D/**",
    "scibert_scivocab_uncased/**",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TC-SRF-AMP model and data assets from Hugging Face Hub into the repository layout."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-root", default=None, help="Defaults to the GitHub repository root.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_ALLOW_PATTERNS,
        help="Hugging Face allow_patterns. Defaults to model/checkpoint/data assets.",
    )
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root

    print(f"Downloading {args.repo_id} -> {output_root}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(output_root),
        allow_patterns=args.include,
    )
    print("Model assets downloaded.")


if __name__ == "__main__":
    main()
