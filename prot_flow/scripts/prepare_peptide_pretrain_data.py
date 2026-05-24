from __future__ import annotations

import argparse
import random
from pathlib import Path


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str, min_len: int, max_len: int) -> str | None:
    seq = seq.strip().upper()
    if not seq:
        return None
    if len(seq) < min_len or len(seq) > max_len:
        return None
    if any(ch not in ALLOWED_AA for ch in seq):
        return None
    return seq


def iter_fasta_records(path: Path):
    header = None
    chunks = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic train/valid FASTA splits for peptide pretraining.")
    parser.add_argument("--input", required=True, help="Path to the source FASTA file.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.995, help="Fraction assigned to train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument("--min-len", type=int, default=5, help="Minimum allowed peptide length.")
    parser.add_argument("--max-len", type=int, default=50, help="Maximum allowed peptide length.")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.fasta"
    valid_path = out_dir / "valid.fasta"
    stats_path = out_dir / "stats.txt"

    rng = random.Random(args.seed)
    train_count = 0
    valid_count = 0
    skipped_count = 0

    with train_path.open("w", encoding="utf-8") as train_out, \
            valid_path.open("w", encoding="utf-8") as valid_out:
        for index, (_, sequence) in enumerate(iter_fasta_records(source), start=1):
            cleaned = clean_sequence(sequence, args.min_len, args.max_len)
            if cleaned is None:
                skipped_count += 1
                continue

            header = f">pep_{index}"
            if rng.random() < args.train_ratio:
                train_out.write(f"{header}\n{cleaned}\n")
                train_count += 1
            else:
                valid_out.write(f"{header}\n{cleaned}\n")
                valid_count += 1

    stats_path.write_text(
        "\n".join(
            [
                f"input={source}",
                f"train={train_count}",
                f"valid={valid_count}",
                f"skipped={skipped_count}",
                f"seed={args.seed}",
                f"min_len={args.min_len}",
                f"max_len={args.max_len}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved train split to: {train_path}")
    print(f"Saved valid split to: {valid_path}")
    print(f"Saved stats to: {stats_path}")
    print({"train": train_count, "valid": valid_count, "skipped": skipped_count})


if __name__ == "__main__":
    main()
