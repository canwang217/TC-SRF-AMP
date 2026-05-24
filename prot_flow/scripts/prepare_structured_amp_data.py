from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_AA = set("AVILMFWYC")
BASIC_AA = set("KR")
ACIDIC_AA = set("DE")


def clean_sequence(sequence: str, min_len: int, max_len: int) -> str | None:
    if sequence is None:
        return None
    seq = str(sequence).strip().upper()
    if len(seq) < min_len or len(seq) > max_len:
        return None
    if any(aa not in ALLOWED_AA for aa in seq):
        return None
    return seq


def infer_activity(text: str) -> str:
    return "inactive" if "inactive" in str(text).lower() else "active"


def infer_target(text: str) -> str:
    lower = str(text).lower()
    if "gram-negative" in lower or "gram negative" in lower:
        return "gram_negative"
    if "gram-positive" in lower or "gram positive" in lower:
        return "gram_positive"
    if "fung" in lower or "candida" in lower:
        return "fungal"
    if "bacteria" in lower or "bacterial" in lower:
        return "bacterial"
    return "unknown"


def infer_toxicity(text: str) -> str:
    lower = str(text).lower()
    if "low toxicity" in lower or "non-toxic" in lower or "nontoxic" in lower or "low hemolysis" in lower:
        return "low"
    if "high toxicity" in lower or "toxic" in lower or "hemolytic" in lower:
        return "high"
    return "unknown"


def length_bin(length: int) -> str:
    if length <= 10:
        return "very_short"
    if length <= 20:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


def charge_value(sequence: str) -> float:
    return (
        sum(sequence.count(aa) for aa in BASIC_AA)
        + 0.1 * sequence.count("H")
        - sum(sequence.count(aa) for aa in ACIDIC_AA)
    )


def charge_bin(charge: float) -> str:
    if charge <= -1:
        return "negative"
    if charge < 2:
        return "neutral"
    if charge < 5:
        return "positive"
    return "high_positive"


def ratio_bin(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def cys_bin(count: int) -> str:
    if count == 0:
        return "none"
    if count <= 2:
        return "low"
    return "high"


def sequence_attributes(sequence: str) -> dict:
    length = len(sequence)
    charge = charge_value(sequence)
    kr_frac = sum(sequence.count(aa) for aa in BASIC_AA) / length
    hydrophobic_frac = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA) / length
    cys = sequence.count("C")
    return {
        "length": length,
        "length_bin": length_bin(length),
        "net_charge": round(charge, 3),
        "charge_bin": charge_bin(charge),
        "kr_frac": round(kr_frac, 4),
        "kr_bin": ratio_bin(kr_frac, 0.10, 0.25),
        "hydrophobic_frac": round(hydrophobic_frac, 4),
        "hydrophobicity_bin": ratio_bin(hydrophobic_frac, 0.35, 0.55),
        "cys_count": cys,
        "cys_bin": cys_bin(cys),
    }


def make_prompt(activity: str, target: str, toxicity: str, attrs: dict, include_numeric: bool) -> str:
    parts = [
        f"activity: {activity}",
        f"target: {target}",
        f"length_bin: {attrs['length_bin']}",
        f"charge: {attrs['charge_bin']}",
        f"kr_ratio: {attrs['kr_bin']}",
        f"hydrophobicity: {attrs['hydrophobicity_bin']}",
        f"cys: {attrs['cys_bin']}",
        f"toxicity: {toxicity}",
    ]
    if include_numeric:
        parts.extend(
            [
                f"length: {attrs['length']}",
                f"net_charge: {attrs['net_charge']}",
                f"kr_frac: {attrs['kr_frac']}",
                f"hydrophobic_frac: {attrs['hydrophobic_frac']}",
                f"cys_count: {attrs['cys_count']}",
            ]
        )
    return "; ".join(parts)


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> dict:
    counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sequence = clean_sequence(obj.get("trg"), args.min_len, args.max_len)
            if sequence is None:
                counts["skipped"] += 1
                continue

            original_src = str(obj.get("src", ""))
            activity = infer_activity(original_src)
            target = infer_target(original_src)
            toxicity = infer_toxicity(original_src)
            attrs = sequence_attributes(sequence)
            prompt = make_prompt(activity, target, toxicity, attrs, args.include_numeric)

            record = {
                "src": prompt,
                "trg": sequence,
                "original_src": original_src,
                "activity": activity,
                "target": target,
                "toxicity": toxicity,
                **attrs,
            }
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")
            counts["kept"] += 1
            counts[f"activity:{activity}"] += 1
            counts[f"target:{target}"] += 1
            counts[f"length_bin:{attrs['length_bin']}"] += 1
            counts[f"charge:{attrs['charge_bin']}"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create structured attribute prompts from AMP TG-style JSONL data.")
    parser.add_argument("--train-input", default="../ensemble_train.jsonl")
    parser.add_argument("--valid-input", default="../ensemble_test.jsonl")
    parser.add_argument("--train-output", default="../ensemble_train_structured.jsonl")
    parser.add_argument("--valid-output", default="../ensemble_test_structured.jsonl")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--include-numeric", action="store_true", help="Include exact numeric attributes in prompts.")
    parser.add_argument("--summary-json", default="../structured_amp_summary.json")
    args = parser.parse_args()

    summary = {
        "train": convert_file(Path(args.train_input), Path(args.train_output), args),
        "valid": convert_file(Path(args.valid_input), Path(args.valid_output), args),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"train_output={args.train_output}")
    print(f"valid_output={args.valid_output}")
    print(f"summary={args.summary_json}")


if __name__ == "__main__":
    main()
