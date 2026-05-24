from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.nl_condition_parser import parse_condition


DEFAULT_CKPT = "checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth"
BASIC_AA = set("KR")
ACIDIC_AA = set("DE")
HYDROPHOBIC_AA = set("AVILMFWYC")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")[:80] or "prompt"


def length_bin(length: int) -> str:
    if length <= 10:
        return "very_short"
    if length <= 20:
        return "short"
    if length <= 35:
        return "medium"
    return "long"


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


def sequence_metrics(sequence: str) -> dict:
    seq = str(sequence).strip().upper()
    length = len(seq)
    net_charge = (
        sum(seq.count(aa) for aa in BASIC_AA)
        + 0.1 * seq.count("H")
        - sum(seq.count(aa) for aa in ACIDIC_AA)
    )
    kr_frac = sum(1 for aa in seq if aa in BASIC_AA) / max(1, length)
    hydrophobic_frac = sum(1 for aa in seq if aa in HYDROPHOBIC_AA) / max(1, length)
    cys_count = seq.count("C")
    return {
        "length": length,
        "net_charge": round(net_charge, 3),
        "kr_frac": round(kr_frac, 4),
        "hydrophobic_frac": round(hydrophobic_frac, 4),
        "cys_count": cys_count,
        "length_bin_observed": length_bin(length),
        "charge_bin_observed": charge_bin(net_charge),
        "kr_bin_observed": ratio_bin(kr_frac, 0.10, 0.25),
        "hydrophobicity_bin_observed": ratio_bin(hydrophobic_frac, 0.35, 0.55),
        "cys_bin_observed": cys_bin(cys_count),
    }


def add_match_flags(record: dict) -> dict:
    record["length_match"] = record["length_bin_observed"] == record["length_bin"]
    record["charge_match"] = record["charge_bin_observed"] == record["charge_bin"]
    record["kr_match"] = record["kr_bin_observed"] == record["kr_bin"]
    record["hydrophobicity_match"] = record["hydrophobicity_bin_observed"] == record["hydrophobicity_bin"]
    record["cys_match"] = record["cys_bin_observed"] == record["cys_bin"]
    record["all_property_match"] = all(
        record[key]
        for key in (
            "length_match",
            "charge_match",
            "kr_match",
            "hydrophobicity_match",
            "cys_match",
        )
    )
    return record


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    numeric_keys = ["length", "net_charge", "kr_frac", "hydrophobic_frac", "cys_count"]
    flag_keys = [
        "length_match",
        "charge_match",
        "kr_match",
        "hydrophobicity_match",
        "cys_match",
        "all_property_match",
    ]
    summary = {
        key: round(sum(float(record[key]) for record in records) / len(records), 4)
        for key in numeric_keys
    }
    summary.update(
        {
            f"{key}_rate": round(sum(1 for record in records if record[key]) / len(records), 4)
            for key in flag_keys
        }
    )
    summary.update({"n": len(records), "unique": len({record["protein"] for record in records})})
    return summary


def effective_guidance_scale(condition: dict, guidance_scale: float, neutral_guidance_scale: float | None) -> float:
    if neutral_guidance_scale is not None and condition["charge_bin"] == "neutral":
        return neutral_guidance_scale
    return guidance_scale


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AMP sequences from one natural-language prompt using the strict "
            "natural language -> structured text -> T_c/c_g TC-SRF path."
        )
    )
    parser.add_argument("--prompt", required=True, help="Natural-language AMP design prompt.")
    parser.add_argument("--name", default="user_prompt", help="Name used in output metadata and default filename.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--num", type=int, default=32, help="Number of sequences to generate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument(
        "--neutral-guidance-scale",
        type=float,
        default=None,
        help="Optional lower guidance scale when the parsed charge condition is neutral.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--print-examples", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-ema", action="store_true", help="Load raw model weights instead of EMA weights.")
    args = parser.parse_args()
    args.prefer_ema = not args.no_ema

    parsed_condition = parse_condition(args.prompt)
    parsed = parsed_condition.with_ids()
    structured_prompt = parsed_condition.structured_prompt()
    used_guidance_scale = effective_guidance_scale(parsed, args.guidance_scale, args.neutral_guidance_scale)

    from scripts.run_strict_nl_prompt_benchmark import load_runner, prepare_conditioning

    set_seed(args.seed)
    runner, load_mode = load_runner(args)
    conditioning = prepare_conditioning(runner, structured_prompt, args.num)
    sequences = runner.generate_text(
        batch_size=args.num,
        conditioning=conditioning,
        guidance_scale=used_guidance_scale,
    )

    records = []
    for seq in sequences:
        record = {
            "prompt_name": args.name,
            "natural_prompt": args.prompt,
            "structured_prompt": structured_prompt,
            "requested_guidance_scale": args.guidance_scale,
            "used_guidance_scale": used_guidance_scale,
            "seed": args.seed,
            "protein": seq,
            "activity": parsed["activity"],
            "target": parsed["target"],
            "length_bin": parsed["length_bin"],
            "charge_bin": parsed["charge_bin"],
            "kr_bin": parsed["kr_bin"],
            "hydrophobicity_bin": parsed["hydrophobicity_bin"],
            "cys_bin": parsed["cys_bin"],
            "toxicity": parsed["toxicity"],
            **sequence_metrics(seq),
        }
        records.append(add_match_flags(record))

    summary = summarize(records)
    output = {
        "strict_path": {
            "natural_language_parser": True,
            "uses_text_encoder": True,
            "uses_token_condition_Tc": True,
            "uses_global_condition_cg": True,
            "attribute_ids_supplied_to_model": False,
            "label_ids_supplied_to_model": False,
        },
        "load_mode": load_mode,
        "prompt_name": args.name,
        "natural_prompt": args.prompt,
        "structured_prompt": structured_prompt,
        "parsed_condition": parsed,
        "requested_guidance_scale": args.guidance_scale,
        "used_guidance_scale": used_guidance_scale,
        "seed": args.seed,
        "summary": summary,
        "records": records,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("generated_seqs") / "user_prompts" / (
            f"{safe_name(args.name)}_g{args.guidance_scale:g}_seed{args.seed}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"load_mode={load_mode}")
    print(f"structured_prompt={structured_prompt}")
    print(f"output_json={out_path}")
    print("summary=" + json.dumps(summary, ensure_ascii=False))
    print("examples")
    for seq in sequences[: max(args.print_examples, 0)]:
        print(seq)


if __name__ == "__main__":
    main()
