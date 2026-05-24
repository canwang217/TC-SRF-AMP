from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from flow_matching_utils.flow_matching_holder import FlowMatchingRunner
from model.ema_model import ExponentialMovingAverage
from scripts.nl_condition_parser import DEFAULT_NL_PROMPTS, load_prompt_file, parse_condition, parse_prompt_arg
from tc_srf_amp_config import create_tc_srf_amp_config
from utils.util import set_seed


BASIC_AA = set("KR")
ACIDIC_AA = set("DE")
HYDROPHOBIC_AA = set("AVILMFWYC")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")[:80] or "prompt"


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
    summary.update(
        {
            "n": len(records),
            "unique": len({record["protein"] for record in records}),
        }
    )
    return summary


def load_runner(args: argparse.Namespace) -> tuple[FlowMatchingRunner, str]:
    os.environ.setdefault("TRANSFORMERS_CACHE", str(WORKSPACE_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HOME", str(WORKSPACE_ROOT / ".cache" / "huggingface"))

    config = create_tc_srf_amp_config()
    config.task.stage = "amp_finetune"
    config.ddp = False
    config.local_rank = 0
    config.training.batch_size_per_gpu = max(args.num, 1)
    config.device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    config.refresh.true = False
    config.text.enabled = True
    config.text.use_cross_attention = True
    config.text.use_adaptive_norm = True

    # Strict PDF-style path: no direct label or attribute ids.
    if hasattr(config.text, "use_label_condition"):
        config.text.use_label_condition = False
    if hasattr(config.text, "use_attribute_condition"):
        config.text.use_attribute_condition = False

    runner = FlowMatchingRunner(config, latent_mode=config.model.embeddings_type)
    load = torch.load(args.ckpt, map_location="cpu")
    load_mode = "unloaded"

    if args.prefer_ema and load.get("ema") is not None:
        try:
            runner.ema = ExponentialMovingAverage(runner.model.parameters(), config.model.ema_rate)
            runner.ema.load_state_dict(load["ema"])
            runner.ema.to(runner.device)
            runner.switch_to_ema()
            load_mode = "ema"
        except Exception as exc:
            print(f"warning: EMA load failed, falling back to model state: {exc}", file=sys.stderr)

    if load_mode == "unloaded":
        missing, unexpected = runner.model.load_state_dict(load["model"], strict=False)
        load_mode = f"model_state_non_strict_missing={len(missing)}_unexpected={len(unexpected)}"

    if runner.text_encoder is not None and load.get("text_encoder") is not None:
        runner.text_encoder.load_state_dict(load["text_encoder"], strict=False)
    if load.get("null_conditioning") is not None:
        runner.null_conditioning.load_state_dict(load["null_conditioning"], strict=False)

    runner.model.eval()
    if runner.text_encoder is not None:
        runner.text_encoder.eval()
    return runner, load_mode


def prepare_conditioning(runner: FlowMatchingRunner, structured_prompt: str, num: int) -> dict:
    conditioning, _ = runner._prepare_conditioning(
        {"text": [structured_prompt] * num},
        batch_size=num,
        seq_len=runner.config.data.max_sequence_len,
        dtype=next(runner.model.parameters()).dtype,
    )
    return conditioning


def load_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    prompts = []
    if args.prompt_file:
        prompts.extend(load_prompt_file(Path(args.prompt_file)))
    prompts.extend(parse_prompt_arg(value) for value in args.prompt)
    if not prompts:
        prompts = list(DEFAULT_NL_PROMPTS.items())
    return prompts


def effective_guidance_scale(condition: dict, guidance_scale: float, neutral_guidance_scale: float | None) -> float:
    if neutral_guidance_scale is not None and condition["charge_bin"] == "neutral":
        return neutral_guidance_scale
    return guidance_scale


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural language -> structured prompt -> strict T_c/c_g conditioned generation."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="generated_seqs/strict_nl_prompt_benchmark")
    parser.add_argument("--prompt", action="append", default=[], help="Literal prompt or name=prompt. Can repeat.")
    parser.add_argument("--prompt-file", default=None, help="JSON/JSONL prompt file.")
    parser.add_argument("--num", type=int, default=64, help="Generated samples per prompt per seed.")
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[4.0])
    parser.add_argument(
        "--neutral-guidance-scale",
        type=float,
        default=None,
        help="Optional lower CFG scale for prompts parsed as charge: neutral.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only parse prompts and write parsed_conditions.json.")
    parser.add_argument("--prefer-ema", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_records = []
    for name, prompt in prompts:
        parsed = parse_condition(prompt).with_ids()
        parsed_records.append(
            {
                "prompt_name": name,
                "natural_prompt": prompt,
                "structured_prompt": parse_condition(prompt).structured_prompt(),
                **parsed,
            }
        )

    parsed_path = out_dir / "parsed_conditions.json"
    parsed_path.write_text(json.dumps(parsed_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"parsed={parsed_path}")
    for record in parsed_records:
        print(json.dumps(record, ensure_ascii=False))

    if args.dry_run:
        return

    runner, load_mode = load_runner(args)
    print(f"load_mode={load_mode}")

    all_records = []
    for parsed in parsed_records:
        for requested_guidance_scale in args.guidance_scale:
            used_guidance_scale = effective_guidance_scale(
                parsed,
                requested_guidance_scale,
                args.neutral_guidance_scale,
            )
            for seed in args.seed:
                set_seed(seed)
                conditioning = prepare_conditioning(runner, parsed["structured_prompt"], args.num)
                seqs = runner.generate_text(
                    batch_size=args.num,
                    conditioning=conditioning,
                    guidance_scale=used_guidance_scale,
                )
                records = []
                for seq in seqs:
                    record = {
                        "prompt_name": parsed["prompt_name"],
                        "natural_prompt": parsed["natural_prompt"],
                        "structured_prompt": parsed["structured_prompt"],
                        "requested_guidance_scale": requested_guidance_scale,
                        "used_guidance_scale": used_guidance_scale,
                        "seed": seed,
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

                output_path = out_dir / f"{safe_name(parsed['prompt_name'])}_g{requested_guidance_scale:g}_seed{seed}.json"
                output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
                print(output_path)
                all_records.extend(records)

    records_path = out_dir / "combined.json"
    records_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")

    grouped = defaultdict(list)
    for record in all_records:
        grouped[(record["prompt_name"], record["requested_guidance_scale"], record["used_guidance_scale"])].append(record)

    summary = []
    for (prompt_name, requested_g, used_g), records in sorted(grouped.items()):
        first = records[0]
        summary.append(
            {
                "prompt_name": prompt_name,
                "natural_prompt": first["natural_prompt"],
                "structured_prompt": first["structured_prompt"],
                "requested_guidance_scale": requested_g,
                "used_guidance_scale": used_g,
                "target_condition": {
                    "length_bin": first["length_bin"],
                    "charge_bin": first["charge_bin"],
                    "kr_bin": first["kr_bin"],
                    "hydrophobicity_bin": first["hydrophobicity_bin"],
                    "cys_bin": first["cys_bin"],
                },
                **summarize(records),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"combined={records_path}")
    print(f"summary={summary_path}")
    print("\nSUMMARY")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
