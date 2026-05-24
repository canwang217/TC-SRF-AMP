from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from flow_matching_utils.flow_matching_holder import FlowMatchingRunner
from model.ema_model import ExponentialMovingAverage
from tc_srf_amp_config import create_tc_srf_amp_config
from utils.util import set_seed


DEFAULT_PROMPTS = {
    "active_short_highpos": (
        "activity: active; target: unknown; length_bin: short; charge: high_positive; "
        "kr_ratio: high; hydrophobicity: medium; cys: none; toxicity: unknown"
    ),
    "active_long_negative": (
        "activity: active; target: unknown; length_bin: long; charge: negative; "
        "kr_ratio: low; hydrophobicity: medium; cys: none; toxicity: unknown"
    ),
    "active_medium_hydrophobic": (
        "activity: active; target: unknown; length_bin: medium; charge: positive; "
        "kr_ratio: medium; hydrophobicity: high; cys: none; toxicity: unknown"
    ),
    "inactive_long_neutral": (
        "activity: inactive; target: unknown; length_bin: long; charge: neutral; "
        "kr_ratio: low; hydrophobicity: medium; cys: none; toxicity: unknown"
    ),
}

BASIC_AA = set("KR")
ACIDIC_AA = set("DE")
HYDROPHOBIC_AA = set("AILMFWVY")


def parse_prompt_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        name, prompt = value.split("=", 1)
        return name.strip(), prompt.strip()
    key = value.strip()
    if key in DEFAULT_PROMPTS:
        return key, DEFAULT_PROMPTS[key]
    safe = "".join(ch if ch.isalnum() else "_" for ch in key.lower()).strip("_")[:80] or "prompt"
    return safe, value


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten()[None], b.flatten()[None]).detach().cpu().item())


def rel_norm_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(b).clamp_min(1e-8)
    return float((num / den).detach().cpu().item())


def sequence_metrics(seq: str) -> dict:
    seq = str(seq).strip().upper()
    length = len(seq)
    charge = sum(1 for aa in seq if aa in BASIC_AA) - sum(1 for aa in seq if aa in ACIDIC_AA)
    kr_frac = sum(1 for aa in seq if aa in BASIC_AA) / max(1, length)
    hydrophobic_frac = sum(1 for aa in seq if aa in HYDROPHOBIC_AA) / max(1, length)
    return {
        "length": length,
        "net_charge": charge,
        "kr_frac": round(kr_frac, 4),
        "hydrophobic_frac": round(hydrophobic_frac, 4),
        "cys_count": seq.count("C"),
    }


def summarize_generation(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ["length", "net_charge", "kr_frac", "hydrophobic_frac", "cys_count"]
    return {
        key: round(sum(float(item[key]) for item in records) / len(records), 4)
        for key in keys
    } | {
        "n": len(records),
        "unique": len({item["protein"] for item in records}),
    }


def build_config(args: argparse.Namespace):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
    config.text.guidance_scale = args.guidance_scale

    # Strict PDF-style diagnostic: no direct label/attribute ids are supplied.
    if hasattr(config.text, "use_label_condition"):
        config.text.use_label_condition = False
    if hasattr(config.text, "use_attribute_condition"):
        config.text.use_attribute_condition = False
    return config


def load_runner(args: argparse.Namespace) -> tuple[FlowMatchingRunner, str]:
    config = build_config(args)
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
        state = load.get("model")
        if state is None:
            raise KeyError("checkpoint has neither loadable `ema` nor `model` state")
        missing, unexpected = runner.model.load_state_dict(state, strict=False)
        load_mode = f"model_state_non_strict_missing={len(missing)}_unexpected={len(unexpected)}"

    if runner.text_encoder is not None and load.get("text_encoder") is not None:
        missing, unexpected = runner.text_encoder.load_state_dict(load["text_encoder"], strict=False)
        if missing or unexpected:
            print(f"warning: text encoder non-strict load missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    if load.get("null_conditioning") is not None:
        runner.null_conditioning.load_state_dict(load["null_conditioning"], strict=False)

    runner.model.eval()
    if runner.text_encoder is not None:
        runner.text_encoder.eval()
    return runner, load_mode


def prepare_text_conditioning(runner: FlowMatchingRunner, prompt: str, batch_size: int, dtype: torch.dtype) -> dict:
    conditioning, _ = runner._prepare_conditioning(
        {"text": [prompt] * batch_size},
        batch_size=batch_size,
        seq_len=runner.config.data.max_sequence_len,
        dtype=dtype,
    )
    return conditioning


def token_only_condition(conditioning: dict, null_conditioning) -> dict:
    return {
        "token_condition": conditioning["token_condition"],
        "global_condition": null_conditioning.global_condition,
        "attention_mask": conditioning["attention_mask"],
    }


def global_only_condition(conditioning: dict, null_conditioning) -> dict:
    return {
        "token_condition": null_conditioning.token_condition,
        "global_condition": conditioning["global_condition"],
        "attention_mask": null_conditioning.attention_mask,
    }


@torch.no_grad()
def vector_field_for_condition(runner: FlowMatchingRunner, x: torch.Tensor, t: torch.Tensor, conditioning: dict) -> torch.Tensor:
    token_mask = runner._get_extended_attention_mask(conditioning["attention_mask"], x.dtype)
    return runner.ddp_model(
        x_t=x,
        time_t=t,
        attention_mask=None,
        cls=None,
        token_condition=conditioning["token_condition"],
        token_condition_mask=token_mask,
        global_condition=conditioning["global_condition"],
    )


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    runner, load_mode = load_runner(args)
    dtype = next(runner.model.parameters()).dtype
    prompts = [parse_prompt_arg(item) for item in args.prompt] if args.prompt else list(DEFAULT_PROMPTS.items())

    conditions = {}
    for name, prompt in prompts:
        conditions[name] = prepare_text_conditioning(runner, prompt, batch_size=1, dtype=dtype)

    text_condition_report = {}
    for name, cond in conditions.items():
        mask = cond["attention_mask"].float()
        token_mean = (cond["token_condition"] * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)[:, None]
        text_condition_report[name] = {
            "prompt": dict(prompts)[name],
            "global_norm": float(torch.linalg.vector_norm(cond["global_condition"]).detach().cpu().item()),
            "token_mean_norm": float(torch.linalg.vector_norm(token_mean).detach().cpu().item()),
        }

    text_pair_report = []
    names = [name for name, _ in prompts]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_cond = conditions[left]
            right_cond = conditions[right]
            left_mask = left_cond["attention_mask"].float()
            right_mask = right_cond["attention_mask"].float()
            left_tok = (left_cond["token_condition"] * left_mask[:, :, None]).sum(dim=1) / left_mask.sum(dim=1).clamp_min(1.0)[:, None]
            right_tok = (right_cond["token_condition"] * right_mask[:, :, None]).sum(dim=1) / right_mask.sum(dim=1).clamp_min(1.0)[:, None]
            text_pair_report.append(
                {
                    "left": left,
                    "right": right,
                    "global_cosine": round(cosine(left_cond["global_condition"], right_cond["global_condition"]), 6),
                    "token_mean_cosine": round(cosine(left_tok, right_tok), 6),
                }
            )

    if runner.config.use_compress:
        shape = (
            1,
            runner.config.data.max_sequence_len,
            runner.config.model.compressed_hidden_size,
        )
    else:
        shape = (
            1,
            runner.config.data.max_sequence_len,
            runner.config.model.hidden_size,
        )
    x = torch.randn(shape, device=runner.device, dtype=dtype) * float(runner.config.fm.sample_std)
    t = torch.full((1,), args.vector_time, device=runner.device, dtype=dtype)
    null_condition = runner.null_conditioning(1, x.shape[1], runner.device, dtype)
    null_dict = {
        "token_condition": null_condition.token_condition,
        "global_condition": null_condition.global_condition,
        "attention_mask": null_condition.attention_mask,
    }
    v_null = vector_field_for_condition(runner, x, t, null_dict)
    vectors = {name: vector_field_for_condition(runner, x, t, cond) for name, cond in conditions.items()}
    vectors_token_only = {
        name: vector_field_for_condition(runner, x, t, token_only_condition(cond, null_condition))
        for name, cond in conditions.items()
    }
    vectors_global_only = {
        name: vector_field_for_condition(runner, x, t, global_only_condition(cond, null_condition))
        for name, cond in conditions.items()
    }

    vector_report = {}
    for name, v in vectors.items():
        v_token = vectors_token_only[name]
        v_global = vectors_global_only[name]
        vector_report[name] = {
            "cosine_vs_null": round(cosine(v, v_null), 6),
            "relative_delta_vs_null": round(rel_norm_delta(v, v_null), 6),
            "vector_norm": float(torch.linalg.vector_norm(v).detach().cpu().item()),
            "token_only_cosine_vs_null": round(cosine(v_token, v_null), 6),
            "token_only_relative_delta_vs_null": round(rel_norm_delta(v_token, v_null), 6),
            "global_only_cosine_vs_null": round(cosine(v_global, v_null), 6),
            "global_only_relative_delta_vs_null": round(rel_norm_delta(v_global, v_null), 6),
            "full_vs_token_only_cosine": round(cosine(v, v_token), 6),
            "full_vs_global_only_cosine": round(cosine(v, v_global), 6),
        }

    vector_pair_report = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            vector_pair_report.append(
                {
                    "left": left,
                    "right": right,
                    "vector_cosine": round(cosine(vectors[left], vectors[right]), 6),
                    "relative_delta": round(rel_norm_delta(vectors[left], vectors[right]), 6),
                }
            )

    generation_records = []
    for name, prompt in prompts:
        for seed in args.generation_seed:
            set_seed(seed)
            cond = prepare_text_conditioning(runner, prompt, batch_size=args.num, dtype=dtype)
            seqs = runner.generate_text(batch_size=args.num, conditioning=cond, guidance_scale=args.guidance_scale)
            for seq in seqs:
                generation_records.append(
                    {
                        "prompt_name": name,
                        "prompt": prompt,
                        "seed": seed,
                        "guidance_scale": args.guidance_scale,
                        "protein": seq,
                        **sequence_metrics(seq),
                    }
                )

    generation_summary = {
        name: summarize_generation([record for record in generation_records if record["prompt_name"] == name])
        for name in names
    }

    return {
        "checkpoint": args.ckpt,
        "load_mode": load_mode,
        "device": str(runner.device),
        "strict_pdf_path": {
            "attribute_ids_supplied": False,
            "label_ids_supplied": False,
            "uses_text_encoder": True,
            "uses_token_condition_Tc": True,
            "uses_global_condition_cg": True,
        },
        "text_condition_report": text_condition_report,
        "text_pair_report": text_pair_report,
        "vector_report": vector_report,
        "vector_pair_report": vector_pair_report,
        "generation_summary": generation_summary,
        "generation_records": generation_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether strict PDF-style T_c/c_g text conditioning controls TC-SRF generation."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="generated_seqs/text_condition_diagnostics")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt key, literal prompt, or name=prompt. Can repeat.")
    parser.add_argument("--num", type=int, default=16, help="Generated samples per prompt per seed.")
    parser.add_argument("--generation-seed", nargs="+", type=int, default=[0], help="Seeds used for generation.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for vector-field diagnostics.")
    parser.add_argument("--vector-time", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--prefer-ema", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    report = diagnose(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "text_condition_diagnostics.json"
    records_path = out_dir / "strict_text_only_generations.json"
    records = report.pop("generation_records")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"report={report_path}")
    print(f"generations={records_path}")
    print("\nTEXT CONDITION PAIRS")
    for row in report["text_pair_report"]:
        print(row)
    print("\nVECTOR FIELD PAIRS")
    for row in report["vector_pair_report"]:
        print(row)
    print("\nGENERATION SUMMARY")
    for name, summary in report["generation_summary"].items():
        print(name, summary)


if __name__ == "__main__":
    main()
