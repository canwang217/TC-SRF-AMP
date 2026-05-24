from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # allows lightweight validation in non-training envs
    class Dataset:  # type: ignore[override]
        pass


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_AA = set("AVILMFWYC")
BASIC_AA = set("KR")
ACIDIC_AA = set("DE")

LENGTH_BIN_TO_ID = {"very_short": 0, "short": 1, "medium": 2, "long": 3}
CHARGE_BIN_TO_ID = {"negative": 0, "neutral": 1, "positive": 2, "high_positive": 3}
RATIO_BIN_TO_ID = {"low": 0, "medium": 1, "high": 2}
CYS_BIN_TO_ID = {"none": 0, "low": 1, "high": 2}

CONDITION_ID_KEYS = (
    "label",
    "length_bin_id",
    "charge_bin_id",
    "kr_bin_id",
    "hydrophobicity_bin_id",
    "cys_bin_id",
)


def clean_peptide_sequence(sequence: str, min_len: int = 2, max_len: int = 50) -> str | None:
    if sequence is None:
        return None
    seq = sequence.strip().upper()
    if len(seq) < min_len or len(seq) > max_len:
        return None
    if any(ch not in ALLOWED_AA for ch in seq):
        return None
    return seq


def normalize_tg_text(text: str) -> str:
    if text is None:
        return "This is a peptide: target inactive"
    cleaned = str(text).replace("&&", ". ")
    cleaned = cleaned.replace("碌", "u")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def infer_label_from_text(text: str) -> int:
    return 0 if "inactive" in str(text).lower() else 1


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
    kr_frac = sum(sequence.count(aa) for aa in BASIC_AA) / max(1, length)
    hydrophobic_frac = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA) / max(1, length)
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


def _field_from_structured_text(text: str, field: str) -> str | None:
    pattern = rf"(?:^|[;,\n])\s*{re.escape(field)}\s*:\s*([a-zA-Z_+-]+)"
    match = re.search(pattern, text)
    return match.group(1).strip().lower() if match else None


def _coerce_choice(value: str | None, mapping: dict[str, int], default: str) -> str:
    if value is None:
        return default
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in mapping else default


def condition_ids_from_prompt(prompt: str) -> dict:
    text = str(prompt)
    lower = text.lower()
    label = 0 if "inactive" in lower else 1

    explicit_activity = _field_from_structured_text(lower, "activity")
    if explicit_activity == "inactive":
        label = 0
    elif explicit_activity == "active":
        label = 1

    explicit_len = _field_from_structured_text(lower, "length_bin")
    if explicit_len is None:
        if "very short" in lower or "very_short" in lower:
            explicit_len = "very_short"
        elif "short" in lower or "20 amino" in lower or "around 20" in lower:
            explicit_len = "short"
        elif "long" in lower:
            explicit_len = "long"
        else:
            explicit_len = "medium"

    explicit_charge = _field_from_structured_text(lower, "charge")
    if explicit_charge is None:
        if "high positive" in lower or "high_positive" in lower or "strongly cationic" in lower:
            explicit_charge = "high_positive"
        elif "cationic" in lower or "positive" in lower:
            explicit_charge = "high_positive"
        elif "negative" in lower or "anionic" in lower:
            explicit_charge = "negative"
        elif "neutral" in lower or label == 0:
            explicit_charge = "neutral"
        else:
            explicit_charge = "positive"

    explicit_kr = (
        _field_from_structured_text(lower, "kr_ratio")
        or _field_from_structured_text(lower, "kr")
        or _field_from_structured_text(lower, "kr_bin")
    )
    if explicit_kr is None:
        explicit_kr = "high" if ("cationic" in lower or "lysine" in lower or "arginine" in lower) else ("low" if label == 0 else "medium")

    explicit_hydro = _field_from_structured_text(lower, "hydrophobicity")
    if explicit_hydro is None:
        explicit_hydro = "high" if "hydrophobic" in lower else "medium"

    explicit_cys = _field_from_structured_text(lower, "cys")
    if explicit_cys is None:
        explicit_cys = "none"

    length_name = _coerce_choice(explicit_len, LENGTH_BIN_TO_ID, "medium")
    charge_name = _coerce_choice(explicit_charge, CHARGE_BIN_TO_ID, "positive" if label == 1 else "neutral")
    kr_name = _coerce_choice(explicit_kr, RATIO_BIN_TO_ID, "medium" if label == 1 else "low")
    hydro_name = _coerce_choice(explicit_hydro, RATIO_BIN_TO_ID, "medium")
    cys_name = _coerce_choice(explicit_cys, CYS_BIN_TO_ID, "none")

    return {
        "label": label,
        "length_bin_id": LENGTH_BIN_TO_ID[length_name],
        "charge_bin_id": CHARGE_BIN_TO_ID[charge_name],
        "kr_bin_id": RATIO_BIN_TO_ID[kr_name],
        "hydrophobicity_bin_id": RATIO_BIN_TO_ID[hydro_name],
        "cys_bin_id": CYS_BIN_TO_ID[cys_name],
        "length_bin": length_name,
        "charge_bin": charge_name,
        "kr_bin": kr_name,
        "hydrophobicity_bin": hydro_name,
        "cys_bin": cys_name,
    }


def _structured_or_inferred(obj: dict, sequence: str, text: str) -> dict:
    attrs = sequence_attributes(sequence)
    activity = str(obj.get("activity") or ("inactive" if infer_label_from_text(text) == 0 else "active")).lower()
    length_name = _coerce_choice(str(obj.get("length_bin") or attrs["length_bin"]), LENGTH_BIN_TO_ID, attrs["length_bin"])
    charge_name = _coerce_choice(str(obj.get("charge_bin") or obj.get("charge") or attrs["charge_bin"]), CHARGE_BIN_TO_ID, attrs["charge_bin"])
    kr_name = _coerce_choice(str(obj.get("kr_bin") or obj.get("kr_ratio") or attrs["kr_bin"]), RATIO_BIN_TO_ID, attrs["kr_bin"])
    hydro_name = _coerce_choice(str(obj.get("hydrophobicity_bin") or obj.get("hydrophobicity") or attrs["hydrophobicity_bin"]), RATIO_BIN_TO_ID, attrs["hydrophobicity_bin"])
    cys_name = _coerce_choice(str(obj.get("cys_bin") or obj.get("cys") or attrs["cys_bin"]), CYS_BIN_TO_ID, attrs["cys_bin"])
    return {
        **attrs,
        "activity": activity,
        "target": obj.get("target", "unknown"),
        "toxicity": obj.get("toxicity", "unknown"),
        "label": 0 if activity == "inactive" else 1,
        "length_bin": length_name,
        "charge_bin": charge_name,
        "kr_bin": kr_name,
        "hydrophobicity_bin": hydro_name,
        "cys_bin": cys_name,
        "length_bin_id": LENGTH_BIN_TO_ID[length_name],
        "charge_bin_id": CHARGE_BIN_TO_ID[charge_name],
        "kr_bin_id": RATIO_BIN_TO_ID[kr_name],
        "hydrophobicity_bin_id": RATIO_BIN_TO_ID[hydro_name],
        "cys_bin_id": CYS_BIN_TO_ID[cys_name],
    }


def load_tg_amp_jsonl(path: str | Path, min_len: int = 2, max_len: int = 50) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sequence = clean_peptide_sequence(obj.get("trg"), min_len=min_len, max_len=max_len)
            if sequence is None:
                continue
            text = normalize_tg_text(obj.get("src", ""))
            attrs = _structured_or_inferred(obj, sequence, text)
            records.append(
                {
                    "sequence": sequence,
                    "text": text,
                    "src": obj.get("src", ""),
                    "trg": obj.get("trg", ""),
                    **attrs,
                }
            )
    return records


class TgAmpJsonlDataset(Dataset):
    def __init__(self, path: str | Path, min_len: int = 2, max_len: int = 50):
        self.path = Path(path)
        self.records = load_tg_amp_jsonl(self.path, min_len=min_len, max_len=max_len)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.records[idx])

