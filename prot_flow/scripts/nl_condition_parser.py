from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


LENGTH_BIN_TO_ID = {"very_short": 0, "short": 1, "medium": 2, "long": 3}
CHARGE_BIN_TO_ID = {"negative": 0, "neutral": 1, "positive": 2, "high_positive": 3}
RATIO_BIN_TO_ID = {"low": 0, "medium": 1, "high": 2}
CYS_BIN_TO_ID = {"none": 0, "low": 1, "high": 2}


@dataclass(frozen=True)
class ParsedCondition:
    activity: str = "active"
    target: str = "unknown"
    length_bin: str = "medium"
    charge_bin: str = "positive"
    kr_bin: str = "medium"
    hydrophobicity_bin: str = "medium"
    cys_bin: str = "none"
    toxicity: str = "unknown"

    @property
    def label(self) -> int:
        return 0 if self.activity == "inactive" else 1

    def with_ids(self) -> dict:
        return {
            **asdict(self),
            "label": self.label,
            "length_bin_id": LENGTH_BIN_TO_ID[self.length_bin],
            "charge_bin_id": CHARGE_BIN_TO_ID[self.charge_bin],
            "kr_bin_id": RATIO_BIN_TO_ID[self.kr_bin],
            "hydrophobicity_bin_id": RATIO_BIN_TO_ID[self.hydrophobicity_bin],
            "cys_bin_id": CYS_BIN_TO_ID[self.cys_bin],
        }

    def structured_prompt(self) -> str:
        return (
            f"activity: {self.activity}; target: {self.target}; "
            f"length_bin: {self.length_bin}; charge: {self.charge_bin}; "
            f"kr_ratio: {self.kr_bin}; hydrophobicity: {self.hydrophobicity_bin}; "
            f"cys: {self.cys_bin}; toxicity: {self.toxicity}"
        )


DEFAULT_NL_PROMPTS = {
    "short_cationic_amp": "Generate a short cationic antimicrobial peptide.",
    "short_kr_rich_no_cys": "Design a short antimicrobial peptide rich in lysine and arginine with no cysteine.",
    "small_positive_synonym": "Create a small positively charged AMP rich in Lys/Arg residues.",
    "long_negative": "Generate a long negatively charged peptide with low lysine and arginine content.",
    "medium_hydrophobic": "Design a medium-length active peptide with high hydrophobicity.",
    "low_hydrophobic": "Generate an active peptide with low hydrophobicity and moderate positive charge.",
    "inactive_long_neutral": "Generate a long inactive-like peptide with neutral charge and low KR content.",
    "local_edit_short": "Generate a short highly cationic antimicrobial peptide with moderate hydrophobicity.",
    "local_edit_long": "Generate a long highly cationic antimicrobial peptide with moderate hydrophobicity.",
}


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("lys/arg", "lys arg")
    lowered = lowered.replace("lys and arg", "lys arg")
    lowered = lowered.replace("lys or arg", "lys arg")
    lowered = lowered.replace("lysine or arginine", "lysine and arginine")
    lowered = lowered.replace("k/r", "kr")
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def structured_field(text: str, *names: str) -> str | None:
    for name in names:
        pattern = rf"(?:^|[;,\n])\s*{re.escape(name)}\s*:\s*([a-zA-Z_+-]+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().lower().replace("-", "_")
    return None


def coerce_choice(value: str | None, choices: dict[str, int], default: str) -> str:
    if value is None:
        return default
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in choices else default


def parse_activity(text: str) -> str:
    explicit = structured_field(text, "activity")
    if explicit in {"active", "inactive"}:
        return explicit
    if has_any(text, [r"\binactive\b", r"\bnon antimicrobial\b", r"\binactive like\b", r"\bnon active\b", r"\bweakly active\b", r"\bweak activity\b", r"无活性"]):
        return "inactive"
    return "active"


def parse_target(text: str) -> str:
    explicit = structured_field(text, "target")
    if explicit:
        return explicit
    if has_any(text, [r"\bgram negative\b", r"\bgramnegative\b", r"\bg negative\b", r"\bg-negative\b", r"革兰阴"]):
        return "gram_negative"
    if has_any(text, [r"\bgram positive\b", r"\bgrampositive\b", r"\bg positive\b", r"\bg-positive\b", r"革兰阳"]):
        return "gram_positive"
    if has_any(text, [r"bacterial", r"bacteria", r"antibacterial", r"细菌"]):
        return "bacterial"
    if has_any(text, [r"fungal", r"antifungal", r"真菌"]):
        return "fungal"
    return "unknown"


def parse_length(text: str) -> str:
    explicit = structured_field(text, "length_bin", "length")
    if explicit:
        return coerce_choice(explicit, LENGTH_BIN_TO_ID, "medium")
    if has_any(text, [r"very short", r"ultra short", r"ultrashort", r"under 10", r"less than 10", r"<\s*10", r"5\s*(aa|amino)", r"短小"]):
        return "very_short"
    if has_any(text, [r"\bshort\b", r"\bsmall\b", r"\bcompact\b", r"\bbrief\b", r"under 20", r"less than 20", r"<\s*20", r"around 20", r"20 amino", r"短肽"]):
        return "short"
    if has_any(text, [r"\blong\b", r"\blonger\b", r"\bextended\b", r"\belongated\b", r"over 30", r"more than 30", r">\s*30", r"30\s*(to|-)\s*50", r"长肽"]):
        return "long"
    if has_any(text, [r"medium length", r"medium sized", r"moderate length", r"moderate sized", r"around 25", r"20\s*(to|-)\s*30", r"中等长度"]):
        return "medium"
    return "medium"


def parse_charge(text: str, activity: str) -> str:
    explicit = structured_field(text, "charge", "charge_bin")
    if explicit:
        return coerce_choice(explicit, CHARGE_BIN_TO_ID, "positive" if activity == "active" else "neutral")
    if has_any(text, [r"negative", r"negatively charged", r"anionic", r"acidic", r"负电", r"酸性"]):
        return "negative"
    if has_any(text, [r"neutral", r"near zero charge", r"near zero net charge", r"zero net charge", r"low charge", r"uncharged", r"中性"]):
        return "neutral"
    if has_any(text, [r"highly cationic", r"strongly cationic", r"\bstrong cationic\b", r"\bcationic\b", r"high positive", r"highly positive", r"strong positive", r"strong positive charge", r"rich in positive charge", r"rich in lysine", r"rich in arginine", r"rich in lys arg", r"lys arg rich", r"many lys arg", r"many lysine and arginine", r"enriched for lys arg", r"enriched for kr", r"lys arg residues", r"kr rich", r"高正电", r"富含赖氨酸", r"富含精氨酸"]):
        return "high_positive"
    if has_any(text, [r"cationic", r"positive", r"positively charged", r"正电"]):
        return "positive"
    return "positive" if activity == "active" else "neutral"


def parse_kr(text: str, charge_bin: str, activity: str) -> str:
    explicit = structured_field(text, "kr_ratio", "kr_bin", "kr")
    if explicit:
        return coerce_choice(explicit, RATIO_BIN_TO_ID, "medium" if activity == "active" else "low")
    if has_any(text, [r"low kr", r"low k r", r"low lysine", r"low arginine", r"few lysine", r"few arginine", r"few lys arg", r"few kr", r"little lysine", r"little arginine", r"very little lysine", r"very little arginine", r"very little lysine and arginine", r"poor in lysine", r"poor in arginine", r"low lysine and arginine", r"low lys arg", r"低kr", r"低赖氨酸", r"低精氨酸"]):
        return "low"
    if has_any(text, [r"moderate kr", r"medium kr", r"balanced kr", r"中等kr"]):
        return "medium"
    if has_any(text, [r"kr rich", r"high kr", r"rich in lysine", r"rich in arginine", r"rich in lys arg", r"lysine and arginine", r"lys arg", r"many lysine", r"many arginine", r"many lys arg", r"many lysine and arginine", r"enriched for kr", r"enriched for lys arg", r"kr amino acids", r"kr residues", r"富含赖氨酸", r"富含精氨酸"]):
        return "high"
    if charge_bin == "high_positive":
        return "high"
    return "medium" if activity == "active" else "low"


def parse_hydrophobicity(text: str) -> str:
    explicit = structured_field(text, "hydrophobicity", "hydrophobicity_bin", "hydro")
    if explicit:
        return coerce_choice(explicit, RATIO_BIN_TO_ID, "medium")
    if has_any(text, [r"low hydrophobic", r"low hydrophobicity", r"\bhydrophilic\b", r"mostly hydrophilic", r"more hydrophilic", r"less hydrophobic", r"低疏水"]):
        return "low"
    if has_any(text, [r"moderately hydrophobic", r"moderate hydrophobicity", r"medium hydrophobic", r"balanced hydrophobic", r"中等疏水"]):
        return "medium"
    if has_any(text, [r"highly hydrophobic", r"high hydrophobicity", r"hydrophobic rich", r"many hydrophobic", r"hydrophobic residues", r"\bhydrophobic\b", r"疏水"]):
        return "high"
    return "medium"


def parse_cys(text: str) -> str:
    explicit = structured_field(text, "cys", "cys_bin", "cysteine")
    if explicit:
        return coerce_choice(explicit, CYS_BIN_TO_ID, "none")
    if has_any(text, [r"no cysteine", r"without cysteine", r"cysteine free", r"no cys", r"without cys", r"无半胱氨酸"]):
        return "none"
    if has_any(text, [r"cysteine rich", r"cys rich", r"disulfide rich", r"many cysteine", r"high cysteine", r"高半胱氨酸"]):
        return "high"
    if has_any(text, [r"low cysteine", r"few cysteine", r"low cys", r"few cys", r"低半胱氨酸"]):
        return "low"
    return "none"


def parse_toxicity(text: str) -> str:
    explicit = structured_field(text, "toxicity")
    if explicit:
        return explicit
    if has_any(text, [r"low toxicity", r"non toxic", r"low toxic", r"low hemolysis", r"non hemolytic", r"低毒", r"低溶血"]):
        return "low"
    if has_any(text, [r"high toxicity", r"toxic", r"hemolytic", r"高毒"]):
        return "high"
    return "unknown"


def parse_condition(prompt: str) -> ParsedCondition:
    text = normalize_text(prompt)
    activity = parse_activity(text)
    charge_bin = parse_charge(text, activity)
    return ParsedCondition(
        activity=activity,
        target=parse_target(text),
        length_bin=parse_length(text),
        charge_bin=charge_bin,
        kr_bin=parse_kr(text, charge_bin, activity),
        hydrophobicity_bin=parse_hydrophobicity(text),
        cys_bin=parse_cys(text),
        toxicity=parse_toxicity(text),
    )


def parse_prompt_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        name, prompt = value.split("=", 1)
        return name.strip(), prompt.strip()
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")[:48] or "prompt"
    return safe, value


def load_prompt_file(path: Path) -> list[tuple[str, str]]:
    rows = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((str(obj.get("name") or f"prompt_{idx}"), str(obj["prompt"])))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [(str(name), str(prompt)) for name, prompt in data.items()]
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            rows.append((str(obj.get("name") or f"prompt_{idx}"), str(obj["prompt"])))
        return rows
    raise ValueError(f"Unsupported prompt file shape: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse natural-language AMP prompts into structured TC-SRF attributes.")
    parser.add_argument("--prompt", action="append", default=[], help="Literal prompt or name=prompt. Can repeat.")
    parser.add_argument("--prompt-file", default=None, help="JSON/JSONL prompt file.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    prompts = []
    if args.prompt_file:
        prompts.extend(load_prompt_file(Path(args.prompt_file)))
    prompts.extend(parse_prompt_arg(value) for value in args.prompt)
    if not prompts:
        prompts = list(DEFAULT_NL_PROMPTS.items())

    records = []
    for name, prompt in prompts:
        parsed = parse_condition(prompt)
        record = {
            "prompt_name": name,
            "prompt": prompt,
            "structured_prompt": parsed.structured_prompt(),
            **parsed.with_ids(),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output={output_path}")


if __name__ == "__main__":
    main()
