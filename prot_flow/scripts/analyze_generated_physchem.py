from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


AA = set("ACDEFGHIKLMNPQRSTVWY")

# Average residue masses in Daltons. A water molecule is added for the full peptide mass.
AA_MASS = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}
WATER_MASS = 18.01528

# Kyte-Doolittle hydropathy scale.
KD_HYDROPATHY = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

PKA = {
    "n_term": 9.69,
    "c_term": 2.34,
    "C": 8.33,
    "D": 3.86,
    "E": 4.25,
    "H": 6.00,
    "K": 10.50,
    "R": 12.40,
    "Y": 10.07,
}

BASIC = set("KRH")
STRONG_BASIC = set("KR")
ACIDIC = set("DE")
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
ALIPHATIC = set("AILV")
POLAR_UNCHARGED = set("STNQCY")


def clean_sequence(value: str) -> str:
    sequence = str(value or "").strip().upper()
    return "".join(aa for aa in sequence if aa in AA)


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("generation_records", "records", "combined"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported input shape: {path}")


def get_sequence(record: dict, sequence_key: str | None) -> str:
    keys = [sequence_key] if sequence_key else []
    keys.extend(["protein", "sequence", "seq", "peptide", "trg"])
    for key in keys:
        if key and key in record:
            sequence = clean_sequence(record[key])
            if sequence:
                return sequence
    return ""


def residue_fraction(counts: Counter, residues: set[str], length: int) -> float:
    return sum(counts[aa] for aa in residues) / max(1, length)


def approximate_net_charge(counts: Counter) -> float:
    return counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]


def charge_at_ph(sequence: str, ph: float) -> float:
    counts = Counter(sequence)
    positive = 1.0 / (1.0 + 10 ** (ph - PKA["n_term"]))
    positive += counts["K"] / (1.0 + 10 ** (ph - PKA["K"]))
    positive += counts["R"] / (1.0 + 10 ** (ph - PKA["R"]))
    positive += counts["H"] / (1.0 + 10 ** (ph - PKA["H"]))

    negative = 1.0 / (1.0 + 10 ** (PKA["c_term"] - ph))
    negative += counts["D"] / (1.0 + 10 ** (PKA["D"] - ph))
    negative += counts["E"] / (1.0 + 10 ** (PKA["E"] - ph))
    negative += counts["C"] / (1.0 + 10 ** (PKA["C"] - ph))
    negative += counts["Y"] / (1.0 + 10 ** (PKA["Y"] - ph))
    return positive - negative


def isoelectric_point(sequence: str) -> float:
    low, high = 0.0, 14.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if charge_at_ph(sequence, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def molecular_weight(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(AA_MASS[aa] for aa in sequence) + WATER_MASS


def gravy(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(KD_HYDROPATHY[aa] for aa in sequence) / len(sequence)


def hydrophobic_moment(sequence: str, angle_degrees: float = 100.0, window: int = 11) -> float:
    if not sequence:
        return 0.0
    angle = math.radians(angle_degrees)

    def moment(subseq: str) -> float:
        x = 0.0
        y = 0.0
        for index, aa in enumerate(subseq):
            hydropathy = KD_HYDROPATHY[aa]
            x += hydropathy * math.cos(index * angle)
            y += hydropathy * math.sin(index * angle)
        return math.sqrt(x * x + y * y) / len(subseq)

    if len(sequence) <= window:
        return moment(sequence)
    return max(moment(sequence[start : start + window]) for start in range(0, len(sequence) - window + 1))


def shannon_entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    entropy = 0.0
    for count in counts.values():
        p = count / len(sequence)
        entropy -= p * math.log(p, 2)
    return entropy


def normalized_entropy(sequence: str) -> float:
    if len(sequence) <= 1:
        return 0.0
    return shannon_entropy(sequence) / math.log(min(len(sequence), len(AA)), 2)


def max_homopolymer_run(sequence: str) -> int:
    best = 0
    current = 0
    last = None
    for aa in sequence:
        if aa == last:
            current += 1
        else:
            current = 1
            last = aa
        best = max(best, current)
    return best


def analyze_sequence(sequence: str) -> dict:
    sequence = clean_sequence(sequence)
    length = len(sequence)
    counts = Counter(sequence)
    return {
        "length": length,
        "molecular_weight_da": round(molecular_weight(sequence), 3),
        "net_charge_simple": round(approximate_net_charge(counts), 3),
        "net_charge_ph7": round(charge_at_ph(sequence, 7.0), 3) if sequence else 0.0,
        "isoelectric_point": round(isoelectric_point(sequence), 3) if sequence else 0.0,
        "gravy": round(gravy(sequence), 4),
        "hydrophobic_moment_alpha": round(hydrophobic_moment(sequence), 4),
        "kr_frac": round(residue_fraction(counts, STRONG_BASIC, length), 4),
        "basic_frac": round(residue_fraction(counts, BASIC, length), 4),
        "acidic_frac": round(residue_fraction(counts, ACIDIC, length), 4),
        "de_frac": round(residue_fraction(counts, ACIDIC, length), 4),
        "hydrophobic_frac": round(residue_fraction(counts, HYDROPHOBIC, length), 4),
        "aromatic_frac": round(residue_fraction(counts, AROMATIC, length), 4),
        "aliphatic_frac": round(residue_fraction(counts, ALIPHATIC, length), 4),
        "polar_uncharged_frac": round(residue_fraction(counts, POLAR_UNCHARGED, length), 4),
        "cys_count": counts["C"],
        "cys_frac": round(counts["C"] / max(1, length), 4),
        "proline_frac": round(counts["P"] / max(1, length), 4),
        "glycine_frac": round(counts["G"] / max(1, length), 4),
        "sequence_entropy": round(shannon_entropy(sequence), 4),
        "normalized_entropy": round(normalized_entropy(sequence), 4),
        "max_homopolymer_run": max_homopolymer_run(sequence),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    return values[lower] * (upper - pos) + values[upper] * (pos - lower)


def summarize_group(records: list[dict], metric_keys: list[str]) -> dict:
    summary = {"n": len(records), "unique": len({record["protein"] for record in records})}
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        summary[f"{key}_mean"] = round(mean(values), 4)
        summary[f"{key}_median"] = round(median(values), 4)
        summary[f"{key}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        summary[f"{key}_p10"] = round(percentile(values, 0.10), 4)
        summary[f"{key}_p90"] = round(percentile(values, 0.90), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute direct physicochemical metrics for generated peptide JSON files.")
    parser.add_argument("--input", required=True, help="JSON/JSONL generated sequence file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-key", default=None, help="Sequence field. Auto-detects protein/sequence/seq/peptide/trg.")
    parser.add_argument("--group-key", default="prompt_name")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(input_path)
    enriched = []
    for index, record in enumerate(raw_records):
        sequence = get_sequence(record, args.sequence_key)
        if not sequence:
            continue
        metrics = analyze_sequence(sequence)
        enriched_record = {
            **record,
            "record_index": index,
            "protein": sequence,
            **metrics,
        }
        enriched.append(enriched_record)

    if not enriched:
        raise ValueError(f"No valid peptide sequences found in {input_path}")

    metric_keys = [
        "length",
        "molecular_weight_da",
        "net_charge_simple",
        "net_charge_ph7",
        "isoelectric_point",
        "gravy",
        "hydrophobic_moment_alpha",
        "kr_frac",
        "basic_frac",
        "acidic_frac",
        "de_frac",
        "hydrophobic_frac",
        "aromatic_frac",
        "aliphatic_frac",
        "polar_uncharged_frac",
        "cys_count",
        "cys_frac",
        "proline_frac",
        "glycine_frac",
        "sequence_entropy",
        "normalized_entropy",
        "max_homopolymer_run",
    ]

    groups = defaultdict(list)
    for record in enriched:
        groups[str(record.get(args.group_key, "all"))].append(record)

    summary = []
    for group_name, records in sorted(groups.items()):
        first = records[0]
        row = {
            "group": group_name,
            "natural_prompt": first.get("natural_prompt"),
            "structured_prompt": first.get("structured_prompt") or first.get("prompt"),
        }
        row.update(summarize_group(records, metric_keys))
        summary.append(row)

    overall = {"group": "ALL", "natural_prompt": None, "structured_prompt": None}
    overall.update(summarize_group(enriched, metric_keys))
    summary.append(overall)

    records_path = output_dir / "physchem_records.json"
    summary_json_path = output_dir / "physchem_summary_by_prompt.json"
    summary_csv_path = output_dir / "physchem_summary_by_prompt.csv"

    records_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(summary[0].keys())
    for row in summary[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"records={records_path}")
    print(f"summary_json={summary_json_path}")
    print(f"summary_csv={summary_csv_path}")
    print("\nSUMMARY")
    for row in summary:
        compact = {
            "group": row["group"],
            "n": row["n"],
            "length_mean": row.get("length_mean"),
            "net_charge_ph7_mean": row.get("net_charge_ph7_mean"),
            "pI_mean": row.get("isoelectric_point_mean"),
            "gravy_mean": row.get("gravy_mean"),
            "hydrophobic_moment_alpha_mean": row.get("hydrophobic_moment_alpha_mean"),
            "kr_frac_mean": row.get("kr_frac_mean"),
            "acidic_frac_mean": row.get("acidic_frac_mean"),
            "aromatic_frac_mean": row.get("aromatic_frac_mean"),
            "normalized_entropy_mean": row.get("normalized_entropy_mean"),
        }
        print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
