#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth}"
OUT="${2:-generated_seqs/diagnostics/strict_nl_parser_g4_fixed}"

python scripts/run_strict_nl_prompt_benchmark.py \
  --ckpt "$CKPT" \
  --output-dir "$OUT" \
  --num 128 \
  --seed 0 1 2 3 \
  --guidance-scale 4 \
  --prompt short_highpos="Generate a short highly cationic antimicrobial peptide rich in lysine and arginine with no cysteine." \
  --prompt long_negative="Generate a long negatively charged peptide with low lysine and arginine content and no cysteine." \
  --prompt medium_hydrophobic="Design a medium-length active peptide with high hydrophobicity and no cysteine." \
  --prompt inactive_long_neutral="Generate a long inactive-like peptide with neutral charge, low KR content, and no cysteine."
