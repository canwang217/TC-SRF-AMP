#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth}"
OUT="${2:-generated_seqs/diagnostics/strict_frozen_text_10k_large}"

python scripts/diagnose_text_conditioning_path.py \
  --ckpt "$CKPT" \
  --output-dir "$OUT" \
  --num 128 \
  --generation-seed 0 1 2 3 4 5 6 7 \
  --guidance-scale 4
