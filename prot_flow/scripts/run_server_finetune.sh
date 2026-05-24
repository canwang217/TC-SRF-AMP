#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/.cache/huggingface}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

export TC_SRF_PRETRAIN_CKPT="${TC_SRF_PRETRAIN_CKPT:-$PWD/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth}"
export TC_SRF_CHECKPOINT_DIR="${TC_SRF_CHECKPOINT_DIR:-$PWD/checkpoints/tc_srf_amp}"
export TC_SRF_BATCH_SIZE="${TC_SRF_BATCH_SIZE:-8}"
export TC_SRF_TRAINING_ITERS="${TC_SRF_TRAINING_ITERS:-50000}"
export TC_SRF_CHECKPOINT_FREQ="${TC_SRF_CHECKPOINT_FREQ:-2000}"
export TC_SRF_EVAL_FREQ="${TC_SRF_EVAL_FREQ:-2000}"
export TC_SRF_GENERATE_FREQ="${TC_SRF_GENERATE_FREQ:-2000}"
export TC_SRF_EVAL_ON_START="${TC_SRF_EVAL_ON_START:-0}"

NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-31345}"

if [[ "$NPROC" == "1" ]]; then
  python train_tc_srf_amp.py
else
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_tc_srf_amp.py
fi

