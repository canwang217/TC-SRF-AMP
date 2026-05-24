#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TC_SRF_MODEL_NAME_OR_PATH="${TC_SRF_MODEL_NAME_OR_PATH:-$PWD/../esm2_t12_35M_UR50D}"
export TC_SRF_TEXT_ENCODER_NAME_OR_PATH="${TC_SRF_TEXT_ENCODER_NAME_OR_PATH:-$PWD/../scibert_scivocab_uncased}"

python train_flow_matching.py 2>&1 | tee train_flow_matching_repro.log
