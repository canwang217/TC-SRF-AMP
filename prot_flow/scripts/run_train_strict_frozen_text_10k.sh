#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export TC_SRF_PRETRAIN_CKPT="${TC_SRF_PRETRAIN_CKPT:-$PWD/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth}"
export TC_SRF_USE_PRETRAIN="${TC_SRF_USE_PRETRAIN:-1}"
export TC_SRF_CHECKPOINT_DIR="${TC_SRF_CHECKPOINT_DIR:-$PWD/checkpoints/tc_srf_amp_strict_frozen_text_10k_repro}"

export TC_SRF_AMP_TRAIN_JSONL="${TC_SRF_AMP_TRAIN_JSONL:-$PWD/../ensemble_train_structured.jsonl}"
export TC_SRF_AMP_VALID_JSONL="${TC_SRF_AMP_VALID_JSONL:-$PWD/../ensemble_test_structured.jsonl}"

export TC_SRF_USE_LABEL_CONDITION=0
export TC_SRF_USE_ATTRIBUTE_CONDITION=0
export TC_SRF_FREEZE_TEXT_ENCODER=1

export TC_SRF_BATCH_SIZE="${TC_SRF_BATCH_SIZE:-8}"
export TC_SRF_TRAINING_ITERS="${TC_SRF_TRAINING_ITERS:-10000}"
export TC_SRF_LR="${TC_SRF_LR:-3e-5}"
export TC_SRF_CFG_DROPOUT="${TC_SRF_CFG_DROPOUT:-0.10}"
export TC_SRF_CC_COEF="${TC_SRF_CC_COEF:-0.3}"
export TC_SRF_COND_COEF="${TC_SRF_COND_COEF:-0.2}"
export TC_SRF_LATENT_REC_COEF="${TC_SRF_LATENT_REC_COEF:-0.1}"

export TC_SRF_CHECKPOINT_FREQ="${TC_SRF_CHECKPOINT_FREQ:-2000}"
export TC_SRF_EVAL_FREQ="${TC_SRF_EVAL_FREQ:-10000}"
export TC_SRF_GENERATE_FREQ="${TC_SRF_GENERATE_FREQ:-10000}"
export TC_SRF_NUM_GEN_TEXTS="${TC_SRF_NUM_GEN_TEXTS:-16}"
export TC_SRF_VALIDATION_ITERS="${TC_SRF_VALIDATION_ITERS:-30}"
export TC_SRF_EVAL_ON_START=0

bash scripts/run_server_finetune.sh 2>&1 | tee train_strict_frozen_text_10k_repro.log
