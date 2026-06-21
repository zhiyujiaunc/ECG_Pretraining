#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-30}"
DATA_DIR="${DATA_DIR:-./data_our_wrist_hr}"
DATA_NAME="${DATA_NAME:-our_wrist_hr}"
CHECKPOINT="${CHECKPOINT:-./model_ckpt/wecg-12lead-pretrain-50ep_bioFAME/ckpt.pt}"
HEAD="${HEAD:-cnn}"
LENGTH="${LENGTH:-3000}"
OUTPUT_DIM="${OUTPUT_DIM:-1}"
MODALITY="${MODALITY:-wrist}"
MOD_CHANNELS="${MOD_CHANNELS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_ckpt/our-wrist-hr-${HEAD}-from-12lead-50ep}"
LR="${LR:-1e-3}"
ENCODER_LR="${ENCODER_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.2}"
LOSS="${LOSS:-mse}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CLIP="${GRAD_CLIP:-0}"
UNFREEZE_LAST_N="${UNFREEZE_LAST_N:-0}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
CHASE_STYLE="${CHASE_STYLE:-0}"
COSINE_LR="${COSINE_LR:-0}"

EXTRA_ARGS=()
if [[ "${FREEZE_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_encoder)
fi
if [[ "${CHASE_STYLE}" == "1" ]]; then
  EXTRA_ARGS+=(--chase_style)
fi
if [[ "${COSINE_LR}" == "1" ]]; then
  EXTRA_ARGS+=(--cosine_lr)
fi

cd "${ML_FAMAE_DIR}"

# shellcheck disable=SC2206
modality_array=(${MODALITY})

"${PYTHON_BIN}" downstream_wrist_hr.py \
  --data_dir "${DATA_DIR}" \
  --data_name "${DATA_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --length "${LENGTH}" \
  --output_dim "${OUTPUT_DIM}" \
  --modality "${modality_array[@]}" \
  --mod_channels "${MOD_CHANNELS}" \
  --head "${HEAD}" \
  --lr "${LR}" \
  --encoder_lr "${ENCODER_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --loss "${LOSS}" \
  --optimizer "${OPTIMIZER}" \
  --grad_clip "${GRAD_CLIP}" \
  --unfreeze_last_n "${UNFREEZE_LAST_N}" \
  "${EXTRA_ARGS[@]}"
