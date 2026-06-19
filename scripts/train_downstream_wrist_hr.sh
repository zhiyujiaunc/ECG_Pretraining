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
OUTPUT_DIR="${OUTPUT_DIR:-./model_ckpt/our-wrist-hr-${HEAD}-from-12lead-50ep}"

cd "${ML_FAMAE_DIR}"

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
  --head "${HEAD}" \
  --freeze_encoder
