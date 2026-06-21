#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HEADS="${HEADS:-cnn lstm unet mlp}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
DATA_DIR="${DATA_DIR:?DATA_DIR is required}"
DATA_NAME="${DATA_NAME:?DATA_NAME is required}"
MODALITY="${MODALITY:?MODALITY is required}"
MOD_CHANNELS="${MOD_CHANNELS:?MOD_CHANNELS is required}"
MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
REPORT_DEVICE="${REPORT_DEVICE:-${DEVICE}}"

for head in ${HEADS}; do
  echo "================================================================================"
  echo "Training downstream head: ${head}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Data: ${DATA_DIR} (${DATA_NAME}; ${MODALITY})"
  echo "================================================================================"
  CHECKPOINT="${CHECKPOINT}" \
  DATA_DIR="${DATA_DIR}" \
  DATA_NAME="${DATA_NAME}" \
  MODALITY="${MODALITY}" \
  MOD_CHANNELS="${MOD_CHANNELS}" \
  LENGTH=5000 \
  OUTPUT_DIM=4 \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  HEAD="${head}" \
  OUTPUT_DIR="${MODEL_ROOT}/${head}" \
  bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_5000_chase_style.sh"

  cd "${ROOT_DIR}/ml-famae"
  "${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv.py \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${MODEL_ROOT}/${head}/best.pt" \
    --device "${REPORT_DEVICE}"
  cd "${ROOT_DIR}"
done
