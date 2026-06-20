#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HEADS="${HEADS:-cnn lstm unet mlp}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CHECKPOINT="${CHECKPOINT:-./model_ckpt/wecg-12lead-pretrain-300ep-bs16_bioFAME/ckpt.pt}"
MODEL_ROOT="${MODEL_ROOT:-./model_ckpt/downstream-head-sweep-hrv-3000-from-12lead-300ep}"

for head in ${HEADS}; do
  echo "================================================================================"
  echo "Training downstream head: ${head}"
  echo "================================================================================"
  CHECKPOINT="${CHECKPOINT}" \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  HEAD="${head}" \
  OUTPUT_DIR="${MODEL_ROOT}/${head}" \
  bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_3000_chase_style.sh"

  cd "${ROOT_DIR}/ml-famae"
  "${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv.py \
    --data_dir ./data_our_wrist_hrv_3000 \
    --checkpoint "${MODEL_ROOT}/${head}/best.pt" \
    --device "${DEVICE}"
  cd "${ROOT_DIR}"
done
