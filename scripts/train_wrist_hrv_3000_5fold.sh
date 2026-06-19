#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-30}"
MODEL_ROOT="${MODEL_ROOT:-./model_ckpt/our-wrist-hrv-3000-5fold}"

for fold in 0 1 2 3 4; do
  echo "================================================================================"
  echo "Training fold ${fold}"
  echo "================================================================================"
  DATA_DIR="./data_our_wrist_hrv_3000_5fold/fold_${fold}" \
  DATA_NAME="our_wrist_hrv_3000" \
  LENGTH=3000 \
  OUTPUT_DIM=3 \
  EPOCHS="${EPOCHS}" \
  DEVICE="${DEVICE}" \
  OUTPUT_DIR="${MODEL_ROOT}/fold_${fold}" \
  bash "${ROOT_DIR}/scripts/train_downstream_wrist_hr.sh"
done

cd "${ROOT_DIR}/ml-famae"
"${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv_5fold.py \
  --fold_root ./data_our_wrist_hrv_3000_5fold \
  --model_root "${MODEL_ROOT}" \
  --device "${DEVICE}" \
  --n_folds 5
