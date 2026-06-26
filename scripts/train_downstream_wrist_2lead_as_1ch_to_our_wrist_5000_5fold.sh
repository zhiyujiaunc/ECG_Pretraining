#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-mps}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
HEAD="${HEAD:-unet}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULT_ROOT="${RESULT_ROOT:-./model_ckpt/downstream-5000-wrist2lead-as-1ch-to-our-wrist-5fold}"
FOLD_ROOT="${FOLD_ROOT:-./data_our_wrist_hrv_5000_5fold}"
CHECKPOINT="${CHECKPOINT:-./model_ckpt/wecg-wrist-2lead-as-1ch-5000-pretrain-300ep-bs64_bioFAME/ckpt.pt}"
TASK_NAME="task4b-wrist2lead-as-1ch-5000-to-our-wrist1ch"

for fold in 0 1 2 3 4; do
  echo "================================================================================"
  echo "Training ${TASK_NAME} | head=${HEAD} | fold=${fold}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Input: our wrist ECG 1-channel"
  echo "================================================================================"
  CHECKPOINT="${CHECKPOINT}" \
  DATA_DIR="${FOLD_ROOT}/fold_${fold}" \
  DATA_NAME="our_wrist_hrv" \
  MODALITY="wrist" \
  MOD_CHANNELS=1 \
  LENGTH=5000 \
  OUTPUT_DIM=4 \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  HEAD="${HEAD}" \
  OUTPUT_DIR="${RESULT_ROOT}/${TASK_NAME}/${HEAD}/fold_${fold}" \
  bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_5000_chase_style.sh"
done

cd "${ROOT_DIR}/ml-famae"
"${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv_5fold.py \
  --fold_root "${FOLD_ROOT}" \
  --model_root "${RESULT_ROOT}/${TASK_NAME}/${HEAD}" \
  --device "${REPORT_DEVICE}" \
  --n_folds 5
