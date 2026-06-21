#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
HEADS="${HEADS:-cnn lstm unet mlp}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULT_ROOT="${RESULT_ROOT:-./model_ckpt/downstream-5000-task2-lead2-5fold}"
FOLD_ROOT="${FOLD_ROOT:-./data_wecg_wrist_hrv_5000_lead2_5fold}"
CHECKPOINT="${CHECKPOINT:-./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt}"
TASK_NAME="task2-12lead5000-to-wecgdb-wrist2ch-lead2gt"

for head in ${HEADS}; do
  for fold in 0 1 2 3 4; do
    echo "================================================================================"
    echo "Training ${TASK_NAME} | head=${head} | fold=${fold}"
    echo "Input: wECGdb wrist 2-channel | Ground truth: 12-lead reference Lead II"
    echo "Checkpoint: ${CHECKPOINT}"
    echo "================================================================================"
    CHECKPOINT="${CHECKPOINT}" \
    DATA_DIR="${FOLD_ROOT}/fold_${fold}" \
    DATA_NAME="wECGdb_wrist_hrv_5000" \
    MODALITY="lead_I lead_wrist" \
    MOD_CHANNELS=2 \
    LENGTH=5000 \
    OUTPUT_DIM=4 \
    DEVICE="${DEVICE}" \
    EPOCHS="${EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    HEAD="${head}" \
    OUTPUT_DIR="${RESULT_ROOT}/${TASK_NAME}/${head}/fold_${fold}" \
    bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_5000_chase_style.sh"
  done

  cd "${ROOT_DIR}/ml-famae"
  "${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv_5fold.py \
    --fold_root "${FOLD_ROOT}" \
    --model_root "${RESULT_ROOT}/${TASK_NAME}/${head}" \
    --device "${REPORT_DEVICE}" \
    --n_folds 5
  cd "${ROOT_DIR}"
done
