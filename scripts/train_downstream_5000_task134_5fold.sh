#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
HEADS="${HEADS:-cnn lstm unet mlp}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULT_ROOT="${RESULT_ROOT:-./model_ckpt/downstream-5000-task134-5fold}"
FOLD_ROOT="${FOLD_ROOT:-./data_our_wrist_hrv_5000_5fold}"

run_task() {
  local task_name="$1"
  local checkpoint="$2"

  for head in ${HEADS}; do
    for fold in 0 1 2 3 4; do
      echo "================================================================================"
      echo "Training ${task_name} | head=${head} | fold=${fold}"
      echo "Checkpoint: ${checkpoint}"
      echo "================================================================================"
      CHECKPOINT="${checkpoint}" \
      DATA_DIR="${FOLD_ROOT}/fold_${fold}" \
      DATA_NAME="our_wrist_hrv" \
      MODALITY="wrist" \
      MOD_CHANNELS=1 \
      LENGTH=5000 \
      OUTPUT_DIM=4 \
      DEVICE="${DEVICE}" \
      EPOCHS="${EPOCHS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      HEAD="${head}" \
      OUTPUT_DIR="${RESULT_ROOT}/${task_name}/${head}/fold_${fold}" \
      bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_5000_chase_style.sh"
    done

    cd "${ROOT_DIR}/ml-famae"
    "${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv_5fold.py \
      --fold_root "${FOLD_ROOT}" \
      --model_root "${RESULT_ROOT}/${task_name}/${head}" \
      --device "${REPORT_DEVICE}" \
      --n_folds 5
    cd "${ROOT_DIR}"
  done
}

run_task \
  "task1-12lead5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt"

run_task \
  "task3-v2v6-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-chest-v2-v6-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt"

run_task \
  "task4-wrist2lead-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-wrist-2lead-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt"
