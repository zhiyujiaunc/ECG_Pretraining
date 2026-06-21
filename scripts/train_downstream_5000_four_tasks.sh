#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
HEADS="${HEADS:-cnn lstm unet mlp}"
BATCH_SIZE_OUR="${BATCH_SIZE_OUR:-32}"
BATCH_SIZE_WECG="${BATCH_SIZE_WECG:-32}"
RESULT_ROOT="${RESULT_ROOT:-./model_ckpt/downstream-5000-four-tasks}"

run_task() {
  local task_name="$1"
  local checkpoint="$2"
  local data_dir="$3"
  local data_name="$4"
  local modality="$5"
  local mod_channels="$6"
  local batch_size="$7"

  echo "################################################################################"
  echo "Running ${task_name}"
  echo "################################################################################"
  CHECKPOINT="${checkpoint}" \
  DATA_DIR="${data_dir}" \
  DATA_NAME="${data_name}" \
  MODALITY="${modality}" \
  MOD_CHANNELS="${mod_channels}" \
  MODEL_ROOT="${RESULT_ROOT}/${task_name}" \
  DEVICE="${DEVICE}" \
  REPORT_DEVICE="${REPORT_DEVICE}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${batch_size}" \
  HEADS="${HEADS}" \
  bash "${ROOT_DIR}/scripts/train_downstream_hrv_5000_head_sweep.sh"
}

run_task \
  "task1-12lead5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt" \
  "./data_our_wrist_hrv_5000" \
  "our_wrist_hrv" \
  "wrist" \
  "1" \
  "${BATCH_SIZE_OUR}"

run_task \
  "task2-12lead5000-to-wecgdb-wrist2ch" \
  "./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt" \
  "./data_wecg_wrist_hrv_5000" \
  "wECGdb_wrist_hrv_5000" \
  "lead_I lead_wrist" \
  "2" \
  "${BATCH_SIZE_WECG}"

run_task \
  "task3-v2v6-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-chest-v2-v6-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt" \
  "./data_our_wrist_hrv_5000" \
  "our_wrist_hrv" \
  "wrist" \
  "1" \
  "${BATCH_SIZE_OUR}"

run_task \
  "task4-wrist2lead-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-wrist-2lead-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt" \
  "./data_our_wrist_hrv_5000" \
  "our_wrist_hrv" \
  "wrist" \
  "1" \
  "${BATCH_SIZE_OUR}"
