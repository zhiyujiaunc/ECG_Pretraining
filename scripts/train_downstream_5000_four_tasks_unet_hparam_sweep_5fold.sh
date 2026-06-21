#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULT_ROOT="${RESULT_ROOT:-./model_ckpt/downstream-5000-four-tasks-unet-hparam-5fold}"
OUR_FOLD_ROOT="${OUR_FOLD_ROOT:-./data_our_wrist_hrv_5000_5fold}"
TASK2_FOLD_ROOT="${TASK2_FOLD_ROOT:-./data_wecg_wrist_hrv_5000_lead2_5fold}"
CONFIGS="${CONFIGS:-baseline lr1e-3 lr1e-4 hidden256 dropout03 unfreeze1}"

config_params() {
  local config="$1"
  case "${config}" in
    baseline)
      echo "LR=5e-4 ENCODER_LR=1e-4 HIDDEN_DIM=128 DROPOUT=0.1 UNFREEZE_LAST_N=0 WEIGHT_DECAY=1e-4"
      ;;
    lr1e-3)
      echo "LR=1e-3 ENCODER_LR=1e-4 HIDDEN_DIM=128 DROPOUT=0.1 UNFREEZE_LAST_N=0 WEIGHT_DECAY=1e-4"
      ;;
    lr1e-4)
      echo "LR=1e-4 ENCODER_LR=5e-5 HIDDEN_DIM=128 DROPOUT=0.1 UNFREEZE_LAST_N=0 WEIGHT_DECAY=1e-4"
      ;;
    hidden256)
      echo "LR=5e-4 ENCODER_LR=1e-4 HIDDEN_DIM=256 DROPOUT=0.1 UNFREEZE_LAST_N=0 WEIGHT_DECAY=1e-4"
      ;;
    dropout03)
      echo "LR=5e-4 ENCODER_LR=1e-4 HIDDEN_DIM=128 DROPOUT=0.3 UNFREEZE_LAST_N=0 WEIGHT_DECAY=1e-4"
      ;;
    unfreeze1)
      echo "LR=5e-4 ENCODER_LR=5e-5 HIDDEN_DIM=128 DROPOUT=0.1 UNFREEZE_LAST_N=1 WEIGHT_DECAY=1e-4"
      ;;
    *)
      echo "Unknown config: ${config}" >&2
      exit 1
      ;;
  esac
}

run_one_task() {
  local task_name="$1"
  local checkpoint="$2"
  local fold_root="$3"
  local data_name="$4"
  local modality="$5"
  local mod_channels="$6"

  for config in ${CONFIGS}; do
    local params
    params="$(config_params "${config}")"
    # shellcheck disable=SC2086
    eval "${params}"

    for fold in 0 1 2 3 4; do
      echo "================================================================================"
      echo "Training ${task_name} | head=unet | config=${config} | fold=${fold}"
      echo "Checkpoint: ${checkpoint}"
      echo "Data: ${fold_root}/fold_${fold} (${data_name}; ${modality}; ${mod_channels}ch)"
      echo "Hyperparams: ${params}"
      echo "================================================================================"
      CHECKPOINT="${checkpoint}" \
      DATA_DIR="${fold_root}/fold_${fold}" \
      DATA_NAME="${data_name}" \
      MODALITY="${modality}" \
      MOD_CHANNELS="${mod_channels}" \
      LENGTH=5000 \
      OUTPUT_DIM=4 \
      DEVICE="${DEVICE}" \
      EPOCHS="${EPOCHS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      HEAD="unet" \
      LR="${LR}" \
      ENCODER_LR="${ENCODER_LR}" \
      HIDDEN_DIM="${HIDDEN_DIM}" \
      DROPOUT="${DROPOUT}" \
      UNFREEZE_LAST_N="${UNFREEZE_LAST_N}" \
      WEIGHT_DECAY="${WEIGHT_DECAY}" \
      OUTPUT_DIR="${RESULT_ROOT}/${task_name}/${config}/unet/fold_${fold}" \
      bash "${ROOT_DIR}/scripts/train_downstream_wrist_hrv_5000_chase_style.sh"
    done

    cd "${ROOT_DIR}/ml-famae"
    "${PYTHON_BIN:-./.venv/bin/python}" report_wrist_hrv_5fold.py \
      --fold_root "${fold_root}" \
      --model_root "${RESULT_ROOT}/${task_name}/${config}/unet" \
      --device "${REPORT_DEVICE}" \
      --n_folds 5
    cd "${ROOT_DIR}"
  done
}

run_one_task \
  "task1-12lead5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt" \
  "${OUR_FOLD_ROOT}" \
  "our_wrist_hrv" \
  "wrist" \
  "1"

run_one_task \
  "task2-12lead5000-to-wecgdb-wrist2ch-lead2gt" \
  "./model_ckpt/wecg-12lead-5000-pretrain-300ep-bs4_bioFAME/ckpt.pt" \
  "${TASK2_FOLD_ROOT}" \
  "wECGdb_wrist_hrv_5000" \
  "lead_I lead_wrist" \
  "2"

run_one_task \
  "task3-v2v6-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-chest-v2-v6-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt" \
  "${OUR_FOLD_ROOT}" \
  "our_wrist_hrv" \
  "wrist" \
  "1"

run_one_task \
  "task4-wrist2lead-5000-to-our-wrist1ch" \
  "./model_ckpt/wecg-wrist-2lead-5000-pretrain-300ep-bs16_bioFAME/ckpt.pt" \
  "${OUR_FOLD_ROOT}" \
  "our_wrist_hrv" \
  "wrist" \
  "1"
