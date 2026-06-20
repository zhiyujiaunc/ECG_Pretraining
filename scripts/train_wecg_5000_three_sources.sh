#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/run_logs}"
mkdir -p "${LOG_DIR}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"

run_one() {
  local name="$1"
  local script="$2"
  local batch_size="$3"

  echo "================================================================================"
  echo "Pretraining ${name}"
  echo "================================================================================"
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
  BATCH_SIZE="${batch_size}" \
  bash "${ROOT_DIR}/scripts/${script}" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_one "wecg-12lead-5000-pretrain" "train_wecg_12lead_5000.sh" "${BATCH_SIZE_12LEAD:-8}"
run_one "wecg-chest-v2-v6-5000-pretrain" "train_wecg_chest_v2_v6_5000.sh" "${BATCH_SIZE_2CH:-32}"
run_one "wecg-wrist-2lead-5000-pretrain" "train_wecg_wrist_2lead_5000.sh" "${BATCH_SIZE_WRIST:-32}"
