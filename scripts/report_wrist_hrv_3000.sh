#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

DEVICE="${DEVICE:-mps}"
DATA_DIR="${DATA_DIR:-./data_our_wrist_hrv_3000}"
CHECKPOINT="${CHECKPOINT:-./model_ckpt/our-wrist-hrv-3000-cnn-run2-from-12lead-50ep/best.pt}"
SPLITS="${SPLITS:-test}"

cd "${ML_FAMAE_DIR}"

# shellcheck disable=SC2206
split_array=(${SPLITS})

"${PYTHON_BIN}" report_wrist_hrv.py \
  --data_dir "${DATA_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --splits "${split_array[@]}" \
  --device "${DEVICE}"
