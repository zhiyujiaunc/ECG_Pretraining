#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

ZIP_PATH="${ZIP_PATH:-${ROOT_DIR}/wECG_dataset.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${ML_FAMAE_DIR}/data_wecg_custom}"
SOURCE="${SOURCE:-reference_12_lead}"
WINDOW_SIZE="${WINDOW_SIZE:-3000}"
STRIDE="${STRIDE:-1500}"
SEED="${SEED:-0}"

cd "${ML_FAMAE_DIR}"

cmd=(
  "${PYTHON_BIN}" data/convert_wecgdb.py
  --zip_path "${ZIP_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --source "${SOURCE}"
  --window_size "${WINDOW_SIZE}"
  --stride "${STRIDE}"
  --seed "${SEED}"
)

if [[ -n "${CHANNEL_NAMES:-}" ]]; then
  # shellcheck disable=SC2206
  channel_names_array=(${CHANNEL_NAMES})
  cmd+=(--channel_names "${channel_names_array[@]}")
fi

if [[ -n "${CHANNEL_INDICES:-}" ]]; then
  # shellcheck disable=SC2206
  channel_indices_array=(${CHANNEL_INDICES})
  cmd+=(--channel_indices "${channel_indices_array[@]}")
fi

cmd+=("$@")

"${cmd[@]}"
