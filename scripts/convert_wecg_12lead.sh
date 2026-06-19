#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

ZIP_PATH="${ZIP_PATH:-${ROOT_DIR}/wECG_dataset.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${ML_FAMAE_DIR}/data_wecg_12lead}"
WINDOW_SIZE="${WINDOW_SIZE:-3000}"
STRIDE="${STRIDE:-1500}"
SOURCE="${SOURCE:-reference_12_lead}"
SEED="${SEED:-0}"

cd "${ML_FAMAE_DIR}"

"${PYTHON_BIN}" data/convert_wecgdb.py \
  --zip_path "${ZIP_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --source "${SOURCE}" \
  --window_size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --seed "${SEED}"
