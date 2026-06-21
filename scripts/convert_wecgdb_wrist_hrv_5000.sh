#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/ml-famae/.venv/bin/python}"

ZIP_PATH="${ZIP_PATH:-${ROOT_DIR}/wECG_dataset.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_wecg_wrist_hrv_5000}"
WINDOW_SIZE="${WINDOW_SIZE:-5000}"
STRIDE="${STRIDE:-500}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_wecgdb_wrist_hrv_5000.py" \
  --zip_path "${ZIP_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --window_size "${WINDOW_SIZE}" \
  --stride "${STRIDE}"
