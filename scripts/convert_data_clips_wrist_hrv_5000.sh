#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/ml-famae/.venv/bin/python}"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_clips}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_our_wrist_hrv_5000}"
SOURCE_FS="${SOURCE_FS:-2000}"
TARGET_FS="${TARGET_FS:-500}"
WINDOW_SIZE="${WINDOW_SIZE:-5000}"
STRIDE="${STRIDE:-500}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_data_clips_wrist_hr.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --source_fs "${SOURCE_FS}" \
  --target_fs "${TARGET_FS}" \
  --window_size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --include_pnn50
