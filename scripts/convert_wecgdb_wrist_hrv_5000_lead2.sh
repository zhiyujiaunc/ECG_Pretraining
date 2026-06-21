#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/convert_wecgdb_wrist_hrv_5000.py \
  --zip_path "${ZIP_PATH:-wECG_dataset.zip}" \
  --output_dir "${OUTPUT_DIR:-ml-famae/data_wecg_wrist_hrv_5000_lead2}" \
  --window_size "${WINDOW_SIZE:-5000}" \
  --stride "${STRIDE:-500}" \
  --label_source reference_12_lead \
  --label_channel 1 \
  --dataset_name "wECGdb Wrist with Lead II GT" \
  --val_ratio "${VAL_RATIO:-0.1}" \
  --test_ratio "${TEST_RATIO:-0.1}" \
  --seed "${SEED:-0}"
