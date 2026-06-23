#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/ml-famae/.venv/bin/python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/flatten_12lead_pt_as_1ch.py \
  --input_dir "${INPUT_DIR:-ml-famae/data_wecg_12lead_5000}" \
  --output_dir "${OUTPUT_DIR:-ml-famae/data_wecg_12lead_as_1ch_5000}"
