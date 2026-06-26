#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/ml-famae/.venv/bin/python}"

cd "${ROOT_DIR}/ml-famae"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_wrist_2lead_as_1ch_pretrain_data.py" \
  --data_dir "${DATA_DIR:-./data_wecg_wrist_2lead_as_1ch_5000}" \
  --data_name "${DATA_NAME:-wECGdb_wrist_2lead_as_1ch}" \
  --length "${LENGTH:-5000}" \
  --batch_size "${BATCH_SIZE:-4}"
