#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

cd "${ML_FAMAE_DIR}"

"${PYTHON_BIN}" make_wrist_hrv_folds.py \
  --data_dir ./data_wecg_wrist_hrv_5000_lead2 \
  --output_root ./data_wecg_wrist_hrv_5000_lead2_5fold \
  --n_folds 5 \
  --split_mode intra_subject \
  --fold_unit window \
  --val_ratio_of_trainval 0.2 \
  --seed "${SEED:-0}"
