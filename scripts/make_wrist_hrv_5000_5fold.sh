#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

cd "${ML_FAMAE_DIR}"

"${PYTHON_BIN}" make_wrist_hrv_folds.py \
  --data_dir ./data_our_wrist_hrv_5000 \
  --output_root ./data_our_wrist_hrv_5000_5fold \
  --n_folds 5 \
  --split_mode intra_subject \
  --val_ratio_of_trainval 0.2 \
  --seed 0
