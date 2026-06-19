#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DATA_DIR="${DATA_DIR:-./data_wecg_probe}"
TB_NAME="${TB_NAME:-wecg-smoke-test}"

cd "${ML_FAMAE_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Smoke-test data directory not found: ${ML_FAMAE_DIR}/${DATA_DIR}" >&2
  echo "Either set DATA_DIR=./data_wecg_12lead, or create a small probe dataset first:" >&2
  echo "  cd ${ROOT_DIR}" >&2
  echo "  WINDOW_SIZE=3000 STRIDE=3000 OUTPUT_DIR=${ML_FAMAE_DIR}/data_wecg_probe bash scripts/convert_wecg_12lead.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" -m main \
  --data_dir "${DATA_DIR}" \
  --data_name wECGdb_12lead \
  --train_length 3000 \
  --test_length 3000 \
  --classes 1 \
  --device "${DEVICE}" \
  --modality I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 \
  --mod_channels 12 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs 1 \
  --reconstruction \
  --skip_validation \
  --skip_periodic_eval \
  --disable_tqdm \
  --save_every_epoch 1 \
  --tb_name "${TB_NAME}"
