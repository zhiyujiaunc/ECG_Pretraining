#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_FAMAE_DIR="${ROOT_DIR}/ml-famae"
PYTHON_BIN="${PYTHON_BIN:-${ML_FAMAE_DIR}/.venv/bin/python}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-300}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
DATA_DIR="${DATA_DIR:-./data_wecg_12lead}"
DATA_NAME="${DATA_NAME:-wECGdb_12lead}"
MODALITY="${MODALITY:-I II III aVR aVL aVF V1 V2 V3 V4 V5 V6}"
MOD_CHANNELS="${MOD_CHANNELS:-12}"
TB_NAME="${TB_NAME:-wecg-12lead-pretrain-${EPOCHS}ep}"

cd "${ML_FAMAE_DIR}"

# shellcheck disable=SC2206
modality_array=(${MODALITY})

"${PYTHON_BIN}" -m main \
  --data_dir "${DATA_DIR}" \
  --data_name "${DATA_NAME}" \
  --train_length 3000 \
  --test_length 3000 \
  --classes 1 \
  --device "${DEVICE}" \
  --modality "${modality_array[@]}" \
  --mod_channels "${MOD_CHANNELS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --reconstruction \
  --skip_validation \
  --skip_periodic_eval \
  --disable_tqdm \
  --save_every_epoch "${SAVE_EVERY_EPOCH}" \
  --tb_name "${TB_NAME}"
