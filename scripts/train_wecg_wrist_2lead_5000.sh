#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LENGTH="${LENGTH:-5000}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
DATA_DIR="${DATA_DIR:-./data_wecg_wrist_2lead_5000}" \
DATA_NAME="${DATA_NAME:-wECGdb_wrist}" \
MODALITY="${MODALITY:-lead_I lead_wrist}" \
MOD_CHANNELS="${MOD_CHANNELS:-2}" \
TB_NAME="${TB_NAME:-wecg-wrist-2lead-5000-pretrain-${EPOCHS:-300}ep-bs${BATCH_SIZE:-32}}" \
bash "${ROOT_DIR}/scripts/train_wecg_12lead.sh"
