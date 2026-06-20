#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LENGTH="${LENGTH:-5000}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
DATA_DIR="${DATA_DIR:-./data_wecg_12lead_5000}" \
DATA_NAME="${DATA_NAME:-wECGdb_12lead}" \
MODALITY="${MODALITY:-I II III aVR aVL aVF V1 V2 V3 V4 V5 V6}" \
MOD_CHANNELS="${MOD_CHANNELS:-12}" \
TB_NAME="${TB_NAME:-wecg-12lead-5000-pretrain-${EPOCHS:-300}ep-bs${BATCH_SIZE:-8}}" \
bash "${ROOT_DIR}/scripts/train_wecg_12lead.sh"
