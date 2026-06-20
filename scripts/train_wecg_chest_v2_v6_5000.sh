#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LENGTH="${LENGTH:-5000}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
DATA_DIR="${DATA_DIR:-./data_wecg_chest_v2_v6_5000}" \
DATA_NAME="${DATA_NAME:-wECGdb_chest_v2_v6}" \
MODALITY="${MODALITY:-V2 V6}" \
MOD_CHANNELS="${MOD_CHANNELS:-2}" \
TB_NAME="${TB_NAME:-wecg-chest-v2-v6-5000-pretrain-${EPOCHS:-300}ep-bs${BATCH_SIZE:-32}}" \
bash "${ROOT_DIR}/scripts/train_wecg_12lead.sh"
