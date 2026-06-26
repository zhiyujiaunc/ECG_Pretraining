#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LENGTH="${LENGTH:-5000}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
DATA_DIR="${DATA_DIR:-./data_wecg_wrist_2lead_as_1ch_5000}" \
DATA_NAME="${DATA_NAME:-wECGdb_wrist_2lead_as_1ch}" \
MODALITY="${MODALITY:-ecg}" \
MOD_CHANNELS="${MOD_CHANNELS:-1}" \
TB_NAME="${TB_NAME:-wecg-wrist-2lead-as-1ch-5000-pretrain-${EPOCHS:-300}ep-bs${BATCH_SIZE:-64}}" \
bash "${ROOT_DIR}/scripts/train_wecg_12lead.sh"
