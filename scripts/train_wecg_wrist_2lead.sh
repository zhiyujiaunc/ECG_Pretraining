#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-./data_wecg_wrist_2lead}" \
DATA_NAME="${DATA_NAME:-wECGdb_wrist}" \
MODALITY="${MODALITY:-lead_I lead_wrist}" \
MOD_CHANNELS="${MOD_CHANNELS:-2}" \
TB_NAME="${TB_NAME:-wecg-wrist-2lead-pretrain-${EPOCHS:-300}ep}" \
bash "${ROOT_DIR}/scripts/train_wecg_12lead.sh"
