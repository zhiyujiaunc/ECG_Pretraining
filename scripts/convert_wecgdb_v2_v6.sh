#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE=reference_12_lead \
CHANNEL_NAMES="V2 V6" \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_wecg_chest_v2_v6}" \
bash "${ROOT_DIR}/scripts/convert_wecgdb.sh"
