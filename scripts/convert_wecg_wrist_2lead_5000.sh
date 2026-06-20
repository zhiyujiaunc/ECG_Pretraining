#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE=wecg \
WINDOW_SIZE="${WINDOW_SIZE:-5000}" \
STRIDE="${STRIDE:-2500}" \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_wecg_wrist_2lead_5000}" \
bash "${ROOT_DIR}/scripts/convert_wecgdb.sh"
