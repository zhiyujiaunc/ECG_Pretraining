#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE=wecg \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_wecg_wrist_2lead}" \
bash "${ROOT_DIR}/scripts/convert_wecgdb.sh"
