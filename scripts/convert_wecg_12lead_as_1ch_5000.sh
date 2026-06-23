#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE=reference_12_lead \
WINDOW_SIZE="${WINDOW_SIZE:-5000}" \
STRIDE="${STRIDE:-2500}" \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ml-famae/data_wecg_12lead_as_1ch_5000}" \
EXTRA_ARGS="${EXTRA_ARGS:-}" \
bash "${ROOT_DIR}/scripts/convert_wecgdb.sh" --flatten_channels_as_samples
