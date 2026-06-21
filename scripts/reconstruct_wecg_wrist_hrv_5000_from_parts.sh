#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTS_DIR="${ROOT_DIR}/ml-famae/data_wecg_wrist_hrv_5000_parts"
OUTPUT_DIR="${ROOT_DIR}/ml-famae/data_wecg_wrist_hrv_5000"

mkdir -p "${OUTPUT_DIR}"
cat "${PARTS_DIR}"/train.pt.part-* > "${OUTPUT_DIR}/train.pt"
cp "${PARTS_DIR}/val.pt" "${OUTPUT_DIR}/val.pt"
cp "${PARTS_DIR}/test.pt" "${OUTPUT_DIR}/test.pt"
cp "${PARTS_DIR}/metadata.json" "${OUTPUT_DIR}/metadata.json"

ls -lh "${OUTPUT_DIR}"
