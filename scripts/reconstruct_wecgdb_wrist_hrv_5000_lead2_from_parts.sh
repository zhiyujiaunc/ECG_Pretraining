#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/ml-famae/data_wecg_wrist_hrv_5000_lead2"
PART_DIR="${ROOT_DIR}/ml-famae/data_wecg_wrist_hrv_5000_lead2_parts"

mkdir -p "${DATA_DIR}"

if [[ -d "${PART_DIR}" ]]; then
  for split in train val test; do
    if compgen -G "${PART_DIR}/${split}.pt.part-*" > /dev/null; then
      cat "${PART_DIR}/${split}.pt.part-"* > "${DATA_DIR}/${split}.pt"
      echo "reconstructed ${DATA_DIR}/${split}.pt"
    elif [[ -f "${PART_DIR}/${split}.pt" ]]; then
      cp "${PART_DIR}/${split}.pt" "${DATA_DIR}/${split}.pt"
      echo "copied ${DATA_DIR}/${split}.pt"
    fi
  done
fi

if [[ -f "${PART_DIR}/metadata.json" ]]; then
  cp "${PART_DIR}/metadata.json" "${DATA_DIR}/metadata.json"
fi
