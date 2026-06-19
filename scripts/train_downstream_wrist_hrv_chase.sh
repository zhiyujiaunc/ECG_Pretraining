#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-./data_our_wrist_hrv}" \
DATA_NAME="${DATA_NAME:-our_wrist_hrv}" \
LENGTH="${LENGTH:-5000}" \
OUTPUT_DIM="${OUTPUT_DIM:-3}" \
OUTPUT_DIR="${OUTPUT_DIR:-./model_ckpt/our-wrist-hrv-${HEAD:-cnn}-from-12lead-50ep}" \
bash "${ROOT_DIR}/scripts/train_downstream_wrist_hr.sh"
