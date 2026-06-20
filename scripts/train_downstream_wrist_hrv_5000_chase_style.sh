#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-./data_our_wrist_hrv_5000}" \
DATA_NAME="${DATA_NAME:-our_wrist_hrv}" \
LENGTH="${LENGTH:-5000}" \
OUTPUT_DIM="${OUTPUT_DIM:-4}" \
HEAD="${HEAD:-unet}" \
FREEZE_ENCODER=0 \
CHASE_STYLE=1 \
LOSS="${LOSS:-smooth_l1}" \
OPTIMIZER="${OPTIMIZER:-adamw}" \
COSINE_LR="${COSINE_LR:-1}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
LR="${LR:-5e-4}" \
ENCODER_LR="${ENCODER_LR:-1e-4}" \
HIDDEN_DIM="${HIDDEN_DIM:-128}" \
DROPOUT="${DROPOUT:-0.1}" \
OUTPUT_DIR="${OUTPUT_DIR:-./model_ckpt/our-wrist-hrv-5000-${HEAD}-from-12lead-300ep}" \
bash "${ROOT_DIR}/scripts/train_downstream_wrist_hr.sh"
