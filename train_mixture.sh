#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CONFIG="${CONFIG:-configs/mixture_full_11datasets.yaml}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EPOCHS="${EPOCHS:-500000}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture}"
PRETRAIN="${PRETRAIN:-/data1/zbf/my_dfrt/outputs_all_noraw/load3240/full/checkpoint_epoch_3460.pth}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python train_mixture.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --output-dir "$OUTPUT_DIR" \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  "$@"
