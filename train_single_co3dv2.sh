#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
CONFIG="${CONFIG:-configs/single_co3dv2.yaml}"
VAL_CONFIG="${VAL_CONFIG:-configs/single_co3dv2_val.yaml}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/single_co3dv2}"
PRETRAIN="${PRETRAIN:-}"
RESUME="${RESUME:-/data/zbf/openclaw/d4rt/outputs/single_co3dv2/checkpoint_latest_15.pth}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.2}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.5}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-0.5}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[single_co3dv2] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[single_co3dv2] CONFIG=$CONFIG"
echo "[single_co3dv2] VAL_CONFIG=$VAL_CONFIG"
echo "[single_co3dv2] OUTPUT_DIR=$OUTPUT_DIR"
echo "[single_co3dv2] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM )))"
echo "[single_co3dv2] PATCH_PROVIDER=$PATCH_PROVIDER"
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[single_co3dv2] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[single_co3dv2] PRETRAIN=$PRETRAIN"
else
  echo "[single_co3dv2] PRETRAIN=<none>"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/python \
  train_mixture.py \
  --config "$CONFIG" \
  --val-config "$VAL_CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --grad-accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --num-frames 48 \
  --output-dir "$OUTPUT_DIR" \
  --patch-provider "$PATCH_PROVIDER" \
  --val-interval "$VAL_INTERVAL" \
  --val-samples "$VAL_SAMPLES" \
  --save-interval "$SAVE_INTERVAL" \
  --loss-w-3d "$LOSS_W_3D" \
  --loss-w-conf "$LOSS_W_CONF" \
  --loss-w-normal "$LOSS_W_NORMAL" \
  --loss-w-static-reprojection "$LOSS_W_STATIC_REPROJ" \
  --loss-3d-mode "$LOSS_3D_MODE" \
  --lr-warmup-steps "$LR_WARMUP_STEPS" \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
