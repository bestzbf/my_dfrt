#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CONFIG="${CONFIG:-configs/mixture_4datasets.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_4datasets}"
PRETRAIN="${PRETRAIN:-}"
RESUME="${RESUME:-}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.2}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.1}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-0.5}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[mixture_4datasets] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_4datasets] CONFIG=$CONFIG"
echo "[mixture_4datasets] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_4datasets] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM )))"
echo "[mixture_4datasets] PATCH_PROVIDER=$PATCH_PROVIDER"
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_4datasets] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_4datasets] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_4datasets] PRETRAIN=<none>"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/python \
  train_mixture.py \
  --config "$CONFIG" \
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
