#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
CONFIG="${CONFIG:-configs/single_kubric.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/kubric_single_gpu_5}"
PRETRAIN="${PRETRAIN:-/data/zbf/my_dfrt/outputs_all_noraw/load3240/full/checkpoint_epoch_3280.pth}"
RESUME="${RESUME:-}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
PATCH_PROVIDER="${PATCH_PROVIDER:-auto}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.0}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.0}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[kubric] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[kubric] OUTPUT_DIR=$OUTPUT_DIR"
echo "[kubric] NUM_WORKERS=$NUM_WORKERS"
echo "[kubric] PATCH_PROVIDER=$PATCH_PROVIDER"
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[kubric] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[kubric] PRETRAIN=$PRETRAIN"
else
  echo "[kubric] PRETRAIN=<none>"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/python \
  train_mixture.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --grad-accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --output-dir "$OUTPUT_DIR" \
  --patch-provider "$PATCH_PROVIDER" \
  --val-interval "$VAL_INTERVAL" \
  --val-samples "$VAL_SAMPLES" \
  --save-interval "$SAVE_INTERVAL" \
  --loss-w-3d "$LOSS_W_3D" \
  --loss-w-conf "$LOSS_W_CONF" \
  --loss-w-normal "$LOSS_W_NORMAL" \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
