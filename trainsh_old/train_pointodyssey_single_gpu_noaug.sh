#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
CONFIG="${CONFIG:-configs/single_pointodyssey_highres_noaug.yaml}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pointodyssey_single_gpu_highres_noaug1}"
PRETRAIN="${PRETRAIN:-/data/zbf/my_dfrt/outputs_all_noraw/load3240/full/checkpoint_epoch_3280.pth}"
RESUME="${RESUME:-}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --loss-w-conf 0.0 \
  --loss-w-3d 1.0 \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
