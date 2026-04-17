#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
CONFIG="${CONFIG:-configs/mixture_minimal_3datasets.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_3datasets_conf_static_form0_gpu6}"
PRETRAIN="${PRETRAIN:-}"
# PRETRAIN="${PRETRAIN:-/data/zbf/my_dfrt/outputs_all_noraw/load3240/full/checkpoint_epoch_3280.pth}"
# PRETRAIN="${PRETRAIN:-/data/zbf/openclaw/d4rt/outputs/mixture_3datasets_finetune_v2_gpu5/checkpoint_latest_45.pth}"
RESUME="${RESUME:-}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.2}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.1}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"  #log_space 
# Warmup: ~300 steps ≈ 2-3 epochs for single-GPU fine-tune (100 epochs × ~126 opt-steps = ~12600 total)
# Default 2500 wastes the first 20 epochs in warmup; 300 gets to full LR by epoch 3.
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[mixture_3datasets] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_3datasets] CONFIG=$CONFIG"
echo "[mixture_3datasets] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_3datasets] NUM_WORKERS=$NUM_WORKERS"
echo "[mixture_3datasets] PATCH_PROVIDER=$PATCH_PROVIDER"
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_3datasets] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_3datasets] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_3datasets] PRETRAIN=<none>"
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
  --loss-3d-mode "$LOSS_3D_MODE" \
  --lr-warmup-steps "$LR_WARMUP_STEPS" \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
