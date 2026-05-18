#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
CONFIG="${CONFIG:-configs/mixture_5datasets_blendedmvs_hdu.yaml}"
VAL_CONFIG="${VAL_CONFIG:-}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_5datasets_blendedmvs_base_1gpu_bs10_hdu}"
DEFAULT_PRETRAIN="/data/zbf/openclaw/d4rt/outputs/mixture_3datasets_finetune_v2_gpu5/checkpoint_latest_50.pth"
PRETRAIN="${PRETRAIN-$DEFAULT_PRETRAIN}"
DEFAULT_RESUME="/data/zbf/openclaw/d4rt/outputs/mixture_5datasets_blendedmvs_base_1gpu_bs8/checkpoint_latest_1.pth"
RESUME="${RESUME-$DEFAULT_RESUME}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
VAL_SAMPLES="${VAL_SAMPLES:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.2}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.1}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-1.0}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-1}"
VARIANT="${VARIANT:-base}"

case "$VARIANT" in
  large)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-large}"
    ;;
  base)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-base}"
    ;;
  *)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-0}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-}"
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[mixture_5datasets_single_gpu_hdu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_5datasets_single_gpu_hdu] CONFIG=$CONFIG"
if [[ -n "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_single_gpu_hdu] VAL_CONFIG=$VAL_CONFIG"
fi
echo "[mixture_5datasets_single_gpu_hdu] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_5datasets_single_gpu_hdu] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM )))"
echo "[mixture_5datasets_single_gpu_hdu] NUM_WORKERS=$NUM_WORKERS"
echo "[mixture_5datasets_single_gpu_hdu] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[mixture_5datasets_single_gpu_hdu] RESET_CONF_HEAD_ON_PRETRAIN=$RESET_CONF_HEAD_ON_PRETRAIN"
echo "[mixture_5datasets_single_gpu_hdu] VARIANT=$VARIANT"
echo "[mixture_5datasets_single_gpu_hdu] USE_VIDEOMAE_V2_INIT=$USE_VIDEOMAE_V2_INIT"
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  echo "[mixture_5datasets_single_gpu_hdu] VIDEOMAE_MODEL=$VIDEOMAE_MODEL"
fi
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_5datasets_single_gpu_hdu] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_5datasets_single_gpu_hdu] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_5datasets_single_gpu_hdu] PRETRAIN=<none>"
fi
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "[mixture_5datasets_single_gpu_hdu] RESUME not found: $RESUME" >&2
    exit 1
  fi
  echo "[mixture_5datasets_single_gpu_hdu] RESUME=$RESUME"
else
  echo "[mixture_5datasets_single_gpu_hdu] RESUME=<none>"
fi
if [[ -n "$VAL_CONFIG" && ! -f "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_single_gpu_hdu] VAL_CONFIG not found: $VAL_CONFIG" >&2
  exit 1
fi

cmd=(
  /root/miniconda3/envs/d4rt/bin/python
  train_mixture.py
  --config "$CONFIG"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --grad-accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --lr "$LR"
  --num-frames 48
  --output-dir "$OUTPUT_DIR"
  --patch-provider "$PATCH_PROVIDER"
  --val-interval "$VAL_INTERVAL"
  --val-samples "$VAL_SAMPLES"
  --save-interval "$SAVE_INTERVAL"
  --loss-w-3d "$LOSS_W_3D"
  --loss-w-conf "$LOSS_W_CONF"
  --loss-w-normal "$LOSS_W_NORMAL"
  --loss-w-static-reprojection "$LOSS_W_STATIC_REPROJ"
  --loss-3d-mode "$LOSS_3D_MODE"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --variant "$VARIANT"
)

if [[ "$USE_VIDEOMAE_V2_INIT" == "1" ]]; then
  cmd+=(--use-videomae-v2-init)
fi
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  cmd+=(--videomae-model "$VIDEOMAE_MODEL")
fi
if [[ "$RESET_CONF_HEAD_ON_PRETRAIN" == "1" ]]; then
  cmd+=(--reset-confidence-head-on-pretrain)
fi
if [[ -n "$VAL_CONFIG" ]]; then
  cmd+=(--val-config "$VAL_CONFIG")
fi
if [[ -n "$PRETRAIN" ]]; then
  cmd+=(--pretrain "$PRETRAIN")
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi
cmd+=("$@")

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
