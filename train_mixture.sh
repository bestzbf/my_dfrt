#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
CONFIG="${CONFIG:-configs/mixture_full_11datasets.yaml}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EPOCHS="${EPOCHS:-500000}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_5_3d}"
PRETRAIN="${PRETRAIN:-/data1/zbf/my_dfrt/outputs_all_augs/256_lossPaper_load2020_con/full/checkpoint_epoch_3240.pth}"
RESUME="${RESUME:-/data1/zbf/my_dfrt/outputs/mixture_5_3d/checkpoint_latest_58.pth}"

export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/torchrun --nproc_per_node=3 --master_port=29510 train_mixture.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --output-dir "$OUTPUT_DIR" \
  --loss-w-conf 0.0 \
  --loss-w-3d 1.0 \
  ${RESUME:+--resume "$RESUME"} \
  "$@"

  # ${PRETRAIN:+--pretrain "$PRETRAIN"} \
