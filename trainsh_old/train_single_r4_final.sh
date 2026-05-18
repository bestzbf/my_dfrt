#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 最终优化版本 - 数据已用encoded_cache
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
CONFIG="${CONFIG:-configs/single_sample_r4_new_f.yaml}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-32}"  # 增加workers
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/single_r4_final_1}"
# PRETRAIN="${PRETRAIN:-/data/zbf/4.6r4/outputs/single_r4_final/checkpoint_epoch_40.pth}"
RESUME="${RESUME:-/data/zbf/4.6r4/outputs/single_r4_final_1/checkpoint_epoch_250.pth}"

echo "=========================================="
echo "最终优化训练配置："
echo "  场景: r4_new_f (850帧)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"
echo "  每个epoch: 动态采样1000个clip"
echo "  Val: 只用1个clip (不是1000个)"
echo "  数据: encoded_cache已启用 (memmap)"
echo "  学习率: $LR"
echo "  输出: $OUTPUT_DIR"
echo "=========================================="

# 启动训练
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/python train_single_sample.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --output-dir "$OUTPUT_DIR" \
  --loss-w-3d 1.0 \
  --loss-w-raw-3d 1.5 \
  --loss-w-2d 0.2 \
  --loss-w-vis 0.1 \
  --loss-w-disp 0.1 \
  --loss-w-conf 0.0 \
  --loss-w-normal 0.1 \
  --save-interval 5 \
  --log-interval 1 \
  ${PRETRAIN:+--pretrain "$PRETRAIN"} \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
