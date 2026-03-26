#!/usr/bin/env bash
# 单场景训练脚本
# 用法:
#   bash train_single_scene.sh <sequence_name> [额外 train.sh 参数...]
#
# 示例:
#   bash train_single_scene.sh ani
#   bash train_single_scene.sh ani --steps 50000
#   bash train_single_scene.sh ani --disable-train-augs --steps 20000
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <sequence_name> [额外参数...]" >&2
    echo "示例: $0 ani" >&2
    echo "示例: $0 ani --steps 50000 --disable-train-augs" >&2
    exit 1
fi

SEQUENCE="$1"
shift

# ── 环境变量（未设置则使用默认值）──────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-/data2/d4rt/datasets/PointOdyssey_fast}"
ENCODER="${ENCODER:-base}"
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-base}"
PATCH_PROVIDER="${PATCH_PROVIDER:-precomputed_highres}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_single_scene/${SEQUENCE}}"

# ── 单场景专属默认值 ──────────────────────────────────────────
STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-10}"

# 预训练权重（可选）：加载模型权重，optimizer/scheduler 重新开始
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"

# dataset_repeat_factor:
#   __len__ 返回序列数（单场景=1），而非 clip 数。
#   repeat_factor = 目标steps/epoch × batch_size
#   例：4000 × bs4 = 1000步/epoch，100000步跑100epoch
DATASET_REPEAT_FACTOR="${DATASET_REPEAT_FACTOR:-4000}"

export CUDA_VISIBLE_DEVICES
export DATA_ROOT
export MODE=normal
export STAGE=full_main
export OUTPUT_ROOT
export ENCODER
export VIDEOMAE_MODEL
export PATCH_PROVIDER
export QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
export BATCH_SIZE
export VAL_EVERY_EPOCHS
export NUM_WORKERS
export PREFETCH_FACTOR
export PYTHON_BIN

echo "====================================="
echo "单场景训练"
echo "  序列:       $SEQUENCE"
echo "  数据根目录: $DATA_ROOT"
echo "  输出目录:   $OUTPUT_ROOT"
echo "  GPU:        $CUDA_VISIBLE_DEVICES"
echo "  batch_size: $BATCH_SIZE"
echo "  steps:      $STEPS  (warmup=$WARMUP_STEPS)"
echo "  repeat_factor: $DATASET_REPEAT_FACTOR"
if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
    echo "  预训练权重:  $PRETRAINED_WEIGHTS"
fi
echo "====================================="

bash train.sh \
    --train-split train \
    --val-split train \
    --train-sequence "$SEQUENCE" \
    --val-sequence "$SEQUENCE" \
    --steps "$STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --dataset-repeat-factor "$DATASET_REPEAT_FACTOR" \
    --lambda-3d 1.0 \
    --lambda-raw-3d 0.0 \
    --lambda-conf 0.0 \
    --lambda-2d 0.1 \
    --lambda-vis 0.1 \
    --lambda-disp 0.1 \
    --lambda-normal 0.5 \
    --conf-weighting-start-step 999999 \
    --conf-ramp-steps 0 \
    --img-size 256 \
    --t-tgt-eq-t-cam-ratio 0.4 \
    --amp \
    --no-gradient-checkpointing \
    --skip-pointodyssey-sanity \
    ${PRETRAINED_WEIGHTS:+--pretrained-weights "$PRETRAINED_WEIGHTS"} \
    "$@"
