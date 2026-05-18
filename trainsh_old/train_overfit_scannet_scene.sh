#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MASTER_PORT="${MASTER_PORT:-28596}"
CONFIG="${CONFIG:-configs/overfit_scannet_scene.yaml}"

BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overfit_scannet_scene0000_00}"
PRETRAIN="${PRETRAIN:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from0/checkpoint_latest_200.pth}"
RESUME="${RESUME:-}"

TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp_overfit_scannet}"
mkdir -p "$TMPDIR" /data1/zbf/d4rt_sample_stage

export TMPDIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONWARNINGS="ignore:mkl-service package failed to import:UserWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"
export D4RT_SERIALIZE_ADAPTER_INIT=1

cmd=(
  "$TORCHRUN_BIN"
  --standalone
  --nproc_per_node=1
  --master_port="$MASTER_PORT"
  train_mixture.py
  --config "$CONFIG"
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --prefetch-factor 2
  --grad-accum 1
  --log-interval 10
  --epochs "$EPOCHS"
  --lr "$LR"
  --lr-warmup-steps 50
  --num-frames 48
  --output-dir "$OUTPUT_DIR"
  --patch-provider sampled_highres
  --val-interval 50
  --val-samples 10
  --save-interval 50
  --loss-w-3d 1.0
  --loss-w-conf 0.1
  --loss-w-normal 0.0
  --loss-w-static-reprojection 0.9
  --loss-3d-mode scale_invariant
  --lr-warmup-steps 50
  --dist-timeout-minutes 30
  --variant large
  --use-videomae-v2-init
  --videomae-model /data1/zbf/pretrained/videomae-v2-large
  --builder-workers 2
  --prefetch-depth 8
  --batch-prefetch-depth 2
)

[[ -n "$PRETRAIN" ]] && cmd+=(--pretrain "$PRETRAIN")
[[ -n "$RESUME"  ]] && cmd+=(--resume "$RESUME")
cmd+=("$@")

echo "[overfit-scannet] config=$CONFIG output=$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
