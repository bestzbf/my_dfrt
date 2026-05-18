#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-${DATASET:-co3dv2_val}}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200}"

find_latest_checkpoint() {
  local latest
  latest="$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_latest_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
  if [[ -z "$latest" ]]; then
    echo "No checkpoint_latest_*.pth found in CHECKPOINT_DIR=$CHECKPOINT_DIR" >&2
    exit 1
  fi
  printf '%s/%s\n' "$CHECKPOINT_DIR" "$latest"
}

CHECKPOINT="${CHECKPOINT:-$(find_latest_checkpoint)}"
MODEL_VARIANT="${MODEL_VARIANT:-large}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_FRAMES="${NUM_FRAMES:-48}"
RESOLUTION="${RESOLUTION:-256}"
DEPTH_STRIDE="${DEPTH_STRIDE:-2}"
SPLIT="${SPLIT:-val}"
START_INDEX="${START_INDEX:-0}"
MAX_SEARCH="${MAX_SEARCH:-300}"
GIF_FPS="${GIF_FPS:-8}"
SEED="${SEED:-42}"

case "$DATASET" in
  scannet_test)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_scannet_test_local.yaml}"
    ;;
  scannetpp_val)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_scannetpp_val_local.yaml}"
    ;;
  dynamic_replica_val|dynamic_replica)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_dynamic_replica_val_local.yaml}"
    ;;
  kubric_val|kubric)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_kubric_val_local.yaml}"
    ;;
  pointodyssey_val|pointodyssey)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_pointodyssey_highres_noaug.yaml}"
    ;;
  blendedmvs_val|blendedmvs)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_blendedmvs_val.yaml}"
    ;;
  co3dv2_val|co3dv2)
    CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_co3dv2_val_cos.yaml}"
    ;;
  *)
    echo "Unknown DATASET=$DATASET" >&2
    echo "Supported: scannet_test, scannetpp_val, dynamic_replica_val, kubric_val, pointodyssey_val, blendedmvs_val, co3dv2_val" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname "$CHECKPOINT")/vis_$(basename "$CHECKPOINT" .pth)_${DATASET}_depth_s${DEPTH_STRIDE}"
fi

echo "[run_depth] DATASET=$DATASET"
echo "[run_depth] CONFIG=$CONFIG"
echo "[run_depth] CHECKPOINT=$CHECKPOINT"
echo "[run_depth] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run_depth] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/visualize_co3dv2_depth.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --model-variant "$MODEL_VARIANT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --num-frames "$NUM_FRAMES" \
  --resolution "$RESOLUTION" \
  --patch-provider "$PATCH_PROVIDER" \
  --stride "$DEPTH_STRIDE" \
  --split "$SPLIT" \
  --seed "$SEED" \
  --start-index "$START_INDEX" \
  --max-search "$MAX_SEARCH" \
  --gif-fps "$GIF_FPS" \
  --device "$DEVICE" \
  "$@"
