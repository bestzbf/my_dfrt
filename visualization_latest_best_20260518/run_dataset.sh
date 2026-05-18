#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-${DATASET:-scannet_test}}"
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
MAX_SEARCH="${MAX_SEARCH:-300}"
START_INDEX="${START_INDEX:-0}"
REFERENCE_FRAME="${REFERENCE_FRAME:--1}"
NUM_POINTS="${NUM_POINTS:-1024}"
NUM_DISPLAY_POINTS="${NUM_DISPLAY_POINTS:-180}"
GIF_FPS="${GIF_FPS:-8}"
DENSE_GT_MAX_POINTS="${DENSE_GT_MAX_POINTS:-0}"
DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-all_finite}"
DENSE_GT_DEPTH_STRIDE="${DENSE_GT_DEPTH_STRIDE:-3}"
DEPTH_GT_MAX_DEPTH="${DEPTH_GT_MAX_DEPTH:-100}"
DENSE_PRED_POINT_CLOUD_STRIDE="${DENSE_PRED_POINT_CLOUD_STRIDE:-2}"
DENSE_PRED_VIS_THRESHOLD="${DENSE_PRED_VIS_THRESHOLD:-0.3}"
DENSE_PRED_CONFIDENCE_PERCENTILE="${DENSE_PRED_CONFIDENCE_PERCENTILE:-10}"
DENSE_PRED_QUERY_BATCH_SIZE="${DENSE_PRED_QUERY_BATCH_SIZE:-8192}"
DENSE_PRED_DEPTH_PERCENTILE="${DENSE_PRED_DEPTH_PERCENTILE:-100}"
DENSE_PRED_QUERY_DEPTH_PERCENTILE="${DENSE_PRED_QUERY_DEPTH_PERCENTILE:-100}"
CAMERA_POSE_GRID_H="${CAMERA_POSE_GRID_H:-8}"
CAMERA_POSE_GRID_W="${CAMERA_POSE_GRID_W:-8}"
CAMERA_POSE_CONFIDENCE_THRESHOLD="${CAMERA_POSE_CONFIDENCE_THRESHOLD:-0.5}"
CAMERA_POSE_RANSAC_ITERS="${CAMERA_POSE_RANSAC_ITERS:-256}"
CAMERA_POSE_INLIER_THRESH="${CAMERA_POSE_INLIER_THRESH:-0.1}"
SPLIT="${SPLIT:-val}"
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
  sintel_geometry10|sintel)
    exec "$SCRIPT_DIR/run_sintel_visualization.sh" "$@"
    ;;
  *)
    echo "Unknown DATASET=$DATASET" >&2
    echo "Supported: scannet_test, scannetpp_val, dynamic_replica_val, kubric_val, pointodyssey_val, blendedmvs_val, co3dv2_val, sintel_geometry10" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  ckpt_dir="$(dirname "$CHECKPOINT")"
  OUTPUT_DIR="$ckpt_dir/vis_$(basename "$CHECKPOINT" .pth)_${DATASET}_camera${NUM_SAMPLES}_densepc_s${DENSE_PRED_POINT_CLOUD_STRIDE}"
fi

echo "[run_dataset] DATASET=$DATASET"
echo "[run_dataset] CONFIG=$CONFIG"
echo "[run_dataset] CHECKPOINT=$CHECKPOINT"
echo "[run_dataset] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run_dataset] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/visualize_dynamic_replica_checkpoint.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --model-variant "$MODEL_VARIANT" \
  --output-dir "$OUTPUT_DIR" \
  --patch-provider "$PATCH_PROVIDER" \
  --resolution 256 \
  --num-frames 48 \
  --num-samples "$NUM_SAMPLES" \
  --start-index "$START_INDEX" \
  --max-search "$MAX_SEARCH" \
  --reference-frame "$REFERENCE_FRAME" \
  --num-points "$NUM_POINTS" \
  --num-display-points "$NUM_DISPLAY_POINTS" \
  --gif-fps "$GIF_FPS" \
  --gt-source auto \
  --dense-gt-max-points "$DENSE_GT_MAX_POINTS" \
  --dense-gt-point-source "$DENSE_GT_POINT_SOURCE" \
  --dense-gt-depth-stride "$DENSE_GT_DEPTH_STRIDE" \
  --depth-gt-max-depth "$DEPTH_GT_MAX_DEPTH" \
  --dense-pred-point-cloud-stride "$DENSE_PRED_POINT_CLOUD_STRIDE" \
  --dense-pred-vis-threshold "$DENSE_PRED_VIS_THRESHOLD" \
  --dense-pred-confidence-percentile "$DENSE_PRED_CONFIDENCE_PERCENTILE" \
  --dense-pred-query-batch-size "$DENSE_PRED_QUERY_BATCH_SIZE" \
  --dense-pred-depth-percentile "$DENSE_PRED_DEPTH_PERCENTILE" \
  --dense-pred-query-depth-percentile "$DENSE_PRED_QUERY_DEPTH_PERCENTILE" \
  --camera-pose-grid-h "$CAMERA_POSE_GRID_H" \
  --camera-pose-grid-w "$CAMERA_POSE_GRID_W" \
  --camera-pose-confidence-threshold "$CAMERA_POSE_CONFIDENCE_THRESHOLD" \
  --camera-pose-ransac-iters "$CAMERA_POSE_RANSAC_ITERS" \
  --camera-pose-inlier-thresh "$CAMERA_POSE_INLIER_THRESH" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --split "$SPLIT" \
  "$@"
