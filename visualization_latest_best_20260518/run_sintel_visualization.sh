#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

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
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/vis_$(basename "$CHECKPOINT" .pth)_sintel_geometry10_densepc_s2}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/visualize_sintel_checkpoint.py" \
  --root "${SINTEL_ROOT:-/data3/dataset/sintel}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --model-variant "${MODEL_VARIANT:-large}" \
  --patch-provider "${PATCH_PROVIDER:-sampled_highres}" \
  --resolution 256 \
  --num-frames 48 \
  --num-scenes "${NUM_SCENES:-10}" \
  --sintel-pass "${SINTEL_PASS:-final}" \
  --reference-frame "${REFERENCE_FRAME:-24}" \
  --dense-stride "${DENSE_STRIDE:-2}" \
  --dense-gt-depth-stride "${DENSE_GT_DEPTH_STRIDE:-3}" \
  --dense-gt-max-points "${DENSE_GT_MAX_POINTS:-0}" \
  --depth-gt-max-depth "${DEPTH_GT_MAX_DEPTH:-100}" \
  --dense-vis-threshold "${DENSE_VIS_THRESHOLD:-0.3}" \
  --dense-confidence-percentile "${DENSE_CONFIDENCE_PERCENTILE:-10}" \
  --dense-query-depth-percentile "${DENSE_QUERY_DEPTH_PERCENTILE:-100}" \
  --dense-query-batch-size "${DENSE_QUERY_BATCH_SIZE:-8192}" \
  --pose-grid-h "${POSE_GRID_H:-8}" \
  --pose-grid-w "${POSE_GRID_W:-8}" \
  --pose-confidence-threshold "${POSE_CONFIDENCE_THRESHOLD:-0.5}" \
  --gif-fps "${GIF_FPS:-8}" \
  --device "$DEVICE" \
  "$@"
