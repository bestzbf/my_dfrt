#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/eval_$(basename "$CHECKPOINT" .pth)_sintel_camera_latest_best}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/eval_sintel_camera_checkpoint.py" \
  --root "${SINTEL_ROOT:-/data3/dataset/sintel}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --model-variant "${MODEL_VARIANT:-large}" \
  --patch-provider "${PATCH_PROVIDER:-sampled_highres}" \
  --resolution "${RESOLUTION:-256}" \
  --num-frames "${NUM_FRAMES:-48}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --sintel-pass "${SINTEL_PASS:-final}" \
  --num-scenes "${NUM_SCENES:-0}" \
  --scene-names "${SCENE_NAMES:-}" \
  --pose-mode "${POSE_MODE:-adjacent}" \
  --reference-frame "${REFERENCE_FRAME:-0}" \
  --pose-grid-h "${POSE_GRID_H:-16}" \
  --pose-grid-w "${POSE_GRID_W:-16}" \
  --pose-margin "${POSE_MARGIN:-0.1}" \
  --pose-confidence-threshold "${POSE_CONFIDENCE_THRESHOLD:-0.0}" \
  --pose-confidence-quantile "${POSE_CONFIDENCE_QUANTILE:-0.4}" \
  --pose-weight-mode "${POSE_WEIGHT_MODE:-product}" \
  --pose-solver "${POSE_SOLVER:-umeyama}" \
  --pose-ransac-iters "${POSE_RANSAC_ITERS:-256}" \
  --pose-inlier-thresh "${POSE_INLIER_THRESH:-0.1}" \
  --intrinsics-grid-h "${INTRINSICS_GRID_H:-16}" \
  --intrinsics-grid-w "${INTRINSICS_GRID_W:-16}" \
  --intrinsics-margin "${INTRINSICS_MARGIN:-0.1}" \
  --intrinsics-confidence-threshold "${INTRINSICS_CONFIDENCE_THRESHOLD:-0.0}" \
  --intrinsics-confidence-quantile "${INTRINSICS_CONFIDENCE_QUANTILE:-0.2}" \
  --intrinsics-frame-stride "${INTRINSICS_FRAME_STRIDE:-1}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-4096}" \
  --device "${DEVICE:-cuda}" \
  "$@"
