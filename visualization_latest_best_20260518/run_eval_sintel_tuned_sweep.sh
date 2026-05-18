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
CHECKPOINT_BASENAME="$(basename "$CHECKPOINT" .pth)"
SWEEP_ROOT="${SWEEP_ROOT:-$(dirname "$CHECKPOINT")/eval_${CHECKPOINT_BASENAME}_sintel_tuned_sweep}"

# name:depth_conf_quantile:pointcloud_conf_quantile:pose_solver:pose_conf_threshold
# Defaults are intentionally small and fast; override for a full 14-scene sweep.
SWEEP_CONFIGS="${SWEEP_CONFIGS:-strict:0.0:0.0:umeyama:0.0 conf20:0.2:0.2:umeyama:0.0 conf40:0.4:0.4:umeyama:0.0 ransac:0.0:0.0:ransac:0.0 ransac_conf20:0.2:0.2:ransac:0.3}"

mkdir -p "$SWEEP_ROOT"
echo "[sweep] checkpoint=$CHECKPOINT"
echo "[sweep] root=$SWEEP_ROOT"
echo "[sweep] configs=$SWEEP_CONFIGS"

for config in $SWEEP_CONFIGS; do
  IFS=":" read -r name depth_q pointcloud_q pose_solver pose_threshold <<< "$config"
  if [[ -z "${name:-}" || -z "${depth_q:-}" || -z "${pointcloud_q:-}" || -z "${pose_solver:-}" || -z "${pose_threshold:-}" ]]; then
    echo "Bad SWEEP_CONFIGS item: $config" >&2
    exit 1
  fi

  run_dir="$SWEEP_ROOT/$name"
  echo
  echo "[sweep] run=$name depth_q=$depth_q pointcloud_q=$pointcloud_q pose=$pose_solver threshold=$pose_threshold"
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$run_dir" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NUM_SCENES="${NUM_SCENES:-1}" \
  DEPTH_STRIDE="${DEPTH_STRIDE:-2}" \
  POINTCLOUD_STRIDE="${POINTCLOUD_STRIDE:-4}" \
  MAX_POINTCLOUD_POINTS="${MAX_POINTCLOUD_POINTS:-50000}" \
  DEPTH_CONFIDENCE_QUANTILE="$depth_q" \
  POINTCLOUD_CONFIDENCE_QUANTILE="$pointcloud_q" \
  POSE_SOLVER="$pose_solver" \
  POSE_CONFIDENCE_THRESHOLD="$pose_threshold" \
  bash "$SCRIPT_DIR/run_eval_sintel.sh" "$@"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_eval_sweep.py" \
  --root "$SWEEP_ROOT" \
  --output-md "$SWEEP_ROOT/sweep_summary.md" \
  --output-json "$SWEEP_ROOT/sweep_summary.json"
