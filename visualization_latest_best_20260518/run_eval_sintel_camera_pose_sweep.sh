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
SWEEP_ROOT="${SWEEP_ROOT:-$(dirname "$CHECKPOINT")/eval_${CHECKPOINT_BASENAME}_sintel_camera_pose_sweep}"

# name:pose_mode:solver:grid_h:grid_w:confidence_quantile:confidence_threshold:weight_mode
SWEEP_CONFIGS="${SWEEP_CONFIGS:-ref_g8_q0:reference:umeyama:8:8:0.0:0.0:product ref_g16_q0:reference:umeyama:16:16:0.0:0.0:product adj_g16_q20:adjacent:umeyama:16:16:0.2:0.0:product adj_g16_q40:adjacent:umeyama:16:16:0.4:0.0:product adj_g16_q40_mean:adjacent:umeyama:16:16:0.4:0.0:mean adj_g16_q40_none:adjacent:umeyama:16:16:0.4:0.0:none adj_g16_ransac_q40:adjacent:ransac:16:16:0.4:0.0:product}"

mkdir -p "$SWEEP_ROOT"
echo "[camera_sweep] checkpoint=$CHECKPOINT"
echo "[camera_sweep] root=$SWEEP_ROOT"
echo "[camera_sweep] configs=$SWEEP_CONFIGS"

for config in $SWEEP_CONFIGS; do
  IFS=":" read -r name pose_mode solver grid_h grid_w conf_q conf_threshold weight_mode <<< "$config"
  if [[ -z "${name:-}" || -z "${pose_mode:-}" || -z "${solver:-}" || -z "${grid_h:-}" || -z "${grid_w:-}" || -z "${conf_q:-}" || -z "${conf_threshold:-}" || -z "${weight_mode:-}" ]]; then
    echo "Bad SWEEP_CONFIGS item: $config" >&2
    exit 1
  fi

  run_dir="$SWEEP_ROOT/$name"
  echo
  echo "[camera_sweep] run=$name mode=$pose_mode solver=$solver grid=${grid_h}x${grid_w} q=$conf_q threshold=$conf_threshold weight=$weight_mode"
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$run_dir" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NUM_SCENES="${NUM_SCENES:-1}" \
  POSE_MODE="$pose_mode" \
  POSE_SOLVER="$solver" \
  POSE_GRID_H="$grid_h" \
  POSE_GRID_W="$grid_w" \
  POSE_CONFIDENCE_QUANTILE="$conf_q" \
  POSE_CONFIDENCE_THRESHOLD="$conf_threshold" \
  POSE_WEIGHT_MODE="$weight_mode" \
  bash "$SCRIPT_DIR/run_eval_sintel_camera.sh" "$@"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_eval_sweep.py" \
  --root "$SWEEP_ROOT" \
  --output-md "$SWEEP_ROOT/camera_pose_sweep_summary.md" \
  --output-json "$SWEEP_ROOT/camera_pose_sweep_summary.json"
