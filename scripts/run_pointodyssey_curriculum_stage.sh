#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stage> [extra train.py args...]" >&2
  exit 1
fi

STAGE="$1"
shift || true

DATA_ROOT="${DATA_ROOT:-/16t/e/d4rt/PointOdyssey}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pointodyssey_curriculum}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
BASE_CONFIG="${BASE_CONFIG:-configs/d4rt_pointodyssey_curriculum_base.yaml}"

case "$STAGE" in
  same_frame_main)
    QUERY_MODE="same_frame"; PREV_STAGE=""; STEPS=4000; WARMUP=200
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.0; LDISP=0.0; LNORM=0.0; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  same_frame_vis)
    QUERY_MODE="same_frame"; PREV_STAGE="same_frame_main"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.1; LDISP=0.0; LNORM=0.0; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  same_frame_aux)
    QUERY_MODE="same_frame"; PREV_STAGE="same_frame_vis"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.1; LDISP=0.1; LNORM=0.5; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  same_frame_full)
    QUERY_MODE="same_frame"; PREV_STAGE="same_frame_aux"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.05; L2D=1.0; LVIS=0.1; LDISP=0.1; LNORM=0.5; LCONF=0.05; CONF_START=1000; CONF_RAMP=1000; CONF_WEIGHT_START=1000; CONF_WEIGHT_RAMP=1000
    ;;
  target_cam_main)
    QUERY_MODE="target_cam"; PREV_STAGE="same_frame_full"; STEPS=4000; WARMUP=200
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.0; LDISP=0.0; LNORM=0.0; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  target_cam_vis)
    QUERY_MODE="target_cam"; PREV_STAGE="target_cam_main"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.1; LDISP=0.0; LNORM=0.0; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  target_cam_aux)
    QUERY_MODE="target_cam"; PREV_STAGE="target_cam_vis"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.1; L2D=1.0; LVIS=0.1; LDISP=0.1; LNORM=0.5; LCONF=0.0; CONF_START=0; CONF_RAMP=0; CONF_WEIGHT_START=0; CONF_WEIGHT_RAMP=0
    ;;
  target_cam_full)
    QUERY_MODE="target_cam"; PREV_STAGE="target_cam_aux"; STEPS=2000; WARMUP=100
    L3D=1.0; LRAW3D=0.05; L2D=1.0; LVIS=0.1; LDISP=0.1; LNORM=0.5; LCONF=0.05; CONF_START=1000; CONF_RAMP=1000; CONF_WEIGHT_START=1000; CONF_WEIGHT_RAMP=1000
    ;;
  full)
    QUERY_MODE="full"; PREV_STAGE=""; STEPS=14000; WARMUP=700
    L3D=1.0; LRAW3D=0.0; L2D=0.1; LVIS=0.1; LDISP=0.1; LNORM=0.5; LCONF=0.05; CONF_START=8000; CONF_RAMP=2000; CONF_WEIGHT_START=8000; CONF_WEIGHT_RAMP=2000
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$STAGE"
mkdir -p "$OUTPUT_ROOT"

CMD=(
  $PYTHON_BIN $TRAIN_SCRIPT
  --config "$BASE_CONFIG"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --query-mode "$QUERY_MODE"
  --steps "$STEPS"
  --warmup-steps "$WARMUP"
  --lambda-3d "$L3D"
  --lambda-raw-3d "$LRAW3D"
  --lambda-2d "$L2D"
  --lambda-vis "$LVIS"
  --lambda-disp "$LDISP"
  --lambda-normal "$LNORM"
  --lambda-conf "$LCONF"
  --conf-ramp-start-step "$CONF_START"
  --conf-ramp-steps "$CONF_RAMP"
  --conf-weighting-start-step "$CONF_WEIGHT_START"
  --conf-weighting-ramp-steps "$CONF_WEIGHT_RAMP"
)

if [[ "$QUERY_MODE" == "full" ]]; then
  CMD+=(--t-tgt-eq-t-cam-ratio 0.4)
fi

if [[ -n "$PREV_STAGE" && "${NO_RESUME_CHAIN:-0}" != "1" ]]; then
  PREV_CKPT="$OUTPUT_ROOT/$PREV_STAGE/checkpoint_latest.pth"
  if [[ -f "$PREV_CKPT" ]]; then
    CMD+=(--pretrained-weights "$PREV_CKPT")
  fi
fi

CMD+=("$@")

printf 'Running stage %s
' "$STAGE"
printf 'Output dir: %s
' "$OUTPUT_DIR"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '
'
exec "${CMD[@]}"
