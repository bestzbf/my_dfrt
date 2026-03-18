#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

resolve_videomae_model() {
  local requested_model="${VIDEOMAE_MODEL:-}"
  local candidates=()

  if [[ -n "$requested_model" ]]; then
    candidates+=("$requested_model")
  fi
  candidates+=(
    "/mnt/d/data/ZBF_Data/d4rt/3090/pretrained/videomae-base"
    "/16t/e/d4rt/pretrained/videomae-base"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested_model" ]]; then
    printf '%s\n' "$requested_model"
    return 0
  fi

  printf '\n'
  return 0
}

TRAIN_SEQUENCE="${TRAIN_SEQUENCE:-${SEQUENCE:-}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  TRAIN_SEQUENCE="$1"
  shift
fi

if [[ -z "$TRAIN_SEQUENCE" ]]; then
  echo "Usage: $0 <sequence_name> [extra train.sh args...]" >&2
  echo "Or set TRAIN_SEQUENCE=<sequence_name> in the environment." >&2
  exit 1
fi

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-$TRAIN_SPLIT}"
VAL_SEQUENCE="${VAL_SEQUENCE:-$TRAIN_SEQUENCE}"
POINTODYSSEY_SANITY_SPLIT="${POINTODYSSEY_SANITY_SPLIT:-$TRAIN_SPLIT}"
POINTODYSSEY_SANITY_SEQUENCE="${POINTODYSSEY_SANITY_SEQUENCE:-$TRAIN_SEQUENCE}"

MODE="${MODE:-normal}"
STAGE="${STAGE:-full_main}"
NO_RESUME_CHAIN="${NO_RESUME_CHAIN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pointodyssey_single_sequence_overfit}"
ENCODER="${ENCODER:-base}"
VIDEOMAE_MODEL="$(resolve_videomae_model)"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-100}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME="${RESUME:-}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-}"

STEPS="${STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LAMBDA_3D="${LAMBDA_3D:-1.0}"
LAMBDA_RAW_3D="${LAMBDA_RAW_3D:-1.0}"
LAMBDA_CONF="${LAMBDA_CONF:-0.0}"
LAMBDA_2D="${LAMBDA_2D:-0.1}"
LAMBDA_VIS="${LAMBDA_VIS:-0.1}"
LAMBDA_DISP="${LAMBDA_DISP:-0.1}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.5}"
CONF_WEIGHTING_START_STEP="${CONF_WEIGHTING_START_STEP:-999999}"
CONF_RAMP_STEPS="${CONF_RAMP_STEPS:-0}"
IMG_SIZE="${IMG_SIZE:-256}"
T_TGT_EQ_T_CAM_RATIO="${T_TGT_EQ_T_CAM_RATIO:-1.0}"

export MODE
export STAGE
export NO_RESUME_CHAIN
export OUTPUT_ROOT
export ENCODER
export VIDEOMAE_MODEL
export PATCH_PROVIDER
export QUERY_CHUNK_SIZE
export BATCH_SIZE
export VAL_EVERY_EPOCHS

CMD=(
  bash train.sh
  --train-split "$TRAIN_SPLIT"
  --val-split "$VAL_SPLIT"
  --train-sequence "$TRAIN_SEQUENCE"
  --val-sequence "$VAL_SEQUENCE"
  --pointodyssey-sanity-split "$POINTODYSSEY_SANITY_SPLIT"
  --pointodyssey-sanity-sequence "$POINTODYSSEY_SANITY_SEQUENCE"
  --steps "$STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --lambda-3d "$LAMBDA_3D"
  --lambda-raw-3d "$LAMBDA_RAW_3D"
  --lambda-conf "$LAMBDA_CONF"
  --lambda-2d "$LAMBDA_2D"
  --lambda-vis "$LAMBDA_VIS"
  --lambda-disp "$LAMBDA_DISP"
  --lambda-normal "$LAMBDA_NORMAL"
  --conf-weighting-start-step "$CONF_WEIGHTING_START_STEP"
  --conf-ramp-steps "$CONF_RAMP_STEPS"
  --disable-train-augs
  --disable-motion-boundary-oversampling
  --img-size "$IMG_SIZE"
  --t-tgt-eq-t-cam-ratio "$T_TGT_EQ_T_CAM_RATIO"
)

if [[ "$AUTO_RESUME" == "1" && -z "$RESUME" ]]; then
  CMD+=(--auto-resume)
fi

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
  CMD+=(--pretrained-weights "$PRETRAINED_WEIGHTS")
fi

if [[ -n "$PRETRAINED_ENCODER" ]]; then
  CMD+=(--pretrained-encoder "$PRETRAINED_ENCODER")
fi

CMD+=("$@")

printf 'Single-sequence training\n'
printf 'Sequence: %s\n' "$TRAIN_SEQUENCE"
printf 'Train split: %s\n' "$TRAIN_SPLIT"
printf 'Val split: %s\n' "$VAL_SPLIT"
if [[ "$AUTO_RESUME" == "1" && -n "$RESUME" ]]; then
  printf 'Auto resume: ignored because RESUME is set\n'
fi
if [[ "$AUTO_RESUME" == "1" && -z "$RESUME" ]]; then
  printf 'Auto resume: enabled\n'
fi
if [[ -n "$RESUME" ]]; then
  printf 'Resume checkpoint: %s\n' "$RESUME"
fi
if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
  printf 'Pretrained weights: %s\n' "$PRETRAINED_WEIGHTS"
fi
if [[ -n "$PRETRAINED_ENCODER" ]]; then
  printf 'Pretrained encoder: %s\n' "$PRETRAINED_ENCODER"
fi
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
