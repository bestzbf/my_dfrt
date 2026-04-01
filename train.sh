#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-normal}"
STAGE="${STAGE:-full}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/data/ZBF_Data/d4rt/test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pointodyssey_curriculum}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-configs/d4rt_pointodyssey_curriculum_base.yaml}"

ENCODER="${ENCODER:-base}"
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

case "$MODE" in
  normal)
    RUNNER="scripts/run_pointodyssey_curriculum_stage.sh"
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Currently only MODE=normal is implemented in 3.16/train.sh." >&2
    exit 1
    ;;
esac

export DATA_ROOT OUTPUT_ROOT PYTHON_BIN BASE_CONFIG

CMD=(
  bash "$RUNNER" "$STAGE"
  --encoder "$ENCODER"
)

if [[ -n "$VIDEOMAE_MODEL" ]]; then
  CMD+=(--videomae-model "$VIDEOMAE_MODEL")
fi

CMD+=(
  --patch-provider "$PATCH_PROVIDER"
  --query-chunk-size "$QUERY_CHUNK_SIZE"
  --batch-size "$BATCH_SIZE"
  --val-every-epochs "$VAL_EVERY_EPOCHS"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
)

CMD+=("$@")

printf 'Mode: %s\n' "$MODE"
printf 'Stage: %s\n' "$STAGE"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
