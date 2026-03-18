#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  local resolved_python
  resolved_python="$(type -P python || true)"
  if [[ -n "$resolved_python" ]]; then
    printf '%s\n' "$resolved_python"
    return 0
  fi

  printf 'python\n'
  return 0
}

resolve_data_root() {
  local requested_root="${DATA_ROOT:-}"
  local candidates=()

  if [[ -n "$requested_root" ]]; then
    candidates+=("$requested_root")
  fi
  candidates+=(
    "/16t/e/d4rt/PointOdyssey"
    "/mnt/D4RT/datasets/PointOdyssey"
    "/home/zbf/16t/e/d4rt/PointOdyssey"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested_root" ]]; then
    echo "DATA_ROOT does not exist: $requested_root" >&2
  fi
  echo "Could not resolve a valid PointOdyssey data root." >&2
  echo "Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

MODE="${MODE:-normal}"
STAGE="${STAGE:-full_main}"
DATA_ROOT="$(resolve_data_root)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pointodyssey_curriculum}"
PYTHON_BIN="$(resolve_python_bin)"
BASE_CONFIG="${BASE_CONFIG:-configs/d4rt_pointodyssey_curriculum_base.yaml}"

ENCODER="${ENCODER:-base}"
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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
)

CMD+=("$@")

printf 'Mode: %s\n' "$MODE"
printf 'Stage: %s\n' "$STAGE"
printf 'Data root: %s\n' "$DATA_ROOT"
printf 'Python bin: %s\n' "$PYTHON_BIN"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
