#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

POLL_SECONDS="${POLL_SECONDS:-120}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data/zbf/BlendedMVS}"
KUBRIC_ROOT="${KUBRIC_ROOT:-/data/zbf/kubric}"
WAIT_LOG="${WAIT_LOG:-logs/wait_and_launch_train_mixture_5datasets_3gpu.log}"

NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_INTERVAL="${VAL_INTERVAL:-0}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

timestamp() {
  date '+%F %T'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$WAIT_LOG"
}

check_dataset_ready() {
  local dataset_name="$1"
  local dataset_root="$2"
  /root/miniconda3/envs/d4rt/bin/python - "$dataset_name" "$dataset_root" <<'PY'
import sys
import time
from pathlib import Path

dataset_name = sys.argv[1]
dataset_root = Path(sys.argv[2])

if not dataset_root.is_dir():
    raise SystemExit(2)

if dataset_name == "blendedmvs":
    from datasets.adapters.blendedmvs import BlendedMVSAdapter

    split_file = dataset_root / "BlendedMVS_training.txt"
    if not split_file.is_file():
        raise SystemExit(3)

    adapter = BlendedMVSAdapter(
        root=str(dataset_root),
        split="train",
        use_masked=False,
        strict=False,
        verbose=False,
        cache_dir=None,
    )
    if len(adapter) != 106:
        raise SystemExit(4)
    seq = adapter.list_sequences()[0]
    rec = adapter._get_record(seq)
    frame_indices = list(range(min(48, rec.num_frames)))
    t0 = time.time()
    adapter.load_clip(seq, frame_indices)
    dt = time.time() - t0
    if dt > 30.0:
        raise SystemExit(5)
    raise SystemExit(0)

if dataset_name == "kubric":
    from datasets.adapters.kubric import KubricAdapter

    adapter = KubricAdapter(
        root=str(dataset_root),
        split="train",
        strict=False,
        verbose=False,
        cache_dir=None,
    )
    if len(adapter) != 718:
        raise SystemExit(4)
    seq = adapter.list_sequences()[0]
    frame_indices = list(range(min(48, adapter.get_num_frames(seq))))
    t0 = time.time()
    adapter.load_clip(seq, frame_indices)
    dt = time.time() - t0
    if dt > 30.0:
        raise SystemExit(5)
    raise SystemExit(0)

raise SystemExit(9)
PY
}

maybe_mark_ready() {
  local dataset_name="$1"
  local dataset_root="$2"
  local marker="$dataset_root/.d4rt_ready"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  if check_dataset_ready "$dataset_name" "$dataset_root"; then
    touch "$marker"
    log "Marked $dataset_name ready: $marker"
    return 0
  fi
  return 1
}

launch_training() {
  local ts log_file pid_file
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="logs/train_mixture_5datasets_3gpu_${ts}.log"
  pid_file="logs/train_mixture_5datasets_3gpu_${ts}.pid"

  nohup setsid env \
    NUM_WORKERS="$NUM_WORKERS" \
    VAL_INTERVAL="$VAL_INTERVAL" \
    DIST_TIMEOUT_MINUTES="$DIST_TIMEOUT_MINUTES" \
    BROADCAST_BUFFERS="$BROADCAST_BUFFERS" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    ALLOW_REMOTE_DATA=0 \
    KUBRIC_ROOT="$KUBRIC_ROOT" \
    BLENDEDMVS_ROOT="$BLENDEDMVS_ROOT" \
    bash "$ROOT_DIR/train_mixture_5datasets_3gpu.sh" >"$log_file" 2>&1 < /dev/null &
  echo $! >"$pid_file"
  log "Launched training: pid=$(cat "$pid_file") log=$log_file pidfile=$pid_file"
}

if pgrep -f "train_mixture.py.*mixture_5datasets_blendedmvs_large_3gpu_bs5" >/dev/null 2>&1; then
  log "Training is already running. Exiting watcher."
  exit 0
fi

log "Watcher started. Waiting for local kubric and BlendedMVS mirrors."
log "kubric root=$KUBRIC_ROOT"
log "blendedmvs root=$BLENDEDMVS_ROOT"
log "poll interval=${POLL_SECONDS}s"

while true; do
  blended_count="$(find "$BLENDEDMVS_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)"
  kubric_count="$(find "$KUBRIC_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)"
  log "progress: blendedmvs_dirs=${blended_count:-0} kubric_dirs=${kubric_count:-0}"

  maybe_mark_ready "blendedmvs" "$BLENDEDMVS_ROOT" || true
  maybe_mark_ready "kubric" "$KUBRIC_ROOT" || true

  if [[ -f "$BLENDEDMVS_ROOT/.d4rt_ready" && -f "$KUBRIC_ROOT/.d4rt_ready" ]]; then
    launch_training
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
