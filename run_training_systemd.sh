#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_training_systemd.sh  --unit d4rt

Options:
  --unit NAME             systemd unit name. Default: d4rt-training-<timestamp>
  --log-dir DIR           log directory. Default: <worktree>/logs
  -h, --help              show this help.

  # 查看训练状态                                                                                                                                                                                                                        
  systemctl status d4rt-training-<timestamp>                                                                                                                                                                                            
                                                                                                                                                                                                                                        
  # 停止训练                                                                                                                                                                                                                            
  systemctl stop d4rt
                                                                                                                                                                                                                                        
  # 查看日志      
  journalctl -u d4rt-training-<timestamp> -f


Examples:
  ./run_training_systemd.sh
  ./run_training_systemd.sh --unit my-training-run
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

shell_safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-' | sed 's/^-//; s/-$//'
}

log_dir=""
unit=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      [[ $# -ge 2 ]] || die "--unit requires a value"
      unit="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir requires a value"
      log_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

command -v systemd-run >/dev/null 2>&1 || die "systemd-run not found in PATH"

work_dir="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration"
script_path="${work_dir}/train_mixture_5datasets_3gpu_cos_planned.sh"

[[ -f "$script_path" ]] || die "training script does not exist: $script_path"

if [[ -z "$log_dir" ]]; then
  log_dir="${work_dir}/logs"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$unit" ]]; then
  unit="d4rt-training-${timestamp}"
fi
unit="$(shell_safe_name "$unit")"

main_log="${log_dir}/training_${timestamp}.log"
# cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"

# Pin the data-loading settings that were validated for the 6-rank, batch-40
# planned pipeline.  The training script still has the same defaults, but
# systemd runs are long-lived enough that inheriting stale shell overrides
# (for example BUILDER_WORKERS=24 / sdk_workers=8 / low_watermark=0.99) can
# silently put the job back onto a slower configuration.
training_env=(
  # "CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
  "NPROC_PER_NODE=6"
  "BATCH_SIZE=40"
  "BUILDER_WORKERS=30"
  "PREFETCH_DEPTH=7680"
  "BATCH_PREFETCH_DEPTH=24"
  "D4RT_PLANNED_RELAXED_ORDER=1"
  "D4RT_PLANNED_RELAXED_LOOKAHEAD=400"
  "DATASET_LOCALITY_SIZE=16"
  "SEQUENCE_LOCALITY_SIZE=96"
  "FRAME_LOCALITY_RADIUS=12"
  "DYNAMIC_REPLICA_ROOT=/data5/d4rt_dataset/Dynamic_Replica"
  "CO3DV2_ROOT=/data4/d4rt_dataset/Co3Dv2"
  "SCANNETPP_ROOT=/data5/d4rt_dataset/scannetpp/data"
  "SCANNETPP_SPLITS_DIR=/data5/d4rt_dataset/scannetpp/splits"
  "SCANNETPP_SCENES_RECORD=/data5/d4rt_dataset/scannetpp/scenes_record.json"
  "SAMPLE_STAGE_DATASETS=pointodyssey,kubric,co3dv2,scannetpp,dynamic_replica"
  "SAMPLE_STAGE_EXTRA_MOUNT_ROOTS=/data4,/data5"
  "SAMPLE_STAGE_SDK_WORKERS=8"
  "SAMPLE_STAGE_REQUEST_TIMEOUT_S=5"
  "SAMPLE_STAGE_REQUEST_RETRIES=1"
  "SAMPLE_STAGE_CACHE_MAX_GB=372"
  "SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO=0.85"
  "SAMPLE_STAGE_EVICTION_MODE=external"
  "SAMPLE_STAGE_PRECLEAN=1"
  "SAMPLE_STAGE_PRECLEAN_FORCE_LOW_WATERMARK=1"
  "D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S=7200"
  "SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S=300"
  "D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE=0"
  "ROLLING_WARM_ENABLE=0"
  "D4RT_ROLLING_WARM_BLOCK_BATCHES=10"
  "D4RT_ROLLING_WARM_TIMEOUT_S=900"
  "ROLLING_WARM_LOOKAHEAD_BLOCKS=12"
  "ROLLING_WARM_MAX_READY_BLOCKS=16"
  "ROLLING_WARM_INITIAL_READY_BLOCKS=8"
  "ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S=1200"
  "ROLLING_WARM_DATASETS=co3dv2,scannetpp"
  "ROLLING_WARM_EXACT_DATASETS="
  "ROLLING_WARM_SDK_EXACT_DATASETS="
  "ROLLING_WARM_COSCLI_ROUTINES=64"
  "ROLLING_WARM_COSCLI_THREAD_NUM=8"
  "ROLLING_WARM_PREFIX_WORKERS=8"
  "ROLLING_WARM_SDK_WORKERS=32"
  "ROLLING_WARM_SKIP_EXISTING=1"
  "ROLLING_WARM_SCANNETPP_RGB_MODE=frame_cache"
  "ROLLING_WARM_SCANNETPP_RGB_DECODE=1"
  "ROLLING_WARM_SCANNETPP_DECODE_WORKERS=8"
  "ROLLING_WARM_SCANNETPP_FRAME_WORKERS=4"
  "ROLLING_WARM_SCANNETPP_DEPTH_WORKERS=8"
  "ROLLING_WARM_SCANNETPP_H5_WORKERS=8"
  "ROLLING_WARM_BLOCK_WORKERS=2"
  "ROLLING_WARM_COMPUTE_S_PER_BATCH=1.5"
  "D4RT_MAX_TRACK_POINTS=4096"
  "D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS=0"
  "D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS=0"
  "PATCH_PROVIDER=sampled_resized"
  "SCANNETPP_RGB_LOAD_WORKERS=1"
  "SCANNETPP_H5_CHUNK_CACHE_DIR=/data1/zbf/d4rt_scannetpp_h5_chunk_cache"
  "SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES=1"
  "SCANNETPP_H5_CHUNK_CACHE_MAX_GB=80"
  "SCANNETPP_RGB_READ_MODE=frames"
  "SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S=12"
  "SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS=16"
  "SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES=67108864"
  "SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES=2097152"
  "D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT=1"
  "DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS=1"
  "D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE=192"
  "D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE=192"
  "D4RT_DYNAMIC_REPLICA_IO_WORKERS=4"
  "CO3DV2_FRAME_CACHE_ITEMS=384"
  "D4RT_CO3D_FRAME_CACHE_ITEMS=384"
  "CO3DV2_IO_WORKERS=4"
  "DATA_WAIT_DETAIL_MAX_SAMPLES=0"
  "D4RT_TRAIN_WATCHDOG_S=120"
  "D4RT_TRAIN_WATCHDOG_INTERVAL_S=60"
  "PROFILE_DATA_LOADING=0"
  "D4RT_SLOW_SAMPLE_THRESHOLD_S=99999"
  "D4RT_EPOCH_STARTUP_SLEEP_S=90"
  "D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S=99999"
)

echo "Working dir: $work_dir"
echo "Script:      $script_path"
echo "Unit:        $unit"
echo "Log:         $main_log"

mkdir -p "$log_dir"
touch "$main_log"

if systemctl list-units --full --all "${unit}.service" --no-legend 2>/dev/null | grep -q .; then
  die "systemd unit already exists: ${unit}. Stop it first or pass --unit with another name."
fi

systemd-run \
  --unit="$unit" \
  --description="D4RT training ${timestamp}" \
  --working-directory="$work_dir" \
  --setenv=HOME=/root \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=TRAINING_LOG="$main_log" \
  "${training_env[@]/#/--setenv=}" \
  --collect \
  /bin/bash -c '
    exec bash train_mixture_5datasets_3gpu_cos_planned.sh >> "$TRAINING_LOG" 2>&1
  '

echo
echo "Training started. Press Ctrl-C to stop watching only; training keeps running under systemd."
echo "Check later with: systemctl status --no-pager $unit"
echo "View logs with:   tail -f $main_log"
echo "Stop training:    systemctl stop $unit"
echo

tail -n +1 -f "$main_log" &
tail_pid=$!

cleanup_tail() {
  kill "$tail_pid" >/dev/null 2>&1 || true
}
trap cleanup_tail EXIT INT TERM

while systemctl is-active --quiet "$unit"; do
  sleep 2
done

sleep 1
cleanup_tail
wait "$tail_pid" 2>/dev/null || true

echo
systemctl status --no-pager "$unit" 2>/dev/null || true
echo
echo "Training service is no longer active. Check log: $main_log"
