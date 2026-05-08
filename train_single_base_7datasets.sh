#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-28590}"
CONFIG="${CONFIG:-configs/mixture_7datasets_single_base.yaml}"

POINTODYSSEY_LOCAL_ROOT="${POINTODYSSEY_LOCAL_ROOT:-/data2/d4rt/datasets/PointOdyssey}"
if [[ -z "${POINTODYSSEY_ROOT:-}" ]]; then
  if [[ -d "$POINTODYSSEY_LOCAL_ROOT" ]]; then
    POINTODYSSEY_ROOT="$POINTODYSSEY_LOCAL_ROOT"
  else
    POINTODYSSEY_ROOT="/data_cos/hdu_datasets/PointOdyssey"
  fi
fi
POINTODYSSEY_FAST_ROOT="${POINTODYSSEY_FAST_ROOT:-}"
POINTODYSSEY_LOCAL_CACHE_DIR="${POINTODYSSEY_LOCAL_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
POINTODYSSEY_STAGE_ANNO_H5="${POINTODYSSEY_STAGE_ANNO_H5:-0}"
POINTODYSSEY_REQUIRE_TRACKS="${POINTODYSSEY_REQUIRE_TRACKS:-1}"
POINTODYSSEY_ASSUME_TRACKS="${POINTODYSSEY_ASSUME_TRACKS:-0}"
POINTODYSSEY_TRACK_WORKERS="${POINTODYSSEY_TRACK_WORKERS:-16}"
KUBRIC_ROOT="${KUBRIC_ROOT:-/data_cos/hdu_datasets/Kubric}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data_cos/hdu_datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data_cos/hdu_datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

# Isolated spool dir to avoid conflict with multi-GPU training
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp_single}"
MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-50}"
BUILDER_WORKERS="${BUILDER_WORKERS:-8}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-128}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-4}"
SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND:-cos_sdk}"
# COS SDK cache is read-only safe to share with multi-GPU training
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/data1/zbf/d4rt_sample_stage}"
BLENDEDMVS_DEPTH_CACHE_DIR="${BLENDEDMVS_DEPTH_CACHE_DIR:-}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-16}"
SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-100}"
SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO="${SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO:-0.9}"
SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S="${SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S:-30}"
SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S="${SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S:-30}"
SAMPLE_STAGE_WINDOW_RADIUS="${SAMPLE_STAGE_WINDOW_RADIUS:-0}"
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-pointodyssey,kubric,dynamic_replica,co3dv2,scannetpp}"
SAMPLE_STAGE_SCENE_PREFETCH_DATASETS="${SAMPLE_STAGE_SCENE_PREFETCH_DATASETS:-}"
SAMPLE_STAGE_MOUNT_ROOT="${SAMPLE_STAGE_MOUNT_ROOT:-/data_cos}"
SAMPLE_STAGE_BUCKET="${SAMPLE_STAGE_BUCKET:-hd-ai-data-1251882982}"
SAMPLE_STAGE_REGION="${SAMPLE_STAGE_REGION:-ap-beijing}"
SAMPLE_STAGE_PASSWD_FILE="${SAMPLE_STAGE_PASSWD_FILE:-/etc/passwd-s3fs-data_cos}"
AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-1}"
WARM_CACHE_ONLY="${WARM_CACHE_ONLY:-0}"
WARM_ONLY_DATASETS="${WARM_ONLY_DATASETS:-}"
WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-8}"
WARM_VAL="${WARM_VAL:-0}"

VAL_CONFIG="${VAL_CONFIG:-configs/mixture_3datasets_val_local.yaml}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
USE_COMPILE="${USE_COMPILE:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
PROFILE_DATA_LOADING="${PROFILE_DATA_LOADING:-0}"
DATA_PROFILE_INTERVAL="${DATA_PROFILE_INTERVAL:-20}"
BUILDER_PROFILE_THRESHOLD_S="${BUILDER_PROFILE_THRESHOLD_S:-5}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-2.0}"
DATA_WAIT_DETAIL="${DATA_WAIT_DETAIL:-1}"
DATA_WAIT_COMPARE_FWD="${DATA_WAIT_COMPARE_FWD:-1}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-8}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_7datasets_single_base}"
PRETRAIN="${PRETRAIN:-/data/zbf/openclaw/d4rt/outputs/mixture_3datasets_finetune_v2_gpu5/checkpoint_latest_50.pth}"
RESUME="${RESUME:-}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.1}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.0}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-0.9}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-1}"
VARIANT="${VARIANT:-base}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"
D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
D4RT_BUILDER_FAULTHANDLER="${D4RT_BUILDER_FAULTHANDLER:-0}"
D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-120}"
D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-0}"
D4RT_PROFILE_BUILDER_ALL="${D4RT_PROFILE_BUILDER_ALL:-0}"
D4RT_VERBOSE_BUILDER="${D4RT_VERBOSE_BUILDER:-0}"
D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-0}"

case "$VARIANT" in
  large)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-large}"
    ;;
  base)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-base}"
    ;;
  *)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-0}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-}"
    ;;
esac

mkdir -p "$TMPDIR"
mkdir -p "$INDEX_CACHE_DIR"
[[ -n "$SAMPLE_STAGE_BACKEND" ]] && mkdir -p "$SAMPLE_STAGE_ROOT"
[[ -n "$BLENDEDMVS_DEPTH_CACHE_DIR" ]] && mkdir -p "$BLENDEDMVS_DEPTH_CACHE_DIR"
[[ -n "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR" ]] && mkdir -p "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"

export TMPDIR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_SCANNETPP_TRUST_ALLOWLIST="${D4RT_SCANNETPP_TRUST_ALLOWLIST:-1}"
export D4RT_BUILDER_FAULTHANDLER
export D4RT_VERBOSE_BUILDER
export D4RT_BUILD_TIMEOUT
export D4RT_PROFILE_BUILDER
export D4RT_PROFILE_BUILDER_ALL
export D4RT_PLANNED_WAIT_LOG
export D4RT_POINTODYSSEY_STAGE_ANNO_H5="$POINTODYSSEY_STAGE_ANNO_H5"
export D4RT_POINTODYSSEY_ANNO_FRAME_CACHE_DIR="$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
export D4RT_BLENDEDMVS_DEPTH_CACHE_DIR="$BLENDEDMVS_DEPTH_CACHE_DIR"
if [[ "$D4RT_SUPPRESS_MKL_WARNING" == "1" ]]; then
  export PYTHONWARNINGS="ignore:mkl-service package failed to import:UserWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"
fi

if [[ "$AUTO_WARM_INDEX_CACHE" == "1" || "$WARM_CACHE_ONLY" == "1" ]]; then
  CONFIG="$CONFIG" \
  INDEX_CACHE_DIR="$INDEX_CACHE_DIR" \
  POINTODYSSEY_ROOT="$POINTODYSSEY_ROOT" \
  POINTODYSSEY_LOCAL_ROOT="$POINTODYSSEY_LOCAL_ROOT" \
  POINTODYSSEY_LOCAL_CACHE_DIR="$POINTODYSSEY_LOCAL_CACHE_DIR" \
  POINTODYSSEY_REQUIRE_TRACKS="$POINTODYSSEY_REQUIRE_TRACKS" \
  POINTODYSSEY_ASSUME_TRACKS="$POINTODYSSEY_ASSUME_TRACKS" \
  POINTODYSSEY_TRACK_WORKERS="$POINTODYSSEY_TRACK_WORKERS" \
  KUBRIC_ROOT="$KUBRIC_ROOT" \
  DYNAMIC_REPLICA_ROOT="$DYNAMIC_REPLICA_ROOT" \
  CO3DV2_ROOT="$CO3DV2_ROOT" \
  BLENDEDMVS_ROOT="$BLENDEDMVS_ROOT" \
  MVSSYNTH_ROOT="$MVSSYNTH_ROOT" \
  SCANNETPP_ROOT="$SCANNETPP_ROOT" \
  SCANNETPP_SPLITS_DIR="$SCANNETPP_SPLITS_DIR" \
  CO3DV2_DENYLIST="$CO3DV2_DENYLIST" \
  INDEX_WORKERS="$WARM_INDEX_WORKERS" \
  WARM_VAL="$WARM_VAL" \
  ONLY_DATASETS="$WARM_ONLY_DATASETS" \
  bash "$ROOT_DIR/cos/warm_index_cache_5datasets_cos.sh"
fi

[[ "$WARM_CACHE_ONLY" == "1" ]] && { echo "WARM_CACHE_ONLY=1, stopping after warmup"; exit 0; }

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/mixture_single_base.XXXXXX.yaml")"
cleanup() { [[ -f "$TEMP_CONFIG" ]] && rm -f "$TEMP_CONFIG"; }
trap cleanup EXIT

"$PYTHON_BIN" - "$CONFIG" "$TEMP_CONFIG" "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" "$BUILDER_WORKERS" "$PREFETCH_DEPTH" "$MAX_SPOOL_BYTES_GB" "$SAMPLE_STAGE_BACKEND" "$SAMPLE_STAGE_ROOT" "$SAMPLE_STAGE_SDK_WORKERS" "$SAMPLE_STAGE_DATASETS" "$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS" "$SAMPLE_STAGE_MOUNT_ROOT" "$SAMPLE_STAGE_BUCKET" "$SAMPLE_STAGE_REGION" "$SAMPLE_STAGE_PASSWD_FILE" "$SAMPLE_STAGE_CACHE_MAX_GB" "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO" "$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S" "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S" "$SAMPLE_STAGE_WINDOW_RADIUS" <<'PY'
from pathlib import Path
import sys
import yaml

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
pointodyssey_root = sys.argv[3]
pointodyssey_fast_root = sys.argv[4]
kubric_root = sys.argv[5]
dynamic_replica_root = sys.argv[6]
co3dv2_root = sys.argv[7]
blendedmvs_root = sys.argv[8]
mvssynth_root = sys.argv[9]
scannetpp_root = sys.argv[10]
scannetpp_splits_dir = sys.argv[11]
scannetpp_scenes_record = sys.argv[12]
index_cache_dir = sys.argv[13]
co3dv2_denylist = sys.argv[14]
builder_workers = int(sys.argv[15])
prefetch_depth = int(sys.argv[16])
max_spool_bytes = int(float(sys.argv[17]) * 1024**3)
sample_stage_backend = sys.argv[18].strip()
sample_stage_root = sys.argv[19].strip()
sample_stage_sdk_workers = int(sys.argv[20])
sample_stage_datasets = sys.argv[21].strip()
sample_stage_scene_prefetch_datasets = sys.argv[22].strip()
sample_stage_mount_root = sys.argv[23].strip()
sample_stage_bucket = sys.argv[24].strip()
sample_stage_region = sys.argv[25].strip()
sample_stage_passwd_file = sys.argv[26].strip()
sample_stage_cache_max_bytes = int(float(sys.argv[27]) * 1024**3)
sample_stage_cache_low_watermark_ratio = float(sys.argv[28])
sample_stage_cache_touch_interval_s = float(sys.argv[29])
sample_stage_cache_scan_interval_s = float(sys.argv[30])
sample_stage_window_radius = int(sys.argv[31])

config = yaml.safe_load(src_path.read_text())
datasets = {item["name"]: item for item in config["datasets"]}

root_map = {
    "pointodyssey": pointodyssey_root,
    "kubric": kubric_root,
    "dynamic_replica": dynamic_replica_root,
    "co3dv2": co3dv2_root,
    "blendedmvs": blendedmvs_root,
    "mvssynth": mvssynth_root,
    "scannetpp": scannetpp_root,
}
for name, root in root_map.items():
    if name in datasets:
        datasets[name]["root"] = root

if "scannetpp" in datasets:
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["splits_dir"] = scannetpp_splits_dir
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["scenes_record"] = scannetpp_scenes_record
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["strict"] = False

datasets["co3dv2"].setdefault("adapter_kwargs", {})["sequence_denylist"] = co3dv2_denylist

if pointodyssey_fast_root:
    datasets["pointodyssey"].setdefault("adapter_kwargs", {})["fast_root"] = pointodyssey_fast_root

config["index_cache_dir"] = index_cache_dir
config["planned_mode"] = True
config["builder_workers"] = builder_workers
config["prefetch_depth"] = prefetch_depth
config["max_spool_bytes"] = max_spool_bytes
if sample_stage_backend:
    config["sample_stage_backend"] = sample_stage_backend
    config["sample_stage_root"] = sample_stage_root
    config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
    config["sample_stage_datasets"] = [s.strip() for s in sample_stage_datasets.split(",") if s.strip()]
    config["sample_stage_scene_prefetch_datasets"] = [s.strip() for s in sample_stage_scene_prefetch_datasets.split(",") if s.strip()]
    config["sample_stage_mount_root"] = sample_stage_mount_root
    config["sample_stage_bucket"] = sample_stage_bucket
    config["sample_stage_region"] = sample_stage_region
    config["sample_stage_passwd_file"] = sample_stage_passwd_file
    config["sample_stage_cache_max_bytes"] = sample_stage_cache_max_bytes
    config["sample_stage_cache_low_watermark_ratio"] = sample_stage_cache_low_watermark_ratio
    config["sample_stage_cache_touch_interval_s"] = sample_stage_cache_touch_interval_s
    config["sample_stage_cache_scan_interval_s"] = sample_stage_cache_scan_interval_s
    config["sample_stage_window_radius"] = sample_stage_window_radius

dst_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

echo "[single-base] EFFECTIVE_CONFIG=$TEMP_CONFIG"

cmd=(
  "$TORCHRUN_BIN"
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  train_mixture.py
  --config "$TEMP_CONFIG"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --grad-accum "$GRAD_ACCUM"
  --log-interval "$LOG_INTERVAL"
  --epochs "$EPOCHS"
  --lr "$LR"
  --num-frames 48
  --output-dir "$OUTPUT_DIR"
  --patch-provider "$PATCH_PROVIDER"
  --val-interval "$VAL_INTERVAL"
  --val-samples "$VAL_SAMPLES"
  --save-interval "$SAVE_INTERVAL"
  --loss-w-3d "$LOSS_W_3D"
  --loss-w-conf "$LOSS_W_CONF"
  --loss-w-normal "$LOSS_W_NORMAL"
  --loss-w-static-reprojection "$LOSS_W_STATIC_REPROJ"
  --loss-3d-mode "$LOSS_3D_MODE"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --dist-timeout-minutes "$DIST_TIMEOUT_MINUTES"
  --variant "$VARIANT"
  --planned-mode
  --builder-workers "$BUILDER_WORKERS"
  --prefetch-depth "$PREFETCH_DEPTH"
  --batch-prefetch-depth "$BATCH_PREFETCH_DEPTH"
)

[[ "$USE_COMPILE" == "1" ]] && cmd+=(--compile)
[[ "$USE_VIDEOMAE_V2_INIT" == "1" ]] && cmd+=(--use-videomae-v2-init)
[[ -n "$VIDEOMAE_MODEL" ]] && cmd+=(--videomae-model "$VIDEOMAE_MODEL")
[[ "$RESET_CONF_HEAD_ON_PRETRAIN" == "1" ]] && cmd+=(--reset-confidence-head-on-pretrain)
[[ "$BROADCAST_BUFFERS" == "1" ]] && cmd+=(--broadcast-buffers)
[[ -n "$VAL_CONFIG" ]] && cmd+=(--val-config "$VAL_CONFIG")
[[ -n "$PRETRAIN" ]] && cmd+=(--pretrain "$PRETRAIN")
[[ -n "$RESUME" ]] && cmd+=(--resume "$RESUME")
cmd+=("$@")

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
