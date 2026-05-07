#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-28580}"
CONFIG="${CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"
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
# CO3DV2_ROOT="${CO3DV2_ROOT:-/data2/d4rt/datasets/Co3Dv2}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data_cos/hdu_datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"
MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"
BUILDER_WORKERS="${BUILDER_WORKERS:-8}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-256}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-4}"
SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND:-cos_sdk}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/data1/zbf/d4rt_sample_stage}"
POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
BLENDEDMVS_DEPTH_CACHE_DIR="${BLENDEDMVS_DEPTH_CACHE_DIR:-}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-32}"
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
WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-16}"
WARM_VAL="${WARM_VAL:-0}"

VAL_CONFIG="${VAL_CONFIG:-configs/mixture_3datasets_val_local.yaml}"
BATCH_SIZE="${BATCH_SIZE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
USE_COMPILE="${USE_COMPILE:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
PROFILE_DATA_LOADING="${PROFILE_DATA_LOADING:-0}"
DATA_PROFILE_INTERVAL="${DATA_PROFILE_INTERVAL:-20}"
BUILDER_PROFILE_THRESHOLD_S="${BUILDER_PROFILE_THRESHOLD_S:-5}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-2.0}"
DATA_WAIT_DETAIL="${DATA_WAIT_DETAIL:-0}"
DATA_WAIT_COMPARE_FWD="${DATA_WAIT_COMPARE_FWD:-0}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-8}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-5e-5}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-1.0}"
DECODER_LR_MULT="${DECODER_LR_MULT:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_6datasets_cos_planned_from200}"
# DEFAULT_PRETRAIN="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from0/checkpoint_latest_200.pth"
DEFAULT_PRETRAIN=""
PRETRAIN="${PRETRAIN-$DEFAULT_PRETRAIN}"
# DEFAULT_RESUME=""
DEFAULT_RESUME="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/checkpoint_latest_385.pth"
RESUME="${RESUME-$DEFAULT_RESUME}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_2D="${LOSS_W_2D:-0.1}"
LOSS_W_VIS="${LOSS_W_VIS:-0.1}"
LOSS_W_DISP="${LOSS_W_DISP:-0.1}"
LOSS_W_CONF="${LOSS_W_CONF:-0.1}"
LOSS_W_CONF_WARMUP_STEPS="${LOSS_W_CONF_WARMUP_STEPS:-0}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.0}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-1.0}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-1}"
VARIANT="${VARIANT:-large}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"
D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
D4RT_BUILDER_FAULTHANDLER="${D4RT_BUILDER_FAULTHANDLER:-0}"
D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-100}"
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
if [[ -n "$SAMPLE_STAGE_BACKEND" ]]; then
  mkdir -p "$SAMPLE_STAGE_ROOT"
fi
if [[ -n "$BLENDEDMVS_DEPTH_CACHE_DIR" ]]; then
  mkdir -p "$BLENDEDMVS_DEPTH_CACHE_DIR"
fi
if [[ -n "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR" ]]; then
  mkdir -p "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
fi

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
  mkl_warning_filter="ignore:mkl-service package failed to import:UserWarning"
  if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${mkl_warning_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="$mkl_warning_filter"
  fi
fi

echo "[mixture_5datasets_cos_planned] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_5datasets_cos_planned] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[mixture_5datasets_cos_planned] MASTER_PORT=$MASTER_PORT"
echo "[mixture_5datasets_cos_planned] CONFIG=$CONFIG"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_ROOT=$POINTODYSSEY_ROOT"
if [[ -n "$POINTODYSSEY_FAST_ROOT" ]]; then
  echo "[mixture_5datasets_cos_planned] POINTODYSSEY_FAST_ROOT=$POINTODYSSEY_FAST_ROOT"
else
  echo "[mixture_5datasets_cos_planned] POINTODYSSEY_FAST_ROOT=<none>"
fi
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_LOCAL_ROOT=$POINTODYSSEY_LOCAL_ROOT"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_LOCAL_CACHE_DIR=$POINTODYSSEY_LOCAL_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_ANNO_FRAME_CACHE_DIR=$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_STAGE_ANNO_H5=$POINTODYSSEY_STAGE_ANNO_H5"
echo "[mixture_5datasets_cos_planned] KUBRIC_ROOT=$KUBRIC_ROOT"
echo "[mixture_5datasets_cos_planned] DYNAMIC_REPLICA_ROOT=$DYNAMIC_REPLICA_ROOT"
echo "[mixture_5datasets_cos_planned] CO3DV2_ROOT=$CO3DV2_ROOT"
echo "[mixture_5datasets_cos_planned] BLENDEDMVS_ROOT=$BLENDEDMVS_ROOT"
echo "[mixture_5datasets_cos_planned] MVSSYNTH_ROOT=$MVSSYNTH_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNETPP_ROOT=$SCANNETPP_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNETPP_SPLITS_DIR=$SCANNETPP_SPLITS_DIR"
echo "[mixture_5datasets_cos_planned] SCANNETPP_SCENES_RECORD=$SCANNETPP_SCENES_RECORD"
if [[ "$CO3DV2_ROOT" == /data_cos/* && ",$SAMPLE_STAGE_DATASETS," != *",co3dv2,"* ]]; then
  echo "[mixture_5datasets_cos_planned] WARNING: Co3Dv2 is on /data_cos but is not enabled in SAMPLE_STAGE_DATASETS; it will read through the mounted filesystem."
fi
echo "[mixture_5datasets_cos_planned] BLENDEDMVS_DEPTH_CACHE_DIR=$BLENDEDMVS_DEPTH_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] INDEX_CACHE_DIR=$INDEX_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] TMPDIR=$TMPDIR"
echo "[mixture_5datasets_cos_planned] MAX_SPOOL_BYTES_GB=$MAX_SPOOL_BYTES_GB"
echo "[mixture_5datasets_cos_planned] BUILDER_WORKERS=$BUILDER_WORKERS"
echo "[mixture_5datasets_cos_planned] PREFETCH_DEPTH=$PREFETCH_DEPTH"
echo "[mixture_5datasets_cos_planned] BATCH_PREFETCH_DEPTH=$BATCH_PREFETCH_DEPTH"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_BACKEND=$SAMPLE_STAGE_BACKEND"
if [[ -n "$SAMPLE_STAGE_BACKEND" ]]; then
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_ROOT=$SAMPLE_STAGE_ROOT"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_SDK_WORKERS=$SAMPLE_STAGE_SDK_WORKERS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_MAX_GB=$SAMPLE_STAGE_CACHE_MAX_GB"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO=$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S=$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S=$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_WINDOW_RADIUS=$SAMPLE_STAGE_WINDOW_RADIUS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_DATASETS=$SAMPLE_STAGE_DATASETS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_SCENE_PREFETCH_DATASETS=$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_BUCKET=$SAMPLE_STAGE_BUCKET"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_REGION=$SAMPLE_STAGE_REGION"
fi
echo "[mixture_5datasets_cos_planned] AUTO_WARM_INDEX_CACHE=$AUTO_WARM_INDEX_CACHE"
echo "[mixture_5datasets_cos_planned] WARM_CACHE_ONLY=$WARM_CACHE_ONLY"
echo "[mixture_5datasets_cos_planned] WARM_INDEX_WORKERS=$WARM_INDEX_WORKERS"
echo "[mixture_5datasets_cos_planned] WARM_VAL=$WARM_VAL"
if [[ -n "$WARM_ONLY_DATASETS" ]]; then
  echo "[mixture_5datasets_cos_planned] WARM_ONLY_DATASETS=$WARM_ONLY_DATASETS"
fi
if [[ -n "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] VAL_CONFIG=$VAL_CONFIG"
fi
echo "[mixture_5datasets_cos_planned] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_5datasets_cos_planned] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE )))"
echo "[mixture_5datasets_cos_planned] NUM_WORKERS=$NUM_WORKERS"
echo "[mixture_5datasets_cos_planned] LOG_INTERVAL=$LOG_INTERVAL"
echo "[mixture_5datasets_cos_planned] PROFILE_DATA_LOADING=$PROFILE_DATA_LOADING"
echo "[mixture_5datasets_cos_planned] DATA_PROFILE_INTERVAL=$DATA_PROFILE_INTERVAL"
echo "[mixture_5datasets_cos_planned] BUILDER_PROFILE_THRESHOLD_S=$BUILDER_PROFILE_THRESHOLD_S"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_THRESHOLD_S=$DATA_WAIT_THRESHOLD_S"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_DETAIL=$DATA_WAIT_DETAIL"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_COMPARE_FWD=$DATA_WAIT_COMPARE_FWD"
echo "[mixture_5datasets_cos_planned] DIST_TIMEOUT_MINUTES=$DIST_TIMEOUT_MINUTES"
echo "[mixture_5datasets_cos_planned] BROADCAST_BUFFERS=$BROADCAST_BUFFERS"
echo "[mixture_5datasets_cos_planned] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[mixture_5datasets_cos_planned] ENCODER_LR_MULT=$ENCODER_LR_MULT"
echo "[mixture_5datasets_cos_planned] DECODER_LR_MULT=$DECODER_LR_MULT"
echo "[mixture_5datasets_cos_planned] LOSS_W_3D=$LOSS_W_3D"
echo "[mixture_5datasets_cos_planned] LOSS_W_2D=$LOSS_W_2D"
echo "[mixture_5datasets_cos_planned] LOSS_W_VIS=$LOSS_W_VIS"
echo "[mixture_5datasets_cos_planned] LOSS_W_DISP=$LOSS_W_DISP"
echo "[mixture_5datasets_cos_planned] LOSS_W_CONF=$LOSS_W_CONF"
echo "[mixture_5datasets_cos_planned] LOSS_W_CONF_WARMUP_STEPS=$LOSS_W_CONF_WARMUP_STEPS"
echo "[mixture_5datasets_cos_planned] LOSS_W_NORMAL=$LOSS_W_NORMAL"
echo "[mixture_5datasets_cos_planned] RESET_CONF_HEAD_ON_PRETRAIN=$RESET_CONF_HEAD_ON_PRETRAIN"
echo "[mixture_5datasets_cos_planned] VARIANT=$VARIANT"
echo "[mixture_5datasets_cos_planned] USE_VIDEOMAE_V2_INIT=$USE_VIDEOMAE_V2_INIT"
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  echo "[mixture_5datasets_cos_planned] VIDEOMAE_MODEL=$VIDEOMAE_MODEL"
fi
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_5datasets_cos_planned] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_5datasets_cos_planned] PRETRAIN=<none>"
fi
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "[mixture_5datasets_cos_planned] RESUME not found: $RESUME" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] RESUME=$RESUME"
else
  echo "[mixture_5datasets_cos_planned] RESUME=<none>"
fi
if [[ -n "$VAL_CONFIG" && ! -f "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] VAL_CONFIG not found: $VAL_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] CONFIG not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CO3DV2_DENYLIST" ]]; then
  echo "[mixture_5datasets_cos_planned] CO3DV2_DENYLIST not found: $CO3DV2_DENYLIST" >&2
  exit 1
fi

TEMP_CONFIG=""
CONFIG_TO_USE="$CONFIG"
cleanup() {
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    rm -f "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT

if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[mixture_5datasets_cos_planned] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TORCHRUN_BIN" ]]; then
  echo "[mixture_5datasets_cos_planned] TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2
  exit 1
fi

if [[ "$AUTO_WARM_INDEX_CACHE" == "1" || "$WARM_CACHE_ONLY" == "1" ]]; then
  if [[ ! -f "$ROOT_DIR/warm_index_cache_5datasets_cos.sh" ]]; then
    echo "[mixture_5datasets_cos_planned] warm script not found: $ROOT_DIR/warm_index_cache_5datasets_cos.sh" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] warming index cache before torchrun"
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
  SCANNETPP_SCENES_RECORD="$SCANNETPP_SCENES_RECORD" \
  CO3DV2_DENYLIST="$CO3DV2_DENYLIST" \
  INDEX_WORKERS="$WARM_INDEX_WORKERS" \
  WARM_VAL="$WARM_VAL" \
  ONLY_DATASETS="$WARM_ONLY_DATASETS" \
  bash "$ROOT_DIR/warm_index_cache_5datasets_cos.sh"
fi

if [[ "$WARM_CACHE_ONLY" == "1" ]]; then
  echo "[mixture_5datasets_cos_planned] WARM_CACHE_ONLY=1, stop after warmup"
  exit 0
fi

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/mixture_5datasets_cos_planned.XXXXXX.yaml")"
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

required_markers = {
    "pointodyssey": ("train",),
    "kubric": (),
    "dynamic_replica": ("train",),
    "co3dv2": ("apple",),
    "blendedmvs": ("BlendedMVS_training.txt",),
    "mvssynth": (),
    "scannetpp": (),
}


def validate_root(dataset_name: str, root: str) -> None:
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[mixture_5datasets_cos_planned] {dataset_name}.root not found or not a directory: {root}"
        )
    for marker in required_markers.get(dataset_name, ()):
        if not (path / marker).exists():
            raise SystemExit(
                f"[mixture_5datasets_cos_planned] {dataset_name}.root is incomplete: missing {path / marker}"
            )
    if dataset_name == "kubric":
        pass  # skip iterdir check — index cache already validates scene existence


def validate_optional_dir(label: str, root: str) -> None:
    if not root:
        return
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[mixture_5datasets_cos_planned] {label} not found or not a directory: {root}"
        )


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

for dataset_name, root in root_map.items():
    if dataset_name not in datasets:
        continue
    validate_root(dataset_name, root)
    datasets[dataset_name]["root"] = root

# scannetpp splits_dir and strict=False for COS mount
if "scannetpp" in datasets:
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["splits_dir"] = scannetpp_splits_dir
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["scenes_record"] = scannetpp_scenes_record
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["strict"] = False

validate_optional_dir("pointodyssey.fast_root", pointodyssey_fast_root)

pointodyssey_cfg = datasets["pointodyssey"].setdefault("adapter_kwargs", {})
if pointodyssey_fast_root:
    pointodyssey_cfg["fast_root"] = pointodyssey_fast_root
else:
    pointodyssey_cfg.pop("fast_root", None)
try:
    point_root_path = Path(pointodyssey_root)
    stage_mount_path = Path(sample_stage_mount_root)
    point_on_stage_mount = point_root_path.is_relative_to(stage_mount_path)
except Exception:
    point_on_stage_mount = False
if point_on_stage_mount:
    pointodyssey_cfg["runtime_sanitize"] = False
else:
    pointodyssey_cfg.pop("runtime_sanitize", None)

co3dv2_cfg = datasets["co3dv2"].setdefault("adapter_kwargs", {})
co3dv2_cfg["sequence_denylist"] = co3dv2_denylist

config["index_cache_dir"] = index_cache_dir
config["planned_mode"] = True
config["builder_workers"] = builder_workers
config["prefetch_depth"] = prefetch_depth
config["max_spool_bytes"] = max_spool_bytes
if sample_stage_backend:
    config["sample_stage_backend"] = sample_stage_backend
    config["sample_stage_root"] = sample_stage_root
    config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
    config["sample_stage_datasets"] = [
        item.strip()
        for item in sample_stage_datasets.split(",")
        if item.strip()
    ]
    config["sample_stage_scene_prefetch_datasets"] = [
        item.strip()
        for item in sample_stage_scene_prefetch_datasets.split(",")
        if item.strip()
    ]
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
CONFIG_TO_USE="$TEMP_CONFIG"
echo "[mixture_5datasets_cos_planned] EFFECTIVE_CONFIG=$CONFIG_TO_USE"

cmd=(
  "$TORCHRUN_BIN"
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  train_mixture.py
  --config "$CONFIG_TO_USE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --grad-accum "$GRAD_ACCUM"
  --log-interval "$LOG_INTERVAL"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder-lr-mult "$ENCODER_LR_MULT"
  --decoder-lr-mult "$DECODER_LR_MULT"
  --num-frames 48
  --output-dir "$OUTPUT_DIR"
  --patch-provider "$PATCH_PROVIDER"
  --val-interval "$VAL_INTERVAL"
  --val-samples "$VAL_SAMPLES"
  --save-interval "$SAVE_INTERVAL"
  --loss-w-3d "$LOSS_W_3D"
  --loss-w-2d "$LOSS_W_2D"
  --loss-w-vis "$LOSS_W_VIS"
  --loss-w-disp "$LOSS_W_DISP"
  --loss-w-conf "$LOSS_W_CONF"
  --loss-conf-warmup-steps "$LOSS_W_CONF_WARMUP_STEPS"
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

if [[ "$USE_COMPILE" == "1" ]]; then
  cmd+=(--compile)
fi
if [[ "$USE_VIDEOMAE_V2_INIT" == "1" ]]; then
  cmd+=(--use-videomae-v2-init)
fi
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  cmd+=(--videomae-model "$VIDEOMAE_MODEL")
fi
if [[ "$RESET_CONF_HEAD_ON_PRETRAIN" == "1" ]]; then
  cmd+=(--reset-confidence-head-on-pretrain)
fi
if [[ "$BROADCAST_BUFFERS" == "1" ]]; then
  cmd+=(--broadcast-buffers)
fi
if [[ "$PROFILE_DATA_LOADING" == "1" ]]; then
  export D4RT_PROFILE_SPOOL="${D4RT_PROFILE_SPOOL:-1}"
  export D4RT_PROFILE_SPOOL_INTERVAL="${D4RT_PROFILE_SPOOL_INTERVAL:-$DATA_PROFILE_INTERVAL}"
  export D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-1}"
  export D4RT_PROFILE_BUILDER_THRESHOLD_S="${D4RT_PROFILE_BUILDER_THRESHOLD_S:-$BUILDER_PROFILE_THRESHOLD_S}"
  export D4RT_PROFILE_DATA_WAIT="${D4RT_PROFILE_DATA_WAIT:-1}"
  export D4RT_DATA_WAIT_THRESHOLD_S="${D4RT_DATA_WAIT_THRESHOLD_S:-$DATA_WAIT_THRESHOLD_S}"
  export D4RT_DATA_WAIT_DETAIL="${D4RT_DATA_WAIT_DETAIL:-$DATA_WAIT_DETAIL}"
  export D4RT_DATA_WAIT_COMPARE_FWD="${D4RT_DATA_WAIT_COMPARE_FWD:-$DATA_WAIT_COMPARE_FWD}"
  export D4RT_DATA_WAIT_DETAIL_MAX_SAMPLES="${D4RT_DATA_WAIT_DETAIL_MAX_SAMPLES:-$DATA_WAIT_DETAIL_MAX_SAMPLES}"
  cmd+=(--profile-data-loading --data-profile-interval "$DATA_PROFILE_INTERVAL")
fi
if [[ -n "$VAL_CONFIG" ]]; then
  cmd+=(--val-config "$VAL_CONFIG")
fi
if [[ -n "$PRETRAIN" ]]; then
  cmd+=(--pretrain "$PRETRAIN")
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi
cmd+=("$@")

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
