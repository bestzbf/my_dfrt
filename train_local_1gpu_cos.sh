#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ===== 本地环境配置 =====
PYTHON_BIN="${PYTHON_BIN:-/home/zbf/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zbf/miniconda3/envs/d4rt/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-28580}"

# ===== 配置文件 =====
CONFIG="${CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"

# ===== 数据集路径 =====
POINTODYSSEY_LOCAL_ROOT="${POINTODYSSEY_LOCAL_ROOT:-$HOME/16t/e/d4rt/PointOdyssey}"
POINTODYSSEY_COS_ROOT="${POINTODYSSEY_COS_ROOT:-/data_cos/hdu_datasets/PointOdyssey}"
if [[ -z "${POINTODYSSEY_ROOT:-}" ]]; then
  if [[ "${POINTODYSSEY_PREFER_LOCAL:-0}" == "1" && -d "$POINTODYSSEY_LOCAL_ROOT" ]]; then
    POINTODYSSEY_ROOT="$POINTODYSSEY_LOCAL_ROOT"
  else
    POINTODYSSEY_ROOT="$POINTODYSSEY_COS_ROOT"
  fi
fi
POINTODYSSEY_FAST_ROOT="${POINTODYSSEY_FAST_ROOT:-}"
KUBRIC_ROOT="${KUBRIC_ROOT:-/data_cos/hdu_datasets/Kubric}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data_cos/hdu_datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data_cos/hdu_datasets/Co3Dv2}"
BLENDEDMVS_LOCAL_ROOT="${BLENDEDMVS_LOCAL_ROOT:-$HOME/16t/e/重建3D数据/BlendedMVS}"
BLENDEDMVS_COS_ROOT="${BLENDEDMVS_COS_ROOT:-/data_cos/hdu_datasets/BlendedMVS}"
if [[ -z "${BLENDEDMVS_ROOT:-}" ]]; then
  if [[ "${BLENDEDMVS_PREFER_LOCAL:-0}" == "1" && -d "$BLENDEDMVS_LOCAL_ROOT" ]]; then
    BLENDEDMVS_ROOT="$BLENDEDMVS_LOCAL_ROOT"
  else
    BLENDEDMVS_ROOT="$BLENDEDMVS_COS_ROOT"
  fi
fi
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
SCANNETPP_USE_PRECOMPUTED_TRACKS="${SCANNETPP_USE_PRECOMPUTED_TRACKS:-1}"
SCANNETPP_PRECOMPUTED_READ_MODE="${SCANNETPP_PRECOMPUTED_READ_MODE:-auto}"
SCANNETPP_H5_RANGE_WORKERS="${SCANNETPP_H5_RANGE_WORKERS:-4}"
SCANNETPP_H5_RANGE_RETRIES="${SCANNETPP_H5_RANGE_RETRIES:-2}"
SCANNETPP_H5_RANGE_MERGE_GAP_MB="${SCANNETPP_H5_RANGE_MERGE_GAP_MB:-1}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/home/zbf/Desktop/d4rt/my_dfrt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

# ===== 缓存和临时目录 (本地SSD) =====
# 使用从服务器下载的缓存文件（路径已指向 /data_cos/hdu_datasets/）
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-$HOME/16t/e/ZBF_Data/0/.index_cache_5datasets_blendedmvs_hdu}"
CO3DV2_FRAME_ANNO_CACHE_DIR="$INDEX_CACHE_DIR/co3dv2_frame_anno"
AUTO_BUILD_CO3DV2_FRAME_ANNO_CACHE="${AUTO_BUILD_CO3DV2_FRAME_ANNO_CACHE:-0}"
TMPDIR="${TMPDIR:-/tmp/d4rt_local}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/tmp/d4rt_sample_stage}"
SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND:-cos_sdk}"

# ===== 权重文件 =====
# Resume权重 (用户指定)
DEFAULT_RESUME="$HOME/16t/e/ZBF_Data/0/model.pth"
RESUME="${RESUME-$DEFAULT_RESUME}"
# VideoMAE预训练模型
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-$HOME/16t/e/ZBF_Data/0/d4rtold/pretrained/videomae-large}"
# 不使用pretrain，只使用resume
DEFAULT_PRETRAIN=""
PRETRAIN="${PRETRAIN-$DEFAULT_PRETRAIN}"

# ===== 训练参数 (针对单GPU 24GB优化) =====
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EPOCHS="${EPOCHS:-500}"
EPOCH_SIZE="${EPOCH_SIZE:-10000}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local_1gpu_cos}"

# ===== 其他参数 =====
VARIANT="${VARIANT:-large}"
USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-0}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-120}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"
VAL_CONFIG="${VAL_CONFIG:-configs/single_scannetpp_val_local.yaml}"
VAL_INTERVAL="${VAL_INTERVAL:-2}"
VAL_SAMPLES="${VAL_SAMPLES:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-precomputed_highres}"
PRECOMPUTE_PATCHES="${PRECOMPUTE_PATCHES:-1}"
PRECOMPUTE_FROM_HIGHRES="${PRECOMPUTE_FROM_HIGHRES:-1}"
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
ENCODER_LR_MULT="${ENCODER_LR_MULT:-1.0}"
DECODER_LR_MULT="${DECODER_LR_MULT:-1.0}"

# COS相关参数
MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-50}"
BUILDER_WORKERS="${BUILDER_WORKERS:-12}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-512}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-6}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-8}"
SAMPLE_STAGE_COS_TIMEOUT_S="${SAMPLE_STAGE_COS_TIMEOUT_S:-20}"
SAMPLE_STAGE_DOWNLOAD_RETRIES="${SAMPLE_STAGE_DOWNLOAD_RETRIES:-2}"
SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-180}"
SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO="${SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO:-0.9}"
SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S="${SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S:-30}"
SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S="${SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S:-30}"
SAMPLE_STAGE_WINDOW_RADIUS="${SAMPLE_STAGE_WINDOW_RADIUS:-8}"
H5_RANGE_CACHE_ROOT="${H5_RANGE_CACHE_ROOT:-$SAMPLE_STAGE_ROOT/shared_raw_cache/data}"
D4RT_H5_RANGE_CACHE_ROOT="${D4RT_H5_RANGE_CACHE_ROOT:-$H5_RANGE_CACHE_ROOT}"
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-pointodyssey,kubric,dynamic_replica,co3dv2,blendedmvs,scannetpp}"
SAMPLE_STAGE_SCENE_PREFETCH_DATASETS="${SAMPLE_STAGE_SCENE_PREFETCH_DATASETS:-co3dv2,kubric,dynamic_replica,blendedmvs,scannetpp}"
SAMPLE_STAGE_MOUNT_ROOT="${SAMPLE_STAGE_MOUNT_ROOT:-/data_cos}"
SAMPLE_STAGE_BUCKET="${SAMPLE_STAGE_BUCKET:-hd-ai-data-1251882982}"
SAMPLE_STAGE_REGION="${SAMPLE_STAGE_REGION:-ap-beijing}"
if [[ -z "${SAMPLE_STAGE_PASSWD_FILE:-}" ]]; then
  if [[ -f /etc/passwd-cosfs ]]; then
    SAMPLE_STAGE_PASSWD_FILE="/etc/passwd-cosfs"
  else
    SAMPLE_STAGE_PASSWD_FILE="/etc/passwd-s3fs-data_cos"
  fi
fi

# PointOdyssey相关
POINTODYSSEY_LOCAL_CACHE_DIR="${POINTODYSSEY_LOCAL_CACHE_DIR:-$INDEX_CACHE_DIR}"
POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
if [[ -z "${POINTODYSSEY_ANNO_READ_MODE:-}" ]]; then
  POINTODYSSEY_ANNO_READ_MODE="auto"
fi
POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB="${POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB:-24}"
POINTODYSSEY_ANNO_RANGE_WORKERS="${POINTODYSSEY_ANNO_RANGE_WORKERS:-8}"
POINTODYSSEY_ANNO_RANGE_RETRIES="${POINTODYSSEY_ANNO_RANGE_RETRIES:-2}"
POINTODYSSEY_ANNO_RANGE_MERGE_GAP_MB="${POINTODYSSEY_ANNO_RANGE_MERGE_GAP_MB:-1}"
POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS="${POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS:-0}"
POINTODYSSEY_PREWARM_ANNO_H5="${POINTODYSSEY_PREWARM_ANNO_H5:-0}"
POINTODYSSEY_PREWARM_ANNO_H5_WORKERS="${POINTODYSSEY_PREWARM_ANNO_H5_WORKERS:-4}"
POINTODYSSEY_STAGE_ANNO_H5="${POINTODYSSEY_STAGE_ANNO_H5:-0}"
POINTODYSSEY_REQUIRE_TRACKS="${POINTODYSSEY_REQUIRE_TRACKS:-1}"
POINTODYSSEY_ASSUME_TRACKS="${POINTODYSSEY_ASSUME_TRACKS:-0}"
POINTODYSSEY_TRACK_WORKERS="${POINTODYSSEY_TRACK_WORKERS:-4}"

# 索引缓存预热（已使用服务器下载的缓存，禁用预热）
AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-0}"
WARM_CACHE_ONLY="${WARM_CACHE_ONLY:-0}"
WARM_ONLY_DATASETS="${WARM_ONLY_DATASETS:-}"
WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-4}"
WARM_VAL="${WARM_VAL:-0}"

# 调试和性能分析
PROFILE_DATA_LOADING="${PROFILE_DATA_LOADING:-0}"
DATA_PROFILE_INTERVAL="${DATA_PROFILE_INTERVAL:-20}"
BUILDER_PROFILE_THRESHOLD_S="${BUILDER_PROFILE_THRESHOLD_S:-5}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-3.0}"
DATA_WAIT_DETAIL="${DATA_WAIT_DETAIL:-0}"
DATA_WAIT_COMPARE_FWD="${DATA_WAIT_COMPARE_FWD:-0}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-8}"

# 环境变量
D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
D4RT_BUILDER_FAULTHANDLER="${D4RT_BUILDER_FAULTHANDLER:-0}"
D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-0}"
D4RT_BUILD_TIMEOUT_GRACE_S="${D4RT_BUILD_TIMEOUT_GRACE_S:-0}"
D4RT_MAX_REQUEUE="${D4RT_MAX_REQUEUE:-3}"
D4RT_BLENDEDMVS_STAGE_PRECOMPUTED="${D4RT_BLENDEDMVS_STAGE_PRECOMPUTED:-0}"
D4RT_BLENDEDMVS_STAGE_PRECOMPUTED_MAX_GB="${D4RT_BLENDEDMVS_STAGE_PRECOMPUTED_MAX_GB:-2.0}"
D4RT_PLANNED_STARTUP_WARMUP_SAMPLES="${D4RT_PLANNED_STARTUP_WARMUP_SAMPLES:-32}"
D4RT_PLANNED_STARTUP_WARMUP_TIMEOUT_S="${D4RT_PLANNED_STARTUP_WARMUP_TIMEOUT_S:-600}"
D4RT_PLANNED_RELAXED_ORDER="${D4RT_PLANNED_RELAXED_ORDER:-1}"
D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-512}"
D4RT_PLANNED_RELAXED_GRACE_S="${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"
D4RT_SLOW_SAMPLE_THRESHOLD_S="${D4RT_SLOW_SAMPLE_THRESHOLD_S:-0}"
D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-0}"
D4RT_PROFILE_BUILDER_ALL="${D4RT_PROFILE_BUILDER_ALL:-0}"
D4RT_VERBOSE_BUILDER="${D4RT_VERBOSE_BUILDER:-0}"
D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-0}"

# ===== 创建必要目录 =====
mkdir -p "$TMPDIR"
mkdir -p "$INDEX_CACHE_DIR"
mkdir -p "$SAMPLE_STAGE_ROOT"
mkdir -p "$OUTPUT_DIR"

# ===== 环境变量导出 =====
export TMPDIR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export D4RT_BUILDER_TORCH_THREADS="${D4RT_BUILDER_TORCH_THREADS:-1}"
export D4RT_BUILDER_TORCH_INTEROP_THREADS="${D4RT_BUILDER_TORCH_INTEROP_THREADS:-1}"
export D4RT_H5_RANGE_CACHE_ROOT
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_SCANNETPP_TRUST_ALLOWLIST="${D4RT_SCANNETPP_TRUST_ALLOWLIST:-1}"
export D4RT_BUILDER_FAULTHANDLER
export D4RT_VERBOSE_BUILDER
export D4RT_BUILD_TIMEOUT
export D4RT_BUILD_TIMEOUT_GRACE_S
export D4RT_MAX_REQUEUE
export D4RT_BLENDEDMVS_STAGE_PRECOMPUTED
export D4RT_BLENDEDMVS_STAGE_PRECOMPUTED_MAX_GB
export D4RT_PLANNED_STARTUP_WARMUP_SAMPLES
export D4RT_PLANNED_STARTUP_WARMUP_TIMEOUT_S
export D4RT_PLANNED_RELAXED_ORDER
export D4RT_PLANNED_RELAXED_LOOKAHEAD
export D4RT_PLANNED_RELAXED_GRACE_S
export D4RT_SLOW_SAMPLE_THRESHOLD_S
export D4RT_PROFILE_BUILDER
export D4RT_PROFILE_BUILDER_ALL
export D4RT_PLANNED_WAIT_LOG
export POINTODYSSEY_ASSUME_TRACKS
export SCANNETPP_USE_PRECOMPUTED_TRACKS
export SCANNETPP_PRECOMPUTED_READ_MODE
export SCANNETPP_H5_RANGE_WORKERS
export SCANNETPP_H5_RANGE_RETRIES
export SCANNETPP_H5_RANGE_MERGE_GAP_MB
export D4RT_POINTODYSSEY_STAGE_ANNO_H5="$POINTODYSSEY_STAGE_ANNO_H5"
export D4RT_POINTODYSSEY_ANNO_FRAME_CACHE_DIR="$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
export POINTODYSSEY_ANNO_READ_MODE
export POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB
export POINTODYSSEY_ANNO_RANGE_WORKERS
export POINTODYSSEY_ANNO_RANGE_RETRIES
export POINTODYSSEY_ANNO_RANGE_MERGE_GAP_MB
export POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS
export POINTODYSSEY_PREWARM_ANNO_H5
export POINTODYSSEY_PREWARM_ANNO_H5_WORKERS

if [[ "$D4RT_SUPPRESS_MKL_WARNING" == "1" ]]; then
  mkl_warning_filter="ignore:mkl-service package failed to import:UserWarning"
  if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${mkl_warning_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="$mkl_warning_filter"
  fi
fi

# ===== 打印配置 =====
echo "=========================================="
echo "  D4RT 本地训练配置 (1x RTX 3090)"
echo "=========================================="
echo "[local_1gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[local_1gpu] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[local_1gpu] CONFIG=$CONFIG"
echo "[local_1gpu] VARIANT=$VARIANT"
echo "[local_1gpu] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE )))"
echo "[local_1gpu] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[local_1gpu] PRECOMPUTE_PATCHES=$PRECOMPUTE_PATCHES  PRECOMPUTE_FROM_HIGHRES=$PRECOMPUTE_FROM_HIGHRES"
echo "[local_1gpu] EPOCH_SIZE=$EPOCH_SIZE"
echo "[local_1gpu] OUTPUT_DIR=$OUTPUT_DIR"
echo "[local_1gpu] RESUME=$RESUME"
echo "[local_1gpu] VIDEOMAE_MODEL=$VIDEOMAE_MODEL"
echo "------------------------------------------"
echo "[local_1gpu] 数据集路径:"
echo "  PointOdyssey: $POINTODYSSEY_ROOT"
echo "  Kubric: $KUBRIC_ROOT"
echo "  Dynamic_Replica: $DYNAMIC_REPLICA_ROOT"
echo "  Co3Dv2: $CO3DV2_ROOT"
echo "  BlendedMVS: $BLENDEDMVS_ROOT"
echo "  ScanNet++: $SCANNETPP_ROOT"
echo "  ScanNet++ precomputed tracks: $SCANNETPP_USE_PRECOMPUTED_TRACKS"
echo "  ScanNet++ precomputed read mode: $SCANNETPP_PRECOMPUTED_READ_MODE"
echo "------------------------------------------"
echo "[local_1gpu] 缓存配置:"
echo "  INDEX_CACHE_DIR=$INDEX_CACHE_DIR"
echo "  SAMPLE_STAGE_ROOT=$SAMPLE_STAGE_ROOT"
echo "  SAMPLE_STAGE_COS_TIMEOUT_S=$SAMPLE_STAGE_COS_TIMEOUT_S"
echo "  SAMPLE_STAGE_DOWNLOAD_RETRIES=$SAMPLE_STAGE_DOWNLOAD_RETRIES"
echo "  SAMPLE_STAGE_CACHE_MAX_GB=$SAMPLE_STAGE_CACHE_MAX_GB"
echo "  SAMPLE_STAGE_WINDOW_RADIUS=$SAMPLE_STAGE_WINDOW_RADIUS"
echo "  MAX_SPOOL_BYTES_GB=$MAX_SPOOL_BYTES_GB"
echo "  D4RT_BUILD_TIMEOUT=$D4RT_BUILD_TIMEOUT"
echo "  D4RT_MAX_REQUEUE=$D4RT_MAX_REQUEUE"
echo "  D4RT_PLANNED_STARTUP_WARMUP_SAMPLES=$D4RT_PLANNED_STARTUP_WARMUP_SAMPLES"
echo "  D4RT_PLANNED_RELAXED_ORDER=$D4RT_PLANNED_RELAXED_ORDER"
echo "  D4RT_PLANNED_RELAXED_LOOKAHEAD=$D4RT_PLANNED_RELAXED_LOOKAHEAD"
echo "  D4RT_H5_RANGE_CACHE_ROOT=$D4RT_H5_RANGE_CACHE_ROOT"
echo "  POINTODYSSEY_ANNO_READ_MODE=$POINTODYSSEY_ANNO_READ_MODE"
echo "  POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB=$POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB"
echo "  POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS=$POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS"
echo "  POINTODYSSEY_PREWARM_ANNO_H5=$POINTODYSSEY_PREWARM_ANNO_H5 workers=$POINTODYSSEY_PREWARM_ANNO_H5_WORKERS"
echo "  CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS builder_torch=$D4RT_BUILDER_TORCH_THREADS"
echo "=========================================="

# ===== 验证必要文件 =====
if [[ ! -f "$RESUME" ]]; then
  echo "[local_1gpu] ERROR: RESUME not found: $RESUME" >&2
  exit 1
fi
echo "[local_1gpu] RESUME 权重文件: $(ls -lh "$RESUME" | awk '{print $5}')"

if [[ ! -d "$VIDEOMAE_MODEL" ]]; then
  echo "[local_1gpu] ERROR: VIDEOMAE_MODEL not found: $VIDEOMAE_MODEL" >&2
  exit 1
fi
echo "[local_1gpu] VideoMAE 模型: $VIDEOMAE_MODEL"

if [[ ! -f "$CONFIG" ]]; then
  echo "[local_1gpu] ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[local_1gpu] ERROR: PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$TORCHRUN_BIN" ]]; then
  echo "[local_1gpu] ERROR: TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2
  exit 1
fi

if [[ "$POINTODYSSEY_PREWARM_ANNO_H5" == "1" && "$POINTODYSSEY_ROOT" == "$SAMPLE_STAGE_MOUNT_ROOT"* ]]; then
  echo "[local_1gpu] 预热 PointOdyssey anno.h5 到本地缓存..."
  "$PYTHON_BIN" - \
    "$POINTODYSSEY_ROOT" \
    "$D4RT_H5_RANGE_CACHE_ROOT" \
    "$POINTODYSSEY_PREWARM_ANNO_H5_WORKERS" \
    "$SAMPLE_STAGE_MOUNT_ROOT" \
    "$SAMPLE_STAGE_BUCKET" \
    "$SAMPLE_STAGE_REGION" \
    "$SAMPLE_STAGE_PASSWD_FILE" \
    "$SAMPLE_STAGE_COS_TIMEOUT_S" \
    "$SAMPLE_STAGE_DOWNLOAD_RETRIES" <<'PY'
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qcloud_cos import CosConfig, CosS3Client

point_root = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
workers = max(1, int(sys.argv[3]))
mount_root = Path(sys.argv[4])
bucket = sys.argv[5]
region = sys.argv[6]
passwd_file = Path(sys.argv[7])
timeout_s = max(1, int(sys.argv[8]))
retries = max(0, int(sys.argv[9]))

cache_dir = cache_root / ".d4rt_pointodyssey_anno_h5"
cache_dir.mkdir(parents=True, exist_ok=True)

parts = passwd_file.read_text().strip().split(":")
if len(parts) == 2:
    secret_id, secret_key = parts
elif len(parts) == 3:
    _bucket, secret_id, secret_key = parts
else:
    raise ValueError(f"Unsupported COS passwd file format: {passwd_file}")

tls = threading.local()

def get_client() -> CosS3Client:
    client = getattr(tls, "client", None)
    if client is None:
        client = CosS3Client(CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Scheme="https",
            Timeout=timeout_s,
        ))
        tls.client = client
    return client

def cos_key(path: Path) -> str:
    try:
        return path.relative_to(mount_root).as_posix()
    except ValueError:
        mount = str(mount_root).rstrip("/") + "/"
        raw = str(path)
        if raw.startswith(mount):
            return raw[len(mount):]
        raise

def cache_path_for(path: Path) -> Path:
    key = cos_key(path)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name)[:80]
    return cache_dir / f"{safe_name}_{digest}.h5"

seq_root = point_root / "train"
tasks: list[tuple[str, Path, int]] = []
total_missing = 0
for seq_dir in seq_root.iterdir():
    if not seq_dir.is_dir():
        continue
    h5 = seq_dir / "anno.h5"
    if not h5.exists():
        continue
    dst = cache_path_for(h5)
    if dst.is_file():
        continue
    size = h5.stat().st_size
    tasks.append((cos_key(h5), dst, size))
    total_missing += size

print(
    f"[PointOdysseyPrewarm] cache_dir={cache_dir} "
    f"missing={len(tasks)} total_missing={total_missing/1024**3:.2f}GB workers={workers}",
    flush=True,
)
if not tasks:
    raise SystemExit(0)

def fetch(task: tuple[str, Path, int]) -> tuple[str, str]:
    key, dst, size = task
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock = dst.with_suffix(dst.suffix + ".lock")
    with open(lock, "a+b") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if dst.is_file():
                return key, "hit"
            tmp = dst.with_name(f".{dst.name}.part.{os.getpid()}.{threading.get_ident()}")
            last_exc: BaseException | None = None
            for attempt in range(retries + 1):
                try:
                    response = get_client().get_object(Bucket=bucket, Key=key)
                    stream = response["Body"].get_raw_stream()
                    with open(tmp, "wb") as f:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    os.replace(tmp, dst)
                    return key, f"downloaded={size/1024**2:.1f}MB"
                except BaseException as exc:
                    last_exc = exc
                    tmp.unlink(missing_ok=True)
                    if attempt < retries:
                        time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                        continue
                    raise
                finally:
                    tmp.unlink(missing_ok=True)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"download failed: {key}")
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(fetch, task) for task in tasks]
    for done, future in enumerate(as_completed(futures), 1):
        key, status = future.result()
        if done % 5 == 0 or done == len(futures):
            elapsed = time.perf_counter() - t0
            print(
                f"[PointOdysseyPrewarm] {done}/{len(futures)} {key} {status} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
print(f"[PointOdysseyPrewarm] done elapsed={time.perf_counter()-t0:.0f}s", flush=True)
PY
fi

# 检查CO3DV2_DENYLIST
if [[ ! -f "$CO3DV2_DENYLIST" ]]; then
  echo "[local_1gpu] WARNING: CO3DV2_DENYLIST not found: $CO3DV2_DENYLIST, will skip" >&2
  CO3DV2_DENYLIST=""
fi

# CO3Dv2 frame_annotations.jgz is very slow to decode from COS inside builder
# processes.  The adapter automatically uses this npz cache when present.
co3d_frame_cache_count=0
if [[ -d "$CO3DV2_FRAME_ANNO_CACHE_DIR" ]]; then
  co3d_frame_cache_count="$(find "$CO3DV2_FRAME_ANNO_CACHE_DIR" -maxdepth 1 -name 'frame_anno_*_v1.npz' | wc -l)"
fi
if [[ "$co3d_frame_cache_count" -lt 51 ]]; then
  if [[ "$AUTO_BUILD_CO3DV2_FRAME_ANNO_CACHE" == "1" ]]; then
    echo "[local_1gpu] 构建 CO3Dv2 frame annotation 快缓存 ($co3d_frame_cache_count/51)..."
    "$PYTHON_BIN" scripts/build_co3dv2_frame_anno_cache.py \
      --root "$CO3DV2_ROOT" \
      --out-dir "$CO3DV2_FRAME_ANNO_CACHE_DIR" \
      --workers "${CO3DV2_FRAME_ANNO_CACHE_WORKERS:-2}"
  else
    echo "[local_1gpu] WARNING: CO3Dv2 frame annotation 快缓存不完整: $CO3DV2_FRAME_ANNO_CACHE_DIR ($co3d_frame_cache_count/51)" >&2
    echo "[local_1gpu] WARNING: 可先运行: AUTO_BUILD_CO3DV2_FRAME_ANNO_CACHE=1 ./train_local_1gpu_cos.sh" >&2
  fi
else
  echo "[local_1gpu] CO3Dv2 frame annotation 快缓存: $co3d_frame_cache_count/51"
fi

# ===== 预热索引缓存 =====
if [[ "$AUTO_WARM_INDEX_CACHE" == "1" || "$WARM_CACHE_ONLY" == "1" ]]; then
  if [[ -f "$ROOT_DIR/cos/warm_index_cache_5datasets_cos.sh" ]]; then
    echo "[local_1gpu] 预热索引缓存..."
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
    bash "$ROOT_DIR/cos/warm_index_cache_5datasets_cos.sh" || {
      echo "[local_1gpu] WARNING: 索引缓存预热失败，继续训练..."
    }
  else
    echo "[local_1gpu] WARNING: warm script not found, skipping cache warmup"
  fi
fi

if [[ "$WARM_CACHE_ONLY" == "1" ]]; then
  echo "[local_1gpu] WARM_CACHE_ONLY=1, 仅预热缓存后退出"
  exit 0
fi

# ===== 生成临时配置 =====
TEMP_CONFIG=""
CONFIG_TO_USE="$CONFIG"
cleanup() {
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    rm -f "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/local_1gpu_cos.XXXXXX.yaml")"
"$PYTHON_BIN" - "$CONFIG" "$TEMP_CONFIG" "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" "$BUILDER_WORKERS" "$PREFETCH_DEPTH" "$MAX_SPOOL_BYTES_GB" "$SAMPLE_STAGE_BACKEND" "$SAMPLE_STAGE_ROOT" "$SAMPLE_STAGE_SDK_WORKERS" "$SAMPLE_STAGE_COS_TIMEOUT_S" "$SAMPLE_STAGE_DOWNLOAD_RETRIES" "$SAMPLE_STAGE_DATASETS" "$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS" "$SAMPLE_STAGE_MOUNT_ROOT" "$SAMPLE_STAGE_BUCKET" "$SAMPLE_STAGE_REGION" "$SAMPLE_STAGE_PASSWD_FILE" "$SAMPLE_STAGE_CACHE_MAX_GB" "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO" "$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S" "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S" "$SAMPLE_STAGE_WINDOW_RADIUS" "$PRECOMPUTE_PATCHES" "$PRECOMPUTE_FROM_HIGHRES" "$EPOCH_SIZE" <<'PY'
from pathlib import Path
import os
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
sample_stage_cos_timeout_s = int(sys.argv[21])
sample_stage_download_retries = int(sys.argv[22])
sample_stage_datasets = sys.argv[23].strip()
sample_stage_scene_prefetch_datasets = sys.argv[24].strip()
sample_stage_mount_root = sys.argv[25].strip()
sample_stage_bucket = sys.argv[26].strip()
sample_stage_region = sys.argv[27].strip()
sample_stage_passwd_file = sys.argv[28].strip()
sample_stage_cache_max_bytes = int(float(sys.argv[29]) * 1024**3)
sample_stage_cache_low_watermark_ratio = float(sys.argv[30])
sample_stage_cache_touch_interval_s = float(sys.argv[31])
sample_stage_cache_scan_interval_s = float(sys.argv[32])
sample_stage_window_radius = int(sys.argv[33])
precompute_patches = sys.argv[34].strip().lower() in {"1", "true", "yes", "on"}
precompute_from_highres = sys.argv[35].strip().lower() in {"1", "true", "yes", "on"}
epoch_size = int(sys.argv[36])

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
            f"[local_1gpu] {dataset_name}.root not found or not a directory: {root}"
        )
    for marker in required_markers.get(dataset_name, ()):
        if not (path / marker).exists():
            raise SystemExit(
                f"[local_1gpu] {dataset_name}.root is incomplete: missing {path / marker}"
            )

def is_under_path(path: str, parent: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(parent).resolve())
    except (OSError, RuntimeError, ValueError):
        return str(path).startswith(str(parent).rstrip("/") + "/")

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
    if not root:
        continue
    validate_root(dataset_name, root)
    datasets[dataset_name]["root"] = root

# scannetpp配置
if "scannetpp" in datasets:
    scannetpp_cfg = datasets["scannetpp"].setdefault("adapter_kwargs", {})
    scannetpp_cfg["splits_dir"] = scannetpp_splits_dir
    scannetpp_cfg["scenes_record"] = scannetpp_scenes_record
    scannetpp_cfg["strict"] = False
    scannetpp_cfg["use_precomputed_tracks"] = os.environ.get(
        "SCANNETPP_USE_PRECOMPUTED_TRACKS",
        "1",
    ).strip().lower() in {"1", "true", "yes", "on"}
    scannetpp_cfg["precomputed_read_mode"] = os.environ.get(
        "SCANNETPP_PRECOMPUTED_READ_MODE",
        "auto",
    ).strip()
    scannetpp_cfg["precomputed_cos_mount_root"] = sample_stage_mount_root
    scannetpp_cfg["precomputed_cos_bucket"] = sample_stage_bucket
    scannetpp_cfg["precomputed_cos_region"] = sample_stage_region
    scannetpp_cfg["precomputed_cos_passwd_file"] = sample_stage_passwd_file
    scannetpp_cfg["precomputed_cos_timeout_s"] = sample_stage_cos_timeout_s
    scannetpp_cfg["precomputed_cos_range_workers"] = int(os.environ.get(
        "SCANNETPP_H5_RANGE_WORKERS",
        "16",
    ))
    scannetpp_cfg["precomputed_cos_range_retries"] = int(os.environ.get(
        "SCANNETPP_H5_RANGE_RETRIES",
        "2",
    ))
    scannetpp_cfg["precomputed_cos_range_merge_gap_bytes"] = int(
        float(os.environ.get("SCANNETPP_H5_RANGE_MERGE_GAP_MB", "1")) * 1024**2
    )

# blendedmvs配置
if "blendedmvs" in datasets:
    blendedmvs_cfg = datasets["blendedmvs"].setdefault("adapter_kwargs", {})
    blendedmvs_io_workers = os.environ.get("BLENDEDMVS_IO_WORKERS", "").strip()
    if blendedmvs_io_workers:
        blendedmvs_cfg["io_workers"] = int(blendedmvs_io_workers)
    blendedmvs_cfg["load_normals"] = os.environ.get(
        "BLENDEDMVS_LOAD_NORMALS",
        "1" if float(os.environ.get("LOSS_W_NORMAL", "0.0")) > 0 else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    blendedmvs_cfg["precomputed_read_mode"] = os.environ.get(
        "BLENDEDMVS_PRECOMPUTED_READ_MODE",
        "auto",
    ).strip()
    blendedmvs_cfg["precomputed_cos_mount_root"] = sample_stage_mount_root
    blendedmvs_cfg["precomputed_cos_bucket"] = sample_stage_bucket
    blendedmvs_cfg["precomputed_cos_region"] = sample_stage_region
    blendedmvs_cfg["precomputed_cos_passwd_file"] = sample_stage_passwd_file
    blendedmvs_cfg["precomputed_cos_timeout_s"] = sample_stage_cos_timeout_s
    blendedmvs_cfg["precomputed_cos_range_workers"] = int(os.environ.get(
        "BLENDEDMVS_H5_RANGE_WORKERS",
        "16",
    ))
    blendedmvs_cfg["precomputed_cos_range_retries"] = int(os.environ.get(
        "BLENDEDMVS_H5_RANGE_RETRIES",
        "2",
    ))
    blendedmvs_cfg["precomputed_cos_range_merge_gap_bytes"] = int(
        float(os.environ.get("BLENDEDMVS_H5_RANGE_MERGE_GAP_MB", "1")) * 1024**2
    )

# pointodyssey配置
if "pointodyssey" in datasets:
    pointodyssey_cfg = datasets["pointodyssey"].setdefault("adapter_kwargs", {})
    if pointodyssey_fast_root:
        pointodyssey_cfg["fast_root"] = pointodyssey_fast_root
    else:
        pointodyssey_cfg.pop("fast_root", None)
    pointodyssey_cfg["anno_read_mode"] = os.environ.get(
        "POINTODYSSEY_ANNO_READ_MODE",
        "auto",
    ).strip()
    pointodyssey_cfg["anno_cos_mount_root"] = sample_stage_mount_root
    pointodyssey_cfg["anno_cos_bucket"] = sample_stage_bucket
    pointodyssey_cfg["anno_cos_region"] = sample_stage_region
    pointodyssey_cfg["anno_cos_passwd_file"] = sample_stage_passwd_file
    pointodyssey_cfg["anno_cos_timeout_s"] = sample_stage_cos_timeout_s
    pointodyssey_cfg["anno_cos_range_workers"] = int(os.environ.get(
        "POINTODYSSEY_ANNO_RANGE_WORKERS",
        "4",
    ))
    pointodyssey_cfg["anno_cos_range_retries"] = int(os.environ.get(
        "POINTODYSSEY_ANNO_RANGE_RETRIES",
        "2",
    ))
    pointodyssey_cfg["anno_cos_range_merge_gap_bytes"] = int(
        float(os.environ.get("POINTODYSSEY_ANNO_RANGE_MERGE_GAP_MB", "1")) * 1024**2
    )
    pointodyssey_cfg["anno_index_window_radius"] = int(os.environ.get(
        "POINTODYSSEY_ANNO_INDEX_WINDOW_RADIUS",
        "0",
    ))
    pointodyssey_cfg["runtime_sanitize"] = not is_under_path(
        pointodyssey_root,
        sample_stage_mount_root,
    )

# co3dv2配置
if "co3dv2" in datasets and co3dv2_denylist:
    co3dv2_cfg = datasets["co3dv2"].setdefault("adapter_kwargs", {})
    co3dv2_cfg["sequence_denylist"] = co3dv2_denylist

config["index_cache_dir"] = index_cache_dir
config["planned_mode"] = True
config["epoch_size"] = epoch_size
config["builder_workers"] = builder_workers
config["prefetch_depth"] = prefetch_depth
config["max_spool_bytes"] = max_spool_bytes
config["precompute_patches"] = precompute_patches
config["precompute_from_highres"] = precompute_from_highres

if sample_stage_backend:
    config["sample_stage_backend"] = sample_stage_backend
    config["sample_stage_root"] = sample_stage_root
    config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
    config["sample_stage_cos_timeout_s"] = sample_stage_cos_timeout_s
    config["sample_stage_download_retries"] = sample_stage_download_retries
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
echo "[local_1gpu] 临时配置: $CONFIG_TO_USE"

# ===== 构建训练命令 =====
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

echo "[local_1gpu] 启动训练..."
echo "[local_1gpu] 命令: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ${cmd[*]}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
