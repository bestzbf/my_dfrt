#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-28580}"
CONFIG="${CONFIG:-configs/mixture_5datasets_local.yaml}"
POINTODYSSEY_ROOT="${POINTODYSSEY_ROOT:-/data2/d4rt/datasets/PointOdyssey}"
KUBRIC_ROOT="${KUBRIC_ROOT:-/data/d4rt/kubric}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data1/d4rt/datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data2/d4rt/datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data/d4rt/data/BlendedMVS}"
ALLOW_REMOTE_DATA="${ALLOW_REMOTE_DATA:-0}"
VAL_CONFIG="${VAL_CONFIG:-}"
BATCH_SIZE="${BATCH_SIZE:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
USE_COMPILE="${USE_COMPILE:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_5datasets_blendedmvs_large_3gpu_bs5}"
DEFAULT_PRETRAIN=""
PRETRAIN="${PRETRAIN-$DEFAULT_PRETRAIN}"
DEFAULT_RESUME="/data/zbf/openclaw/d4rt/outputs/mixture_5datasets_blendedmvs_large_3gpu_bs5/checkpoint_latest_47.pth"
RESUME="${RESUME-$DEFAULT_RESUME}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
VAL_SAMPLES="${VAL_SAMPLES:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_CONF="${LOSS_W_CONF:-0.2}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.0}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-1.0}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-1}"
VARIANT="${VARIANT:-large}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"

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

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"

echo "[mixture_5datasets] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_5datasets] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[mixture_5datasets] MASTER_PORT=$MASTER_PORT"
echo "[mixture_5datasets] CONFIG=$CONFIG"
echo "[mixture_5datasets] POINTODYSSEY_ROOT=$POINTODYSSEY_ROOT"
echo "[mixture_5datasets] KUBRIC_ROOT=$KUBRIC_ROOT"
echo "[mixture_5datasets] DYNAMIC_REPLICA_ROOT=$DYNAMIC_REPLICA_ROOT"
echo "[mixture_5datasets] CO3DV2_ROOT=$CO3DV2_ROOT"
echo "[mixture_5datasets] BLENDEDMVS_ROOT=$BLENDEDMVS_ROOT"
echo "[mixture_5datasets] ALLOW_REMOTE_DATA=$ALLOW_REMOTE_DATA"
if [[ -n "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets] VAL_CONFIG=$VAL_CONFIG"
fi
echo "[mixture_5datasets] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_5datasets] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE )))"
echo "[mixture_5datasets] NUM_WORKERS=$NUM_WORKERS"
echo "[mixture_5datasets] LOG_INTERVAL=$LOG_INTERVAL"
echo "[mixture_5datasets] DIST_TIMEOUT_MINUTES=$DIST_TIMEOUT_MINUTES"
echo "[mixture_5datasets] BROADCAST_BUFFERS=$BROADCAST_BUFFERS"
echo "[mixture_5datasets] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[mixture_5datasets] RESET_CONF_HEAD_ON_PRETRAIN=$RESET_CONF_HEAD_ON_PRETRAIN"
echo "[mixture_5datasets] VARIANT=$VARIANT"
echo "[mixture_5datasets] USE_VIDEOMAE_V2_INIT=$USE_VIDEOMAE_V2_INIT"
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  echo "[mixture_5datasets] VIDEOMAE_MODEL=$VIDEOMAE_MODEL"
fi
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_5datasets] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_5datasets] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_5datasets] PRETRAIN=<none>"
fi
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "[mixture_5datasets] RESUME not found: $RESUME" >&2
    exit 1
  fi
  echo "[mixture_5datasets] RESUME=$RESUME"
else
  echo "[mixture_5datasets] RESUME=<none>"
fi
if [[ -n "$VAL_CONFIG" && ! -f "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets] VAL_CONFIG not found: $VAL_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[mixture_5datasets] CONFIG not found: $CONFIG" >&2
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

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/mixture_5datasets_3gpu.XXXXXX.yaml")"
/root/miniconda3/envs/d4rt/bin/python - "$CONFIG" "$TEMP_CONFIG" "$ALLOW_REMOTE_DATA" "$POINTODYSSEY_ROOT" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" <<'PY'
from pathlib import Path
import sys

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
allow_remote = sys.argv[3] == "1"
root_map = {
    "pointodyssey": sys.argv[4],
    "kubric": sys.argv[5],
    "dynamic_replica": sys.argv[6],
    "co3dv2": sys.argv[7],
    "blendedmvs": sys.argv[8],
}

required_markers = {
    "pointodyssey": ("train",),
    "kubric": (".d4rt_ready",),
    "dynamic_replica": ("train",),
    "co3dv2": ("apple",),
    "blendedmvs": ("BlendedMVS_training.txt", ".d4rt_ready"),
}

lines = src_path.read_text().splitlines()
out_lines = []
current_dataset = None
replaced = set()


def validate_root(dataset_name: str, root: str) -> None:
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[mixture_5datasets] {dataset_name}.root not found or not a directory: {root}"
        )
    for marker in required_markers.get(dataset_name, ()):
        if not (path / marker).exists():
            raise SystemExit(
                f"[mixture_5datasets] {dataset_name}.root is incomplete: missing {path / marker}"
            )
    if not allow_remote and str(path).startswith("/data_cos/"):
        raise SystemExit(
            f"[mixture_5datasets] refusing remote s3fs root for {dataset_name}: {root}\n"
            "Measured single-sample load time on /data_cos is too slow for training. "
            "Finish syncing this dataset locally first, or set ALLOW_REMOTE_DATA=1 to override."
        )

for line in lines:
    stripped = line.strip()
    if stripped.startswith("- name: "):
        current_dataset = stripped.split(":", 1)[1].strip()
        out_lines.append(line)
        continue
    if current_dataset is not None and line and not line.startswith(" "):
        current_dataset = None
    if current_dataset is not None and stripped.startswith("root:") and current_dataset in root_map:
        new_root = root_map[current_dataset]
        validate_root(current_dataset, new_root)
        indent = line[: len(line) - len(line.lstrip())]
        out_lines.append(f"{indent}root: {new_root}")
        replaced.add(current_dataset)
        continue
    out_lines.append(line)

missing = sorted(set(root_map) - replaced)
if missing:
    raise SystemExit(
        f"[mixture_5datasets] failed to rewrite dataset roots in config {src_path}: missing {missing}"
    )

dst_path.write_text("\n".join(out_lines) + "\n")
PY
CONFIG_TO_USE="$TEMP_CONFIG"
echo "[mixture_5datasets] EFFECTIVE_CONFIG=$CONFIG_TO_USE"

cmd=(
  /root/miniconda3/envs/d4rt/bin/torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  train_mixture.py
  --config "$CONFIG_TO_USE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  $( [[ "$USE_COMPILE" == "1" ]] && printf '%s' "--compile" || true )
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
