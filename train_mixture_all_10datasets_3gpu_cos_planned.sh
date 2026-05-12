#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-configs/mixture_all_10datasets_cos_planned.yaml}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"

POINTODYSSEY_LOCAL_ROOT="${POINTODYSSEY_LOCAL_ROOT:-/data2/d4rt/datasets/PointOdyssey}"
if [[ -z "${POINTODYSSEY_ROOT:-}" ]]; then
  if [[ -d "$POINTODYSSEY_LOCAL_ROOT" ]]; then
    POINTODYSSEY_ROOT="$POINTODYSSEY_LOCAL_ROOT"
  else
    POINTODYSSEY_ROOT="/data_cos/hdu_datasets/PointOdyssey"
  fi
fi
POINTODYSSEY_FAST_ROOT="${POINTODYSSEY_FAST_ROOT:-}"
KUBRIC_LOCAL_ROOT="${KUBRIC_LOCAL_ROOT:-/data3/Kubric}"
if [[ -z "${KUBRIC_ROOT:-}" ]]; then
  if [[ -d "$KUBRIC_LOCAL_ROOT" ]]; then
    KUBRIC_ROOT="$KUBRIC_LOCAL_ROOT"
  else
    KUBRIC_ROOT="/data_cos/hdu_datasets/Kubric"
  fi
fi
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data_cos/hdu_datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data_cos/hdu_datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
SCANNET_ROOT="${SCANNET_ROOT:-/data3/dataset/scannet}"
TARTANAIR_ROOT="${TARTANAIR_ROOT:-/data_cos/hdu_datasets/tartanairv1}"
VKITTI2_ROOT="${VKITTI2_ROOT:-/data3/dataset/VirtualKitti}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_10datasets_local}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_10datasets_cos_planned_from422}"
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-pointodyssey,kubric,dynamic_replica,co3dv2,scannetpp}"

mkdir -p "$TMPDIR" "$INDEX_CACHE_DIR"
if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[mixture_all_10datasets_cos_planned] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[mixture_all_10datasets_cos_planned] BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi

TEMP_CONFIG="$(mktemp "${TMPDIR}/mixture_all_10datasets_cos_planned.wrapper.XXXXXX.yaml")"
cleanup() {
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$BASE_CONFIG" "$TEMP_CONFIG" \
  "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" \
  "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" \
  "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" \
  "$SCANNET_ROOT" "$TARTANAIR_ROOT" "$VKITTI2_ROOT" "$MVSSYNTH_ROOT" \
  "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" <<'PY'
from pathlib import Path
import sys

import yaml

(
    src,
    dst,
    pointodyssey_root,
    pointodyssey_fast_root,
    kubric_root,
    dynamic_replica_root,
    co3dv2_root,
    blendedmvs_root,
    scannetpp_root,
    scannetpp_splits_dir,
    scannetpp_scenes_record,
    scannet_root,
    tartanair_root,
    vkitti2_root,
    mvssynth_root,
    index_cache_dir,
    co3dv2_denylist,
) = sys.argv[1:18]

cfg = yaml.safe_load(Path(src).read_text())
datasets = {item["name"]: item for item in cfg["datasets"]}

root_map = {
    "pointodyssey": pointodyssey_root,
    "kubric": kubric_root,
    "dynamic_replica": dynamic_replica_root,
    "co3dv2": co3dv2_root,
    "blendedmvs": blendedmvs_root,
    "scannetpp": scannetpp_root,
    "scannet": scannet_root,
    "tartanair": tartanair_root,
    "vkitti2": vkitti2_root,
    "mvssynth": mvssynth_root,
}

for name, root in root_map.items():
    if name not in datasets:
        continue
    root_path = Path(root)
    if not root_path.is_dir():
        raise SystemExit(
            f"[mixture_all_10datasets_cos_planned] {name}.root not found: {root}"
        )
    datasets[name]["root"] = root

if "pointodyssey" in datasets:
    kwargs = datasets["pointodyssey"].setdefault("adapter_kwargs", {})
    if pointodyssey_fast_root:
        kwargs["fast_root"] = pointodyssey_fast_root
    else:
        kwargs.pop("fast_root", None)

if "co3dv2" in datasets:
    datasets["co3dv2"].setdefault("adapter_kwargs", {})["sequence_denylist"] = co3dv2_denylist

if "scannetpp" in datasets:
    kwargs = datasets["scannetpp"].setdefault("adapter_kwargs", {})
    kwargs["splits_dir"] = scannetpp_splits_dir
    kwargs["scenes_record"] = scannetpp_scenes_record
    kwargs["strict"] = False

for name in ("scannet", "tartanair", "vkitti2", "mvssynth"):
    if name in datasets:
        kwargs = datasets[name].setdefault("adapter_kwargs", {})
        kwargs["precompute_root"] = datasets[name]["root"]

cfg["index_cache_dir"] = index_cache_dir
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

export CONFIG="$TEMP_CONFIG"
export INDEX_CACHE_DIR
export OUTPUT_DIR
export SAMPLE_STAGE_DATASETS
export POINTODYSSEY_ROOT
export POINTODYSSEY_LOCAL_ROOT
export POINTODYSSEY_FAST_ROOT
export KUBRIC_ROOT
export DYNAMIC_REPLICA_ROOT
export CO3DV2_ROOT
export BLENDEDMVS_ROOT
export MVSSYNTH_ROOT
export SCANNETPP_ROOT
export SCANNETPP_SPLITS_DIR
export SCANNETPP_SCENES_RECORD
export CO3DV2_DENYLIST

echo "[mixture_all_10datasets_cos_planned] BASE_CONFIG=$BASE_CONFIG"
echo "[mixture_all_10datasets_cos_planned] WRAPPER_CONFIG=$TEMP_CONFIG"
echo "[mixture_all_10datasets_cos_planned] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_all_10datasets_cos_planned] SAMPLE_STAGE_DATASETS=$SAMPLE_STAGE_DATASETS"
echo "[mixture_all_10datasets_cos_planned] extra dataset roots: SCANNET_ROOT=$SCANNET_ROOT TARTANAIR_ROOT=$TARTANAIR_ROOT VKITTI2_ROOT=$VKITTI2_ROOT MVSSYNTH_ROOT=$MVSSYNTH_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[mixture_all_10datasets_cos_planned] DRY_RUN=1, not launching training"
  cat "$TEMP_CONFIG"
  exit 0
fi

bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh"
