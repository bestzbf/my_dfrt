#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"
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

DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data5/d4rt_dataset/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data4/d4rt_dataset/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data5/d4rt_dataset/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data5/d4rt_dataset/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data5/d4rt_dataset/scannetpp/scenes_record.json}"
SCANNET_ROOT="${SCANNET_ROOT:-/data4/d4rt_dataset/scannet}"
TARTANAIR_ROOT="${TARTANAIR_ROOT:-/data5/d4rt_dataset/tartanairv1}"
VKITTI2_ROOT="${VKITTI2_ROOT:-/data5/d4rt_dataset/VirtualKitti}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_9datasets_local}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_9datasets_cos_planned_from457}"
BALANCE_DATASET_WEIGHTS="${BALANCE_DATASET_WEIGHTS:-1}"
WARM_EXTRA_INDEX_CACHE="${WARM_EXTRA_INDEX_CACHE:-1}"

# The existing 5-dataset launcher rewrites this into the effective config.
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-pointodyssey,kubric,dynamic_replica,co3dv2,scannetpp,scannet,tartanair,vkitti2}"
SAMPLE_STAGE_EXTRA_MOUNT_ROOTS="${SAMPLE_STAGE_EXTRA_MOUNT_ROOTS:-/data4,/data5}"

mkdir -p "$TMPDIR" "$INDEX_CACHE_DIR"

if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[mixture_9datasets_cos_planned] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[mixture_9datasets_cos_planned] BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi

TEMP_CONFIG="$(mktemp "${TMPDIR}/mixture_9datasets_cos_planned.wrapper.XXXXXX.yaml")"
cleanup() {
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$BASE_CONFIG" "$TEMP_CONFIG" \
  "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" \
  "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" \
  "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" \
  "$SCANNET_ROOT" "$TARTANAIR_ROOT" "$VKITTI2_ROOT" \
  "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" "$SAMPLE_STAGE_DATASETS" \
  "$SAMPLE_STAGE_EXTRA_MOUNT_ROOTS" "$BALANCE_DATASET_WEIGHTS" <<'PY'
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
    mvssynth_root,
    scannetpp_root,
    scannetpp_splits_dir,
    scannetpp_scenes_record,
    scannet_root,
    tartanair_root,
    vkitti2_root,
    index_cache_dir,
    co3dv2_denylist,
    sample_stage_datasets,
    sample_stage_extra_mount_roots,
    balance_dataset_weights,
) = sys.argv[1:21]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_dir(label: str, root: str) -> None:
    if not Path(root).is_dir():
        raise SystemExit(
            f"[mixture_9datasets_cos_planned] {label} not found or not a directory: {root}"
        )


def require_any(label: str, root: str, markers: tuple[str, ...]) -> None:
    path = Path(root)
    if not any((path / marker).exists() for marker in markers):
        marker_text = " or ".join(str(path / marker) for marker in markers)
        raise SystemExit(
            f"[mixture_9datasets_cos_planned] {label} is incomplete: missing {marker_text}"
        )


for label, root in (
    ("pointodyssey.root", pointodyssey_root),
    ("kubric.root", kubric_root),
    ("dynamic_replica.root", dynamic_replica_root),
    ("co3dv2.root", co3dv2_root),
    ("blendedmvs.root", blendedmvs_root),
    ("scannetpp.root", scannetpp_root),
    ("scannet.root", scannet_root),
    ("tartanair.root", tartanair_root),
    ("vkitti2.root", vkitti2_root),
):
    require_dir(label, root)

if pointodyssey_fast_root:
    require_dir("pointodyssey.fast_root", pointodyssey_fast_root)
require_any("scannet.root", scannet_root, ("scans", "scene0000_00"))
require_any("tartanair.root", tartanair_root, ("abandonedfactory", "abandonedfactory_night"))
require_any("vkitti2.root", vkitti2_root, ("Scene01", "Scene02"))

cfg = yaml.safe_load(Path(src).read_text())
datasets = {item["name"]: item for item in cfg["datasets"]}


def upsert_dataset(name: str, root: str, weight: float, adapter_kwargs: dict | None = None) -> None:
    if name not in datasets:
        entry = {"name": name, "root": root, "weight": weight}
        cfg["datasets"].append(entry)
        datasets[name] = entry
    datasets[name]["root"] = root
    if adapter_kwargs:
        kwargs = datasets[name].setdefault("adapter_kwargs", {})
        kwargs.update(adapter_kwargs)


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

if "pointodyssey" in datasets:
    kwargs = datasets["pointodyssey"].setdefault("adapter_kwargs", {})
    kwargs["require_tracks"] = True
    if pointodyssey_fast_root:
        kwargs["fast_root"] = pointodyssey_fast_root
    else:
        kwargs.pop("fast_root", None)

if "dynamic_replica" in datasets:
    kwargs = datasets["dynamic_replica"].setdefault("adapter_kwargs", {})
    kwargs["skip_depth_when_tracks"] = True
    kwargs["prefer_trajectory_npz"] = True
    kwargs["io_workers"] = 4

if "co3dv2" in datasets:
    kwargs = datasets["co3dv2"].setdefault("adapter_kwargs", {})
    kwargs["sequence_denylist"] = co3dv2_denylist
    kwargs["frame_cache_items"] = 384
    kwargs["io_workers"] = 4

if "scannetpp" in datasets:
    kwargs = datasets["scannetpp"].setdefault("adapter_kwargs", {})
    kwargs["splits_dir"] = scannetpp_splits_dir
    kwargs["scenes_record"] = scannetpp_scenes_record
    kwargs["strict"] = False

upsert_dataset(
    "scannet",
    scannet_root,
    1.0,
    {
        "precompute_root": scannet_root,
    },
)
upsert_dataset(
    "tartanair",
    tartanair_root,
    1.0,
    {
        "camera": "left",
        "precompute_root": tartanair_root,
        # Current TartanAir precomputed caches may be partial. Keep static
        # RGB/depth supervision unless full-sequence caches are rebuilt.
        "load_precomputed": False,
    },
)
upsert_dataset(
    "vkitti2",
    vkitti2_root,
    1.0,
    {
        "camera": "Camera_0",
        "precompute_root": vkitti2_root,
        "load_normals": False,
        "load_flow": False,
    },
)

target_names = [
    name
    for name in (
        "pointodyssey",
        "kubric",
        "dynamic_replica",
        "co3dv2",
        "blendedmvs",
        "scannetpp",
        "scannet",
        "tartanair",
        "vkitti2",
    )
    if name in datasets
]
if parse_bool(balance_dataset_weights):
    weight = 1.0 / len(target_names)
    for name in target_names:
        datasets[name]["weight"] = round(weight, 6)

cfg["index_cache_dir"] = index_cache_dir
cfg["sample_stage_datasets"] = [
    item.strip() for item in sample_stage_datasets.split(",") if item.strip()
]
cfg["sample_stage_extra_mount_roots"] = [
    item.strip() for item in sample_stage_extra_mount_roots.split(",") if item.strip()
]

Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

export CONFIG="$TEMP_CONFIG"
export INDEX_CACHE_DIR
export OUTPUT_DIR
export SAMPLE_STAGE_DATASETS
export SAMPLE_STAGE_EXTRA_MOUNT_ROOTS
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
export SCANNET_ROOT
export TARTANAIR_ROOT
export VKITTI2_ROOT
export CO3DV2_DENYLIST

echo "[mixture_9datasets_cos_planned] BASE_CONFIG=$BASE_CONFIG"
echo "[mixture_9datasets_cos_planned] WRAPPER_CONFIG=$TEMP_CONFIG"
echo "[mixture_9datasets_cos_planned] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_9datasets_cos_planned] INDEX_CACHE_DIR=$INDEX_CACHE_DIR"
echo "[mixture_9datasets_cos_planned] SAMPLE_STAGE_DATASETS=$SAMPLE_STAGE_DATASETS"
echo "[mixture_9datasets_cos_planned] roots: SCANNET_ROOT=$SCANNET_ROOT TARTANAIR_ROOT=$TARTANAIR_ROOT VKITTI2_ROOT=$VKITTI2_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[mixture_9datasets_cos_planned] DRY_RUN=1, not launching training"
  cat "$TEMP_CONFIG"
  exit 0
fi

if [[ "$WARM_EXTRA_INDEX_CACHE" == "1" || "${WARM_CACHE_ONLY:-0}" == "1" ]]; then
  echo "[mixture_9datasets_cos_planned] warming added dataset index caches"
  "$PYTHON_BIN" - "$TEMP_CONFIG" "$INDEX_CACHE_DIR" "${WARM_ONLY_DATASETS:-}" <<'PY'
from pathlib import Path
import sys
import time

import yaml

from datasets.registry import create_adapter

config_path = Path(sys.argv[1])
index_cache_dir = Path(sys.argv[2])
only = {item.strip() for item in sys.argv[3].split(",") if item.strip()}
extra_names = {"scannet", "tartanair", "vkitti2"}

cfg = yaml.safe_load(config_path.read_text())
for ds_cfg in cfg["datasets"]:
    name = ds_cfg["name"]
    if name not in extra_names:
        continue
    if only and name not in only:
        continue
    kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
    kwargs.update(
        {
            "root": ds_cfg["root"],
            "split": ds_cfg.get("split", "train"),
            "cache_dir": str(index_cache_dir),
            "verbose": True,
        }
    )
    print(f"[mixture_9datasets_cos_planned] warming {name} root={ds_cfg['root']}", flush=True)
    t0 = time.time()
    adapter = create_adapter(name, **kwargs)
    print(
        f"[mixture_9datasets_cos_planned] ready {name}: {len(adapter)} sequences in {time.time() - t0:.1f}s",
        flush=True,
    )
PY
fi

bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
