#!/usr/bin/env python3
"""Build stride-aware offline motion-boundary caches for PointOdyssey sequences."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (
    FAST_ANNOTATION_DIRNAME,
    MOTION_BOUNDARY_BITORDER,
    MOTION_BOUNDARY_CACHE_VERSION,
    MOTION_BOUNDARY_META_TEMPLATE,
    MOTION_BOUNDARY_PACKED_TEMPLATE,
    compute_motion_boundary_mask_for_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True, help="PointOdyssey root containing train/val/test")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional sequence name filter within each split",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Temporal strides to precompute for",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 4),
        help="Number of parallel conversion workers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing motion-boundary caches",
    )
    return parser.parse_args()


def list_sequence_paths(data_root: Path, splits: list[str], sequence_name: str | None) -> list[Path]:
    sequence_paths = []
    for split in splits:
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split not found: {split_root}")
        for seq_path in sorted(split_root.iterdir()):
            if not seq_path.is_dir():
                continue
            if sequence_name is not None and seq_path.name != sequence_name:
                continue
            sequence_paths.append(seq_path)
    return sequence_paths


def resolve_annotation_path(seq_path: Path) -> Path:
    anno_path = seq_path / "anno.npz"
    if anno_path.exists():
        return anno_path
    npzs = sorted(seq_path.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No annotation found in {seq_path}")
    return npzs[0]


def resolve_fast_annotation_path(seq_path: Path, key: str) -> Path | None:
    path = seq_path / FAST_ANNOTATION_DIRNAME / f"{key}.npy"
    return path if path.exists() else None


def load_trajectory_arrays(seq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    traj_path = resolve_fast_annotation_path(seq_path, "trajs_2d")
    valids_path = resolve_fast_annotation_path(seq_path, "valids")
    if traj_path is not None and valids_path is not None:
        trajs_2d = np.load(traj_path, mmap_mode="r", allow_pickle=False)
        valids = np.load(valids_path, mmap_mode="r", allow_pickle=False)
        return trajs_2d, valids

    anno_path = resolve_annotation_path(seq_path)
    with np.load(anno_path, allow_pickle=True) as anno:
        return anno["trajs_2d"], anno["valids"]


def find_first_frame(seq_path: Path) -> Path:
    rgb_dir = seq_path / "rgbs"
    candidates = []
    for pattern in ("*.jpg", "*.png"):
        candidates.extend(sorted(rgb_dir.glob(pattern)))
    for path in candidates:
        if re.search(r"(\d+)(?=\.[^.]+$)", path.name):
            return path
    raise FileNotFoundError(f"No RGB frame found under {rgb_dir}")


def resolve_frame_size(seq_path: Path) -> tuple[int, int]:
    first_frame = find_first_frame(seq_path)
    image = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(first_frame)
    height, width = image.shape[:2]
    return int(height), int(width)


def cache_is_ready(seq_path: Path, stride: int) -> bool:
    fast_dir = seq_path / FAST_ANNOTATION_DIRNAME
    packed_path = fast_dir / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
    meta_path = fast_dir / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
    return packed_path.exists() and meta_path.exists()


def build_stride_cache(seq_path: Path, trajs_2d: np.ndarray, valids: np.ndarray, height: int, width: int, stride: int) -> dict[str, float | int]:
    fast_dir = seq_path / FAST_ANNOTATION_DIRNAME
    fast_dir.mkdir(parents=True, exist_ok=True)

    packed_path = fast_dir / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
    meta_path = fast_dir / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
    tmp_packed_path = packed_path.with_suffix(".tmp.npy")
    tmp_meta_path = meta_path.with_suffix(".tmp.json")

    packed_width = (width + 7) // 8
    packed = np.lib.format.open_memmap(
        tmp_packed_path,
        mode="w+",
        dtype=np.uint8,
        shape=(trajs_2d.shape[0], height, packed_width),
    )
    for frame_idx in range(trajs_2d.shape[0]):
        mask = compute_motion_boundary_mask_for_frame(
            trajs_2d=trajs_2d,
            valids=valids,
            frame_idx=frame_idx,
            height=height,
            width=width,
            temporal_step=int(stride),
        )
        packed[frame_idx] = np.packbits(mask, axis=-1, bitorder=MOTION_BOUNDARY_BITORDER)
    del packed

    meta = {
        "version": MOTION_BOUNDARY_CACHE_VERSION,
        "stride": int(stride),
        "frames": int(trajs_2d.shape[0]),
        "height": int(height),
        "width": int(width),
        "packed_width": int(packed_width),
        "bitorder": MOTION_BOUNDARY_BITORDER,
    }
    with open(tmp_meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)

    os.replace(tmp_packed_path, packed_path)
    os.replace(tmp_meta_path, meta_path)
    return {
        "stride": int(stride),
        "packed_mb": round(packed_path.stat().st_size / 1024 / 1024, 2),
    }


def convert_sequence(seq_path_str: str, strides: list[int], overwrite: bool) -> dict[str, object]:
    seq_path = Path(seq_path_str)
    pending_strides = [int(stride) for stride in strides if overwrite or not cache_is_ready(seq_path, int(stride))]
    if not pending_strides:
        return {"sequence": seq_path.name, "status": "skipped"}

    trajs_2d, valids = load_trajectory_arrays(seq_path)
    height, width = resolve_frame_size(seq_path)

    built = []
    for stride in pending_strides:
        built.append(build_stride_cache(seq_path, trajs_2d, valids, height, width, stride))

    return {
        "sequence": seq_path.name,
        "status": "converted",
        "height": height,
        "width": width,
        "built": built,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sequence_paths = list_sequence_paths(data_root, args.splits, args.sequence)
    if not sequence_paths:
        raise RuntimeError("No sequences matched the requested split/filter")

    strides = sorted({int(stride) for stride in args.strides if int(stride) > 0})
    if not strides:
        raise RuntimeError("At least one positive stride is required")

    print(f"Building motion-boundary cache under {data_root}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Sequences: {len(sequence_paths)}")
    print(f"Strides: {', '.join(str(stride) for stride in strides)}")
    print(f"Workers: {args.workers}")

    converted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(convert_sequence, str(seq_path), strides, args.overwrite): seq_path
            for seq_path in sequence_paths
        }
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "converted":
                converted += 1
                built_str = ", ".join(
                    f"s{item['stride']}={item['packed_mb']:.2f}MB"
                    for item in result["built"]
                )
                print(
                    f"[converted] {result['sequence']} "
                    f"({result['height']}x{result['width']}; {built_str})"
                )
            else:
                skipped += 1
                print(f"[skipped] {result['sequence']}")

    print(f"Done. Converted: {converted}, skipped: {skipped}")


if __name__ == "__main__":
    main()
