#!/usr/bin/env python3
"""Build sequence-level frame caches for PointOdyssey to avoid small-file IO."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
    FAST_FRAME_CACHE_DEPTH_FILENAME,
    FAST_FRAME_CACHE_META_FILENAME,
    FAST_FRAME_CACHE_NORMAL_FILENAME,
    FAST_FRAME_CACHE_NORMAL_VALIDS_FILENAME,
    FAST_FRAME_CACHE_RGB_FILENAME,
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
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 4),
        help="Number of parallel conversion workers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing frame caches",
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


def find_frame_files(seq_path: Path, subdir: str, patterns: tuple[str, ...]) -> list[Path]:
    directory = seq_path / subdir
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    return []


def cache_paths(seq_path: Path) -> dict[str, Path]:
    fast_dir = seq_path / FAST_ANNOTATION_DIRNAME
    return {
        "fast_dir": fast_dir,
        "meta": fast_dir / FAST_FRAME_CACHE_META_FILENAME,
        "rgb": fast_dir / FAST_FRAME_CACHE_RGB_FILENAME,
        "depth": fast_dir / FAST_FRAME_CACHE_DEPTH_FILENAME,
        "normal": fast_dir / FAST_FRAME_CACHE_NORMAL_FILENAME,
        "normal_valids": fast_dir / FAST_FRAME_CACHE_NORMAL_VALIDS_FILENAME,
    }


def cache_is_ready(seq_path: Path) -> bool:
    paths = cache_paths(seq_path)
    required = (paths["meta"], paths["rgb"], paths["depth"], paths["normal"], paths["normal_valids"])
    return all(path.exists() for path in required)


def load_rgb_uint8(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth_raw(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    return depth


def load_normal_storage(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32), "float32"
    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if normal is None:
        raise FileNotFoundError(path)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    return normal, "rgb_uint8"


def open_memmap(path: Path, dtype, shape):
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def relative_paths(seq_path: Path, files: list[Path]) -> list[str]:
    return [os.path.relpath(path, seq_path) for path in files]


def convert_sequence(seq_path_str: str, overwrite: bool) -> dict[str, object]:
    seq_path = Path(seq_path_str)
    if cache_is_ready(seq_path) and not overwrite:
        return {"sequence": seq_path.name, "status": "skipped"}

    rgb_files = find_frame_files(seq_path, "rgbs", ("*.jpg", "*.png"))
    depth_files = find_frame_files(seq_path, "depths", ("*.png", "*.npy"))
    normal_files = find_frame_files(seq_path, "normals", ("*.jpg", "*.png", "*.npy"))
    if not rgb_files or not depth_files or not normal_files:
        raise RuntimeError(f"Missing rgb/depth/normal files under {seq_path}")
    if not (len(rgb_files) == len(depth_files) == len(normal_files)):
        raise RuntimeError(
            f"Frame-count mismatch under {seq_path}: rgb={len(rgb_files)}, depth={len(depth_files)}, normal={len(normal_files)}"
        )

    first_rgb = load_rgb_uint8(rgb_files[0])
    first_depth = load_depth_raw(depth_files[0])

    normal_storage_mode = None
    first_normal = None
    first_normal_valid = False
    for normal_path in normal_files:
        try:
            first_normal, normal_storage_mode = load_normal_storage(normal_path)
            first_normal_valid = True
            break
        except (FileNotFoundError, OSError, ValueError):
            continue
    if normal_storage_mode is None:
        normal_storage_mode = "float32"
        first_normal = np.zeros(first_rgb.shape, dtype=np.float32)

    num_frames = len(rgb_files)
    height, width = first_rgb.shape[:2]

    paths = cache_paths(seq_path)
    paths["fast_dir"].mkdir(parents=True, exist_ok=True)
    tmp_dir = paths["fast_dir"] / "frame_cache_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_rgb = tmp_dir / FAST_FRAME_CACHE_RGB_FILENAME
    tmp_depth = tmp_dir / FAST_FRAME_CACHE_DEPTH_FILENAME
    tmp_normal = tmp_dir / FAST_FRAME_CACHE_NORMAL_FILENAME
    tmp_normal_valids = tmp_dir / FAST_FRAME_CACHE_NORMAL_VALIDS_FILENAME
    tmp_meta = tmp_dir / FAST_FRAME_CACHE_META_FILENAME

    rgb_cache = open_memmap(tmp_rgb, dtype=first_rgb.dtype, shape=(num_frames, height, width, 3))
    depth_cache = open_memmap(tmp_depth, dtype=first_depth.dtype, shape=(num_frames, *first_depth.shape))
    normal_cache = open_memmap(tmp_normal, dtype=first_normal.dtype, shape=(num_frames, *first_normal.shape))
    normal_valids = open_memmap(tmp_normal_valids, dtype=np.uint8, shape=(num_frames,))

    for frame_idx, (rgb_path, depth_path, normal_path) in enumerate(zip(rgb_files, depth_files, normal_files)):
        rgb = load_rgb_uint8(rgb_path)
        depth = load_depth_raw(depth_path)
        if rgb.shape != first_rgb.shape:
            raise RuntimeError(f"Inconsistent RGB shape in {rgb_path}: {rgb.shape} vs {first_rgb.shape}")
        if depth.shape != first_depth.shape or depth.dtype != first_depth.dtype:
            raise RuntimeError(
                f"Inconsistent depth tensor in {depth_path}: shape={depth.shape}, dtype={depth.dtype}; "
                f"expected shape={first_depth.shape}, dtype={first_depth.dtype}"
            )
        rgb_cache[frame_idx] = rgb
        depth_cache[frame_idx] = depth

        try:
            normal, storage_mode = load_normal_storage(normal_path)
            if storage_mode != normal_storage_mode or normal.shape != first_normal.shape or normal.dtype != first_normal.dtype:
                raise RuntimeError(
                    f"Inconsistent normal tensor in {normal_path}: mode={storage_mode}, shape={normal.shape}, dtype={normal.dtype}; "
                    f"expected mode={normal_storage_mode}, shape={first_normal.shape}, dtype={first_normal.dtype}"
                )
            normal_cache[frame_idx] = normal
            normal_valids[frame_idx] = 1
        except (FileNotFoundError, OSError, ValueError):
            normal_cache[frame_idx] = np.zeros_like(first_normal)
            normal_valids[frame_idx] = 0

    del rgb_cache
    del depth_cache
    del normal_cache
    del normal_valids

    meta = {
        "version": 1,
        "frames": num_frames,
        "height": int(height),
        "width": int(width),
        "rgb_dtype": str(first_rgb.dtype),
        "depth_dtype": str(first_depth.dtype),
        "normal_dtype": str(first_normal.dtype),
        "normal_storage": normal_storage_mode,
        "rgb_files": relative_paths(seq_path, rgb_files),
        "depth_files": relative_paths(seq_path, depth_files),
        "normal_files": relative_paths(seq_path, normal_files),
    }
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)

    os.replace(tmp_rgb, paths["rgb"])
    os.replace(tmp_depth, paths["depth"])
    os.replace(tmp_normal, paths["normal"])
    os.replace(tmp_normal_valids, paths["normal_valids"])
    os.replace(tmp_meta, paths["meta"])
    shutil.rmtree(tmp_dir)

    total_mb = sum(
        path.stat().st_size for path in (paths["rgb"], paths["depth"], paths["normal"], paths["normal_valids"])
    ) / 1024 / 1024
    return {
        "sequence": seq_path.name,
        "status": "converted",
        "frames": num_frames,
        "shape": f"{height}x{width}",
        "storage_mb": round(total_mb, 2),
        "normal_storage": normal_storage_mode,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sequence_paths = list_sequence_paths(data_root, args.splits, args.sequence)
    if not sequence_paths:
        raise RuntimeError("No sequences matched the requested split/filter")

    print(f"Building frame cache under {data_root}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Sequences: {len(sequence_paths)}")
    print(f"Workers: {args.workers}")

    converted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(convert_sequence, str(seq_path), args.overwrite): seq_path
            for seq_path in sequence_paths
        }
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "converted":
                converted += 1
                print(
                    f"[converted] {result['sequence']} "
                    f"({result['frames']} frames, {result['shape']}, {result['storage_mb']:.2f}MB, normal={result['normal_storage']})"
                )
            else:
                skipped += 1
                print(f"[skipped] {result['sequence']}")

    print(f"Done. Converted: {converted}, skipped: {skipped}")


if __name__ == "__main__":
    main()
