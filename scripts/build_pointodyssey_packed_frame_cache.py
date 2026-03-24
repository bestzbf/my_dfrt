#!/usr/bin/env python3
"""Build packed compressed-frame caches for PointOdyssey sequences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from data.dataset import (
    FAST_ANNOTATION_DIRNAME,
    FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_META_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME,
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
        help="Rebuild existing packed frame caches",
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


def packed_cache_paths(seq_path: Path) -> dict[str, Path]:
    fast_dir = seq_path / FAST_ANNOTATION_DIRNAME
    return {
        "fast_dir": fast_dir,
        "meta": fast_dir / FAST_ENCODED_FRAME_CACHE_META_FILENAME,
        "rgb_bin": fast_dir / FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME,
        "rgb_offsets": fast_dir / FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME,
        "depth_bin": fast_dir / FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME,
        "depth_offsets": fast_dir / FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME,
        "normal_bin": fast_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME,
        "normal_offsets": fast_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME,
        "normal_valids": fast_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME,
    }


def packed_cache_is_ready(seq_path: Path) -> bool:
    paths = packed_cache_paths(seq_path)
    required = (
        paths["meta"],
        paths["rgb_bin"],
        paths["rgb_offsets"],
        paths["depth_bin"],
        paths["depth_offsets"],
        paths["normal_bin"],
        paths["normal_offsets"],
        paths["normal_valids"],
    )
    return all(path.exists() for path in required)


def relative_paths(seq_path: Path, files: list[Path]) -> list[str]:
    return [os.path.relpath(path, seq_path) for path in files]


def write_packed_stream(bin_path: Path, files: list[Path]) -> np.ndarray:
    offsets = np.zeros((len(files) + 1,), dtype=np.uint64)
    current = 0
    with open(bin_path, "wb") as handle:
        for idx, path in enumerate(files):
            payload = path.read_bytes()
            handle.write(payload)
            current += len(payload)
            offsets[idx + 1] = current
    return offsets


def write_optional_packed_stream(bin_path: Path, files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(files) + 1,), dtype=np.uint64)
    valids = np.zeros((len(files),), dtype=np.uint8)
    current = 0
    with open(bin_path, "wb") as handle:
        for idx, path in enumerate(files):
            try:
                payload = path.read_bytes()
                handle.write(payload)
                current += len(payload)
                valids[idx] = 1
            except OSError:
                pass
            offsets[idx + 1] = current
    return offsets, valids


def convert_sequence(seq_path_str: str, overwrite: bool) -> dict[str, object]:
    seq_path = Path(seq_path_str)
    if packed_cache_is_ready(seq_path) and not overwrite:
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

    paths = packed_cache_paths(seq_path)
    paths["fast_dir"].mkdir(parents=True, exist_ok=True)
    tmp_dir = paths["fast_dir"] / "packed_frame_cache_tmp"
    if tmp_dir.exists():
        for child in tmp_dir.iterdir():
            child.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_rgb_bin = tmp_dir / FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME
    tmp_depth_bin = tmp_dir / FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME
    tmp_normal_bin = tmp_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME
    tmp_rgb_offsets = tmp_dir / FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME
    tmp_depth_offsets = tmp_dir / FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME
    tmp_normal_offsets = tmp_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME
    tmp_normal_valids = tmp_dir / FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME
    tmp_meta = tmp_dir / FAST_ENCODED_FRAME_CACHE_META_FILENAME

    rgb_offsets = write_packed_stream(tmp_rgb_bin, rgb_files)
    depth_offsets = write_packed_stream(tmp_depth_bin, depth_files)
    normal_offsets, normal_valids = write_optional_packed_stream(tmp_normal_bin, normal_files)

    np.save(tmp_rgb_offsets, rgb_offsets, allow_pickle=False)
    np.save(tmp_depth_offsets, depth_offsets, allow_pickle=False)
    np.save(tmp_normal_offsets, normal_offsets, allow_pickle=False)
    np.save(tmp_normal_valids, normal_valids, allow_pickle=False)

    meta = {
        "version": 1,
        "frames": len(rgb_files),
        "rgb_files": relative_paths(seq_path, rgb_files),
        "depth_files": relative_paths(seq_path, depth_files),
        "normal_files": relative_paths(seq_path, normal_files),
    }
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)

    os.replace(tmp_rgb_bin, paths["rgb_bin"])
    os.replace(tmp_depth_bin, paths["depth_bin"])
    os.replace(tmp_normal_bin, paths["normal_bin"])
    os.replace(tmp_rgb_offsets, paths["rgb_offsets"])
    os.replace(tmp_depth_offsets, paths["depth_offsets"])
    os.replace(tmp_normal_offsets, paths["normal_offsets"])
    os.replace(tmp_normal_valids, paths["normal_valids"])
    os.replace(tmp_meta, paths["meta"])
    tmp_dir.rmdir()

    total_mb = sum(
        path.stat().st_size
        for path in (
            paths["rgb_bin"],
            paths["depth_bin"],
            paths["normal_bin"],
            paths["rgb_offsets"],
            paths["depth_offsets"],
            paths["normal_offsets"],
            paths["normal_valids"],
        )
    ) / 1024 / 1024
    return {
        "sequence": seq_path.name,
        "status": "converted",
        "frames": len(rgb_files),
        "storage_mb": round(total_mb, 2),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sequence_paths = list_sequence_paths(data_root, args.splits, args.sequence)
    if not sequence_paths:
        raise RuntimeError("No sequences matched the requested split/filter")

    print(f"Building packed frame cache under {data_root}")
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
                    f"({result['frames']} frames, {result['storage_mb']:.2f}MB)"
                )
            else:
                skipped += 1
                print(f"[skipped] {result['sequence']}")

    print(f"Done. Converted: {converted}, skipped: {skipped}")


if __name__ == "__main__":
    main()
