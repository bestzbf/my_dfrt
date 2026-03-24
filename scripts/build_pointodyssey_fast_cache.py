#!/usr/bin/env python3
"""Build a fast memmap-friendly annotation cache for PointOdyssey sequences."""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


FAST_ANNOTATION_DIRNAME = "anno_fast"
FAST_FRAME_MANIFEST_FILENAME = "frame_manifest.json"
FAST_REQUIRED_KEYS = (
    "trajs_2d",
    "trajs_3d",
    "valids",
    "intrinsics",
    "extrinsics",
)
FAST_OPTIONAL_KEYS = ("visibs",)


def parse_args():
    parser = argparse.ArgumentParser(description="Build PointOdyssey fast annotation cache")
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
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel conversion workers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing anno_fast directories",
    )
    return parser.parse_args()


def resolve_annotation_path(seq_path: Path) -> Path:
    anno_path = seq_path / "anno.npz"
    if anno_path.exists():
        return anno_path

    npzs = sorted(seq_path.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No annotation found in {seq_path}")
    return npzs[0]


def find_frame_files(seq_path: Path, subdir: str, patterns: tuple[str, ...]) -> list[str]:
    directory = seq_path / subdir
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return [os.path.relpath(path, seq_path) for path in files]
    return []


def cache_is_ready(fast_dir: Path) -> bool:
    if not fast_dir.is_dir():
        return False
    for key in FAST_REQUIRED_KEYS:
        if not (fast_dir / f"{key}.npy").exists():
            return False
    return (fast_dir / FAST_FRAME_MANIFEST_FILENAME).exists()


def convert_sequence(seq_path_str: str, overwrite: bool) -> dict:
    seq_path = Path(seq_path_str)
    fast_dir = seq_path / FAST_ANNOTATION_DIRNAME
    if cache_is_ready(fast_dir) and not overwrite:
        return {"sequence": seq_path.name, "status": "skipped"}

    tmp_dir = seq_path / f"{FAST_ANNOTATION_DIRNAME}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    anno_path = resolve_annotation_path(seq_path)
    with np.load(anno_path, allow_pickle=True) as anno:
        for key in FAST_REQUIRED_KEYS:
            np.save(tmp_dir / f"{key}.npy", anno[key], allow_pickle=False)
        for key in FAST_OPTIONAL_KEYS:
            if key in anno:
                np.save(tmp_dir / f"{key}.npy", anno[key], allow_pickle=False)

    manifest = {
        "version": 1,
        "source_annotation": os.path.relpath(anno_path, seq_path),
        "rgb_files": find_frame_files(seq_path, "rgbs", ("*.jpg", "*.png")),
        "depth_files": find_frame_files(seq_path, "depths", ("*.png", "*.npy")),
        "normal_files": find_frame_files(seq_path, "normals", ("*.jpg", "*.png", "*.npy")),
    }
    with open(tmp_dir / FAST_FRAME_MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True)

    if fast_dir.exists():
        shutil.rmtree(fast_dir)
    os.replace(tmp_dir, fast_dir)

    return {
        "sequence": seq_path.name,
        "status": "converted",
        "annotation_mb": round(anno_path.stat().st_size / 1024 / 1024, 2),
    }


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


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sequence_paths = list_sequence_paths(data_root, args.splits, args.sequence)
    if not sequence_paths:
        raise RuntimeError("No sequences matched the requested split/filter")

    print(f"Building anno_fast cache under {data_root}")
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
                print(f"[converted] {result['sequence']} ({result['annotation_mb']:.2f} MB npz)")
            else:
                skipped += 1
                print(f"[skipped] {result['sequence']}")

    print(f"Done. Converted: {converted}, skipped: {skipped}")


if __name__ == "__main__":
    main()
