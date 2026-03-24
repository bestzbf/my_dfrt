#!/usr/bin/env python3
"""Prepare an SSD-friendly PointOdyssey dataset root with all runtime caches materialized.

The default output is a compact "cache-only" dataset that keeps only the files needed by the
current training/inference pipeline:
  - anno_fast/*.npy annotation cache
  - anno_fast/*_frames.bin + offsets packed frame cache
  - anno_fast/motion_boundary_stride_*.{json,npy}

This avoids carrying over the original per-frame JPG/PNG/NPY tree unless explicitly requested.


/home/zbf/miniconda3/envs/d4rt/bin/python scripts/prepare_pointodyssey_local_dataset.py \
  --src-root /home/zbf/16t/e/d4rt/PointOdyssey \
  --dst-root /mnt/nvme/PointOdyssey_local_fast \
  --splits train val test \
  --workers 4 \
  --verify-sample

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    MOTION_BOUNDARY_BITORDER,
    MOTION_BOUNDARY_CACHE_VERSION,
    MOTION_BOUNDARY_META_TEMPLATE,
    MOTION_BOUNDARY_PACKED_TEMPLATE,
    compute_motion_boundary_mask_for_frame,
)


FAST_FRAME_MANIFEST_FILENAME = "frame_manifest.json"
FAST_REQUIRED_KEYS = ("trajs_2d", "trajs_3d", "valids", "intrinsics", "extrinsics")
FAST_OPTIONAL_KEYS = ("visibs",)
PACKED_FRAME_CACHE_FILES = (
    FAST_ENCODED_FRAME_CACHE_META_FILENAME,
    FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME,
    FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME,
)
RAW_SEQUENCE_DIRS = ("rgbs", "depths", "normals")
DATASET_SUMMARY_FILENAME = "prepared_local_dataset_meta.json"


@dataclass(frozen=True)
class SequenceJob:
    split: str
    sequence: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=str, required=True, help="Original PointOdyssey root")
    parser.add_argument("--dst-root", type=str, required=True, help="New local dataset root to materialize")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to prepare",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional single sequence name to prepare within each split",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Motion-boundary temporal strides to cache",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 4),
        help="Number of parallel sequence workers",
    )
    parser.add_argument(
        "--include-raw-files",
        action="store_true",
        help="Also copy rgbs/depths/normals into the destination dataset",
    )
    parser.add_argument(
        "--include-anno-npz",
        action="store_true",
        help="Also copy anno.npz into the destination sequence directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild destination outputs even if they already exist",
    )
    parser.add_argument(
        "--verify-sample",
        action="store_true",
        help="After preparation, load one sample from the destination dataset to verify it works",
    )
    return parser.parse_args()


def list_sequence_jobs(src_root: Path, splits: list[str], sequence_name: str | None) -> list[SequenceJob]:
    jobs: list[SequenceJob] = []
    for split in splits:
        split_root = src_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split not found: {split_root}")
        for seq_path in sorted(split_root.iterdir()):
            if not seq_path.is_dir():
                continue
            if sequence_name is not None and seq_path.name != sequence_name:
                continue
            jobs.append(SequenceJob(split=split, sequence=seq_path.name))
    return jobs


def resolve_annotation_path(seq_path: Path) -> Path:
    anno_path = seq_path / "anno.npz"
    if anno_path.exists():
        return anno_path
    npzs = sorted(seq_path.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No annotation found in {seq_path}")
    return npzs[0]


def find_frame_files(seq_path: Path, subdir: str, patterns: tuple[str, ...]) -> list[Path]:
    directory = seq_path / subdir
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    return []


def relative_paths(seq_path: Path, files: list[Path]) -> list[str]:
    return [os.path.relpath(path, seq_path) for path in files]


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        if not overwrite:
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, overwrite: bool) -> int:
    if not src.exists():
        return 0
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    if dst.exists():
        return 0
    shutil.copytree(src, dst)
    return 1


def fast_annotation_ready(fast_dir: Path) -> bool:
    if not fast_dir.is_dir():
        return False
    required = [fast_dir / f"{key}.npy" for key in FAST_REQUIRED_KEYS]
    required.append(fast_dir / FAST_FRAME_MANIFEST_FILENAME)
    return all(path.exists() for path in required)


def packed_frame_cache_ready(fast_dir: Path) -> bool:
    if not fast_dir.is_dir():
        return False
    return all((fast_dir / filename).exists() for filename in PACKED_FRAME_CACHE_FILES)


def motion_cache_ready(fast_dir: Path, stride: int) -> bool:
    packed_path = fast_dir / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
    meta_path = fast_dir / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
    return packed_path.exists() and meta_path.exists()


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def copy_or_build_fast_annotations(src_seq: Path, dst_seq: Path, overwrite: bool) -> dict[str, object]:
    dst_fast = dst_seq / FAST_ANNOTATION_DIRNAME
    dst_fast.mkdir(parents=True, exist_ok=True)
    src_fast = src_seq / FAST_ANNOTATION_DIRNAME

    reused = False
    copied_files = 0
    if fast_annotation_ready(src_fast):
        for key in FAST_REQUIRED_KEYS:
            copied_files += int(copy_file(src_fast / f"{key}.npy", dst_fast / f"{key}.npy", overwrite))
        for key in FAST_OPTIONAL_KEYS:
            copied_files += int(copy_file(src_fast / f"{key}.npy", dst_fast / f"{key}.npy", overwrite))
        copied_files += int(copy_file(src_fast / FAST_FRAME_MANIFEST_FILENAME, dst_fast / FAST_FRAME_MANIFEST_FILENAME, overwrite))
        reused = fast_annotation_ready(dst_fast)

    if not fast_annotation_ready(dst_fast):
        anno_path = resolve_annotation_path(src_seq)
        with np.load(anno_path, allow_pickle=True) as anno:
            for key in FAST_REQUIRED_KEYS:
                np.save(dst_fast / f"{key}.npy", anno[key], allow_pickle=False)
            for key in FAST_OPTIONAL_KEYS:
                if key in anno:
                    np.save(dst_fast / f"{key}.npy", anno[key], allow_pickle=False)

        manifest = {
            "version": 1,
            "source_annotation": os.path.relpath(anno_path, src_seq),
            "rgb_files": relative_paths(src_seq, find_frame_files(src_seq, "rgbs", ("*.jpg", "*.png"))),
            "depth_files": relative_paths(src_seq, find_frame_files(src_seq, "depths", ("*.png", "*.npy"))),
            "normal_files": relative_paths(src_seq, find_frame_files(src_seq, "normals", ("*.jpg", "*.png", "*.npy"))),
        }
        write_json(dst_fast / FAST_FRAME_MANIFEST_FILENAME, manifest)

    return {
        "reused": reused,
        "copied_files": copied_files,
        "ready": fast_annotation_ready(dst_fast),
    }


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


def copy_or_build_packed_frame_cache(src_seq: Path, dst_seq: Path, overwrite: bool) -> dict[str, object]:
    dst_fast = dst_seq / FAST_ANNOTATION_DIRNAME
    dst_fast.mkdir(parents=True, exist_ok=True)
    src_fast = src_seq / FAST_ANNOTATION_DIRNAME

    reused = False
    copied_files = 0
    if packed_frame_cache_ready(src_fast):
        for filename in PACKED_FRAME_CACHE_FILES:
            copied_files += int(copy_file(src_fast / filename, dst_fast / filename, overwrite))
        reused = packed_frame_cache_ready(dst_fast)

    if not packed_frame_cache_ready(dst_fast):
        rgb_files = find_frame_files(src_seq, "rgbs", ("*.jpg", "*.png"))
        depth_files = find_frame_files(src_seq, "depths", ("*.png", "*.npy"))
        normal_files = find_frame_files(src_seq, "normals", ("*.jpg", "*.png", "*.npy"))
        if not rgb_files or not depth_files or not normal_files:
            raise RuntimeError(f"Missing rgb/depth/normal files under {src_seq}")
        if not (len(rgb_files) == len(depth_files) == len(normal_files)):
            raise RuntimeError(
                f"Frame-count mismatch under {src_seq}: "
                f"rgb={len(rgb_files)}, depth={len(depth_files)}, normal={len(normal_files)}"
            )

        tmp_dir = dst_fast / "packed_frame_cache_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
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
            "rgb_files": relative_paths(src_seq, rgb_files),
            "depth_files": relative_paths(src_seq, depth_files),
            "normal_files": relative_paths(src_seq, normal_files),
        }
        write_json(tmp_meta, meta)

        for filename in PACKED_FRAME_CACHE_FILES:
            os.replace(tmp_dir / filename, dst_fast / filename)
        tmp_dir.rmdir()

    return {
        "reused": reused,
        "copied_files": copied_files,
        "ready": packed_frame_cache_ready(dst_fast),
    }


def load_trajs_and_valids(src_seq: Path) -> tuple[np.ndarray, np.ndarray]:
    src_fast = src_seq / FAST_ANNOTATION_DIRNAME
    traj_path = src_fast / "trajs_2d.npy"
    valids_path = src_fast / "valids.npy"
    if traj_path.exists() and valids_path.exists():
        return (
            np.load(traj_path, mmap_mode="r", allow_pickle=False),
            np.load(valids_path, mmap_mode="r", allow_pickle=False),
        )

    anno_path = resolve_annotation_path(src_seq)
    with np.load(anno_path, allow_pickle=True) as anno:
        return anno["trajs_2d"], anno["valids"]


def find_first_frame(src_seq: Path) -> Path:
    rgb_dir = src_seq / "rgbs"
    for pattern in ("*.jpg", "*.png"):
        for path in sorted(rgb_dir.glob(pattern)):
            if re.search(r"(\d+)(?=\.[^.]+$)", path.name):
                return path
    raise FileNotFoundError(f"No RGB frame found under {rgb_dir}")


def resolve_frame_size(src_seq: Path) -> tuple[int, int]:
    first_frame = find_first_frame(src_seq)
    image = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(first_frame)
    height, width = image.shape[:2]
    return int(height), int(width)


def build_motion_cache(dst_fast: Path, trajs_2d: np.ndarray, valids: np.ndarray, height: int, width: int, stride: int):
    packed_path = dst_fast / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
    meta_path = dst_fast / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
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
    write_json(tmp_meta_path, meta)
    os.replace(tmp_packed_path, packed_path)
    os.replace(tmp_meta_path, meta_path)


def copy_or_build_motion_caches(src_seq: Path, dst_seq: Path, strides: list[int], overwrite: bool) -> dict[str, object]:
    dst_fast = dst_seq / FAST_ANNOTATION_DIRNAME
    dst_fast.mkdir(parents=True, exist_ok=True)
    src_fast = src_seq / FAST_ANNOTATION_DIRNAME

    reused = 0
    built = 0
    for stride in strides:
        src_packed = src_fast / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
        src_meta = src_fast / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
        dst_packed = dst_fast / MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride)
        dst_meta = dst_fast / MOTION_BOUNDARY_META_TEMPLATE.format(stride=stride)
        if src_packed.exists() and src_meta.exists():
            copy_file(src_packed, dst_packed, overwrite)
            copy_file(src_meta, dst_meta, overwrite)
            if motion_cache_ready(dst_fast, stride):
                reused += 1
                continue

        if overwrite or not motion_cache_ready(dst_fast, stride):
            trajs_2d, valids = load_trajs_and_valids(src_seq)
            height, width = resolve_frame_size(src_seq)
            build_motion_cache(dst_fast, trajs_2d, valids, height, width, stride)
            built += 1

    return {"reused": reused, "built": built}


def copy_optional_raw_payloads(src_seq: Path, dst_seq: Path, include_raw_files: bool, include_anno_npz: bool, overwrite: bool) -> dict[str, int]:
    copied_dirs = 0
    copied_files = 0
    if include_raw_files:
        for dirname in RAW_SEQUENCE_DIRS:
            copied_dirs += copy_tree(src_seq / dirname, dst_seq / dirname, overwrite)
    if include_anno_npz:
        copied_files += int(copy_file(resolve_annotation_path(src_seq), dst_seq / "anno.npz", overwrite))
    return {"copied_dirs": copied_dirs, "copied_files": copied_files}


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def prepare_sequence(
    src_root_str: str,
    dst_root_str: str,
    split: str,
    sequence: str,
    strides: list[int],
    include_raw_files: bool,
    include_anno_npz: bool,
    overwrite: bool,
) -> dict[str, object]:
    src_root = Path(src_root_str)
    dst_root = Path(dst_root_str)
    src_seq = src_root / split / sequence
    dst_seq = dst_root / split / sequence
    dst_seq.mkdir(parents=True, exist_ok=True)

    raw_stats = copy_optional_raw_payloads(src_seq, dst_seq, include_raw_files, include_anno_npz, overwrite)
    fast_stats = copy_or_build_fast_annotations(src_seq, dst_seq, overwrite)
    packed_stats = copy_or_build_packed_frame_cache(src_seq, dst_seq, overwrite)
    motion_stats = copy_or_build_motion_caches(src_seq, dst_seq, strides, overwrite)

    return {
        "split": split,
        "sequence": sequence,
        "status": "prepared",
        "include_raw_files": include_raw_files,
        "include_anno_npz": include_anno_npz,
        "raw": raw_stats,
        "fast": fast_stats,
        "packed": packed_stats,
        "motion": motion_stats,
        "size_mb": round(directory_size_bytes(dst_seq) / 1024 / 1024, 2),
    }


def verify_prepared_dataset(dst_root: Path, split: str, sequence: str):
    from data.dataset import PointOdysseyDataset

    dataset = PointOdysseyDataset(
        dataset_location=str(dst_root),
        dset=split,
        sequence_name=sequence,
        use_augs=False,
        S=48,
        strides=[1],
        img_size=256,
        num_queries=128,
        cache_boundaries=True,
        precompute_local_patches=True,
        return_query_video=False,
        local_patch_source="highres",
        return_aux_tensors=False,
        verbose=True,
    )
    sample, success = dataset[0]
    if not success:
        raise RuntimeError(f"Verification failed for {split}/{sequence}")
    return {
        "keys": sorted(sample.keys()),
        "video_shape": list(sample["video"].shape),
        "local_patches_shape": list(sample["local_patches"].shape) if "local_patches" in sample else None,
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    strides = sorted({int(stride) for stride in args.strides if int(stride) > 0})
    if not strides:
        raise RuntimeError("At least one positive stride is required")

    jobs = list_sequence_jobs(src_root, args.splits, args.sequence)
    if not jobs:
        raise RuntimeError("No sequences matched the requested split/filter")

    print(f"Preparing local PointOdyssey dataset")
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Sequences: {len(jobs)}")
    print(f"Motion strides: {', '.join(str(s) for s in strides)}")
    print(f"Workers: {args.workers}")
    print(f"Include raw files: {args.include_raw_files}")
    print(f"Include anno.npz: {args.include_anno_npz}")

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                prepare_sequence,
                str(src_root),
                str(dst_root),
                job.split,
                job.sequence,
                strides,
                args.include_raw_files,
                args.include_anno_npz,
                args.overwrite,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[prepared] {job.split}/{job.sequence} "
                f"(size={result['size_mb']:.2f}MB, "
                f"fast_reused={result['fast']['reused']}, "
                f"packed_reused={result['packed']['reused']}, "
                f"motion_reused={result['motion']['reused']}, "
                f"motion_built={result['motion']['built']})"
            )

    results.sort(key=lambda item: (str(item["split"]), str(item["sequence"])))
    summary = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src_root),
        "destination_root": str(dst_root),
        "splits": args.splits,
        "sequence_filter": args.sequence,
        "strides": strides,
        "include_raw_files": bool(args.include_raw_files),
        "include_anno_npz": bool(args.include_anno_npz),
        "sequence_count": len(results),
        "results": results,
    }
    write_json(dst_root / DATASET_SUMMARY_FILENAME, summary)
    print(f"Wrote dataset summary to {dst_root / DATASET_SUMMARY_FILENAME}")

    if args.verify_sample:
        first = results[0]
        verify_info = verify_prepared_dataset(dst_root, str(first["split"]), str(first["sequence"]))
        print(
            f"[verified] {first['split']}/{first['sequence']} "
            f"(video={verify_info['video_shape']}, local_patches={verify_info['local_patches_shape']})"
        )


if __name__ == "__main__":
    main()
