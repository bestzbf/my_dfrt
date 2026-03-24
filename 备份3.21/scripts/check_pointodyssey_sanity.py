#!/usr/bin/env python3
"""Lightweight geometry/data sanity checks for PointOdyssey."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import PointOdysseyDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check PointOdyssey geometry and sample loading.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/val/test/sample")
    parser.add_argument("--split", type=str, required=True, help="Split name to inspect")
    parser.add_argument("--sequence", type=str, default=None, help="Optional sequence to restrict checks to")
    parser.add_argument("--img-size", type=int, default=256, help="Dataset image size for loader smoke test")
    parser.add_argument("--num-frames", type=int, default=8, help="Clip length for loader smoke test")
    parser.add_argument("--num-queries", type=int, default=256, help="Number of queries for loader smoke test")
    parser.add_argument("--patch-size", type=int, default=9, help="Patch size for loader smoke test")
    parser.add_argument(
        "--max-frames-check",
        type=int,
        default=6,
        help="Maximum number of frames to sample for reprojection checks",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=1024,
        help="Maximum number of points to sample per checked frame",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_split_root(data_root: str, split: str) -> Path:
    split_root = Path(data_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    return split_root


def list_sequences(split_root: Path, sequence_name: str | None) -> list[Path]:
    if sequence_name is not None:
        seq_path = split_root / sequence_name
        if not seq_path.is_dir():
            raise FileNotFoundError(f"Sequence not found: {seq_path}")
        return [seq_path]

    sequences = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not sequences:
        raise FileNotFoundError(f"No sequences found in split: {split_root}")
    return sequences


def load_annotation(seq_path: Path) -> dict[str, np.ndarray]:
    anno_path = seq_path / "anno.npz"
    if not anno_path.exists():
        candidates = sorted(seq_path.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No annotation file found in {seq_path}")
        anno_path = candidates[0]

    anno = np.load(anno_path, allow_pickle=True)
    required_keys = ("trajs_2d", "trajs_3d", "intrinsics", "extrinsics", "valids")
    missing = [key for key in required_keys if key not in anno]
    if missing:
        raise KeyError(f"Missing annotation keys in {anno_path}: {missing}")

    visibilities = anno["visibs"] if "visibs" in anno else anno["valids"]
    result = {
        "trajs_2d": anno["trajs_2d"],
        "trajs_3d": anno["trajs_3d"],
        "intrinsics": anno["intrinsics"],
        "extrinsics": anno["extrinsics"],
        "valids": anno["valids"],
        "visibs": visibilities,
    }
    return result


def validate_annotation_shapes(seq_name: str, anno: dict[str, np.ndarray]):
    trajs_2d = anno["trajs_2d"]
    trajs_3d = anno["trajs_3d"]
    intrinsics = anno["intrinsics"]
    extrinsics = anno["extrinsics"]
    valids = anno["valids"]
    visibs = anno["visibs"]

    if trajs_2d.ndim != 3 or trajs_2d.shape[-1] != 2:
        raise ValueError(f"{seq_name}: trajs_2d must have shape (T, N, 2), got {trajs_2d.shape}")
    if trajs_3d.ndim != 3 or trajs_3d.shape[-1] != 3:
        raise ValueError(f"{seq_name}: trajs_3d must have shape (T, N, 3), got {trajs_3d.shape}")
    if intrinsics.shape != (trajs_2d.shape[0], 3, 3):
        raise ValueError(f"{seq_name}: intrinsics must have shape (T, 3, 3), got {intrinsics.shape}")
    if extrinsics.shape != (trajs_2d.shape[0], 4, 4):
        raise ValueError(f"{seq_name}: extrinsics must have shape (T, 4, 4), got {extrinsics.shape}")
    if valids.shape != trajs_2d.shape[:2]:
        raise ValueError(f"{seq_name}: valids must match (T, N), got {valids.shape}")
    if visibs.shape != trajs_2d.shape[:2]:
        raise ValueError(f"{seq_name}: visibs must match (T, N), got {visibs.shape}")

    if not np.isfinite(trajs_3d).all():
        raise ValueError(f"{seq_name}: trajs_3d contains non-finite values")
    if not np.isfinite(intrinsics).all():
        raise ValueError(f"{seq_name}: intrinsics contains non-finite values")
    if not np.isfinite(extrinsics).all():
        raise ValueError(f"{seq_name}: extrinsics contains non-finite values")

    focal_x = intrinsics[:, 0, 0]
    focal_y = intrinsics[:, 1, 1]
    if not np.all(focal_x > 0.0) or not np.all(focal_y > 0.0):
        raise ValueError(f"{seq_name}: intrinsics focal lengths must be positive")

    rotation = extrinsics[:, :3, :3]
    rotation_error = np.linalg.norm(rotation @ np.transpose(rotation, (0, 2, 1)) - np.eye(3), axis=(1, 2))
    if float(np.nanmax(rotation_error)) > 1e-1:
        raise ValueError(f"{seq_name}: extrinsics rotation matrices are not close to orthonormal")


def sample_frame_indices(num_frames: int, max_frames_check: int) -> np.ndarray:
    if num_frames <= 0:
        return np.empty((0,), dtype=np.int64)
    count = min(num_frames, max_frames_check)
    return np.unique(np.linspace(0, num_frames - 1, num=count, dtype=np.int64))


def compute_reprojection_metrics(
    seq_name: str,
    anno: dict[str, np.ndarray],
    max_frames_check: int,
    max_points_per_frame: int,
    seed: int,
) -> dict[str, float]:
    trajs_2d = anno["trajs_2d"].astype(np.float64, copy=False)
    trajs_3d = anno["trajs_3d"].astype(np.float64, copy=False)
    intrinsics = anno["intrinsics"].astype(np.float64, copy=False)
    extrinsics = anno["extrinsics"].astype(np.float64, copy=False)
    valids = anno["valids"].astype(bool, copy=False)
    visibs = anno["visibs"].astype(bool, copy=False)

    rng = np.random.default_rng(seed)
    errors = []
    checked_frames = 0

    for frame_idx in sample_frame_indices(trajs_2d.shape[0], max_frames_check):
        points_2d = trajs_2d[frame_idx]
        points_3d = trajs_3d[frame_idx]
        valid_mask = (
            valids[frame_idx]
            & visibs[frame_idx]
            & np.isfinite(points_2d).all(axis=-1)
            & np.isfinite(points_3d).all(axis=-1)
        )
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            continue

        checked_frames += 1
        if valid_indices.size > max_points_per_frame:
            valid_indices = rng.choice(valid_indices, size=max_points_per_frame, replace=False)

        points_world_h = np.concatenate(
            [points_3d[valid_indices], np.ones((valid_indices.size, 1), dtype=np.float64)],
            axis=1,
        )
        camera_points = (extrinsics[frame_idx] @ points_world_h.T).T[:, :3]
        z = camera_points[:, 2]
        positive_depth = np.abs(z) > 1e-8
        if not np.any(positive_depth):
            continue

        camera_points = camera_points[positive_depth]
        z = z[positive_depth]
        targets_2d = points_2d[valid_indices][positive_depth]
        k = intrinsics[frame_idx]

        projected = np.empty_like(targets_2d)
        projected[:, 0] = k[0, 0] * camera_points[:, 0] / z + k[0, 2]
        projected[:, 1] = k[1, 1] * camera_points[:, 1] / z + k[1, 2]
        frame_errors = np.linalg.norm(projected - targets_2d, axis=1)
        if frame_errors.size:
            errors.append(frame_errors)

    if not errors:
        raise ValueError(f"{seq_name}: no valid reprojection samples found")

    errors_flat = np.concatenate(errors, axis=0)
    metrics = {
        "checked_frames": float(checked_frames),
        "num_points": float(errors_flat.size),
        "mean": float(np.mean(errors_flat)),
        "median": float(np.median(errors_flat)),
        "p95": float(np.percentile(errors_flat, 95)),
        "max": float(np.max(errors_flat)),
    }

    if metrics["mean"] > 2.0 or metrics["p95"] > 5.0:
        raise ValueError(
            f"{seq_name}: reprojection error too high "
            f"(mean={metrics['mean']:.3f}px, p95={metrics['p95']:.3f}px, max={metrics['max']:.3f}px)"
        )

    return metrics


def smoke_test_dataset_loader(args, seq_name: str):
    dataset = PointOdysseyDataset(
        dataset_location=args.data_root,
        dset=args.split,
        use_augs=False,
        S=args.num_frames,
        img_size=args.img_size,
        num_queries=args.num_queries,
        patch_size=args.patch_size,
        verbose=False,
        sequence_name=seq_name,
        query_mode="full",
        precompute_local_patches=False,
    )

    if len(dataset) != 1:
        raise ValueError(f"{seq_name}: expected sequence-filtered dataset length 1, got {len(dataset)}")

    sample, ok = dataset[0]
    if not ok:
        raise RuntimeError(f"{seq_name}: dataset loader returned ok=False")
    if "video" not in sample or tuple(sample["video"].shape) != (args.num_frames, 3, args.img_size, args.img_size):
        raise ValueError(
            f"{seq_name}: unexpected video tensor shape {tuple(sample.get('video', torch.empty(0)).shape)}"
        )
    if "coords" not in sample or sample["coords"].shape[0] != args.num_queries:
        raise ValueError(f"{seq_name}: unexpected coords tensor shape {tuple(sample.get('coords', torch.empty(0)).shape)}")


def main():
    args = parse_args()
    split_root = resolve_split_root(args.data_root, args.split)
    sequences = list_sequences(split_root, args.sequence)

    print(
        f"PointOdyssey sanity check: split={args.split}, sequences={len(sequences)}, "
        f"img_size={args.img_size}, num_frames={args.num_frames}, num_queries={args.num_queries}"
    )

    failures = []
    for seq_path in sequences:
        seq_name = seq_path.name
        try:
            anno = load_annotation(seq_path)
            validate_annotation_shapes(seq_name, anno)
            metrics = compute_reprojection_metrics(
                seq_name=seq_name,
                anno=anno,
                max_frames_check=args.max_frames_check,
                max_points_per_frame=args.max_points_per_frame,
                seed=args.seed,
            )
            smoke_test_dataset_loader(args, seq_name)
            print(
                f"[OK] {seq_name}: reproj_mean={metrics['mean']:.3f}px, "
                f"reproj_p95={metrics['p95']:.3f}px, points={int(metrics['num_points'])}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{seq_name}: {exc}")
            print(f"[FAIL] {seq_name}: {exc}", file=sys.stderr)

    if failures:
        print("PointOdyssey sanity check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("PointOdyssey sanity check passed.")


if __name__ == "__main__":
    main()
