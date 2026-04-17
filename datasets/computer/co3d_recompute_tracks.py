"""
Regenerate Co3D precomputed tracks from ``pointcloud.ply``.

The previous Co3D recompute path sampled points from one reference frame's
depth map and then projected those points to all other frames.  That works
for datasets with dense, view-consistent depth, but it fails for Co3D:

- the depth maps are sparse;
- a single frame only observes a thin visible surface patch of the object;
- relaxing the depth check turns most in-bounds projections into false
  positives, so the saved ``valids`` become nearly useless;
- the resulting point cloud is typically a small front surface that looks
  planar in rerun.

Co3D already ships a sequence-level ``pointcloud.ply`` in the same world
coordinates as the cameras.  We therefore use the point cloud itself as the
3D track source and only use per-frame depth for normals and optional sparse
consistency filtering.

Usage:
    conda run -n d4rt python datasets/computer/co3d_recompute_tracks.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        --categories apple toaster \\
        --num-points 8000 \\
        --workers 4 \\
        --overwrite

    # All categories (may take a long time):
    conda run -n d4rt python datasets/computer/co3d_recompute_tracks.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        --num-points 8000 \\
        --workers 8 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.co3dv2 import Co3Dv2Adapter
from computer.co3d_pointcloud_to_tracks import (
    project_pointcloud_to_frames,
    read_ply_pointcloud,
)
from computer.depth_to_normals import compute_normals_sequence


def compute_tracks_co3d(
    pointcloud: np.ndarray,
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: tuple[int, int],
    num_points: int = 8000,
    depth_consistency_thresh: float = 0.20,
    rng_seed: int = 42,
) -> dict:
    """Project sampled PLY points to all frames.

    The sampled 3D points come directly from the sequence point cloud, so the
    saved tracks cover the full object shape instead of a single view's
    visible surface patch.
    """
    rng = np.random.default_rng(rng_seed)
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError(f"pointcloud must have shape [N,3], got {pointcloud.shape}")

    if len(pointcloud) == 0:
        T = len(depths)
        return {
            "trajs_2d": np.zeros((T, 0, 2), dtype=np.float32),
            "trajs_3d_world": np.zeros((T, 0, 3), dtype=np.float32),
            "valids": np.zeros((T, 0), dtype=bool),
            "visibs": np.zeros((T, 0), dtype=bool),
            "ref_frame": 0,
            "num_points": 0,
        }

    if len(pointcloud) > num_points:
        keep = rng.choice(len(pointcloud), num_points, replace=False)
        pointcloud = pointcloud[keep]

    tracks = project_pointcloud_to_frames(
        pointcloud=pointcloud,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
        depths=depths,
        depth_consistency_thresh=depth_consistency_thresh,
    )
    tracks["track_source"] = "pointcloud.ply"
    return tracks


def process_sequence(args: tuple) -> tuple[str, str | None]:
    """Process one Co3D sequence and write precomputed.npz."""
    seq_name, adapter_kwargs, output_root, num_points, overwrite = args

    out_path = Path(output_root) / seq_name / "precomputed.npz"
    if out_path.exists() and not overwrite:
        return seq_name, None

    try:
        adapter = Co3Dv2Adapter(**adapter_kwargs)
        info = adapter.get_sequence_info(seq_name)
        num_frames = info["num_frames"]

        all_indices = list(range(num_frames))
        clip = adapter.load_clip(seq_name, all_indices)

        normals = compute_normals_sequence(clip.depths, clip.intrinsics)

        ply_path = Path(info["sequence_root"]) / "pointcloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing pointcloud.ply for {seq_name}: {ply_path}")
        pointcloud = read_ply_pointcloud(ply_path)

        tracks = compute_tracks_co3d(
            pointcloud=pointcloud,
            depths=clip.depths,
            intrinsics=clip.intrinsics,
            extrinsics=clip.extrinsics,
            image_size=clip.image_size,
            num_points=num_points,
            depth_consistency_thresh=0.20,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            normals=normals.astype(np.float16),
            trajs_2d=tracks["trajs_2d"].astype(np.float32),
            trajs_3d_world=tracks["trajs_3d_world"].astype(np.float32),
            valids=tracks["valids"],
            visibs=tracks["visibs"],
            ref_frame=np.array(tracks["ref_frame"], dtype=np.int32),
            num_frames=np.array(num_frames, dtype=np.int32),
            num_points=np.array(tracks["num_points"], dtype=np.int32),
            track_source=np.array(tracks["track_source"]),
        )
        n_valid_frames = int((tracks["valids"].sum(axis=1) > 0).sum())
        return seq_name, None

    except Exception:
        return seq_name, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Co3D precomputed tracks with depth-scale fix."
    )
    parser.add_argument("--root", required=True, help="Co3Dv2 dataset root")
    parser.add_argument("--output-root", default=None, help="Output root (defaults to --root)")
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to process (default: all)")
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(root=args.root, categories=args.categories, verbose=True)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[co3d_recompute] {total} sequences → {output_root}")

    adapter_kwargs = {
        "root": str(args.root),
        "categories": adapter.categories,
        "subset_name": adapter.subset_name,
        "split": adapter.split,
        "verbose": False,
    }

    job_args = [
        (seq, adapter_kwargs, str(output_root), args.num_points, args.overwrite)
        for seq in sequences
    ]

    done = 0
    failed = []

    if args.workers <= 1:
        for job_arg in tqdm(job_args, desc="co3d_recompute"):
            seq_name, err = process_sequence(job_arg)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [FAIL] {seq_name}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_sequence, a): a[0] for a in job_args}
            with tqdm(total=total, desc="co3d_recompute") as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [FAIL] {seq_name}: {err.splitlines()[-1]}")

    print(f"\n[co3d_recompute] done {done - len(failed)}/{total}, failed {len(failed)}")
    if failed:
        print("[co3d_recompute] First 10 failures:")
        for sn, err in failed[:10]:
            print(f"  {sn}: {err.splitlines()[-1]}")


if __name__ == "__main__":
    main()
