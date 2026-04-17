"""
Generate precomputed tracks for Co3D from pointcloud.ply instead of sparse depth maps.

Usage:
    python co3d_pointcloud_to_tracks.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        --output-root /data2/d4rt/datasets/Co3Dv2 \\
        --num-points 8000 \\
        --workers 4 \\
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
from computer.depth_to_normals import compute_normals_sequence


def read_ply_pointcloud(ply_path: Path) -> np.ndarray:
    """Read XYZ coordinates from a PLY file (simple ASCII or binary parser)."""
    with open(ply_path, 'rb') as f:
        # Read header
        line = f.readline().decode('latin-1').strip()
        if line != 'ply':
            raise ValueError(f"Not a PLY file: {ply_path}")

        format_type = None
        num_vertices = 0
        properties = []

        while True:
            line = f.readline().decode('latin-1').strip()
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break

        # Find x, y, z indices
        prop_names = [p[0] for p in properties]
        if 'x' not in prop_names or 'y' not in prop_names or 'z' not in prop_names:
            raise ValueError(f"PLY file missing x/y/z properties: {ply_path}")

        x_idx = prop_names.index('x')
        y_idx = prop_names.index('y')
        z_idx = prop_names.index('z')

        # Read binary data
        if format_type == 'binary_little_endian':
            import struct
            # Assume float for x,y,z and uchar for colors
            vertex_size = 0
            for prop_name, prop_type in properties:
                if prop_type == 'float':
                    vertex_size += 4
                elif prop_type == 'uchar':
                    vertex_size += 1

            data = f.read()
            points = []
            for i in range(num_vertices):
                offset = i * vertex_size
                # Read x, y, z (first 3 floats)
                x = struct.unpack('<f', data[offset:offset+4])[0]
                y = struct.unpack('<f', data[offset+4:offset+8])[0]
                z = struct.unpack('<f', data[offset+8:offset+12])[0]
                points.append([x, y, z])

            return np.array(points, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported PLY format: {format_type}")


def project_pointcloud_to_frames(
    pointcloud: np.ndarray,  # [N, 3] world coordinates
    intrinsics: np.ndarray,  # [T, 3, 3]
    extrinsics: np.ndarray,  # [T, 4, 4] w2c
    image_size: tuple[int, int],  # (H, W)
    depths: list[np.ndarray],  # [T][H, W] for depth consistency check
    depth_consistency_thresh: float = 0.2,  # Relaxed for Co3D
) -> dict:
    """Project point cloud to all frames and check visibility."""
    T = len(intrinsics)
    H, W = image_size
    N = len(pointcloud)

    trajs_2d = np.zeros((T, N, 2), dtype=np.float32)
    valids = np.zeros((T, N), dtype=bool)

    # Find reference frame (most points visible)
    ref_valid_counts = []
    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]

        # World to camera
        P_hom = np.concatenate([pointcloud, np.ones((N, 1), dtype=np.float32)], axis=1)
        P_cam = (E_t @ P_hom.T).T[:, :3]
        z = P_cam[:, 2]

        # Project to 2D
        fx, fy, cx, cy = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
        u = P_cam[:, 0] / z * fx + cx
        v = P_cam[:, 1] / z * fy + cy

        # In-bounds check
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        ref_valid_counts.append(in_bounds.sum())

    ref_frame = int(np.argmax(ref_valid_counts))

    # Project to all frames
    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]
        depth_t = depths[t]

        # World to camera
        P_hom = np.concatenate([pointcloud, np.ones((N, 1), dtype=np.float32)], axis=1)
        P_cam = (E_t @ P_hom.T).T[:, :3]
        z_proj = P_cam[:, 2]

        # Project to 2D
        fx, fy, cx, cy = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
        u = P_cam[:, 0] / z_proj * fx + cx
        v = P_cam[:, 1] / z_proj * fy + cy

        # In-bounds check
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_proj > 0)

        # Depth consistency check (relaxed for Co3D's sparse depth)
        px = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        py = np.clip(np.round(v).astype(np.int32), 0, H - 1)
        sampled_d = depth_t[py, px]

        # Only check if depth is valid (not zero)
        depth_valid = (sampled_d > 0) & np.isfinite(sampled_d)
        depth_ok = np.ones(N, dtype=bool)  # Default to True
        depth_ok[depth_valid] = (
            np.abs(sampled_d[depth_valid] - z_proj[depth_valid]) /
            np.maximum(z_proj[depth_valid], 1e-6) < depth_consistency_thresh
        )

        valid_t = in_bounds & depth_ok
        trajs_2d[t] = np.stack([u, v], axis=1)
        valids[t] = valid_t

    trajs_3d_world = np.broadcast_to(pointcloud[None], (T, N, 3)).copy()
    visibs = valids.copy()

    return {
        "trajs_2d": trajs_2d,
        "trajs_3d_world": trajs_3d_world,
        "valids": valids,
        "visibs": visibs,
        "ref_frame": ref_frame,
        "num_points": N,
    }


def process_sequence(args: tuple) -> tuple[str, str | None]:
    """Process one Co3D sequence."""
    seq_name, adapter_state, output_root, num_points, overwrite = args

    out_path = Path(output_root) / seq_name / "precomputed.npz"
    if out_path.exists() and not overwrite:
        return seq_name, None

    try:
        # Reconstruct adapter
        adapter_cls, adapter_kwargs = adapter_state
        adapter = adapter_cls(**adapter_kwargs)

        info = adapter.get_sequence_info(seq_name)
        num_frames = info["num_frames"]

        # Load all frames
        all_indices = list(range(num_frames))
        clip = adapter.load_clip(seq_name, all_indices)

        # Check for pointcloud.ply
        ply_path = Path(info["sequence_root"]) / "pointcloud.ply"
        if not ply_path.exists():
            return seq_name, "no pointcloud.ply"

        # Read point cloud
        pointcloud = read_ply_pointcloud(ply_path)

        # Sample points if too many
        if len(pointcloud) > num_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pointcloud), num_points, replace=False)
            pointcloud = pointcloud[idx]

        # Compute normals from depth
        normals = compute_normals_sequence(clip.depths, clip.intrinsics)

        # Project point cloud to all frames
        tracks = project_pointcloud_to_frames(
            pointcloud=pointcloud,
            intrinsics=clip.intrinsics,
            extrinsics=clip.extrinsics,
            image_size=clip.image_size,
            depths=clip.depths,
            depth_consistency_thresh=0.2,  # Relaxed for Co3D
        )

        # Save
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
        )
        return seq_name, None

    except Exception:
        return seq_name, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None, help="Specific categories to process")
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(root=args.root, categories=args.categories)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[co3d_pointcloud] {total} sequences → {output_root}")

    # Build adapter state for multiprocessing
    adapter_state = (
        Co3Dv2Adapter,
        {
            "root": str(adapter.root),
            "categories": adapter.categories,
            "subset_name": adapter.subset_name,
            "split": adapter.split,
        }
    )

    job_args = [
        (seq, adapter_state, str(output_root), args.num_points, args.overwrite)
        for seq in sequences
    ]

    done = 0
    failed = []

    if args.workers <= 1:
        for job_arg in tqdm(job_args, desc="co3d_pointcloud"):
            seq_name, err = process_sequence(job_arg)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_sequence, a): a[0] for a in job_args}
            with tqdm(total=total, desc="co3d_pointcloud") as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")

    print(f"[co3d_pointcloud] done {done - len(failed)}/{total}, failed {len(failed)}")
    if failed:
        print("[co3d_pointcloud] Failed sequences:")
        for seq_name, err in failed[:10]:
            print(f"  {seq_name}: {err.splitlines()[-1]}")


if __name__ == "__main__":
    main()
