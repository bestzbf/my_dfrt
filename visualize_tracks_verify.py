#!/usr/bin/env python3
"""
Visualize 2D and 3D trajectories side-by-side for verification.

Left panel:  2D trajectories from trajs_2d
Right panel: 3D trajectories reprojected to 2D using camera parameters

If 3D data is correct, both panels should match perfectly.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter


def make_colors(n: int):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


def project_3d_to_2d(pts_world, intrinsic, extrinsic_w2c):
    """
    Project 3D world points to 2D pixel coordinates.

    Args:
        pts_world: (N, 3) world coordinates
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic_w2c: (4, 4) world-to-camera transformation

    Returns:
        uv: (N, 2) pixel coordinates [u, v]
        valid: (N,) bool, True if point is in front of camera
    """
    N = pts_world.shape[0]
    pts_h = np.concatenate([pts_world, np.ones((N, 1))], axis=1)  # (N, 4)
    pts_cam = (extrinsic_w2c @ pts_h.T).T  # (N, 4)

    z = pts_cam[:, 2]
    valid = z > 1e-3

    z_safe = np.where(valid, z, 1.0)
    x_n = pts_cam[:, 0] / z_safe
    y_n = pts_cam[:, 1] / z_safe

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * x_n + cx
    v = fy * y_n + cy

    return np.stack([u, v], axis=1), valid


def draw_trails(frame, positions, visibles, colors, trail_len, t):
    """Draw trajectory trails on frame."""
    H, W = frame.shape[:2]
    for idx, (pos, vis) in enumerate(zip(positions, visibles)):
        color = colors[idx]

        # Collect visible points in trail window
        start_t = max(0, t - trail_len + 1)
        pts = []
        for tt in range(start_t, t + 1):
            if vis[tt]:
                px = int(np.clip(pos[tt, 0], 0, W - 1))
                py = int(np.clip(pos[tt, 1], 0, H - 1))
                pts.append((px, py))

        # Draw lines between consecutive visible points
        for i in range(len(pts) - 1):
            alpha = (i + 1) / max(len(pts), 1)
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

        # Draw current point
        if vis[t]:
            px = int(np.clip(pos[t, 0], 0, W - 1))
            py = int(np.clip(pos[t, 1], 0, H - 1))
            cv2.circle(frame, (px, py), 4, (0, 0, 0), -1)
            cv2.circle(frame, (px, py), 3, color, -1)


def visualize_clip(adapter, seq_name, frame_indices, out_path, max_tracks=80, trail_len=12):
    """Create side-by-side visualization of 2D and 3D trajectories."""
    clip = adapter.load_clip(seq_name, frame_indices)

    if clip.trajs_2d is None:
        print(f"  Skip: no trajs_2d")
        return

    images = clip.images
    trajs_2d = clip.trajs_2d
    visibs = clip.visibs
    trajs_3d = clip.trajs_3d_world
    intrinsics = clip.intrinsics
    extrinsics = clip.extrinsics

    T, N = trajs_2d.shape[:2]
    H, W = images[0].shape[:2]
    has_3d = trajs_3d is not None

    # Sample tracks (select most visible tracks instead of random)
    vis_counts = visibs.sum(axis=0)  # (N,)
    valid_mask = vis_counts >= 1
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print(f"  Skip: no valid tracks")
        return

    # Sort by visibility and take top tracks
    sorted_indices = valid_indices[np.argsort(-vis_counts[valid_indices])]
    track_ids = sorted_indices[:min(max_tracks, len(sorted_indices))].tolist()

    colors = make_colors(len(track_ids))
    print(f"  {seq_name[:40]}  T={T}  tracks={len(track_ids)}  3D={'yes' if has_3d else 'no'}")

    # Convert 2D trajectories to pixel coordinates
    pos_2d, vis_2d = [], []
    for tid in track_ids:
        uv = trajs_2d[:, tid, :]  # (T, 2) normalized [0,1]
        px = uv[:, 0] * (W - 1)
        py = uv[:, 1] * (H - 1)
        pos_2d.append(np.stack([px, py], axis=1))
        vis_2d.append(visibs[:, tid])

    # Reproject 3D trajectories to pixel coordinates
    pos_3d, vis_3d = [], []
    if has_3d:
        for tid in track_ids:
            pts_world = trajs_3d[:, tid, :]  # (T, 3)
            pixel_seq = np.zeros((T, 2), dtype=np.float32)
            valid_seq = np.zeros(T, dtype=bool)
            for t in range(T):
                uv, valid = project_3d_to_2d(pts_world[t:t+1], intrinsics[t], extrinsics[t])
                pixel_seq[t] = uv[0]
                valid_seq[t] = valid[0] and visibs[t, tid]
            pos_3d.append(pixel_seq)
            vis_3d.append(valid_seq)

    # Render frames
    frames = []
    for t in range(T):
        img = images[t].copy()
        # Ensure uint8 format for OpenCV
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Left: 2D trajectories
        left = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_trails(left, pos_2d, vis_2d, colors, trail_len, t)
        cv2.putText(left, f"2D trajs t={t}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        if has_3d:
            # Right: 3D reprojected
            right = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            draw_trails(right, pos_3d, vis_3d, colors, trail_len, t)
            cv2.putText(right, f"3D->2D t={t}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            # Show reprojection error
            for idx, (p2, p3, v2, v3) in enumerate(zip(pos_2d, pos_3d, vis_2d, vis_3d)):
                if v2[t] and v3[t]:
                    err = np.linalg.norm(p2[t] - p3[t])
                    if err < 2:
                        err_color = (0, 255, 0)  # green
                    elif err < 5:
                        err_color = (0, 255, 255)  # yellow
                    else:
                        err_color = (0, 0, 255)  # red
                    px = int(np.clip(p2[t,0], 0, W-1))
                    py = int(np.clip(p2[t,1], 0, H-1))
                    cv2.circle(left, (px, py), 6, err_color, 1)

            frame = np.concatenate([left, right], axis=1)
        else:
            frame = left

        frames.append(frame)

    # Save video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 3.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mixture_10datasets.yaml")
    parser.add_argument("--output-dir", default="outputs/vis_verify")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--max-tracks", type=int, default=150)
    parser.add_argument("--trail-len", type=int, default=12)
    parser.add_argument("--max-stride", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    rng = np.random.default_rng(42)

    for ds in config["datasets"]:
        name, root = ds["name"], ds["root"]
        print(f"\n[{name}]")

        try:
            adapter = create_adapter(name, root=root, split="train")
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        sequences = adapter.list_sequences()
        sample_count = 0

        for seq_name in sequences:
            if sample_count >= args.num_samples:
                break

            try:
                info = adapter.get_sequence_info(seq_name)
                num_frames = info["num_frames"]
                if num_frames < args.clip_len:
                    continue

                # Static reconstruction datasets need stride=1 for better overlap
                static_datasets = {'scannet', 'co3dv2', 'blendedmvs', 'mvssynth'}
                if name in static_datasets:
                    stride = 1
                else:
                    max_stride = min(args.max_stride, max(1, (num_frames - 1) // (args.clip_len - 1)))
                    stride = rng.integers(1, max_stride + 1)

                max_start = num_frames - (args.clip_len - 1) * stride
                start = rng.integers(0, max(1, max_start))
                frame_indices = [start + i * stride for i in range(args.clip_len)]

                out_path = out_dir / name / f"{sample_count:03d}_{seq_name[:30]}.mp4"
                visualize_clip(adapter, seq_name, frame_indices, out_path,
                             max_tracks=args.max_tracks, trail_len=args.trail_len)
                sample_count += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
