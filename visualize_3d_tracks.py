#!/usr/bin/env python3
"""Visualize 3D trajectories as point cloud animation."""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter


def create_3d_animation(adapter, seq_name, frame_indices, out_path: Path,
                       max_tracks: int = 100):
    """Create 3D point cloud animation."""
    clip = adapter.load_clip(seq_name, frame_indices)

    if clip.trajs_3d_world is None:
        print(f"  Warning: No 3D trajectories available")
        return

    trajs_3d = clip.trajs_3d_world  # [T, N, 3] in world coords
    visibs = clip.visibs  # [T, N]
    T, N = trajs_3d.shape[:2]

    # Sample tracks
    if N > max_tracks:
        rng = np.random.default_rng(0)
        track_ids = rng.choice(N, max_tracks, replace=False)
    else:
        track_ids = np.arange(N)

    # Filter valid tracks
    valid_tracks = []
    for tid in track_ids:
        if visibs[:, tid].sum() >= 2:
            valid_tracks.append(tid)

    if len(valid_tracks) == 0:
        print(f"  Warning: No valid 3D tracks")
        return

    print(f"  Visualizing {len(valid_tracks)} tracks")

    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute bounds
    all_pts = []
    for tid in valid_tracks:
        for t in range(T):
            if visibs[t, tid]:
                all_pts.append(trajs_3d[t, tid])
    all_pts = np.array(all_pts)

    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Colors
    colors = plt.cm.hsv(np.linspace(0, 1, len(valid_tracks)))

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Trajectories - Frame {frame_idx}/{T-1}')

        for idx, tid in enumerate(valid_tracks):
            # Draw trail
            trail_pts = []
            for t in range(frame_idx + 1):
                if visibs[t, tid]:
                    trail_pts.append(trajs_3d[t, tid])

            if len(trail_pts) > 1:
                trail_pts = np.array(trail_pts)
                ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                       color=colors[idx], alpha=0.5, linewidth=1)

            # Draw current point
            if visibs[frame_idx, tid]:
                pt = trajs_3d[frame_idx, tid]
                ax.scatter(pt[0], pt[1], pt[2], color=colors[idx], s=50)

    anim = FuncAnimation(fig, update, frames=T, interval=500)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=2)
    anim.save(str(out_path), writer=writer)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mixture_10datasets.yaml")
    parser.add_argument("--output-dir", default="outputs/vis_3d_tracks")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--max-tracks", type=int, default=50)
    parser.add_argument("--max-stride", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading adapters...")
    adapters = []
    for ds in config["datasets"]:
        name, root = ds["name"], ds["root"]
        try:
            a = create_adapter(name, root=root, split="train")
            info = a.get_sequence_info(a.get_sequence_name(0))
            if info.get('has_trajs_3d_world'):
                adapters.append((name, a))
                print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\nGenerating {args.num_samples} 3D animations...\n")
    out_dir = Path(args.output_dir)

    rng = np.random.default_rng(42)
    sample_count = 0

    for ds_name, adapter in adapters:
        if sample_count >= args.num_samples:
            break

        sequences = adapter.list_sequences()
        for seq_name in sequences[:2]:
            if sample_count >= args.num_samples:
                break

            try:
                info = adapter.get_sequence_info(seq_name)
                num_frames = info['num_frames']

                if num_frames < args.clip_len:
                    continue

                max_stride = min(args.max_stride, max(1, (num_frames - 1) // (args.clip_len - 1)))
                stride = rng.integers(1, max_stride + 1)
                max_start = num_frames - (args.clip_len - 1) * stride
                start = rng.integers(0, max(1, max_start))
                frame_indices = [start + i * stride for i in range(args.clip_len)]

                out_path = out_dir / f"3d_{sample_count:03d}_{ds_name}.mp4"
                print(f"[{sample_count}] {ds_name} - {seq_name[:30]}")

                create_3d_animation(adapter, seq_name, frame_indices, out_path,
                                  max_tracks=args.max_tracks)
                sample_count += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nDone. View videos in {out_dir}/")


if __name__ == "__main__":
    main()
