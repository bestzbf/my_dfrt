#!/usr/bin/env python3
"""
Visualize point tracking trajectories from MixtureDataset.
Shows temporal motion paths of tracked points overlaid on video frames.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset


def get_colors(n, cmap='hsv'):
    """Generate n distinct colors."""
    cm = plt.get_cmap(cmap)
    return (cm(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)


def draw_tracks_on_frames(frames, coords, t_src, t_tgt, pos_2d, mask_2d, max_tracks=50):
    """
    Draw point trajectories on video frames with cumulative trails.
    """
    T = len(frames)
    H, W = frames[0].shape[:2]

    # Build tracks: group by source point
    track_dict = {}
    for i in range(len(coords)):
        if not mask_2d[i]:
            continue
        ts = int(t_src[i])
        tt = int(t_tgt[i])
        u_src, v_src = coords[i]
        u_tgt, v_tgt = pos_2d[i]

        px_src = int(np.clip(u_src * (W - 1), 0, W - 1))
        py_src = int(np.clip(v_src * (H - 1), 0, H - 1))
        px_tgt = int(np.clip(u_tgt * (W - 1), 0, W - 1))
        py_tgt = int(np.clip(v_tgt * (H - 1), 0, H - 1))

        key = (ts, px_src, py_src)
        if key not in track_dict:
            track_dict[key] = []
        track_dict[key].append((tt, px_tgt, py_tgt))

    # Convert to list and sample
    tracks = [(ts, px, py, pts) for (ts, px, py), pts in track_dict.items()]
    if len(tracks) > max_tracks:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(tracks), max_tracks, replace=False)
        tracks = [tracks[i] for i in indices]

    colors = get_colors(len(tracks))
    vis_frames = [f.copy() for f in frames]

    # Draw tracks with cumulative trails
    for track_idx, (ts, px_src, py_src, points) in enumerate(tracks):
        color = tuple(int(c) for c in colors[track_idx])

        # Build full trajectory
        traj = [(ts, px_src, py_src)] + sorted(points)

        # Draw on each frame
        for t in range(T):
            # Draw all points up to current time
            prev_pt = None
            for tt, px, py in traj:
                if tt <= t:
                    # Draw point
                    cv2.circle(vis_frames[t], (px, py), 3, (0, 0, 0), -1)
                    cv2.circle(vis_frames[t], (px, py), 2, color, -1)
                    # Draw line from previous
                    if prev_pt is not None:
                        cv2.line(vis_frames[t], prev_pt, (px, py), color, 2, cv2.LINE_AA)
                    prev_pt = (px, py)

            # Mark source with larger circle
            cv2.circle(vis_frames[t], (px_src, py_src), 5, (255, 255, 255), 1)

    return vis_frames


def visualize_sample(sample, out_path: Path, max_tracks: int = 50):
    """Render tracking visualization for one sample."""
    video = sample.video.numpy()  # [T,3,H,W]
    T, _, H, W = video.shape
    frames = [(video[t].transpose(1, 2, 0) * 255).astype(np.uint8) for t in range(T)]

    coords = sample.coords.numpy()
    t_src = sample.t_src.numpy()
    t_tgt = sample.t_tgt.numpy()
    pos_2d = sample.targets['pos_2d'].numpy()
    mask_2d = sample.targets['mask_2d'].numpy().astype(bool)

    # Draw tracks
    vis_frames = draw_tracks_on_frames(frames, coords, t_src, t_tgt, pos_2d,
                                       mask_2d, max_tracks)

    # Create grid
    n_cols = min(8, T)
    n_rows = (T + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        if i < T:
            ax.imshow(vis_frames[i])
            ax.set_title(f't={i}', fontsize=10)
        ax.axis('off')

    # Title
    mask_2d_sum = int(mask_2d.sum())
    total_q = len(coords)
    info = (f"{sample.dataset_name} | {sample.sequence_name} | "
            f"T={T} Q={total_q} valid={mask_2d_sum}")
    fig.suptitle(info, fontsize=12, y=0.98)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mixture_10datasets.yaml")
    p.add_argument("--output-dir", default="outputs/vis_mixture_tracks")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=512)
    p.add_argument("--max-tracks", type=int, default=50,
                   help="Max tracks to visualize per sample")
    p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_adapters_from_config(config, dataset_filter):
    adapters, weights = [], []
    for ds in config["datasets"]:
        name = ds["name"]
        if dataset_filter and name != dataset_filter:
            continue
        try:
            a = create_adapter(name, root=ds["root"], split="train")
            adapters.append(a)
            weights.append(ds.get("weight", 1.0))
            print(f"  ✓ {name:20s} seqs={len(a.list_sequences())}")
        except Exception as e:
            print(f"  ✗ {name:20s} SKIP: {e}")
    return adapters, weights


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print("Loading adapters...")
    adapters, weights = build_adapters_from_config(config, args.dataset)
    if not adapters:
        print("No adapters loaded.")
        sys.exit(1)

    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=args.clip_len,
        img_size=256,
        num_queries=args.num_queries,
        use_augs=False,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        seed=args.seed,
    )
    print(f"Dataset ready: {dataset.get_dataset_names()}")
    print(f"Visualizing {args.num_samples} samples → {args.output_dir}/\n")

    out_dir = Path(args.output_dir)
    for i in range(args.num_samples):
        sample = dataset[i]
        out_path = out_dir / f"tracks_{i:03d}_{sample.dataset_name}.png"
        print(f"[{i}] {sample.dataset_name:20s} seq={sample.sequence_name[:30]}")
        visualize_sample(sample, out_path, max_tracks=args.max_tracks)

    print(f"\nDone. Open {out_dir}/ to inspect.")


if __name__ == "__main__":
    main()
