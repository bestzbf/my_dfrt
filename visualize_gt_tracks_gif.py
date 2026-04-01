#!/usr/bin/env python3
"""Generate GIF animations showing GT point tracking from MixtureDataset."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset

try:
    import imageio
except ImportError:
    print("Error: imageio not installed. Run: pip install imageio")
    sys.exit(1)


def create_tracking_gif(sample, out_path: Path, max_tracks: int = 30, trail_len: int = 10):
    """Create video with fading trails that disappear when points become invisible."""
    video = sample.video.numpy()
    T, _, H, W = video.shape
    frames = [(video[t].transpose(1, 2, 0) * 255).astype(np.uint8) for t in range(T)]

    coords = sample.coords.numpy()
    t_src = sample.t_src.numpy()
    t_tgt = sample.t_tgt.numpy()
    pos_2d = sample.targets['pos_2d'].numpy()
    mask_2d = sample.targets['mask_2d'].numpy().astype(bool)

    # Build tracks with visibility
    track_dict = {}
    for i in range(len(coords)):
        ts, tt = int(t_src[i]), int(t_tgt[i])
        is_visible = mask_2d[i]
        u_src, v_src = coords[i]
        u_tgt, v_tgt = pos_2d[i]

        px_src = int(np.clip(u_src * (W - 1), 0, W - 1))
        py_src = int(np.clip(v_src * (H - 1), 0, H - 1))
        px_tgt = int(np.clip(u_tgt * (W - 1), 0, W - 1))
        py_tgt = int(np.clip(v_tgt * (H - 1), 0, H - 1))

        key = (ts, px_src, py_src)
        if key not in track_dict:
            track_dict[key] = []
        track_dict[key].append((tt, px_tgt, py_tgt, is_visible))

    # Sample tracks
    tracks = [(ts, px, py, pts) for (ts, px, py), pts in track_dict.items()]
    if len(tracks) > max_tracks:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(tracks), max_tracks, replace=False)
        tracks = [tracks[i] for i in indices]

    # Colors
    colors = []
    for i in range(len(tracks)):
        hue = int(180 * i / len(tracks))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))

    # Generate frames
    gif_frames = []
    for t in range(T):
        frame = cv2.cvtColor(frames[t].copy(), cv2.COLOR_RGB2BGR)

        for track_idx, (ts, px_src, py_src, points) in enumerate(tracks):
            color = colors[track_idx]
            traj = [(ts, px_src, py_src, True)] + sorted(points, key=lambda x: x[0])

            # Only show trail for recent visible points within bounds
            visible_pts = []
            for tt, px, py, vis in traj:
                # Check if point is within frame bounds
                in_bounds = (0 <= px < W and 0 <= py < H)
                if tt <= t and vis and in_bounds:
                    visible_pts.append((tt, px, py))
                elif tt <= t and (not vis or not in_bounds):
                    # Point disappeared or out of bounds, clear trail
                    visible_pts = []

            # Keep only recent trail
            if len(visible_pts) > trail_len:
                visible_pts = visible_pts[-trail_len:]

            # Draw fading trail
            for i in range(len(visible_pts) - 1):
                alpha = (i + 1) / len(visible_pts)
                thickness = max(1, int(2 * alpha))
                pt1 = (visible_pts[i][1], visible_pts[i][2])
                pt2 = (visible_pts[i+1][1], visible_pts[i+1][2])
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

            # Draw current point
            if visible_pts:
                tt, px, py = visible_pts[-1]
                cv2.circle(frame, (px, py), 4, (0, 0, 0), -1)
                cv2.circle(frame, (px, py), 3, color, -1)

        cv2.putText(frame, f't={t}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as MP4 video for smoother playback
    if str(out_path).endswith('.gif'):
        out_path = out_path.with_suffix('.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, 2.0, (W, H))
    for frame in gif_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mixture_10datasets.yaml")
    parser.add_argument("--output-dir", default="outputs/vis_tracks_gif")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--trail-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print("Loading adapters...")
    adapters, weights = [], []
    for ds in config["datasets"]:
        name, root = ds["name"], ds["root"]
        try:
            a = create_adapter(name, root=root, split="train")
            adapters.append(a)
            weights.append(ds.get("weight", 1.0))
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    dataset = MixtureDataset(
        adapters=adapters, dataset_weights=weights,
        clip_len=args.clip_len, img_size=256, num_queries=512,
        use_augs=False, seed=args.seed, sampling_mode='stride')

    print(f"Generating {args.num_samples} GIFs...\n")
    out_dir = Path(args.output_dir)
    for i in range(args.num_samples):
        sample = dataset[i]
        out_path = out_dir / f"track_{i:03d}_{sample.dataset_name}.gif"
        print(f"[{i}] {sample.dataset_name} - {sample.sequence_name[:30]}")
        create_tracking_gif(sample, out_path, max_tracks=args.max_tracks, trail_len=args.trail_len)

    print(f"\nDone. View GIFs in {out_dir}/")


if __name__ == "__main__":
    main()
