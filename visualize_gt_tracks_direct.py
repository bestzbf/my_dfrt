#!/usr/bin/env python3
"""Generate tracking videos directly from adapter trajectory data."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter

def create_tracking_video(adapter, seq_name, frame_indices, out_path: Path,
                         max_tracks: int = 100, trail_len: int = 10):
    """Create video with tracking from original trajectory data."""
    clip = adapter.load_clip(seq_name, frame_indices)

    # Get data
    images = clip.images  # List of [H, W, 3] uint8
    trajs_2d = clip.trajs_2d  # [T, N, 2] in normalized [0,1] coords
    visibs = clip.visibs  # [T, N] bool

    T = len(images)
    H, W = images[0].shape[:2]

    # Sample tracks
    N = trajs_2d.shape[1]
    if N > max_tracks:
        rng = np.random.default_rng(0)
        track_ids = rng.choice(N, max_tracks, replace=False)
    else:
        track_ids = np.arange(N)

    # Filter tracks that are visible in at least 2 frames
    valid_tracks = []
    for tid in track_ids:
        if visibs[:, tid].sum() >= 2:
            valid_tracks.append(tid)

    if len(valid_tracks) == 0:
        print(f"  Warning: No valid tracks found")
        return

    # Colors
    colors = []
    for i in range(len(valid_tracks)):
        hue = int(180 * i / len(valid_tracks))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))

    # Generate frames
    gif_frames = []
    for t in range(T):
        frame = cv2.cvtColor(images[t].copy(), cv2.COLOR_RGB2BGR)

        for idx, tid in enumerate(valid_tracks):
            color = colors[idx]

            # Get visible trajectory up to current frame
            visible_pts = []
            for tt in range(t + 1):
                if visibs[tt, tid]:
                    u, v = trajs_2d[tt, tid]
                    px = int(np.clip(u * (W - 1), 0, W - 1))
                    py = int(np.clip(v * (H - 1), 0, H - 1))
                    visible_pts.append((px, py))
                else:
                    visible_pts = []  # Clear trail when invisible

            # Keep only recent trail
            if len(visible_pts) > trail_len:
                visible_pts = visible_pts[-trail_len:]

            # Draw trail
            for i in range(len(visible_pts) - 1):
                alpha = (i + 1) / len(visible_pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(frame, visible_pts[i], visible_pts[i+1], color, thickness, cv2.LINE_AA)

            # Draw current point
            if visible_pts:
                px, py = visible_pts[-1]
                cv2.circle(frame, (px, py), 4, (0, 0, 0), -1)
                cv2.circle(frame, (px, py), 3, color, -1)

        cv2.putText(frame, f't={t}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--output-dir", default="outputs/vis_tracks_direct")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--clip-len", type=int, default=24)
    parser.add_argument("--max-tracks", type=int, default=200)
    parser.add_argument("--trail-len", type=int, default=15)
    parser.add_argument("--max-stride", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print("Loading adapters...")
    adapters = []
    for ds in config["datasets"]:
        name, root = ds["name"], ds["root"]
        try:
            a = create_adapter(name, root=root, split="train")
            adapters.append((name, a))
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\nGenerating {args.num_samples} videos...\n")
    out_dir = Path(args.output_dir)

    rng = np.random.default_rng(42)
    sample_count = 0

    for ds_name, adapter in adapters:
        if sample_count >= args.num_samples:
            break

        sequences = adapter.list_sequences()
        for seq_name in sequences[:2]:  # Max 2 per dataset
            if sample_count >= args.num_samples:
                break

            try:
                info = adapter.get_sequence_info(seq_name)
                num_frames = info['num_frames']

                if num_frames < args.clip_len:
                    continue

                # Sample with stride
                max_stride = min(args.max_stride, max(1, (num_frames - 1) // (args.clip_len - 1)))
                stride = rng.integers(1, max_stride + 1)
                max_start = num_frames - (args.clip_len - 1) * stride
                start = rng.integers(0, max(1, max_start))
                frame_indices = [start + i * stride for i in range(args.clip_len)]

                out_path = out_dir / f"track_{sample_count:03d}_{ds_name}.mp4"
                print(f"[{sample_count}] {ds_name} - {seq_name[:30]} (stride={stride})")

                create_tracking_video(adapter, seq_name, frame_indices, out_path,
                                    max_tracks=args.max_tracks, trail_len=args.trail_len)
                sample_count += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nDone. View videos in {out_dir}/")


if __name__ == "__main__":
    main()
