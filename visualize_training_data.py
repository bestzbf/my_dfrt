#!/usr/bin/env python3
"""
Visualize actual training data with query sampling.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.factory import create_training_dataset


def make_colors(n: int):
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


def visualize_sample(sample, out_path):
    """Visualize one training sample."""
    video = sample.video    # [T, 3, H, W]
    coords = sample.coords  # [Q, 2]
    t_src = sample.t_src    # [Q]
    targets = sample.targets

    T, _, H, W = video.shape
    Q = coords.shape[0]

    print(f"  T={T}, Q={Q}, H={H}, W={W}")

    # Convert video to numpy
    video_np = video.permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)

    coords_np = coords.numpy()
    t_src_np = t_src.numpy()

    colors = make_colors(min(Q, 150))  # Limit colors

    # Render frames
    frames = []
    for t in range(T):
        frame = cv2.cvtColor(video_np[t].copy(), cv2.COLOR_RGB2BGR)

        # Draw query points on their source frames
        for qi in range(min(Q, 150)):
            if t_src_np[qi] == t:
                color = colors[qi]
                px = int(np.clip(coords_np[qi, 0] * (W - 1), 0, W - 1))
                py = int(np.clip(coords_np[qi, 1] * (H - 1), 0, H - 1))
                cv2.circle(frame, (px, py), 4, (0, 0, 0), -1)
                cv2.circle(frame, (px, py), 3, color, -1)

        cv2.putText(frame, f"t={t} Q={Q}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
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
    parser.add_argument("--output-dir", default="outputs/vis_training")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Add default training params
    config['clip_len'] = config.get('clip_len', 16)
    config['img_size'] = config.get('img_size', 256)
    config['num_queries'] = config.get('num_queries', 2048)

    out_dir = Path(args.output_dir)

    print(f"Creating training dataset...")
    dataset = create_training_dataset(config, split='train')
    print(f"Dataset size: {len(dataset)}")

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\nSample {i}:")
        sample = dataset[i]

        dataset_name = sample.dataset_name
        seq_name = sample.sequence_name
        out_path = out_dir / f"{i:03d}_{dataset_name}_{seq_name[:20]}.mp4"

        visualize_sample(sample, out_path)

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()

