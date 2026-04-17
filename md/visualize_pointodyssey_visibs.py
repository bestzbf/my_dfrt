#!/usr/bin/env python3
"""Visualize PointOdyssey valids/visibs over a frame window."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from datasets.factory import create_training_dataset


def _load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compute_inbounds(trajs_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        (trajs_2d[..., 0] >= 0.0)
        & (trajs_2d[..., 0] < float(width))
        & (trajs_2d[..., 1] >= 0.0)
        & (trajs_2d[..., 1] < float(height))
    )


def _draw_points(
    ax,
    image: np.ndarray,
    pts_xy: np.ndarray,
    vis_mask: np.ndarray,
    max_points: int,
    rng: random.Random,
) -> None:
    ax.imshow(image)
    ax.axis("off")

    if pts_xy.shape[0] == 0:
        return

    if pts_xy.shape[0] > max_points:
        chosen = rng.sample(range(pts_xy.shape[0]), k=max_points)
        pts_xy = pts_xy[chosen]
        vis_mask = vis_mask[chosen]

    inv_mask = ~vis_mask
    if np.any(inv_mask):
        ax.scatter(
            pts_xy[inv_mask, 0],
            pts_xy[inv_mask, 1],
            s=6,
            c="#ff3b30",
            alpha=0.55,
            linewidths=0.0,
        )
    if np.any(vis_mask):
        ax.scatter(
            pts_xy[vis_mask, 0],
            pts_xy[vis_mask, 1],
            s=8,
            c="#34c759",
            alpha=0.85,
            linewidths=0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_pointodyssey_highres_noaug.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence", default="cnb_dlab_0215_3rd")
    parser.add_argument("--start", type=int, default=724)
    parser.add_argument("--end", type=int, default=776)
    parser.add_argument("--context", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument(
        "--output",
        default="tmp/cnb_dlab_0215_3rd_visibs_window_724_776.png",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split=args.split)
    adapter = dataset.adapters[0]
    record = adapter.get_record(args.sequence)
    anno = adapter._load_anno(args.sequence)

    trajs_2d = np.asarray(anno["trajs_2d"])
    valids = np.asarray(anno["valids"]).astype(bool)
    visibs = np.asarray(anno["visibs"]).astype(bool)

    height, width = record.image_size
    inbounds = _compute_inbounds(trajs_2d, width=width, height=height)

    per_frame_valids = valids.sum(axis=1)
    per_frame_visibs = visibs.sum(axis=1)
    per_frame_valids_inbounds = (valids & inbounds).sum(axis=1)
    per_frame_visibs_inbounds = (visibs & inbounds).sum(axis=1)

    lo = max(0, args.start - args.context)
    hi = min(len(per_frame_visibs), args.end + args.context + 1)
    frame_ids = np.arange(lo, hi)

    selected_frames = [
        max(0, args.start - 4),
        max(0, args.start - 1),
        args.start,
        min(args.end, args.start + (args.end - args.start) // 2),
        args.end - 1,
        args.end,
        min(len(per_frame_visibs) - 1, args.end + 1),
        min(len(per_frame_visibs) - 1, args.end + 4),
    ]
    selected_frames = list(dict.fromkeys(selected_frames))

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0])

    ax_curve = fig.add_subplot(grid[0, :])
    ax_curve.plot(frame_ids, per_frame_valids[lo:hi], label="valids", color="#007aff", linewidth=2.0)
    ax_curve.plot(frame_ids, per_frame_visibs[lo:hi], label="visibs", color="#ff2d55", linewidth=2.0)
    ax_curve.plot(
        frame_ids,
        per_frame_valids_inbounds[lo:hi],
        label="valids_inbounds",
        color="#5ac8fa",
        linestyle="--",
        linewidth=1.6,
    )
    ax_curve.plot(
        frame_ids,
        per_frame_visibs_inbounds[lo:hi],
        label="visibs_inbounds",
        color="#ff9500",
        linestyle="--",
        linewidth=1.6,
    )
    ax_curve.axvspan(args.start, args.end, color="#ff3b30", alpha=0.10, label="window")
    ax_curve.set_title(
        f"{args.sequence} ({args.split})  visibs/valids around frames {args.start}-{args.end}"
    )
    ax_curve.set_xlabel("frame")
    ax_curve.set_ylabel("num points")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper left", ncol=5, fontsize=9)

    draw_rng = random.Random(42)
    for idx, frame_idx in enumerate(selected_frames[:8]):
        row = 1 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(grid[row, col])

        image = _load_image_rgb(record.rgb_paths[frame_idx])
        valid_inbounds = valids[frame_idx] & inbounds[frame_idx]
        pts_xy = trajs_2d[frame_idx][valid_inbounds]
        vis_mask = visibs[frame_idx][valid_inbounds]
        _draw_points(ax, image, pts_xy, vis_mask, args.max_points, draw_rng)
        ax.set_title(
            "\n".join(
                [
                    f"frame {frame_idx}",
                    f"valids={int(per_frame_valids[frame_idx])}  visibs={int(per_frame_visibs[frame_idx])}",
                    f"inbounds={int(per_frame_valids_inbounds[frame_idx])}  vis_inb={int(per_frame_visibs_inbounds[frame_idx])}",
                ]
            ),
            fontsize=10,
        )

    fig.suptitle(
        "Green: visible points, Red: valid-but-not-visible points",
        fontsize=14,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(output.resolve())


if __name__ == "__main__":
    main()
