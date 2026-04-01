#!/usr/bin/env python3
"""
Visualize GT data loaded from MixtureDataset.

For each sample, renders:
  - Row 1: video frames (all T frames)
  - Row 2: depth maps (if available)
  - Row 3: normal maps (if available)
  - Row 4: source-frame query points (colored by query index)
  - Row 5: target projections — pos_2d overlaid on target frame,
           colored by mask_3d / mask_2d
  - Row 6: 3D tracks — 2D projection of pos_3d for track datasets

Saves one PNG per sample to --output-dir.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_uint8(img_float):
    """float32 [0,1] HWC → uint8 HWC."""
    return np.clip(img_float * 255, 0, 255).astype(np.uint8)


def colormap_depth(depth):
    """depth [H,W] float32 → uint8 [H,W,3] (jet colormap, invalid=gray)."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.full((*depth.shape, 3), 128, dtype=np.uint8)
    d = depth.copy()
    d[~valid] = np.nan
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    if hi <= lo:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0, 1)
    d_norm[~valid] = 0
    cmap = matplotlib.colormaps.get_cmap("jet")
    colored = (cmap(d_norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 128
    return colored


def colormap_normal(normal):
    """normal [H,W,3] float32 → uint8 [H,W,3] (xyz → rgb)."""
    n = (normal + 1.0) / 2.0
    return np.clip(n * 255, 0, 255).astype(np.uint8)


def draw_points_on_frame(frame_uint8, xy_norm, colors_rgb, radius=3):
    """Draw colored dots at normalized (u,v)∈[0,1] coords."""
    img = frame_uint8.copy()
    H, W = img.shape[:2]
    for i, (uv, c) in enumerate(zip(xy_norm, colors_rgb)):
        px = int(np.clip(uv[0] * (W - 1), 0, W - 1))
        py = int(np.clip(uv[1] * (H - 1), 0, H - 1))
        cv2.circle(img, (px, py), radius + 1, (0, 0, 0), -1)
        cv2.circle(img, (px, py), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return img


def sample_query_subset(coords, t_src, t_tgt, targets, max_q=200, seed=0):
    """Sub-sample queries for legible visualization."""
    rng = np.random.default_rng(seed)
    Q = coords.shape[0]
    idx = rng.choice(Q, size=min(max_q, Q), replace=False)
    return (coords[idx], t_src[idx], t_tgt[idx],
            {k: v[idx] for k, v in targets.items()})


# ---------------------------------------------------------------------------
# Per-sample visualization
# ---------------------------------------------------------------------------

def visualize_sample(sample, out_path: Path, max_q: int = 150):
    """Render one QuerySample to a PNG grid."""
    video   = sample.video.numpy()          # [T,3,H,W] float32
    T, _, H, W = video.shape
    frames  = [video[t].transpose(1, 2, 0) for t in range(T)]   # list [H,W,3]

    depths  = sample.depths                 # [T,1,H,W] or None
    normals = sample.normals                # [T,3,H,W] or None

    coords  = sample.coords.numpy()         # [Q,2]
    t_src   = sample.t_src.numpy()          # [Q]
    t_tgt   = sample.t_tgt.numpy()          # [Q]
    targets = {k: v.numpy() for k, v in sample.targets.items()}

    # Sub-sample queries
    coords_s, t_src_s, t_tgt_s, tgt_s = sample_query_subset(
        coords, t_src, t_tgt, targets, max_q=max_q)
    Q = coords_s.shape[0]

    # Per-query colors (tab20)
    cmap20 = matplotlib.colormaps.get_cmap("tab20")
    qcolors = (cmap20(np.linspace(0, 1, Q))[:, :3] * 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Build figure rows                                                    #
    # ------------------------------------------------------------------ #
    has_depth   = depths  is not None
    has_normals = normals is not None
    has_tracks  = sample.targets["mask_2d"].any().item()

    row_titles = ["Video frames"]
    rows_data  = [frames]

    # Depth row
    if has_depth:
        depth_np = depths.numpy()[:, 0]           # [T,H,W]
        depth_vis = [colormap_depth(depth_np[t]) for t in range(T)]
        rows_data.append(depth_vis)
        row_titles.append("Depth")

    # Normal row
    if has_normals:
        normal_np = normals.numpy().transpose(0, 2, 3, 1)  # [T,H,W,3]
        normal_vis = [colormap_normal(normal_np[t]) for t in range(T)]
        rows_data.append(normal_vis)
        row_titles.append("Normals")

    # Source query points (one panel per unique source frame)
    src_panels = []
    for t in range(T):
        mask = t_src_s == t
        frame_u8 = to_uint8(frames[t])
        if mask.any():
            frame_u8 = draw_points_on_frame(
                frame_u8, coords_s[mask], qcolors[mask], radius=3)
        src_panels.append(frame_u8)
    rows_data.append(src_panels)
    row_titles.append("Source queries (colored by query idx)")

    # Target pos_2d overlay (green=mask_2d, red=no_mask_2d, blue=mask_3d_only)
    tgt_panels = []
    for t in range(T):
        frame_u8 = to_uint8(frames[t])
        mask_t   = t_tgt_s == t
        if mask_t.any():
            pos2d  = tgt_s["pos_2d"][mask_t]         # [M,2] normalized
            m2d    = tgt_s["mask_2d"][mask_t].astype(bool)
            m3d    = tgt_s["mask_3d"][mask_t].astype(bool)
            pts_colors = np.where(
                m2d[:, None],
                np.array([[0, 220, 0]]),   # green  = 2D valid
                np.where(
                    m3d[:, None],
                    np.array([[0, 120, 255]]),  # blue = 3D only
                    np.array([[220, 0, 0]]),    # red  = invalid
                )
            ).astype(np.uint8)
            frame_u8 = draw_points_on_frame(frame_u8, pos2d, pts_colors, radius=3)
        tgt_panels.append(frame_u8)
    rows_data.append(tgt_panels)
    row_titles.append("Target pos_2d  (green=mask_2d  blue=mask_3d_only  red=invalid)")

    # 3D track projection: re-project pos_3d via intrinsics onto target frame
    if has_tracks:
        K_np = sample.intrinsics.numpy()    # [T,3,3]
        track_panels = []
        for t in range(T):
            frame_u8 = to_uint8(frames[t])
            mask_t   = (t_tgt_s == t) & tgt_s["mask_3d"].astype(bool)
            if mask_t.any():
                p3d = tgt_s["pos_3d"][mask_t]   # [M,3] camera coords
                K   = K_np[t]                   # [3,3]
                # project
                z    = np.clip(p3d[:, 2], 1e-4, None)
                u    = (p3d[:, 0] / z * K[0, 0] + K[0, 2]) / (W - 1)
                v    = (p3d[:, 1] / z * K[1, 1] + K[1, 2]) / (H - 1)
                uv   = np.stack([u, v], axis=1)
                # depth-color: closer=warmer
                d_vals = z
                d_norm = np.clip((d_vals - d_vals.min()) /
                                 (d_vals.max() - d_vals.min() + 1e-6), 0, 1)
                jet = matplotlib.colormaps.get_cmap("jet")
                pt_colors = (jet(d_norm)[:, :3] * 255).astype(np.uint8)
                frame_u8 = draw_points_on_frame(frame_u8, uv, pt_colors, radius=3)
            track_panels.append(frame_u8)
        rows_data.append(track_panels)
        row_titles.append("3D pos_3d reprojected to target frame (jet=depth)")

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #
    n_rows = len(rows_data)
    n_cols = T
    cell_h, cell_w = H, W
    fig_w = n_cols * cell_w / 100
    fig_h = n_rows * cell_h / 100
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.patch.set_facecolor("#111111")

    for r, (row_frames, row_title) in enumerate(zip(rows_data, row_titles)):
        for c in range(n_cols):
            ax = axes[r][c]
            img = row_frames[c] if c < len(row_frames) else np.zeros((H, W, 3), np.uint8)
            # float → uint8
            if img.dtype != np.uint8:
                img = to_uint8(img)
            ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_title(row_title, color="white", fontsize=7,
                             loc="left", pad=3)
            if r == 0:
                ax.set_title(f"t={c}", color="yellow", fontsize=7, pad=2)

    # Header info
    mask_3d_sum = int(sample.targets["mask_3d"].sum().item())
    mask_2d_sum = int(sample.targets["mask_2d"].sum().item())
    total_q     = sample.coords.shape[0]
    info = (f"dataset={sample.dataset_name}  seq={sample.sequence_name}  "
            f"T={T}  Q={total_q}  "
            f"mask_3d={mask_3d_sum}/{total_q}  mask_2d={mask_2d_sum}/{total_q}")
    fig.suptitle(info, color="white", fontsize=9, y=0.995,
                 fontfamily="monospace", backgroundcolor="#333333")

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.995])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/mixture_10datasets.yaml")
    p.add_argument("--output-dir",  default="outputs/vis_mixture_gt")
    p.add_argument("--num-samples", type=int, default=5,
                   help="How many samples to visualize (one PNG each)")
    p.add_argument("--clip-len",    type=int, default=8,
                   help="Frames per clip (override config for faster vis)")
    p.add_argument("--num-queries", type=int, default=512,
                   help="Queries per sample (override config)")
    p.add_argument("--dataset",     default=None,
                   help="Restrict to one dataset name, e.g. pointodyssey")
    p.add_argument("--seed",        type=int, default=42)
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
        print("No adapters loaded — check paths / --dataset filter.")
        sys.exit(1)

    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=args.clip_len,
        img_size=256,
        num_queries=args.num_queries,
        use_augs=False,          # no color aug → easier to inspect GT
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        seed=args.seed,
    )
    print(f"Dataset ready: {dataset.get_dataset_names()}")
    print(f"Visualizing {args.num_samples} samples → {args.output_dir}/\n")

    out_dir = Path(args.output_dir)
    for i in range(args.num_samples):
        sample = dataset[i]
        out_path = out_dir / f"sample_{i:03d}_{sample.dataset_name}.png"
        print(f"[{i}] {sample.dataset_name:20s}  seq={sample.sequence_name[:30]}")
        visualize_sample(sample, out_path, max_q=150)

    print(f"\nDone. Open {out_dir}/ to inspect.")


if __name__ == "__main__":
    main()
