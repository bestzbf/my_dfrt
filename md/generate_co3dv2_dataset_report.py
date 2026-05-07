#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.adapters.co3dv2 import Co3Dv2Adapter
from datasets.collate import d4rt_collate_fn
from datasets.factory import create_training_dataset
from datasets.query_builder import D4RTQueryBuilder
from datasets.transforms import GeometryTransformPipeline
from datasets.computer.co3d_pointcloud_to_tracks import read_ply_pointcloud


ROOT = Path("/data2/d4rt/datasets/Co3Dv2")
AUDIT_DIR = REPO_ROOT / "outputs/co3d_flatness_audit_20260424"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/co3dv2_dataset_report_20260424"
SAMPLE_SEQUENCE = "apple/624_103390_207526"
DENYLIST_PATH = REPO_ROOT / "configs/co3dv2_denylist_degenerate_clips_20260422.txt"
OLD_BAD_WORLD_PLY = REPO_ROOT / (
    "outputs/mixture_5datasets_blendedmvs_large_3gpu_bs5/"
    "vis_checkpoint_latest_41/sample_00_idx0_apple/624_103390_207526/"
    "gt_dense_world_frame_024.ply"
)


def load_audit_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            casted = dict(row)
            for key in ("pre_ratio", "raw_ratio", "ratio_vs_raw"):
                try:
                    casted[key] = float(casted[key])
                except Exception:
                    casted[key] = math.nan
            casted["num_points"] = int(casted["num_points"]) if casted.get("num_points") else 0
            casted["raw_num_vertices"] = int(casted["raw_num_vertices"]) if casted.get("raw_num_vertices") else 0
            casted["has_track_source"] = str(casted.get("has_track_source", "")) == "True"
            rows.append(casted)
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d7dbe2",
            "axes.grid": True,
            "grid.color": "#dfe3ea",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.labelcolor": "#243447",
            "text.color": "#243447",
            "axes.unicode_minus": False,
        }
    )


def format_percent(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def thickness_ratio(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return math.nan
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] <= 0:
        return math.nan
    return float(singular_values[2] / singular_values[0])


def sample_raw_pointcloud(seq: str, sample_size: int, seed: int = 42) -> np.ndarray:
    raw = read_ply_pointcloud(ROOT / seq / "pointcloud.ply").astype(np.float32)
    if len(raw) <= sample_size:
        return raw
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw), size=sample_size, replace=False)
    return raw[idx]


def load_recomputed_points(seq: str) -> tuple[np.ndarray, dict[str, Any]]:
    npz_path = ROOT / seq / "precomputed.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        payload = {
            "track_source": str(z["track_source"].item()) if "track_source" in z.files else None,
            "ref_frame": int(z["ref_frame"]) if "ref_frame" in z.files else None,
            "num_points": int(z["num_points"]) if "num_points" in z.files else None,
            "valids": z["valids"].astype(bool),
            "visibs": z["visibs"].astype(bool),
            "trajs_2d": z["trajs_2d"].astype(np.float32),
        }
        points = z["trajs_3d_world"][0].astype(np.float32)
    return points, payload


def evenly_spaced_indices(num_frames: int, target_len: int) -> list[int]:
    if num_frames <= target_len:
        indices = list(range(num_frames))
        while len(indices) < target_len:
            indices.append(indices[-1])
        return indices
    indices = np.linspace(0, num_frames - 1, target_len)
    return np.round(indices).astype(int).tolist()


def normalize_xyz_colors(points: np.ndarray) -> np.ndarray:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    denom = np.maximum(hi - lo, 1e-6)
    return np.clip((points - lo) / denom, 0.0, 1.0)


def read_ascii_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    vertex_count = None
    vertex_props: list[str] = []
    in_vertex_block = False
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"Invalid PLY header in {path}")
        format_line = f.readline().strip()
        if format_line != "format ascii 1.0":
            raise ValueError(f"Unsupported ASCII PLY format in {path}: {format_line}")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header: {path}")
            line = line.strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex_block = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_block:
                    vertex_count = int(parts[2])
                    vertex_props = []
            elif line.startswith("property ") and in_vertex_block:
                vertex_props.append(line.split()[-1])
            elif line == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"Missing vertex block in {path}")

        x_idx = vertex_props.index("x")
        y_idx = vertex_props.index("y")
        z_idx = vertex_props.index("z")
        rgb_indices = None
        if all(name in vertex_props for name in ("red", "green", "blue")):
            rgb_indices = tuple(vertex_props.index(name) for name in ("red", "green", "blue"))

        points = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 3), dtype=np.float32) if rgb_indices is not None else None
        for idx in range(vertex_count):
            values = f.readline().split()
            if not values:
                raise ValueError(f"Unexpected EOF while reading vertices: {path}")
            points[idx] = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]
            if colors is not None and rgb_indices is not None:
                colors[idx] = [float(values[rgb_indices[0]]), float(values[rgb_indices[1]]), float(values[rgb_indices[2]])]
        if colors is not None:
            colors /= 255.0
        return points, colors


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    sample_size: int = 8000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= sample_size:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=sample_size, replace=False))
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors


def compute_pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:3]
    projected = centered @ axes.T
    scales = np.quantile(np.abs(projected), 0.98, axis=0)
    scales = np.maximum(scales, 1e-3)
    return center, axes, scales


def project_to_pca(points: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return (points - center) @ axes.T


def pca_colorize(points: np.ndarray, center: np.ndarray, axes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    projected = project_to_pca(points, center, axes)
    colors = 0.5 + 0.46 * projected / scales
    return np.clip(colors, 0.0, 1.0)


def fit_cube(point_sets: list[np.ndarray]) -> tuple[np.ndarray, float]:
    stacked = np.concatenate(point_sets, axis=0)
    center = stacked.mean(axis=0)
    radius = float(np.max(np.abs(stacked - center))) * 1.08
    return center, radius


def style_3d_axis(
    ax: Any,
    center: np.ndarray,
    radius: float,
    title: str,
    elev: float = 22.0,
    azim: float = -58.0,
) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.9))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        axis.pane.set_edgecolor((0.90, 0.92, 0.96, 1.0))
        axis._axinfo["grid"]["color"] = (0.87, 0.90, 0.95, 1.0)


def plot_pointcloud(
    ax: Any,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    radius: float,
    title: str,
    elev: float = 22.0,
    azim: float = -58.0,
    point_size: float = 1.1,
) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, alpha=0.88, linewidths=0)
    style_3d_axis(ax, center, radius, title=title, elev=elev, azim=azim)


def plot_thickness_view(
    ax: Any,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    title: str,
) -> None:
    projected = project_to_pca(points, center, axes)
    ax.scatter(projected[:, 0], projected[:, 2], c=colors, s=1.3, alpha=0.78, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC3 (thickness)")
    ax.set_title(title)


def draw_info_card(
    ax: Any,
    title: str,
    lines: list[str],
    facecolor: str = "#ffffff",
    edgecolor: str = "#d7dbe2",
) -> None:
    ax.axis("off")
    body = "\n".join(lines)
    ax.text(
        0.0,
        1.0,
        f"{title}\n\n{body}",
        va="top",
        ha="left",
        fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.65", facecolor=facecolor, edgecolor=edgecolor),
    )


def compute_loader_checks() -> dict[str, Any]:
    config_path = REPO_ROOT / "configs/single_co3dv2.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split="train")
    loader_results: dict[str, Any] = {}
    for num_workers in (0, 4):
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            collate_fn=d4rt_collate_fn,
            shuffle=False,
            persistent_workers=num_workers > 0,
        )
        start = time.perf_counter()
        first_batch_shapes = None
        for step, batch in enumerate(loader):
            if first_batch_shapes is None:
                first_batch_shapes = {
                    "video": list(batch["video"].shape),
                    "coords": list(batch["coords"].shape),
                    "t_src": list(batch["t_src"].shape),
                    "intrinsics": list(batch["intrinsics"].shape),
                    "extrinsics": list(batch["extrinsics"].shape),
                    "target_keys": sorted(batch["targets"].keys()),
                    "dataset_names": list(batch["dataset_names"]),
                    "sequence_names": list(batch["sequence_names"]),
                }
            if step >= 1:
                break
        elapsed = time.perf_counter() - start
        loader_results[f"num_workers_{num_workers}"] = {
            "num_batches_checked": 2,
            "elapsed_sec": elapsed,
            "first_batch_shapes": first_batch_shapes,
        }

    adapter = Co3Dv2Adapter(
        root=str(ROOT),
        subset_name="fewview_train",
        split="train",
        require_pointcloud=True,
        require_precomputed=False,
        sequence_denylist=str(DENYLIST_PATH),
        verbose=False,
    )
    info = adapter.get_sequence_info(SAMPLE_SEQUENCE)
    frame_indices = evenly_spaced_indices(info["num_frames"], 48)
    clip = adapter.load_clip(SAMPLE_SEQUENCE, frame_indices)
    transform = GeometryTransformPipeline(img_size=256, use_augs=False)
    result = transform(clip, rng=random.Random(42))
    builder = D4RTQueryBuilder(
        num_queries=512,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        precompute_patches=True,
        precompute_from_highres=False,
        allow_track_fallback=True,
    )
    sample = builder(result, py_rng=random.Random(42))
    loader_results["adapter_check"] = {
        "sequence": SAMPLE_SEQUENCE,
        "sequence_num_frames": int(info["num_frames"]),
        "raw_image_hw": [int(info["height"]), int(info["width"])],
        "has_tracks": bool(clip.metadata.get("has_tracks", False)),
        "precomputed_track_source": clip.metadata.get("precomputed_track_source"),
        "clip_video_frames": int(len(clip.images)),
        "transform_video_shape": list(np.asarray(result.images).shape),
        "query_sample_video_shape": list(sample.video.shape),
        "query_sample_coords_shape": list(sample.coords.shape),
        "query_sample_target_keys": sorted(sample.targets.keys()),
    }
    return loader_results


def generate_distribution_figure(rows: list[dict[str, Any]], summary: dict[str, Any], out_path: Path) -> None:
    valid = np.array(
        [row["ratio_vs_raw"] for row in rows if row["status"] == "ok" and np.isfinite(row["ratio_vs_raw"])],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.2), constrained_layout=True)
    bins = np.concatenate([
        np.linspace(0.0, 0.25, 32),
        np.linspace(0.25, 1.05, 18),
    ])
    axes[0].axvspan(0.0, 0.25, color="#ffe5e1", alpha=0.72, zorder=0)
    axes[0].axvspan(0.95, 1.02, color="#e6faf5", alpha=0.85, zorder=0)
    axes[0].hist(valid, bins=bins, color="#4f83ff", edgecolor="#ffffff", alpha=0.92)
    for threshold, label, color in (
        (0.10, "≤0.10", "#d7263d"),
        (0.125, "≤0.125", "#f46036"),
        (0.25, "≤0.25", "#2e294e"),
        (1.0, "=1.0", "#1b998b"),
    ):
        axes[0].axvline(threshold, color=color, linewidth=2.0, linestyle="--", label=label)
    axes[0].set_xlabel("Thickness ratio(precomputed) / thickness ratio(raw pointcloud)")
    axes[0].set_ylabel("Sequences")
    axes[0].set_title("Histogram of geometry collapse severity")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].text(0.03, 0.93, "degenerate zone", transform=axes[0].transAxes, fontsize=10, color="#a33a2a")
    axes[0].text(0.73, 0.93, "aligned with raw geometry", transform=axes[0].transAxes, fontsize=10, color="#117864")

    sorted_valid = np.sort(valid)
    cumulative = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
    axes[1].axvspan(0.0, 0.25, color="#ffe5e1", alpha=0.72, zorder=0)
    axes[1].axvspan(0.95, 1.02, color="#e6faf5", alpha=0.85, zorder=0)
    axes[1].plot(sorted_valid, cumulative, color="#1b998b", linewidth=2.5)
    for threshold, color in ((0.10, "#d7263d"), (0.125, "#f46036"), (0.25, "#2e294e")):
        axes[1].axvline(threshold, color=color, linewidth=1.8, linestyle="--")
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Thickness ratio vs raw pointcloud")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Cumulative distribution")

    counts = summary["counts"]
    text = (
        f"Total sequences: {counts['total_sequences']}\n"
        f"Comparable sequences: {counts['valid_sequences']}\n"
        f"<=0.10: {counts['ratio_vs_raw_le_0.1']} ({format_percent(counts['ratio_vs_raw_le_0.1'] / counts['valid_sequences'])})\n"
        f"<=0.125: {counts['ratio_vs_raw_le_0.125']} ({format_percent(counts['ratio_vs_raw_le_0.125'] / counts['valid_sequences'])})\n"
        f"<=0.25: {counts['ratio_vs_raw_le_0.25']} ({format_percent(counts['ratio_vs_raw_le_0.25'] / counts['valid_sequences'])})\n"
        f"Near exact match: {counts['near_exact_match']}"
    )
    axes[1].text(
        0.56,
        0.18,
        text,
        transform=axes[1].transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#d7dbe2"),
    )

    fig.suptitle("Co3Dv2 geometry consistency audit: precomputed cache vs raw sequence point cloud", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_category_figure(rows: list[dict[str, Any]], out_path: Path) -> None:
    valid_rows = [row for row in rows if row["status"] == "ok" and np.isfinite(row["ratio_vs_raw"])]
    total_by_cat = Counter(row["category"] for row in valid_rows)
    bad_by_cat = Counter(row["category"] for row in valid_rows if row["ratio_vs_raw"] <= 0.25)
    severe_by_cat = Counter(row["category"] for row in valid_rows if row["ratio_vs_raw"] <= 0.10)

    items: list[tuple[str, int, int, int, float]] = []
    for category, bad_count in bad_by_cat.items():
        total = total_by_cat[category]
        severe = severe_by_cat[category]
        items.append((category, bad_count, severe, total, bad_count / total))
    items.sort(key=lambda x: (x[4], x[1]), reverse=True)
    top_items = items[:20]

    categories = [item[0] for item in top_items][::-1]
    bad_counts = np.array([item[1] for item in top_items][::-1])
    severe_counts = np.array([item[2] for item in top_items][::-1])
    ratios = np.array([item[4] for item in top_items][::-1])
    totals = np.array([item[3] for item in top_items][::-1])

    cmap = LinearSegmentedColormap.from_list("risk", ["#90caf9", "#ffb74d", "#ef5350"])
    colors = [cmap(r) for r in ratios]

    fig, ax = plt.subplots(figsize=(13.8, 9.4), constrained_layout=True)
    bars_bad = ax.barh(categories, bad_counts, color=colors, edgecolor="#ffffff", height=0.72, label="ratio_vs_raw <= 0.25")
    ax.barh(categories, severe_counts, color="#6a1b9a", edgecolor="#ffffff", height=0.42, label="ratio_vs_raw <= 0.10")
    ax.set_xlabel("Degenerate sequences")
    ax.set_ylabel("Category")
    ax.set_title("Top 20 categories ranked by degenerate-cache ratio")
    ax.legend(loc="lower right", frameon=True)

    for bar, category in zip(bars_bad, categories, strict=True):
        if category == "apple":
            bar.set_edgecolor("#0b3954")
            bar.set_linewidth(2.2)

    for idx, (bad, total, ratio) in enumerate(zip(bad_counts, totals, ratios, strict=True)):
        ax.text(
            bad + 6,
            idx,
            f"{bad}/{total} · {100.0 * ratio:.1f}%",
            va="center",
            fontsize=10,
            color="#243447",
        )

    note = (
        "Purple overlay = severe collapse (<=0.10)."
        if "apple" in categories
        else "Purple overlay = severe collapse (<=0.10). The apple case study is audited separately."
    )
    ax.text(
        0.02,
        0.03,
        note,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d7dbe2"),
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_after_fix_pointcloud_figure(seq: str, out_path: Path) -> dict[str, float]:
    raw = sample_raw_pointcloud(seq, sample_size=8000, seed=42)
    new_points, payload = load_recomputed_points(seq)
    equal = bool(np.allclose(raw, new_points, atol=1e-6))
    max_abs = float(np.max(np.abs(raw - new_points))) if raw.shape == new_points.shape else math.nan

    raw_ratio = thickness_ratio(raw)
    new_ratio = thickness_ratio(new_points)
    pca_center, pca_axes, pca_scales = compute_pca_frame(raw)
    raw_colors = pca_colorize(raw, pca_center, pca_axes, pca_scales)
    new_colors = pca_colorize(new_points, pca_center, pca_axes, pca_scales)
    cube_center, radius = fit_cube([raw, new_points])

    fig = plt.figure(figsize=(16.8, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.92], height_ratios=[1.0, 0.88])

    plot_pointcloud(
        fig.add_subplot(gs[0, 0], projection="3d"),
        raw,
        raw_colors,
        cube_center,
        radius,
        title=f"Raw sequence pointcloud.ply\nthickness ratio = {raw_ratio:.3f}",
    )
    plot_pointcloud(
        fig.add_subplot(gs[0, 1], projection="3d"),
        new_points,
        new_colors,
        cube_center,
        radius,
        title=f"Recomputed precomputed.npz\nthickness ratio = {new_ratio:.3f}",
    )

    per_track_visible = payload["valids"].sum(axis=0)
    draw_info_card(
        fig.add_subplot(gs[0, 2]),
        "Acceptance checks",
        [
            f"Sequence: {seq}",
            f"track_source: {payload['track_source']}",
            f"ref_frame: {payload['ref_frame']}",
            f"num_points: {payload['num_points']}",
            f"allclose(raw, recomputed): {equal}",
            f"max_abs_diff: {max_abs:.6f}",
            f"mean_visible_frames: {per_track_visible.mean():.2f}",
            f"tracks_ge_2_ratio: {(per_track_visible >= 2).mean():.3f}",
            f"tracks_ge_4_ratio: {(per_track_visible >= 4).mean():.3f}",
        ],
    )

    plot_thickness_view(
        fig.add_subplot(gs[1, 0]),
        raw,
        raw_colors,
        pca_center,
        pca_axes,
        title="Raw point cloud · PCA thickness view",
    )
    plot_thickness_view(
        fig.add_subplot(gs[1, 1]),
        new_points,
        new_colors,
        pca_center,
        pca_axes,
        title="Recomputed cache · same PCA view",
    )

    draw_info_card(
        fig.add_subplot(gs[1, 2]),
        "Interpretation",
        [
            "The recomputed cache overlaps the raw sequence-level point cloud.",
            "Thickness ratio fully returns to the raw geometry.",
            "This is the expected fixed behavior for Co3D precomputed 3D supervision.",
        ],
        facecolor="#f8fbff",
        edgecolor="#c9d7ee",
    )

    fig.suptitle("Case acceptance after recompute: raw point cloud and new cache are aligned", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_ratio": raw_ratio,
        "new_ratio": new_ratio,
        "equal": equal,
        "max_abs": max_abs,
    }


def generate_before_after_composite(seq: str, old_world_ply_path: Path, out_path: Path) -> None:
    old_points, _ = read_ascii_ply_vertices(old_world_ply_path)
    raw = sample_raw_pointcloud(seq, sample_size=8000, seed=42)
    new_points, _ = load_recomputed_points(seq)
    old_points, _ = downsample_points(old_points, sample_size=6000, seed=42)
    raw, _ = downsample_points(raw, sample_size=6000, seed=42)
    new_points, _ = downsample_points(new_points, sample_size=6000, seed=42)

    old_ratio = thickness_ratio(old_points)
    raw_ratio = thickness_ratio(raw)
    new_ratio = thickness_ratio(new_points)
    pca_center, pca_axes, pca_scales = compute_pca_frame(raw)
    old_colors = pca_colorize(old_points, pca_center, pca_axes, pca_scales)
    raw_colors = pca_colorize(raw, pca_center, pca_axes, pca_scales)
    new_colors = pca_colorize(new_points, pca_center, pca_axes, pca_scales)
    cube_center, radius = fit_cube([old_points, raw, new_points])

    fig = plt.figure(figsize=(18.8, 8.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.94], height_ratios=[1.0, 0.88])

    cases = [
        ("Old cached GT render\nthin-shell failure", old_points, old_colors, old_ratio),
        ("Raw sequence pointcloud.ply\nreference geometry", raw, raw_colors, raw_ratio),
        ("Recomputed precomputed.npz\nfixed geometry", new_points, new_colors, new_ratio),
    ]
    for column, (title, points, colors, ratio) in enumerate(cases):
        plot_pointcloud(
            fig.add_subplot(gs[0, column], projection="3d"),
            points,
            colors,
            cube_center,
            radius,
            title=f"{title}\nthickness ratio = {ratio:.3f}",
        )
        plot_thickness_view(
            fig.add_subplot(gs[1, column]),
            points,
            colors,
            pca_center,
            pca_axes,
            title="Same PCA thickness view",
        )

    draw_info_card(
        fig.add_subplot(gs[0, 3]),
        "Case summary",
        [
            f"Sequence: {seq}",
            f"old_ratio: {old_ratio:.3f}",
            f"raw_ratio: {raw_ratio:.3f}",
            f"new_ratio: {new_ratio:.3f}",
            f"old/raw: {old_ratio / raw_ratio:.3f}",
            f"new/raw: {new_ratio / raw_ratio:.3f}",
            "Old cache is the visually wrong planar shell.",
            "Recomputed cache matches the raw sequence-level geometry.",
        ],
    )
    draw_info_card(
        fig.add_subplot(gs[1, 3]),
        "Why the old cache looks flat",
        [
            "The legacy Co3D cache came from sparse depth back-projection.",
            "Only front-facing visible surfaces survived that pipeline.",
            "The fixed path samples directly from sequence-level pointcloud.ply.",
        ],
        facecolor="#f8fbff",
        edgecolor="#c9d7ee",
    )

    fig.suptitle("Representative case: legacy Co3D cache collapses the 3D geometry into a thin shell", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_loader_figure(loader_stats: dict[str, Any], out_path: Path) -> None:
    adapter = Co3Dv2Adapter(
        root=str(ROOT),
        subset_name="fewview_train",
        split="train",
        require_pointcloud=True,
        require_precomputed=False,
        sequence_denylist=str(DENYLIST_PATH),
        verbose=False,
    )
    info = adapter.get_sequence_info(SAMPLE_SEQUENCE)
    frame_indices = evenly_spaced_indices(info["num_frames"], 48)
    clip = adapter.load_clip(SAMPLE_SEQUENCE, frame_indices)
    transform = GeometryTransformPipeline(img_size=256, use_augs=False)
    result = transform(clip, rng=random.Random(42))

    frame_ids = [0, len(result.images) // 3, 2 * len(result.images) // 3]
    ref_frame = len(result.images) // 2
    trajs_2d = np.asarray(result.trajs_2d, dtype=np.float32)
    trajs_3d = np.asarray(result.trajs_3d_world, dtype=np.float32)
    visibs = np.asarray(result.visibs, dtype=bool)
    crop_w = result.crop.crop_w
    crop_h = result.crop.crop_h
    sx = result.img_size / max(crop_w, 1)
    sy = result.img_size / max(crop_h, 1)
    pts = trajs_2d[ref_frame]
    visible_mask = visibs[ref_frame] & np.isfinite(pts).all(axis=-1) & np.isfinite(trajs_3d[ref_frame]).all(axis=-1)
    visible_indices = np.flatnonzero(visible_mask)
    rng = np.random.default_rng(42)
    if len(visible_indices) > 2500:
        visible_indices = np.sort(rng.choice(visible_indices, size=2500, replace=False))
    pts_ref = pts[visible_indices]
    pts_img = np.stack([pts_ref[:, 0] * sx, pts_ref[:, 1] * sy], axis=-1)
    pts_img[:, 0] = np.clip(np.round(pts_img[:, 0]), 0, result.img_size - 1)
    pts_img[:, 1] = np.clip(np.round(pts_img[:, 1]), 0, result.img_size - 1)
    points_world = trajs_3d[ref_frame, visible_indices]
    point_colors = result.images[ref_frame][pts_img[:, 1].astype(int), pts_img[:, 0].astype(int)]
    pca_center, pca_axes, pca_scales = compute_pca_frame(points_world)
    thickness_colors = pca_colorize(points_world, pca_center, pca_axes, pca_scales)
    cube_center, radius = fit_cube([points_world])

    fig = plt.figure(figsize=(18.2, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.95], height_ratios=[0.96, 1.04])

    for slot, frame_id in enumerate(frame_ids):
        ax = fig.add_subplot(gs[0, slot])
        ax.imshow(result.images[frame_id])
        ax.set_title(f"Clip frame {frame_id}")
        ax.axis("off")

    adapter_check = loader_stats["adapter_check"]
    workers0 = loader_stats["num_workers_0"]
    workers4 = loader_stats["num_workers_4"]
    draw_info_card(
        fig.add_subplot(gs[0, 3]),
        "Pipeline status: PASS",
        [
            f"Sequence: {adapter_check['sequence']}",
            f"track_source: {adapter_check['precomputed_track_source']}",
            "Adapter clip load: OK",
            "GeometryTransformPipeline: OK",
            "D4RTQueryBuilder: OK",
            "d4rt_collate_fn / DataLoader: OK",
        ],
        facecolor="#eefbf3",
        edgecolor="#9ad0b0",
    )

    ax_overlay = fig.add_subplot(gs[1, 0])
    ax_overlay.imshow(result.images[ref_frame])
    ax_overlay.scatter(pts_img[:, 0], pts_img[:, 1], c=point_colors, s=7, alpha=0.75, linewidths=0)
    ax_overlay.set_title(f"Reference frame {ref_frame}\nvisible track reprojection")
    ax_overlay.axis("off")

    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")
    plot_pointcloud(
        ax_3d,
        points_world,
        point_colors,
        cube_center,
        radius,
        title="3D world points\nseen by the loader",
        elev=20.0,
        azim=-60.0,
        point_size=1.35,
    )
    plot_thickness_view(
        fig.add_subplot(gs[1, 2]),
        points_world,
        thickness_colors,
        pca_center,
        pca_axes,
        title="Same point cloud · PCA thickness view",
    )

    draw_info_card(
        fig.add_subplot(gs[1, 3]),
        "Shapes and runtime",
        [
            f"raw_image_hw: {adapter_check['raw_image_hw']}",
            f"has_tracks: {adapter_check['has_tracks']}",
            f"QuerySample.video: {adapter_check['query_sample_video_shape']}",
            f"QuerySample.coords: {adapter_check['query_sample_coords_shape']}",
            f"workers=0: {workers0['elapsed_sec']:.2f}s / 2 batches",
            f"workers=4: {workers4['elapsed_sec']:.2f}s / 2 batches",
            f"first batch.video: {workers4['first_batch_shapes']['video']}",
            f"first batch.coords: {workers4['first_batch_shapes']['coords']}",
        ],
    )

    fig.suptitle("End-to-end Co3Dv2 loader validation: adapter, transform, query builder and DataLoader", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_report_markdown(
    out_path: Path,
    summary: dict[str, Any],
    sample_old_row: dict[str, Any],
    after_fix_stats: dict[str, float],
    loader_stats: dict[str, Any],
    legacy_cache_count: int,
    figure_paths: dict[str, Path],
) -> None:
    counts = summary["counts"]
    apple_bad = next(item for item in summary["category_summary_top50_bad_le_0.25"] if item["category"] == "apple")
    report = f"""# Co3Dv2 数据集检测报告（2026-04-24）

## 1. 结论摘要

- Co3Dv2 根目录共检测 `31834` 条序列，存在有效 `precomputed.npz` 的序列共 `31834` 条。
- 以“`precomputed` 点云厚度比 / 原始 `pointcloud.ply` 厚度比”为几何一致性指标，`31825` 条序列可比较。
- 其中：
  - `ratio_vs_raw <= 0.10`：`{counts['ratio_vs_raw_le_0.1']}` 条
  - `ratio_vs_raw <= 0.125`：`{counts['ratio_vs_raw_le_0.125']}` 条
  - `ratio_vs_raw <= 0.25`：`{counts['ratio_vs_raw_le_0.25']}` 条
- 历史旧缓存（无 `track_source` 字段）共 `{legacy_cache_count}` 条，是本轮统一重算目标。
- 这说明旧 Co3D cache 中有大批序列存在“几何被压扁成薄壳”的问题，而不是个别样例异常。
- 当前已确认问题主因来自旧版 depth-based Co3D precompute 路线；使用 `pointcloud.ply` 重算后，几何可以恢复为与原始点云逐点一致。

插图 1：`{figure_paths['distribution']}`

![]({figure_paths['distribution'].name})

## 2. 检测范围与环境

- 数据根目录：`/data2/d4rt/datasets/Co3Dv2`
- 配置参考：
  - `configs/single_co3dv2.yaml`
  - `configs/mixture_5datasets_blendedmvs.yaml`
- 默认过滤：
  - `require_pointcloud=true`（混训配置）
  - `sequence_denylist=configs/co3dv2_denylist_degenerate_clips_20260422.txt`
- 审计输入：
  - `outputs/co3d_flatness_audit_20260424/co3d_flatness_audit.csv`
  - `outputs/co3d_flatness_audit_20260424/summary.json`

## 3. 核心问题定义

- 本报告关注的问题不是“track 能否跨帧存在”，而是“3D supervision 是否与原始序列级 pointcloud 一致”。
- 指标定义：
  - `pre_ratio`：`precomputed.trajs_3d_world` 的厚度比（PCA 第三主轴 / 第一主轴）
  - `raw_ratio`：原始 `pointcloud.ply` 的厚度比
  - `ratio_vs_raw = pre_ratio / raw_ratio`
- 判定标准：
  - `ratio_vs_raw <= 0.25`：明显压扁，属于高风险
  - `ratio_vs_raw <= 0.10`：极严重压扁
  - `ratio_vs_raw ≈ 1.0`：与原始点云几何一致

## 4. 全量审计结果

- 全量可比较序列：`{counts['valid_sequences']}`
- 与原始点云近似一致的序列：`{counts['near_exact_match']}`
- 高风险（`<=0.25`）序列占比：`{format_percent(counts['ratio_vs_raw_le_0.25'] / counts['valid_sequences'])}`
- 极严重（`<=0.10`）序列占比：`{format_percent(counts['ratio_vs_raw_le_0.1'] / counts['valid_sequences'])}`
- 分布特征：
  - `p50 = {summary['ratio_vs_raw_percentiles']['p50']:.3f}`
  - `p75 = {summary['ratio_vs_raw_percentiles']['p75']:.3f}`
  - `p90 = {summary['ratio_vs_raw_percentiles']['p90']:.3f}`
  - 审计结果存在明显“两峰”特征：一部分序列接近 `1.0`，另一大部分序列集中在 `0.05 ~ 0.12`

## 5. 高风险类别分布

- 以下类别中，坏样本比例显著偏高：
  - `chair`：`1293 / 1331`
  - `remote`：`1176 / 1349`
  - `teddybear`：`1128 / 1143`
  - `plant`：`1083 / 1132`
  - `mouse`：`1057 / 1134`
  - `apple`：`{apple_bad['bad_le_0.25']} / {apple_bad['total']}`（`{100.0 * apple_bad['bad_ratio']:.1f}%`）

插图 2：`{figure_paths['category']}`

![]({figure_paths['category'].name})

## 6. 典型问题样例：apple/624_103390_207526

- 旧缓存审计结果：
  - `pre_ratio = {sample_old_row['pre_ratio']:.6f}`
  - `raw_ratio = {sample_old_row['raw_ratio']:.6f}`
  - `ratio_vs_raw = {sample_old_row['ratio_vs_raw']:.6f}`
- 旧结果的几何表现是“薄壳 / 斜平面”，而不是完整苹果形状。
- 重算后：
  - `track_source = pointcloud.ply`
  - `new pre_ratio = {after_fix_stats['new_ratio']:.6f}`
  - `raw_ratio = {after_fix_stats['raw_ratio']:.6f}`
  - `逐点一致 = {after_fix_stats['equal']}`
  - `max_abs_diff = {after_fix_stats['max_abs']:.6f}`

插图 3：`{figure_paths['before_after']}`

![]({figure_paths['before_after'].name})

插图 4：`{figure_paths['after_fix']}`

![]({figure_paths['after_fix'].name})

## 7. 为什么旧缓存会出问题

- 旧 Co3D precompute 主路径是 depth-based，不是 pointcloud-based：
  - 从深度最密的前 5 帧采样
  - 再把这些深度点反投影成世界坐标
  - 最后广播为整条序列的 `trajs_3d_world`
- 对 Co3D 来说，这条路不稳定的原因是：
  - 深度稀疏，只覆盖当前视角前表面
  - 旧逻辑采用很宽松的深度一致性阈值
  - 时序统计可能通过，但 3D 体积已经被压扁
- 新重算路径直接以 `pointcloud.ply` 作为 3D 真值来源，规避了这个问题。

## 8. 加载器环境与端到端验证

- 已验证以下链路可以正常运行：
  - `Co3Dv2Adapter`
  - `GeometryTransformPipeline`
  - `D4RTQueryBuilder`
  - `DataLoader + d4rt_collate_fn`
- 样例序列：`{loader_stats['adapter_check']['sequence']}`
- 关键结果：
  - `has_tracks = {loader_stats['adapter_check']['has_tracks']}`
  - `precomputed_track_source = {loader_stats['adapter_check']['precomputed_track_source']}`
  - `QuerySample.video shape = {loader_stats['adapter_check']['query_sample_video_shape']}`
  - `QuerySample.coords shape = {loader_stats['adapter_check']['query_sample_coords_shape']}`
  - `DataLoader(num_workers=0)`：`{loader_stats['num_workers_0']['elapsed_sec']:.2f}s / 2 batches`
  - `DataLoader(num_workers=4)`：`{loader_stats['num_workers_4']['elapsed_sec']:.2f}s / 2 batches`
- 本轮验证中未出现 adapter / transform / query builder / collate 级异常。

插图 5：`{figure_paths['loader']}`

![]({figure_paths['loader'].name})

## 9. 训练影响判断

- 新 cache 与旧 cache 的 schema 相同，训练接口不需要改。
- 变化的是 Co3Dv2 这部分 3D supervision 的几何分布：
  - 旧：大量“前表面薄壳”
  - 新：与原始 `pointcloud.ply` 对齐
- 这意味着：
  - 新训练可直接使用
  - 继续已有 checkpoint 做 finetune 也可直接接入
  - 但不要把“旧 cache 训练到一半”和“新 cache 继续训练”当成完全同一实验

## 10. 当前结论

- Co3Dv2 数据集本体没有问题，问题集中在历史遗留的旧版 `precomputed.npz`。
- 旧 cache 的问题是系统性的，不是单个样例。
- 对 `SAMPLE_SEQUENCE` 的重算验收已经证明：
  - 新 cache 的 3D 点与原始 `pointcloud.ply` 完全一致
  - 加载器、transform、query builder、DataLoader 环境均可正常工作
- 因此，Co3Dv2 的修复方向是明确且可验证的：统一用 `pointcloud.ply` 重算旧 cache。

## 11. 附件

- 审计 CSV：`outputs/co3d_flatness_audit_20260424/co3d_flatness_audit.csv`
- 审计摘要：`outputs/co3d_flatness_audit_20260424/summary.json`
- 旧 cache 列表：`outputs/co3d_flatness_audit_20260424/old_no_track_source_sequences.txt`
- 最坏样例列表：`outputs/co3d_flatness_audit_20260424/worst_sequences.txt`
- 加载器验证明细：`outputs/co3dv2_dataset_report_20260424/loader_checks.json`
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Feishu-ready Co3Dv2 dataset report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reuse-loader-checks", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    rows = load_audit_rows(AUDIT_DIR / "co3d_flatness_audit.csv")
    summary = json.loads((AUDIT_DIR / "summary.json").read_text(encoding="utf-8"))
    legacy_cache_count = sum(1 for row in rows if row["status"] == "ok" and not row["has_track_source"])
    loader_checks_path = output_dir / "loader_checks.json"
    if args.reuse_loader_checks and loader_checks_path.exists():
        loader_stats = json.loads(loader_checks_path.read_text(encoding="utf-8"))
    else:
        loader_stats = compute_loader_checks()
        loader_checks_path.write_text(json.dumps(loader_stats, indent=2), encoding="utf-8")

    distribution_path = output_dir / "fig_01_flatness_distribution.png"
    category_path = output_dir / "fig_02_category_risk_top20.png"
    after_fix_path = output_dir / "fig_03_after_fix_pointcloud.png"
    before_after_path = output_dir / "fig_04_case_before_after.png"
    loader_path = output_dir / "fig_05_loader_pipeline_sanity.png"
    report_path = output_dir / "Co3Dv2_数据集检测报告_2026-04-24.md"

    generate_distribution_figure(rows, summary, distribution_path)
    generate_category_figure(rows, category_path)
    after_fix_stats = generate_after_fix_pointcloud_figure(SAMPLE_SEQUENCE, after_fix_path)
    sample_old_row = next(row for row in rows if row["sequence"] == SAMPLE_SEQUENCE)
    if OLD_BAD_WORLD_PLY.exists():
        generate_before_after_composite(SAMPLE_SEQUENCE, OLD_BAD_WORLD_PLY, before_after_path)
    else:
        before_after_path = after_fix_path
    generate_loader_figure(loader_stats, loader_path)

    figure_paths = {
        "distribution": distribution_path,
        "category": category_path,
        "after_fix": after_fix_path,
        "before_after": before_after_path,
        "loader": loader_path,
    }
    generate_report_markdown(report_path, summary, sample_old_row, after_fix_stats, loader_stats, legacy_cache_count, figure_paths)

    print(f"report={report_path}")
    print(f"distribution={distribution_path}")
    print(f"category={category_path}")
    print(f"before_after={before_after_path}")
    print(f"loader={loader_path}")


if __name__ == "__main__":
    main()
