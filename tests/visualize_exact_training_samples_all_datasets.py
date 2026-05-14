#!/usr/bin/env python3
"""Visualize exact training-loader samples for multiple datasets.

This follows the same diagnostic pattern as the Waymo exact-sample visualizer:
instantiate the real adapter + MixtureDataset path, capture the TransformResult
seen by QueryBuilder, then render sampled 2D/3D supervision and consistency
statistics.  It is intended for quick regression checks when adapter or
precomputed-track logic changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.factory import create_training_dataset  # noqa: E402


VIS_REASON_ORDER = [
    "visible",
    "occluded",
    "out_of_view",
    "depth_match_hidden",
    "missing_depth",
    "depth_mismatch",
    "invalid_3d",
]
VIS_REASON_LABELS = {
    "visible": "visible",
    "occluded": "occluded",
    "out_of_view": "out",
    "depth_match_hidden": "dm-hidden",
    "missing_depth": "no depth",
    "depth_mismatch": "d-mismatch",
    "invalid_3d": "invalid",
}
VIS_REASON_SHORT = {
    "visible": "vis",
    "occluded": "occ",
    "out_of_view": "out",
    "depth_match_hidden": "dmh",
    "missing_depth": "nod",
    "depth_mismatch": "dif",
    "invalid_3d": "inv",
}
VIS_REASON_COLORS = {
    "visible": (39, 174, 96),
    "occluded": (230, 92, 40),
    "out_of_view": (52, 120, 230),
    "depth_match_hidden": (241, 196, 15),
    "missing_depth": (142, 68, 173),
    "depth_mismatch": (213, 45, 125),
    "invalid_3d": (0, 150, 150),
}
VIS_REASON_CODES = {name: idx for idx, name in enumerate(VIS_REASON_ORDER)}


class CapturingQueryBuilder:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_result = None

    def __call__(self, result: Any, *args: Any, **kwargs: Any) -> Any:
        self.last_result = result
        return self.inner(result, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class CapturingAdapter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_sequence_name: str | None = None
        self.last_frame_indices: list[int] | None = None
        self.last_clip = None

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> Any:
        self.last_sequence_name = sequence_name
        self.last_frame_indices = list(frame_indices)
        self.last_clip = self.inner.load_clip(sequence_name, frame_indices)
        return self.last_clip

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def percentiles(values: np.ndarray, qs: list[float]) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0 for _ in qs]
    return [float(np.percentile(values, q)) for q in qs]


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    arr = video.detach().cpu()
    if arr.dtype == torch.uint8:
        out = arr.permute(0, 2, 3, 1).numpy()
    else:
        out = arr.float().permute(0, 2, 3, 1).numpy()
        if out.size and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def norm_to_px(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    xy = np.where(np.isfinite(xy), xy, 0.0)
    out = np.empty_like(xy, dtype=np.int32)
    out[..., 0] = np.clip(np.round(xy[..., 0] * float(width - 1)), 0, width - 1)
    out[..., 1] = np.clip(np.round(xy[..., 1] * float(height - 1)), 0, height - 1)
    return out


def save_grid(images: list[Image.Image], out_path: Path, title: str, cols: int = 4) -> None:
    if not images:
        return
    w, h = images[0].size
    label_h = 22
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), title, fill=(0, 0, 0))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * (h + label_h) + label_h
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def draw_query_sources(sample: Any, frames: np.ndarray, out_path: Path) -> dict[str, Any]:
    t_src = sample.t_src.detach().cpu().numpy().astype(np.int64)
    coords = sample.coords.detach().cpu().numpy().astype(np.float32)
    pts = norm_to_px(coords, frames.shape[2], frames.shape[1])
    frame_ids = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int)
    thumbs = []
    counts = []
    for frame_id in frame_ids:
        img = Image.fromarray(frames[frame_id]).convert("RGB")
        draw = ImageDraw.Draw(img)
        q = t_src == frame_id
        counts.append(int(q.sum()))
        for x, y in pts[q]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(31, 119, 180))
        draw.rectangle((0, 0, img.width, 22), fill=(255, 255, 255))
        draw.text((4, 4), f"source local={frame_id} queries={int(q.sum())}", fill=(0, 0, 0))
        thumbs.append(img)
    save_grid(thumbs, out_path, "exact training source queries", cols=4)
    return {"shown_frame_counts": counts}


def draw_query_targets(sample: Any, frames: np.ndarray, out_path: Path) -> dict[str, Any]:
    t_tgt = sample.t_tgt.detach().cpu().numpy().astype(np.int64)
    mask = sample.targets["mask_2d"].detach().cpu().numpy().astype(bool)
    pts = norm_to_px(
        sample.targets["pos_2d"].detach().cpu().numpy().astype(np.float32),
        frames.shape[2],
        frames.shape[1],
    )
    frame_ids = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int)
    thumbs = []
    counts = []
    for frame_id in frame_ids:
        img = Image.fromarray(frames[frame_id]).convert("RGB")
        draw = ImageDraw.Draw(img)
        q = (t_tgt == frame_id) & mask
        counts.append(int(q.sum()))
        for x, y in pts[q]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(214, 98, 53))
        draw.rectangle((0, 0, img.width, 22), fill=(255, 255, 255))
        draw.text((4, 4), f"target local={frame_id} mask_2d={int(q.sum())}", fill=(0, 0, 0))
        thumbs.append(img)
    save_grid(thumbs, out_path, "exact training target queries", cols=4)
    return {"shown_frame_counts": counts}


def scaled_tracks(result: Any, point_indices: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(result.trajs_2d[:, point_indices], dtype=np.float32)
    scale = np.array(
        [
            float(width - 1) / float(max(result.crop.crop_w - 1, 1)),
            float(height - 1) / float(max(result.crop.crop_h - 1, 1)),
        ],
        dtype=np.float32,
    )
    pts_img = pts * scale[None, None, :]
    world = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    vis = (
        np.asarray(result.valids[:, point_indices]).astype(bool)
        & np.asarray(result.visibs[:, point_indices]).astype(bool)
        & np.isfinite(pts_img).all(axis=-1)
        & np.isfinite(world).all(axis=-1)
    )
    return pts_img, vis


def select_render_points(point_indices: np.ndarray, max_render: int, seed: int = 42) -> np.ndarray:
    point_indices = np.asarray(point_indices, dtype=np.int64)
    if len(point_indices) <= max_render:
        return point_indices
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(len(point_indices)), size=max_render, replace=False)
    return point_indices[np.sort(ids)]


def draw_all_query_tracks(
    sample: Any,
    result: Any,
    frames: np.ndarray,
    out_gif: Path,
    out_sheet: Path,
    max_render: int,
) -> dict[str, Any]:
    point_indices_all = sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64)
    pts_all, vis_all = scaled_tracks(result, point_indices_all, frames.shape[2], frames.shape[1])
    counts = vis_all.sum(axis=1)

    point_indices = select_render_points(point_indices_all, max_render)
    pts, vis = scaled_tracks(result, point_indices, frames.shape[2], frames.shape[1])
    out_frames = []
    for t in range(len(frames)):
        img = Image.fromarray(frames[t]).convert("RGB")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        start = max(0, t - 8)
        for q in range(len(point_indices)):
            last = None
            for tt in range(start, t + 1):
                if not vis[tt, q]:
                    last = None
                    continue
                cur = (float(pts[tt, q, 0]), float(pts[tt, q, 1]))
                if last is not None:
                    draw.line([last, cur], fill=(31, 119, 180, 100), width=1)
                last = cur
            if vis[t, q]:
                x, y = pts[t, q]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(31, 119, 180, 210))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, img.width, 22), fill=(255, 255, 255))
        d.text((4, 4), f"local={t} stored tracks visible={int(vis_all[t].sum())}", fill=(0, 0, 0))
        out_frames.append(img)
    out_frames[0].save(out_gif, save_all=True, append_images=out_frames[1:], duration=420, loop=0, optimize=False)
    ids = np.linspace(0, len(out_frames) - 1, num=min(12, len(out_frames)), dtype=int)
    save_grid([out_frames[i] for i in ids], out_sheet, "stored 2D query tracks", cols=4)
    return {
        "rendered_query_entries": int(len(point_indices)),
        "visible_entries_per_frame_min": int(counts.min()),
        "visible_entries_per_frame_p50": float(np.percentile(counts, 50)),
        "visible_entries_per_frame_max": int(counts.max()),
    }


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return (extrinsic[:3, :3].astype(np.float32) @ points.T).T + extrinsic[:3, 3].astype(np.float32)


def project_world(points_world: np.ndarray, intrinsics: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = world_to_camera(points_world, extrinsic)
    z = cam[:, 2].astype(np.float32)
    uv = np.full((len(cam), 2), np.nan, dtype=np.float32)
    ok = z > 1e-6
    if ok.any():
        k = np.asarray(intrinsics, dtype=np.float32)
        uv[ok, 0] = cam[ok, 0] / z[ok] * k[0, 0] + k[0, 2]
        uv[ok, 1] = cam[ok, 1] / z[ok] * k[1, 1] + k[1, 2]
    return uv, z


def projected_3d_tracks(
    result: Any,
    point_indices: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    pts = np.full((trajs.shape[0], trajs.shape[1], 2), np.nan, dtype=np.float32)
    zvals = np.full((trajs.shape[0], trajs.shape[1]), np.nan, dtype=np.float32)
    for t in range(trajs.shape[0]):
        pts[t], zvals[t] = project_world(trajs[t], result.intrinsics[t], result.extrinsics[t])
    finite = np.isfinite(trajs).all(axis=-1) & np.isfinite(pts).all(axis=-1)
    in_bounds = (pts[..., 0] >= 0) & (pts[..., 0] < width) & (pts[..., 1] >= 0) & (pts[..., 1] < height)
    vis = (
        np.asarray(result.valids[:, point_indices]).astype(bool)
        & np.asarray(result.visibs[:, point_indices]).astype(bool)
        & finite
        & in_bounds
        & (zvals > 1e-6)
    )
    stored, stored_vis = scaled_tracks(result, point_indices, width, height)
    common = vis & stored_vis & np.isfinite(stored).all(axis=-1)
    stats: dict[str, Any] = {}
    if common.any():
        err = np.linalg.norm(pts[common] - stored[common], axis=-1)
        stats.update(
            {
                "projected_vs_stored_2d_error_px_p50": float(np.percentile(err, 50)),
                "projected_vs_stored_2d_error_px_p95": float(np.percentile(err, 95)),
                "projected_vs_stored_2d_error_px_max": float(np.max(err)),
                "projected_vs_stored_2d_common_entries": int(common.sum()),
            }
        )
    counts = vis.sum(axis=1)
    stats.update(
        {
            "projected_visible_entries_per_frame_min": int(counts.min()),
            "projected_visible_entries_per_frame_p50": float(np.percentile(counts, 50)),
            "projected_visible_entries_per_frame_max": int(counts.max()),
            "projected_inbounds_entries": int(in_bounds.sum()),
        }
    )
    return pts, vis, stats


def draw_projected_3d_tracks_2d(
    sample: Any,
    result: Any,
    frames: np.ndarray,
    out_gif: Path,
    out_sheet: Path,
    max_render: int,
) -> dict[str, Any]:
    point_indices_all = sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64)
    _, _, all_stats = projected_3d_tracks(result, point_indices_all, frames.shape[2], frames.shape[1])
    point_indices = select_render_points(point_indices_all, max_render)
    pts, vis, _ = projected_3d_tracks(result, point_indices, frames.shape[2], frames.shape[1])
    out_frames = []
    for t in range(len(frames)):
        img = Image.fromarray(frames[t]).convert("RGB")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        start = max(0, t - 8)
        for q in range(len(point_indices)):
            last = None
            for tt in range(start, t + 1):
                if not vis[tt, q]:
                    last = None
                    continue
                cur = (float(pts[tt, q, 0]), float(pts[tt, q, 1]))
                if last is not None:
                    draw.line([last, cur], fill=(44, 160, 44, 110), width=1)
                last = cur
            if vis[t, q]:
                x, y = pts[t, q]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(44, 160, 44, 220))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, img.width, 22), fill=(255, 255, 255))
        d.text((4, 4), f"local={t} 3D-projected visible={int(vis[t].sum())}", fill=(0, 0, 0))
        out_frames.append(img)
    out_frames[0].save(out_gif, save_all=True, append_images=out_frames[1:], duration=420, loop=0, optimize=False)
    ids = np.linspace(0, len(out_frames) - 1, num=min(12, len(out_frames)), dtype=int)
    save_grid([out_frames[i] for i in ids], out_sheet, "3D tracks projected to 2D", cols=4)
    all_stats["rendered_query_entries"] = int(len(point_indices))
    return all_stats


def figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return Image.fromarray(rgba[..., :3].copy())


def axis_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=-1)]
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    radius = max(float((hi - lo).max()) * 0.55, 1e-3)
    return center, radius


def set_axes(ax: Any, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)


def draw_query_tracks_3d(
    sample: Any,
    result: Any,
    out_png: Path,
    out_gif: Path,
    out_sheet: Path,
    max_points: int,
) -> dict[str, Any]:
    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    if len(point_indices) > max_points:
        rng = np.random.default_rng(42)
        point_indices = np.sort(rng.choice(point_indices, size=max_points, replace=False))
    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    vis = (
        np.asarray(result.valids[:, point_indices]).astype(bool)
        & np.asarray(result.visibs[:, point_indices]).astype(bool)
        & np.isfinite(trajs).all(axis=-1)
    )
    keep = vis.sum(axis=0) >= 2
    trajs, vis, point_indices = trajs[:, keep], vis[:, keep], point_indices[keep]
    if len(point_indices) == 0:
        return {"has_query_3d_tracks": False}
    center, radius = axis_limits(trajs[vis])
    colors = plt.cm.turbo(np.linspace(0, 1, len(point_indices)))
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for q in range(len(point_indices)):
        pts = trajs[vis[:, q], q]
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[q], linewidth=0.8, alpha=0.75)
            ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], c=[colors[q]], s=8)
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], c=[colors[q]], s=12, marker="^")
    ax.view_init(elev=20, azim=-62)
    set_axes(ax, center, radius)
    fig.suptitle(f"query 3D tracks, points={len(point_indices)}")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    gif_frames = []
    for t in range(len(trajs)):
        fig = plt.figure(figsize=(6.4, 5.8), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        for q in range(len(point_indices)):
            pts = trajs[: t + 1, q][vis[: t + 1, q]]
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[q], linewidth=0.7, alpha=0.6)
        cur = vis[t]
        if cur.any():
            pts = trajs[t, cur]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[cur], s=10)
        ax.view_init(elev=20, azim=-62)
        set_axes(ax, center, radius)
        fig.suptitle(f"query 3D local={t}, visible={int(cur.sum())}")
        gif_frames.append(figure_to_image(fig))
        plt.close(fig)
    gif_frames[0].save(out_gif, save_all=True, append_images=gif_frames[1:], duration=250, loop=0, optimize=False)
    ids = np.linspace(0, len(gif_frames) - 1, num=min(12, len(gif_frames)), dtype=int)
    save_grid([gif_frames[i] for i in ids], out_sheet, "query 3D track frames", cols=4)
    counts = vis.sum(axis=1)
    return {
        "has_query_3d_tracks": True,
        "query_3d_track_points_rendered": int(len(point_indices)),
        "query_3d_track_visible_per_frame_min": int(counts.min()),
        "query_3d_track_visible_per_frame_p50": float(np.percentile(counts, 50)),
        "query_3d_track_visible_per_frame_max": int(counts.max()),
    }


def sample_colors(frames: np.ndarray, frame_id: int, points_img: np.ndarray) -> np.ndarray:
    image = frames[frame_id]
    colors = np.full((len(points_img), 3), 160, dtype=np.uint8)
    if len(points_img) == 0:
        return colors
    xy = np.round(points_img).astype(np.int32)
    h, w = image.shape[:2]
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if ok.any():
        colors[ok] = image[xy[ok, 1], xy[ok, 0]]
    return colors


def depth_world_points(result: Any, frame_id: int, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.depths is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)
    depth = np.asarray(result.depths[frame_id], dtype=np.float32)
    h, w = depth.shape[:2]
    valid = np.isfinite(depth) & (depth > 1e-3)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)
    ids = np.arange(len(xs))
    if len(ids) > max_points:
        ids = rng.choice(ids, size=max_points, replace=False)
    xs = xs[ids].astype(np.float32)
    ys = ys[ids].astype(np.float32)
    z = depth[ys.astype(np.int32), xs.astype(np.int32)]
    K = np.asarray(result.intrinsics[frame_id], dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_c = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1).astype(np.float32)
    E_inv = np.linalg.inv(np.asarray(result.extrinsics[frame_id], dtype=np.float64)).astype(np.float32)
    pts_h = np.concatenate([pts_c, np.ones((len(pts_c), 1), dtype=np.float32)], axis=1)
    pts_w = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)
    uv = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts_w, pts_c, uv


def draw_dense_pointclouds(
    result: Any,
    frames: np.ndarray,
    out_world: Path,
    out_camera: Path,
    max_points: int,
    stride: int,
) -> dict[str, Any]:
    frame_ids = list(range(0, len(frames), stride))
    if frame_ids[-1] != len(frames) - 1:
        frame_ids.append(len(frames) - 1)
    scale = np.array(
        [
            float(frames.shape[2] - 1) / float(max(result.crop.crop_w - 1, 1)),
            float(frames.shape[1] - 1) / float(max(result.crop.crop_h - 1, 1)),
        ],
        dtype=np.float32,
    )
    entries = []
    rng = np.random.default_rng(7)
    has_tracks = result.trajs_3d_world is not None and result.valids is not None and result.visibs is not None
    for t in frame_ids:
        if has_tracks:
            pts_w_all = np.asarray(result.trajs_3d_world[t], dtype=np.float32)
            pts_img_all = np.asarray(result.trajs_2d[t], dtype=np.float32) * scale[None, :]
            vis = (
                np.asarray(result.valids[t]).astype(bool)
                & np.asarray(result.visibs[t]).astype(bool)
                & np.isfinite(pts_w_all).all(axis=-1)
            )
            ids = np.flatnonzero(vis)
            candidate_count = int(len(ids))
            if len(ids) > max_points:
                ids = np.sort(rng.choice(ids, size=max_points, replace=False))
            pts_w = pts_w_all[ids]
            pts_c = world_to_camera(pts_w, result.extrinsics[t])
            uv = pts_img_all[ids]
        else:
            pts_w, pts_c, uv = depth_world_points(result, t, max_points, rng)
            candidate_count = int(len(pts_w))
        cols = sample_colors(frames, t, uv).astype(np.float32) / 255.0
        entries.append((t, candidate_count, pts_w, pts_c, cols))

    all_w = np.concatenate([e[2] for e in entries if len(e[2])], axis=0) if any(len(e[2]) for e in entries) else np.zeros((0, 3), np.float32)
    all_c = np.concatenate([e[3] for e in entries if len(e[3])], axis=0) if any(len(e[3]) for e in entries) else np.zeros((0, 3), np.float32)

    def render(points_key: int, center_points: np.ndarray, out_path: Path, camera_view: bool) -> None:
        center, radius = axis_limits(center_points)
        imgs = []
        for t, candidate_count, pts_w, pts_c, cols in entries:
            pts = pts_c if points_key == 3 else pts_w
            fig = plt.figure(figsize=(6.4, 5.8), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.0, alpha=0.9, linewidths=0)
            ax.view_init(elev=-90, azim=-90 if camera_view else -62)
            set_axes(ax, center, radius)
            fig.suptitle(f"dense pointcloud local={t}, candidates={candidate_count}, rendered={len(pts)}")
            imgs.append(figure_to_image(fig))
            plt.close(fig)
        ids = np.linspace(0, len(imgs) - 1, num=min(12, len(imgs)), dtype=int)
        save_grid([imgs[i] for i in ids], out_path, "dense 3D point cloud", cols=4)

    render(2, all_w, out_world, camera_view=False)
    render(3, all_c, out_camera, camera_view=True)
    counts = np.array([e[1] for e in entries], dtype=np.float32)
    return {
        "dense_3d_source": "tracks" if has_tracks else "depth",
        "dense_3d_rendered_frames": int(len(entries)),
        "dense_3d_visible_candidates_p50": float(np.percentile(counts, 50)) if len(counts) else 0.0,
        "dense_3d_visible_candidates_max": int(counts.max()) if len(counts) else 0,
    }


def track_diagnostics(sample: Any, result: Any, width: int, height: int) -> dict[str, Any]:
    if result.trajs_2d is None or result.trajs_3d_world is None:
        return {"has_tracks": False}
    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    if len(point_indices) == 0:
        return {"has_tracks": True, "unique_selected_points": 0}

    pts, vis = scaled_tracks(result, point_indices, width, height)
    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    finite_2d = np.isfinite(pts).all(axis=-1)
    inbounds = finite_2d & (pts[..., 0] >= 0) & (pts[..., 0] < width) & (pts[..., 1] >= 0) & (pts[..., 1] < height)
    vis_changes = np.abs(vis[1:].astype(np.int8) - vis[:-1].astype(np.int8)).sum(axis=0)
    inbounds_changes = np.abs(inbounds[1:].astype(np.int8) - inbounds[:-1].astype(np.int8)).sum(axis=0)

    step_2d = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    step_2d_visible = step_2d[vis[:-1] & vis[1:]]
    step_3d = np.linalg.norm(np.diff(trajs, axis=0), axis=-1)
    step_3d_finite = step_3d[np.isfinite(step_3d)]
    per_point_3d_p95 = np.array(
        [np.percentile(v[np.isfinite(v)], 95) if np.isfinite(v).any() else np.inf for v in step_3d.T],
        dtype=np.float32,
    )
    static_like = per_point_3d_p95 < 1e-3
    hidden_inbounds = inbounds & ~vis

    out: dict[str, Any] = {
        "has_tracks": True,
        "unique_selected_points": int(len(point_indices)),
        "visible_entries": int(vis.sum()),
        "inbounds_entries": int(inbounds.sum()),
        "hidden_inbounds_entries": int(hidden_inbounds.sum()),
        "vis_changes_per_point_min_p50_p95_max": [
            int(vis_changes.min()),
            *percentiles(vis_changes, [50, 95]),
            int(vis_changes.max()),
        ],
        "inbounds_changes_per_point_min_p50_p95_max": [
            int(inbounds_changes.min()),
            *percentiles(inbounds_changes, [50, 95]),
            int(inbounds_changes.max()),
        ],
        "stored_2d_step_px_visible_p50_p95_max": percentiles(step_2d_visible, [50, 95, 100]),
        "world_3d_step_p50_p95_max": percentiles(step_3d_finite, [50, 95, 100]),
        "static_like_points_world_step_p95_lt_1e-3": int(static_like.sum()),
    }
    if static_like.any():
        out.update(
            {
                "static_like_vis_changes_p50_p95_max": percentiles(vis_changes[static_like], [50, 95, 100]),
                "static_like_inbounds_changes_p50_p95_max": percentiles(inbounds_changes[static_like], [50, 95, 100]),
                "static_like_hidden_inbounds_entries": int(hidden_inbounds[:, static_like].sum()),
            }
        )

    track_types = result.metadata.get("track_types") if isinstance(result.metadata, dict) else None
    if track_types is not None:
        try:
            selected_types = np.asarray(track_types)[point_indices]
            vals, counts = np.unique(selected_types, return_counts=True)
            out["track_type_hist_selected_unique_points"] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
        except Exception:
            pass
    return out


def draw_static_flicker_focus(
    sample: Any,
    result: Any,
    frames: np.ndarray,
    out_dir: Path,
    max_points: int = 8,
    crop_radius: int = 42,
) -> dict[str, Any]:
    if result.trajs_2d is None or result.trajs_3d_world is None:
        return {"has_static_flicker_focus": False}

    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    if len(point_indices) == 0:
        return {"has_static_flicker_focus": False, "reason": "no selected points"}

    pts, vis = scaled_tracks(result, point_indices, frames.shape[2], frames.shape[1])
    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    finite_2d = np.isfinite(pts).all(axis=-1)
    inbounds = (
        finite_2d
        & (pts[..., 0] >= 0)
        & (pts[..., 0] < frames.shape[2])
        & (pts[..., 1] >= 0)
        & (pts[..., 1] < frames.shape[1])
    )
    step_3d = np.linalg.norm(np.diff(trajs, axis=0), axis=-1)
    per_point_3d_p95 = np.array(
        [np.percentile(v[np.isfinite(v)], 95) if np.isfinite(v).any() else np.inf for v in step_3d.T],
        dtype=np.float32,
    )
    static_like = per_point_3d_p95 < 1e-3
    vis_changes = np.abs(vis[1:].astype(np.int8) - vis[:-1].astype(np.int8)).sum(axis=0)
    inb_changes = np.abs(inbounds[1:].astype(np.int8) - inbounds[:-1].astype(np.int8)).sum(axis=0)
    hidden_inbounds = inbounds & ~vis
    hidden_counts = hidden_inbounds.sum(axis=0)
    score = hidden_counts.astype(np.float32) + 2.0 * np.maximum(vis_changes - inb_changes, 0)
    eligible = static_like & (inbounds.sum(axis=0) >= max(3, len(frames) // 3)) & (score > 0)
    if not eligible.any():
        return {
            "has_static_flicker_focus": False,
            "static_like_points": int(static_like.sum()),
            "eligible_points": 0,
        }

    eligible_ids = np.flatnonzero(eligible)
    order = eligible_ids[np.argsort(-score[eligible_ids])]
    chosen_local = order[: min(max_points, len(order))]
    chosen_points = point_indices[chosen_local]

    # Full-frame contact sheet with selected points.
    frame_ids = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int)
    thumbs = []
    for frame_id in frame_ids:
        img = Image.fromarray(frames[frame_id]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for rank, q in enumerate(chosen_local):
            if not inbounds[frame_id, q]:
                continue
            x, y = pts[frame_id, q]
            color = (35, 170, 80) if vis[frame_id, q] else (225, 55, 55)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text((float(x) + 5, float(y) - 7), str(rank), fill=(255, 255, 255))
        draw.rectangle((0, 0, img.width, 22), fill=(255, 255, 255))
        draw.text((4, 4), f"static-like focus local={frame_id}", fill=(0, 0, 0))
        thumbs.append(img)
    focus_sheet = out_dir / "static_flicker_focus_contact_sheet.png"
    save_grid(thumbs, focus_sheet, "green=visible red=hidden while in-bounds", cols=4)

    # Compact visibility timeline.
    cell_w, cell_h = 10, 18
    timeline = Image.new("RGB", (cell_w * len(frames), cell_h * len(chosen_local)), "white")
    draw = ImageDraw.Draw(timeline)
    for row, q in enumerate(chosen_local):
        for t in range(len(frames)):
            if vis[t, q]:
                color = (35, 170, 80)
            elif inbounds[t, q]:
                color = (225, 55, 55)
            else:
                color = (80, 80, 80)
            x0 = t * cell_w
            y0 = row * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), fill=color)
        draw.text((2, row * cell_h + 2), str(row), fill=(255, 255, 255))
    timeline_path = out_dir / "static_flicker_timeline.png"
    timeline.save(timeline_path)

    # Crop strips around the worst selected points.
    crop_paths = []
    for rank, q in enumerate(chosen_local[: min(4, len(chosen_local))]):
        change_frames = np.flatnonzero(np.abs(np.diff(vis[:, q].astype(np.int8))) > 0)
        event_frames = sorted(set(int(np.clip(t, 0, len(frames) - 1)) for cf in change_frames for t in (cf, cf + 1)))
        if len(event_frames) < 12:
            extra = np.linspace(0, len(frames) - 1, num=12, dtype=int).tolist()
            event_frames = sorted(set(event_frames + extra))
        event_frames = event_frames[:16]
        crops = []
        for t in event_frames:
            img = Image.fromarray(frames[t]).convert("RGB")
            x, y = pts[t, q]
            if np.isfinite([x, y]).all():
                cx = int(np.clip(round(float(x)), 0, img.width - 1))
                cy = int(np.clip(round(float(y)), 0, img.height - 1))
            else:
                cx, cy = img.width // 2, img.height // 2
            x0 = max(0, cx - crop_radius)
            y0 = max(0, cy - crop_radius)
            x1 = min(img.width, cx + crop_radius)
            y1 = min(img.height, cy + crop_radius)
            crop = img.crop((x0, y0, x1, y1)).resize((crop_radius * 2, crop_radius * 2))
            d = ImageDraw.Draw(crop)
            local_x = (cx - x0) * (crop_radius * 2) / max(x1 - x0, 1)
            local_y = (cy - y0) * (crop_radius * 2) / max(y1 - y0, 1)
            if vis[t, q]:
                color = (35, 170, 80)
                state = "vis"
            elif inbounds[t, q]:
                color = (225, 55, 55)
                state = "hidden"
            else:
                color = (80, 80, 80)
                state = "out"
            d.ellipse((local_x - 5, local_y - 5, local_x + 5, local_y + 5), fill=color)
            d.rectangle((0, 0, crop.width, 18), fill=(255, 255, 255))
            d.text((3, 3), f"t={t} {state}", fill=(0, 0, 0))
            crops.append(crop)
        path = out_dir / f"static_flicker_point_{rank:02d}_crops.png"
        save_grid(crops, path, f"point rank {rank} crops", cols=4)
        crop_paths.append(str(path))

    return {
        "has_static_flicker_focus": True,
        "static_like_points": int(static_like.sum()),
        "eligible_points": int(eligible.sum()),
        "selected_point_indices": [int(x) for x in chosen_points.tolist()],
        "selected_scores": [float(score[q]) for q in chosen_local],
        "selected_hidden_inbounds_counts": [int(hidden_counts[q]) for q in chosen_local],
        "selected_vis_changes": [int(vis_changes[q]) for q in chosen_local],
        "selected_inbounds_changes": [int(inb_changes[q]) for q in chosen_local],
        "outputs": {
            "focus_contact_sheet": str(focus_sheet),
            "timeline": str(timeline_path),
            "crop_sheets": crop_paths,
        },
    }


def sample_min_depth(depth: np.ndarray, uv: np.ndarray, radius: int = 2) -> np.ndarray:
    h, w = depth.shape[:2]
    out = np.zeros((len(uv),), dtype=np.float32)
    if len(uv) == 0:
        return out
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    for i, (x, y) in enumerate(zip(px, py)):
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 1e-3) & np.isfinite(patch)]
        if valid.size:
            out[i] = float(valid.min())
    return out


def depth_visibility_reason_diagnostics(sample: Any, result: Any, width: int, height: int) -> dict[str, Any]:
    if result.trajs_3d_world is None or result.valids is None or result.visibs is None:
        return {"has_depth_visibility_reason": False, "reason": "no tracks"}
    if result.depths is None:
        return {"has_depth_visibility_reason": False, "reason": "no depth"}

    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    if len(point_indices) == 0:
        return {"has_depth_visibility_reason": False, "reason": "no selected points"}

    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    valids = np.asarray(result.valids[:, point_indices]).astype(bool)
    visibs = np.asarray(result.visibs[:, point_indices]).astype(bool)
    projected = np.full((trajs.shape[0], trajs.shape[1], 2), np.nan, dtype=np.float32)
    zvals = np.full((trajs.shape[0], trajs.shape[1]), np.nan, dtype=np.float32)
    for t in range(trajs.shape[0]):
        projected[t], zvals[t] = project_world(trajs[t], result.intrinsics[t], result.extrinsics[t])

    finite = np.isfinite(projected).all(axis=-1) & np.isfinite(zvals)
    inbounds = finite & (zvals > 1e-6) & (projected[..., 0] >= 0) & (projected[..., 0] < width) & (projected[..., 1] >= 0) & (projected[..., 1] < height)
    visible = valids & visibs & inbounds
    hidden_inbounds = valids & (~visibs) & inbounds

    hidden_pixel_depths = []
    hidden_min_depths = []
    hidden_z = []
    visible_pixel_depths = []
    visible_min_depths = []
    visible_z = []
    for t in range(trajs.shape[0]):
        depth = np.asarray(result.depths[t], dtype=np.float32)
        h_idx = np.flatnonzero(hidden_inbounds[t])
        if len(h_idx):
            xy = np.round(projected[t, h_idx]).astype(np.int32)
            xy[:, 0] = np.clip(xy[:, 0], 0, depth.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, depth.shape[0] - 1)
            hidden_pixel_depths.append(depth[xy[:, 1], xy[:, 0]].astype(np.float32))
            hidden_min_depths.append(sample_min_depth(depth, projected[t, h_idx], radius=2))
            hidden_z.append(zvals[t, h_idx])
        v_idx = np.flatnonzero(visible[t])
        if len(v_idx):
            xy = np.round(projected[t, v_idx]).astype(np.int32)
            xy[:, 0] = np.clip(xy[:, 0], 0, depth.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, depth.shape[0] - 1)
            visible_pixel_depths.append(depth[xy[:, 1], xy[:, 0]].astype(np.float32))
            visible_min_depths.append(sample_min_depth(depth, projected[t, v_idx], radius=2))
            visible_z.append(zvals[t, v_idx])

    if hidden_min_depths:
        hpd = np.concatenate(hidden_pixel_depths).astype(np.float32)
        hd = np.concatenate(hidden_min_depths).astype(np.float32)
        hz = np.concatenate(hidden_z).astype(np.float32)
    else:
        hpd = np.zeros((0,), dtype=np.float32)
        hd = np.zeros((0,), dtype=np.float32)
        hz = np.zeros((0,), dtype=np.float32)
    if visible_min_depths:
        vpd = np.concatenate(visible_pixel_depths).astype(np.float32)
        vd = np.concatenate(visible_min_depths).astype(np.float32)
        vz = np.concatenate(visible_z).astype(np.float32)
    else:
        vpd = np.zeros((0,), dtype=np.float32)
        vd = np.zeros((0,), dtype=np.float32)
        vz = np.zeros((0,), dtype=np.float32)

    hp_has_depth = (hpd > 1e-3) & np.isfinite(hpd) & np.isfinite(hz)
    hp_rel_err = np.full_like(hpd, np.inf, dtype=np.float32)
    if hp_has_depth.any():
        hp_rel_err[hp_has_depth] = np.abs(hpd[hp_has_depth] - hz[hp_has_depth]) / np.maximum(hz[hp_has_depth], 1e-6)
    hidden_pixel_match_05 = hp_has_depth & (hp_rel_err < 0.05)
    hidden_pixel_match_10 = hp_has_depth & (hp_rel_err < 0.10)
    hidden_pixel_occluded = hp_has_depth & (hpd + np.maximum(0.05, 0.02 * np.maximum(hz, 0.0)) < hz)

    h_has_depth = (hd > 1e-3) & np.isfinite(hd) & np.isfinite(hz)
    h_margin = np.maximum(0.05, 0.02 * np.maximum(hz, 0.0))
    hidden_occluded = h_has_depth & (hd + h_margin < hz)
    hidden_depth_match = h_has_depth & (np.abs(hd - hz) <= np.maximum(0.05, 0.03 * np.maximum(hz, 0.0)))
    hidden_missing = ~h_has_depth
    hidden_other_mismatch = h_has_depth & ~(hidden_occluded | hidden_depth_match)

    vp_has_depth = (vpd > 1e-3) & np.isfinite(vpd) & np.isfinite(vz)
    vp_rel_err = np.abs(vpd[vp_has_depth] - vz[vp_has_depth]) / np.maximum(vz[vp_has_depth], 1e-6)
    v_has_depth = (vd > 1e-3) & np.isfinite(vd) & np.isfinite(vz)
    v_rel_err = np.abs(vd[v_has_depth] - vz[v_has_depth]) / np.maximum(vz[v_has_depth], 1e-6)
    h_rel_err = np.abs(hd[h_has_depth] - hz[h_has_depth]) / np.maximum(hz[h_has_depth], 1e-6)

    return {
        "has_depth_visibility_reason": True,
        "selected_unique_points": int(len(point_indices)),
        "visible_inbounds_entries": int(visible.sum()),
        "hidden_inbounds_entries": int(hidden_inbounds.sum()),
        "single_pixel_hidden_depth_valid": int(hp_has_depth.sum()),
        "single_pixel_hidden_depth_match_rel05": int(hidden_pixel_match_05.sum()),
        "single_pixel_hidden_depth_match_rel10": int(hidden_pixel_match_10.sum()),
        "single_pixel_hidden_depth_occluded": int(hidden_pixel_occluded.sum()),
        "single_pixel_hidden_depth_match_rel05_fraction": float(hidden_pixel_match_05.sum() / max(1, len(hpd))),
        "single_pixel_hidden_depth_occluded_fraction": float(hidden_pixel_occluded.sum() / max(1, len(hpd))),
        "single_pixel_hidden_depth_relerr_p50_p95_max": percentiles(hp_rel_err[hp_has_depth], [50, 95, 100]),
        "single_pixel_visible_depth_relerr_p50_p95_max": percentiles(vp_rel_err, [50, 95, 100]),
        "hidden_depth_missing": int(hidden_missing.sum()),
        "hidden_depth_occluded_by_nearer_surface": int(hidden_occluded.sum()),
        "hidden_depth_matches_surface_potential_bad_vis": int(hidden_depth_match.sum()),
        "hidden_depth_other_mismatch": int(hidden_other_mismatch.sum()),
        "hidden_depth_match_fraction": float(hidden_depth_match.sum() / max(1, len(hd))),
        "hidden_occluded_fraction": float(hidden_occluded.sum() / max(1, len(hd))),
        "hidden_depth_relerr_p50_p95_max": percentiles(h_rel_err, [50, 95, 100]),
        "visible_depth_relerr_p50_p95_max": percentiles(v_rel_err, [50, 95, 100]),
    }


def classify_visibility_reasons(
    sample: Any,
    result: Any,
    width: int,
    height: int,
    depth_radius: int = 2,
) -> dict[str, Any]:
    """Classify selected track entries into visible / occluded / out-of-view reasons."""
    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    if (
        len(point_indices) == 0
        or result.trajs_3d_world is None
        or result.valids is None
        or result.visibs is None
    ):
        return {"has_visibility_reasons": False, "reason": "missing tracks"}

    trajs = np.asarray(result.trajs_3d_world[:, point_indices], dtype=np.float32)
    valids = np.asarray(result.valids[:, point_indices]).astype(bool)
    visibs = np.asarray(result.visibs[:, point_indices]).astype(bool)

    projected = np.full((trajs.shape[0], trajs.shape[1], 2), np.nan, dtype=np.float32)
    zvals = np.full((trajs.shape[0], trajs.shape[1]), np.nan, dtype=np.float32)
    for t in range(trajs.shape[0]):
        projected[t], zvals[t] = project_world(trajs[t], result.intrinsics[t], result.extrinsics[t])

    finite = np.isfinite(projected).all(axis=-1) & np.isfinite(zvals)
    in_front = finite & (zvals > 1e-6)
    inbounds = (
        in_front
        & (projected[..., 0] >= 0)
        & (projected[..., 0] < width)
        & (projected[..., 1] >= 0)
        & (projected[..., 1] < height)
    )
    label_visible = valids & visibs & inbounds
    label_hidden = valids & (~visibs) & inbounds

    min_depth = np.zeros_like(zvals, dtype=np.float32)
    depth_valid = np.zeros_like(inbounds, dtype=bool)
    if result.depths is not None:
        for t in range(trajs.shape[0]):
            idx = np.flatnonzero(inbounds[t])
            if len(idx) == 0:
                continue
            depth = np.asarray(result.depths[t], dtype=np.float32)
            sampled = sample_min_depth(depth, projected[t, idx], radius=depth_radius)
            min_depth[t, idx] = sampled
            depth_valid[t, idx] = (sampled > 1e-3) & np.isfinite(sampled)

    occ_margin = np.maximum(0.05, 0.02 * np.maximum(zvals, 0.0))
    match_margin = np.maximum(0.05, 0.03 * np.maximum(zvals, 0.0))
    depth_nearer = depth_valid & (min_depth + occ_margin < zvals)
    depth_match = depth_valid & (np.abs(min_depth - zvals) <= match_margin)

    occluded = inbounds & (~label_visible) & depth_nearer
    depth_match_hidden = label_hidden & (~occluded) & depth_match
    out_of_view = in_front & (~inbounds)
    missing_depth = inbounds & (~label_visible) & (~occluded) & (~depth_match_hidden) & (~depth_valid)
    depth_mismatch = (
        inbounds
        & (~label_visible)
        & (~occluded)
        & (~depth_match_hidden)
        & depth_valid
    )
    invalid_3d = ~(label_visible | occluded | out_of_view | depth_match_hidden | missing_depth | depth_mismatch)

    codes = np.full(inbounds.shape, VIS_REASON_CODES["invalid_3d"], dtype=np.int16)
    codes[invalid_3d] = VIS_REASON_CODES["invalid_3d"]
    codes[depth_mismatch] = VIS_REASON_CODES["depth_mismatch"]
    codes[missing_depth] = VIS_REASON_CODES["missing_depth"]
    codes[out_of_view] = VIS_REASON_CODES["out_of_view"]
    codes[depth_match_hidden] = VIS_REASON_CODES["depth_match_hidden"]
    codes[occluded] = VIS_REASON_CODES["occluded"]
    codes[label_visible] = VIS_REASON_CODES["visible"]

    per_frame_counts: dict[str, list[int]] = {}
    per_point_counts: dict[str, np.ndarray] = {}
    for name in VIS_REASON_ORDER:
        mask = codes == VIS_REASON_CODES[name]
        per_frame_counts[name] = [int(x) for x in mask.sum(axis=1).tolist()]
        per_point_counts[name] = mask.sum(axis=0).astype(np.int32)

    total_entries = int(codes.size)
    total_counts = {name: int(np.sum(codes == VIS_REASON_CODES[name])) for name in VIS_REASON_ORDER}
    fractions = {name: float(total_counts[name] / max(1, total_entries)) for name in VIS_REASON_ORDER}
    return {
        "has_visibility_reasons": True,
        "point_indices": point_indices,
        "projected": projected,
        "zvals": zvals,
        "codes": codes,
        "inbounds": inbounds,
        "depth_valid": depth_valid,
        "min_depth": min_depth,
        "has_visibility_supervision": bool(result.metadata.get("has_visibility", True))
        if isinstance(result.metadata, dict)
        else True,
        "per_frame_counts": per_frame_counts,
        "per_point_counts": per_point_counts,
        "total_counts": total_counts,
        "fractions": fractions,
    }


def select_visibility_reason_points(reasons: dict[str, Any], max_points: int) -> np.ndarray:
    point_indices = np.asarray(reasons["point_indices"], dtype=np.int64)
    if len(point_indices) <= max_points:
        return np.arange(len(point_indices), dtype=np.int64)

    counts = reasons["per_point_counts"]
    visible = counts["visible"]
    chosen: list[int] = []

    def add_top(name: str, quota: int, prefer_transition: bool = True) -> None:
        nonlocal chosen
        cls = counts[name]
        if prefer_transition:
            score = cls.astype(np.float32) + (visible > 0).astype(np.float32) * 1000.0
        else:
            score = cls.astype(np.float32)
        ids = np.flatnonzero(cls > 0)
        if len(ids) == 0:
            return
        order = ids[np.argsort(-score[ids])]
        for idx in order:
            i = int(idx)
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= max_points or sum(counts[name][j] > 0 for j in chosen) >= quota:
                break

    quotas = {
        "occluded": max(4, max_points // 3),
        "out_of_view": max(4, max_points // 3),
        "depth_match_hidden": max(2, max_points // 6),
    }
    add_top("occluded", quotas["occluded"])
    add_top("out_of_view", quotas["out_of_view"])
    add_top("depth_match_hidden", quotas["depth_match_hidden"])
    add_top("missing_depth", max(2, max_points // 8), prefer_transition=False)
    add_top("depth_mismatch", max(2, max_points // 8), prefer_transition=False)
    add_top("invalid_3d", max(1, max_points // 12), prefer_transition=False)

    if len(chosen) < max_points:
        score = visible.astype(np.float32)
        for idx in np.argsort(-score):
            i = int(idx)
            if score[i] <= 0:
                break
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= max_points:
                break

    return np.array(chosen[:max_points], dtype=np.int64)


def draw_visibility_legend(draw: ImageDraw.ImageDraw, x: int, y: int, scale: int = 1) -> None:
    cursor = x
    box = 10 * scale
    for name in VIS_REASON_ORDER:
        color = VIS_REASON_COLORS[name]
        draw.rectangle((cursor, y, cursor + box, y + box), fill=color, outline=(0, 0, 0))
        draw.text((cursor + box + 4, y - 1), VIS_REASON_LABELS[name], fill=(0, 0, 0))
        cursor += box + 4 + 7 * len(VIS_REASON_LABELS[name])


def marker_position(
    xy: np.ndarray,
    status_name: str,
    width: int,
    height: int,
    display_scale: int,
    header_h: int,
) -> tuple[float, float]:
    x = float(xy[0]) if np.isfinite(xy[0]) else width * 0.5
    y = float(xy[1]) if np.isfinite(xy[1]) else height * 0.5
    if status_name == "out_of_view":
        x = float(np.clip(x, 0, width - 1))
        y = float(np.clip(y, 0, height - 1))
    return x * display_scale, header_h + y * display_scale


def draw_visibility_marker(
    draw: ImageDraw.ImageDraw,
    xy: np.ndarray,
    status_name: str,
    rank: int,
    width: int,
    height: int,
    display_scale: int,
    header_h: int,
    radius: int = 5,
) -> None:
    color = VIS_REASON_COLORS[status_name]
    x = float(xy[0]) if np.isfinite(xy[0]) else width * 0.5
    y = float(xy[1]) if np.isfinite(xy[1]) else height * 0.5
    sx, sy = marker_position(xy, status_name, width, height, display_scale, header_h)
    r = radius * display_scale
    if status_name == "out_of_view":
        pad = 4 * display_scale
        if x < 0:
            pts = [(pad, sy), (pad + 3 * r, sy - r), (pad + 3 * r, sy + r)]
        elif x >= width:
            frame_w = width * display_scale
            pts = [(frame_w - pad, sy), (frame_w - pad - 3 * r, sy - r), (frame_w - pad - 3 * r, sy + r)]
        elif y < 0:
            pts = [(sx, header_h + pad), (sx - r, header_h + pad + 3 * r), (sx + r, header_h + pad + 3 * r)]
        else:
            frame_h = height * display_scale
            pts = [(sx, header_h + frame_h - pad), (sx - r, header_h + frame_h - pad - 3 * r), (sx + r, header_h + frame_h - pad - 3 * r)]
        draw.polygon(pts, fill=color, outline=(0, 0, 0))
        tx, ty = pts[0]
    else:
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=color, outline=(0, 0, 0), width=max(1, display_scale))
        tx, ty = sx + r + 2, sy - r
    draw.text((tx + 1, ty + 1), str(rank), fill=(0, 0, 0))
    draw.text((tx, ty), str(rank), fill=(255, 255, 255))


def render_visibility_reason_frame(
    frame: np.ndarray,
    reasons: dict[str, Any],
    selected_local: np.ndarray,
    frame_id: int,
    display_scale: int,
    trail: int = 5,
) -> Image.Image:
    height, width = frame.shape[:2]
    header_h = 46
    base = Image.fromarray(frame).convert("RGB").resize(
        (width * display_scale, height * display_scale),
        Image.BILINEAR,
    )
    canvas = Image.new("RGB", (base.width, base.height + header_h), "white")
    canvas.paste(base, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    counts = reasons["per_frame_counts"]
    count_text = " ".join(f"{VIS_REASON_SHORT[name]}={counts[name][frame_id]}" for name in VIS_REASON_ORDER)
    draw.text((4, 4), f"local={frame_id} {count_text}", fill=(0, 0, 0))
    draw_visibility_legend(draw, 4, 22, scale=display_scale)

    pts = reasons["projected"]
    codes = reasons["codes"]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    start = max(0, frame_id - trail)
    for rank, q in enumerate(selected_local):
        last: tuple[float, float] | None = None
        for tt in range(start, frame_id + 1):
            name = VIS_REASON_ORDER[int(codes[tt, q])]
            if name == "out_of_view":
                last = None
                continue
            xy = pts[tt, q]
            if not np.isfinite(xy).all():
                last = None
                continue
            cur = (
                float(xy[0]) * display_scale,
                header_h + float(xy[1]) * display_scale,
            )
            if last is not None:
                color = VIS_REASON_COLORS[name] + (110,)
                odraw.line([last, cur], fill=color, width=max(1, display_scale))
            last = cur
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for rank, q in enumerate(selected_local):
        name = VIS_REASON_ORDER[int(codes[frame_id, q])]
        draw_visibility_marker(
            draw,
            pts[frame_id, q],
            name,
            rank,
            width,
            height,
            display_scale,
            header_h,
            radius=4,
        )
    return canvas


def draw_visibility_reason_timeline(
    reasons: dict[str, Any],
    selected_local: np.ndarray,
    out_path: Path,
) -> None:
    codes = reasons["codes"][:, selected_local]
    point_indices = np.asarray(reasons["point_indices"], dtype=np.int64)[selected_local]
    T, N = codes.shape
    cell_w, cell_h = 14, 22
    label_w, top_h = 170, 48
    img = Image.new("RGB", (label_w + T * cell_w, top_h + N * cell_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), "visibility reason timeline", fill=(0, 0, 0))
    draw_visibility_legend(draw, 4, 24, scale=1)
    for q in range(N):
        y0 = top_h + q * cell_h
        hist = {name: int((codes[:, q] == VIS_REASON_CODES[name]).sum()) for name in VIS_REASON_ORDER}
        major = max(hist.items(), key=lambda kv: kv[1])[0]
        draw.text((4, y0 + 4), f"{q:02d} pt={int(point_indices[q])} {VIS_REASON_SHORT[major]}", fill=(0, 0, 0))
        for t in range(T):
            name = VIS_REASON_ORDER[int(codes[t, q])]
            color = VIS_REASON_COLORS[name]
            x0 = label_w + t * cell_w
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), fill=color)
    img.save(out_path)


def draw_visibility_reason_counts(reasons: dict[str, Any], out_path: Path) -> None:
    counts = reasons["per_frame_counts"]
    T = len(next(iter(counts.values()))) if counts else 0
    x = np.arange(T)
    bottom = np.zeros((T,), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for name in VIS_REASON_ORDER:
        vals = np.asarray(counts[name], dtype=np.float32)
        color = np.asarray(VIS_REASON_COLORS[name], dtype=np.float32) / 255.0
        ax.bar(x, vals, bottom=bottom, label=VIS_REASON_LABELS[name], color=color, width=0.9)
        bottom += vals
    ax.set_xlabel("local frame")
    ax.set_ylabel("unique sampled points")
    ax.set_title("visibility reason counts per frame")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def count_reason_codes(codes: np.ndarray) -> dict[str, int]:
    codes = np.asarray(codes, dtype=np.int64)
    return {name: int((codes == VIS_REASON_CODES[name]).sum()) for name in VIS_REASON_ORDER}


def build_training_target_reason_summary(
    sample: Any,
    reasons: dict[str, Any],
) -> dict[str, Any]:
    """Summarize reason classes for the actual target entries emitted by QueryBuilder."""
    point_indices = np.asarray(reasons["point_indices"], dtype=np.int64)
    point_to_local = {int(p): i for i, p in enumerate(point_indices.tolist())}
    query_points = sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64)
    t_tgt = sample.t_tgt.detach().cpu().numpy().astype(np.int64)
    mask_vis = sample.targets["mask_vis"].detach().cpu().numpy().astype(bool)
    mask_2d = sample.targets["mask_2d"].detach().cpu().numpy().astype(bool)
    mask_3d = sample.targets["mask_3d"].detach().cpu().numpy().astype(bool)

    local = np.array([point_to_local.get(int(p), -1) for p in query_points], dtype=np.int64)
    valid = (local >= 0) & (t_tgt >= 0) & (t_tgt < reasons["codes"].shape[0])
    query_codes = np.full((len(query_points),), VIS_REASON_CODES["invalid_3d"], dtype=np.int16)
    query_codes[valid] = reasons["codes"][t_tgt[valid], local[valid]]

    def pack(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool) & valid
        counts = count_reason_codes(query_codes[mask])
        total = int(mask.sum())
        return {
            "total": total,
            "counts": counts,
            "fractions": {
                name: float(counts[name] / max(1, total))
                for name in VIS_REASON_ORDER
            },
        }

    return {
        "all_queries": pack(valid),
        "mask_vis_queries": pack(mask_vis),
        "mask_2d_visible_queries": pack(mask_2d),
        "mask_3d_queries": pack(mask_3d),
    }


def draw_visibility_reason_point_crops(
    frames: np.ndarray,
    reasons: dict[str, Any],
    selected_local: np.ndarray,
    out_dir: Path,
    display_scale: int,
    crop_radius: int,
    max_points: int = 12,
) -> list[str]:
    codes = reasons["codes"]
    pts = reasons["projected"]
    height, width = frames.shape[1:3]
    paths: list[str] = []

    crop_positions: list[int] = []
    for name in [
        "occluded",
        "out_of_view",
        "depth_match_hidden",
        "missing_depth",
        "depth_mismatch",
        "invalid_3d",
        "visible",
    ]:
        ids = [
            i
            for i, q in enumerate(selected_local)
            if int((codes[:, q] == VIS_REASON_CODES[name]).sum()) > 0
        ]
        ids.sort(
            key=lambda i: int((codes[:, selected_local[i]] == VIS_REASON_CODES[name]).sum()),
            reverse=True,
        )
        quota = 3 if name in {"occluded", "out_of_view"} else 2
        for i in ids[:quota]:
            if i not in crop_positions:
                crop_positions.append(i)
            if len(crop_positions) >= max_points:
                break
        if len(crop_positions) >= max_points:
            break
    if len(crop_positions) < max_points:
        for i in range(len(selected_local)):
            if i not in crop_positions:
                crop_positions.append(i)
            if len(crop_positions) >= max_points:
                break

    for rank_pos in crop_positions[:max_points]:
        rank = int(rank_pos)
        q = selected_local[rank_pos]
        event_frames: list[int] = []
        for name in VIS_REASON_ORDER:
            ids = np.flatnonzero(codes[:, q] == VIS_REASON_CODES[name])
            if len(ids):
                take = np.linspace(0, len(ids) - 1, num=min(3, len(ids)), dtype=int)
                event_frames.extend(int(ids[i]) for i in take)
        changes = np.flatnonzero(np.diff(codes[:, q]) != 0)
        event_frames.extend(int(np.clip(t, 0, len(frames) - 1)) for c in changes[:8] for t in (c, c + 1))
        event_frames = sorted(set(event_frames))
        if len(event_frames) > 16:
            keep = np.linspace(0, len(event_frames) - 1, num=16, dtype=int)
            event_frames = [event_frames[i] for i in keep]
        crops: list[Image.Image] = []
        for t in event_frames:
            name = VIS_REASON_ORDER[int(codes[t, q])]
            xy = pts[t, q]
            x = float(xy[0]) if np.isfinite(xy[0]) else width * 0.5
            y = float(xy[1]) if np.isfinite(xy[1]) else height * 0.5
            cx = int(np.clip(round(x), 0, width - 1))
            cy = int(np.clip(round(y), 0, height - 1))
            x0 = max(0, cx - crop_radius)
            y0 = max(0, cy - crop_radius)
            x1 = min(width, cx + crop_radius)
            y1 = min(height, cy + crop_radius)
            crop = Image.fromarray(frames[t]).convert("RGB").crop((x0, y0, x1, y1))
            crop = crop.resize((crop_radius * 2 * display_scale, crop_radius * 2 * display_scale), Image.BILINEAR)
            header_h = 22
            canvas = Image.new("RGB", (crop.width, crop.height + header_h), "white")
            canvas.paste(crop, (0, header_h))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, canvas.width, header_h), fill=(255, 255, 255))
            draw.text((3, 4), f"t={t} {VIS_REASON_LABELS[name]}", fill=(0, 0, 0))
            local_xy = np.array(
                [
                    (x - x0) * crop.width / max(x1 - x0, 1),
                    (y - y0) * crop.height / max(y1 - y0, 1),
                ],
                dtype=np.float32,
            )
            draw_visibility_marker(
                draw,
                local_xy / max(display_scale, 1),
                name,
                rank,
                crop.width // display_scale,
                crop.height // display_scale,
                display_scale,
                header_h,
                radius=4,
            )
            crops.append(canvas)
        if crops:
            path = out_dir / f"visibility_reason_point_{rank:02d}_crops.png"
            save_grid(crops, path, f"visibility reason point {rank}", cols=4)
            paths.append(str(path))
    return paths


def draw_visibility_reason_analysis(
    sample: Any,
    result: Any,
    frames: np.ndarray,
    out_dir: Path,
    max_points: int,
    display_scale: int,
    crop_radius: int,
) -> dict[str, Any]:
    reasons = classify_visibility_reasons(sample, result, frames.shape[2], frames.shape[1])
    if not reasons.get("has_visibility_reasons", False):
        return reasons
    selected_local = select_visibility_reason_points(reasons, max_points=max_points)
    if len(selected_local) == 0:
        return {"has_visibility_reasons": False, "reason": "no selected reason points"}

    out_frames = [
        render_visibility_reason_frame(
            frame=frames[t],
            reasons=reasons,
            selected_local=selected_local,
            frame_id=t,
            display_scale=display_scale,
        )
        for t in range(len(frames))
    ]
    gif_path = out_dir / "visibility_reason_tracks.gif"
    out_frames[0].save(gif_path, save_all=True, append_images=out_frames[1:], duration=650, loop=0, optimize=False)
    ids = np.linspace(0, len(out_frames) - 1, num=min(12, len(out_frames)), dtype=int)
    contact_path = out_dir / "visibility_reason_contact_sheet.png"
    save_grid([out_frames[i] for i in ids], contact_path, "visibility reasons", cols=2)

    timeline_path = out_dir / "visibility_reason_timeline.png"
    draw_visibility_reason_timeline(reasons, selected_local, timeline_path)

    counts_path = out_dir / "visibility_reason_counts.png"
    draw_visibility_reason_counts(reasons, counts_path)

    crop_paths = draw_visibility_reason_point_crops(
        frames=frames,
        reasons=reasons,
        selected_local=selected_local,
        out_dir=out_dir,
        display_scale=display_scale,
        crop_radius=crop_radius,
    )

    point_indices = np.asarray(reasons["point_indices"], dtype=np.int64)[selected_local]
    selected_counts: list[dict[str, Any]] = []
    for rank, q in enumerate(selected_local):
        selected_counts.append(
            {
                "rank": int(rank),
                "point_index": int(point_indices[rank]),
                **{
                    name: int(reasons["per_point_counts"][name][q])
                    for name in VIS_REASON_ORDER
                },
            }
        )
    training_target_reasons = build_training_target_reason_summary(sample, reasons)

    summary = {
        "has_visibility_reasons": True,
        "has_visibility_supervision": bool(reasons.get("has_visibility_supervision", True)),
        "num_unique_points_classified": int(len(reasons["point_indices"])),
        "num_points_rendered": int(len(selected_local)),
        "total_counts": reasons["total_counts"],
        "fractions": reasons["fractions"],
        "training_target_reasons": training_target_reasons,
        "selected_points": selected_counts,
        "outputs": {
            "gif": str(gif_path),
            "contact_sheet": str(contact_path),
            "timeline": str(timeline_path),
            "counts": str(counts_path),
            "crop_sheets": crop_paths,
        },
    }
    (out_dir / "visibility_reason_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def raw_clip_visibility_reason_diagnostics(sample: Any, clip: Any | None) -> dict[str, Any]:
    if clip is None:
        return {"has_raw_clip_visibility_reason": False, "reason": "no captured clip"}
    if clip.trajs_3d_world is None or clip.trajs_2d is None or clip.valids is None or clip.visibs is None:
        return {"has_raw_clip_visibility_reason": False, "reason": "no tracks"}
    if clip.depths is None:
        return {"has_raw_clip_visibility_reason": False, "reason": "no depth"}

    point_indices = np.unique(sample.targets["point_indices"].detach().cpu().numpy().astype(np.int64))
    max_n = clip.trajs_2d.shape[1]
    point_indices = point_indices[(point_indices >= 0) & (point_indices < max_n)]
    if len(point_indices) == 0:
        return {"has_raw_clip_visibility_reason": False, "reason": "no selected raw point indices"}

    valids = np.asarray(clip.valids[:, point_indices]).astype(bool)
    visibs = np.asarray(clip.visibs[:, point_indices]).astype(bool)
    uv = np.asarray(clip.trajs_2d[:, point_indices], dtype=np.float32)
    trajs = np.asarray(clip.trajs_3d_world[:, point_indices], dtype=np.float32)

    hidden_rel = []
    visible_rel = []
    hidden_depth_valid = 0
    hidden_count = 0
    hidden_match05 = 0
    hidden_match10 = 0
    hidden_occluded = 0
    hidden_missing = 0

    for t in range(uv.shape[0]):
        depth = np.asarray(clip.depths[t], dtype=np.float32)
        h, w = depth.shape[:2]
        xy = uv[t]
        finite = np.isfinite(xy).all(axis=-1)
        inbounds = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        z = world_to_camera(trajs[t], np.asarray(clip.extrinsics[t], dtype=np.float32))[:, 2]
        z_ok = np.isfinite(z) & (z > 1e-6)
        hidden = valids[t] & (~visibs[t]) & inbounds & z_ok
        visible = valids[t] & visibs[t] & inbounds & z_ok

        for label, mask, dst in [("hidden", hidden, hidden_rel), ("visible", visible, visible_rel)]:
            ids = np.flatnonzero(mask)
            if len(ids) == 0:
                continue
            px = np.clip(np.round(xy[ids, 0]).astype(np.int32), 0, w - 1)
            py = np.clip(np.round(xy[ids, 1]).astype(np.int32), 0, h - 1)
            d = depth[py, px].astype(np.float32)
            has_depth = (d > 1e-3) & np.isfinite(d)
            rel = np.full((len(ids),), np.inf, dtype=np.float32)
            rel[has_depth] = np.abs(d[has_depth] - z[ids][has_depth]) / np.maximum(z[ids][has_depth], 1e-6)
            dst.append(rel)
            if label == "hidden":
                hidden_count += int(len(ids))
                hidden_depth_valid += int(has_depth.sum())
                hidden_match05 += int((has_depth & (rel < 0.05)).sum())
                hidden_match10 += int((has_depth & (rel < 0.10)).sum())
                hidden_occluded += int((has_depth & (d + np.maximum(0.05, 0.02 * np.maximum(z[ids], 0.0)) < z[ids])).sum())
                hidden_missing += int((~has_depth).sum())

    hrel = np.concatenate(hidden_rel) if hidden_rel else np.zeros((0,), dtype=np.float32)
    vrel = np.concatenate(visible_rel) if visible_rel else np.zeros((0,), dtype=np.float32)
    hrel_f = hrel[np.isfinite(hrel)]
    vrel_f = vrel[np.isfinite(vrel)]
    return {
        "has_raw_clip_visibility_reason": True,
        "selected_unique_points": int(len(point_indices)),
        "hidden_inbounds_entries": int(hidden_count),
        "hidden_depth_valid": int(hidden_depth_valid),
        "hidden_depth_missing": int(hidden_missing),
        "hidden_depth_match_rel05": int(hidden_match05),
        "hidden_depth_match_rel10": int(hidden_match10),
        "hidden_depth_occluded": int(hidden_occluded),
        "hidden_depth_match_rel05_fraction": float(hidden_match05 / max(1, hidden_count)),
        "hidden_depth_occluded_fraction": float(hidden_occluded / max(1, hidden_count)),
        "hidden_depth_relerr_p50_p95_max": percentiles(hrel_f, [50, 95, 100]),
        "visible_depth_relerr_p50_p95_max": percentiles(vrel_f, [50, 95, 100]),
    }


def build_single_config(config: dict[str, Any], dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    ds_conf = None
    for item in config["datasets"]:
        if item["name"] == dataset_name:
            ds_conf = item
            break
    if ds_conf is None:
        raise ValueError(f"Dataset {dataset_name!r} not found in config")

    common_keys = [
        "clip_len",
        "img_size",
        "num_queries",
        "use_augs",
        "boundary_ratio",
        "use_motion_boundaries",
        "t_tgt_eq_t_cam_ratio",
        "seed",
        "allow_track_fallback",
        "dataset_locality_size",
        "sequence_locality_size",
        "frame_locality_radius",
        "precompute_patches",
        "precompute_from_highres",
        "return_highres_video",
        "store_video_uint8",
        "store_auxiliary_tensors",
        "color_aug_after_resize",
        "keep_cropped_images",
        "motion_boundary_on_resized",
        "index_cache_dir",
        "index_workers",
        "sampling_mode",
        "epoch_size",
    ]
    out = {k: copy.deepcopy(config[k]) for k in common_keys if k in config}
    out.update(
        {
            "mode": "single",
            "name": ds_conf["name"],
            "root": ds_conf["root"],
            "adapter_kwargs": copy.deepcopy(ds_conf.get("adapter_kwargs", {})),
            "weight": 1.0,
        }
    )
    if args.clip_len is not None:
        out["clip_len"] = args.clip_len
    if args.img_size is not None:
        out["img_size"] = args.img_size
    if args.num_queries is not None:
        out["num_queries"] = args.num_queries
    if args.no_augs:
        out["use_augs"] = False
    return out


def run_one_dataset(dataset_name: str, config: dict[str, Any], args: argparse.Namespace, sample_index: int) -> dict[str, Any]:
    ds_cfg = build_single_config(config, dataset_name, args)
    ds = create_training_dataset(ds_cfg, split=args.split)
    adapter_capture = CapturingAdapter(ds.adapters[0])
    ds.adapters[0] = adapter_capture
    ds.mixture_sampler.samplers[0].adapter = adapter_capture
    capture = CapturingQueryBuilder(ds.query_builder)
    ds.query_builder = capture

    sample = ds[sample_index]
    result = capture.last_result
    if result is None:
        raise RuntimeError(f"{dataset_name}: failed to capture TransformResult")

    out_dir = Path(args.out_dir) / dataset_name / f"sample_{sample_index:04d}_{sample.sequence_name.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = video_to_uint8(sample.video)

    source_stats = draw_query_sources(sample, frames, out_dir / "exact_source_queries.png")
    target_stats = draw_query_targets(sample, frames, out_dir / "exact_target_queries.png")

    has_tracks = result.trajs_2d is not None and result.trajs_3d_world is not None
    track_stats: dict[str, Any] = {"has_tracks": bool(has_tracks)}
    projected_3d_stats: dict[str, Any] = {"has_tracks": bool(has_tracks)}
    query_3d_stats: dict[str, Any] = {"has_tracks": bool(has_tracks)}
    if has_tracks:
        track_stats = draw_all_query_tracks(
            sample,
            result,
            frames,
            out_dir / "exact_all_query_tracks.gif",
            out_dir / "exact_all_query_tracks_contact_sheet.png",
            max_render=args.max_query_render,
        )
        projected_3d_stats = draw_projected_3d_tracks_2d(
            sample,
            result,
            frames,
            out_dir / "exact_projected_3d_tracks_2d.gif",
            out_dir / "exact_projected_3d_tracks_2d_contact_sheet.png",
            max_render=args.max_query_render,
        )
        query_3d_stats = draw_query_tracks_3d(
            sample,
            result,
            out_dir / "exact_query_tracks_3d.png",
            out_dir / "exact_query_tracks_3d.gif",
            out_dir / "exact_query_tracks_3d_contact_sheet.png",
            max_points=args.max_3d_track_points,
        )
        static_flicker_focus_stats = draw_static_flicker_focus(
            sample,
            result,
            frames,
            out_dir,
            max_points=args.max_static_focus_points,
        )
        visibility_reason_stats = draw_visibility_reason_analysis(
            sample,
            result,
            frames,
            out_dir,
            max_points=args.max_visibility_reason_points,
            display_scale=args.visibility_display_scale,
            crop_radius=args.visibility_crop_radius,
        )
    else:
        static_flicker_focus_stats = {"has_static_flicker_focus": False, "reason": "no tracks"}
        visibility_reason_stats = {"has_visibility_reasons": False, "reason": "no tracks"}

    dense_3d_stats = draw_dense_pointclouds(
        result,
        frames,
        out_dir / "exact_dense_3d_world_pointcloud_contact_sheet.png",
        out_dir / "exact_dense_3d_camera_pointcloud_contact_sheet.png",
        max_points=args.max_dense_points,
        stride=args.dense_stride,
    )

    targets = sample.targets
    point_indices = targets["point_indices"].detach().cpu().numpy().astype(np.int64)
    t_src = sample.t_src.detach().cpu().numpy().astype(np.int64)
    t_tgt = sample.t_tgt.detach().cpu().numpy().astype(np.int64)
    mask_2d = targets["mask_2d"].detach().cpu().numpy().astype(bool)
    mask_3d = targets["mask_3d"].detach().cpu().numpy().astype(bool)
    mask_vis = targets["mask_vis"].detach().cpu().numpy().astype(bool)
    is_static = targets.get("is_static_reprojection")
    is_static_np = (
        is_static.detach().cpu().numpy().astype(bool)
        if is_static is not None
        else (t_src == t_tgt)
    )

    report = {
        "dataset_name": sample.dataset_name,
        "sequence_name": sample.sequence_name,
        "sample_index": int(sample_index),
        "frame_indices": adapter_capture.last_frame_indices,
        "clip_len": int(sample.video.shape[0]),
        "num_queries": int(sample.coords.shape[0]),
        "unique_point_indices": int(np.unique(point_indices).size),
        "query_semantics": sample.metadata.get("query_semantics"),
        "has_temporal_supervision": bool(sample.metadata.get("has_temporal_supervision", False)),
        "mask_2d": int(mask_2d.sum()),
        "mask_3d": int(mask_3d.sum()),
        "mask_vis": int(mask_vis.sum()),
        "mask_disp": int(targets["mask_disp"].sum().item()),
        "static_reprojection_queries": int(is_static_np.sum()),
        "source_is_boundary": int(targets["source_is_boundary"].sum().item()),
        "source_is_depth_boundary": int(targets["source_is_depth_boundary"].sum().item()),
        "source_is_motion_boundary": int(targets["source_is_motion_boundary"].sum().item()),
        "source_frame_hist": {str(i): int((t_src == i).sum()) for i in range(int(sample.video.shape[0]))},
        "target_frame_mask2d_hist": {
            str(i): int(((t_tgt == i) & mask_2d).sum())
            for i in range(int(sample.video.shape[0]))
        },
        "track_diagnostics": track_diagnostics(sample, result, frames.shape[2], frames.shape[1]),
        "depth_visibility_reason": depth_visibility_reason_diagnostics(sample, result, frames.shape[2], frames.shape[1]),
        "raw_clip_visibility_reason": raw_clip_visibility_reason_diagnostics(sample, adapter_capture.last_clip),
        "source_stats": source_stats,
        "target_stats": target_stats,
        "track_stats": track_stats,
        "projected_3d_stats": projected_3d_stats,
        "query_3d_stats": query_3d_stats,
        "static_flicker_focus_stats": static_flicker_focus_stats,
        "visibility_reason_stats": visibility_reason_stats,
        "dense_3d_stats": dense_3d_stats,
        "adapter_root": ds_cfg["root"],
        "adapter_kwargs": ds_cfg.get("adapter_kwargs", {}),
        "outputs": {
            "source_queries": str(out_dir / "exact_source_queries.png"),
            "target_queries": str(out_dir / "exact_target_queries.png"),
            "all_query_tracks_gif": str(out_dir / "exact_all_query_tracks.gif") if has_tracks else None,
            "all_query_tracks_contact_sheet": str(out_dir / "exact_all_query_tracks_contact_sheet.png") if has_tracks else None,
            "projected_3d_tracks_2d_gif": str(out_dir / "exact_projected_3d_tracks_2d.gif") if has_tracks else None,
            "projected_3d_tracks_2d_contact_sheet": str(out_dir / "exact_projected_3d_tracks_2d_contact_sheet.png") if has_tracks else None,
            "query_tracks_3d_png": str(out_dir / "exact_query_tracks_3d.png") if has_tracks else None,
            "query_tracks_3d_gif": str(out_dir / "exact_query_tracks_3d.gif") if has_tracks else None,
            "query_tracks_3d_contact_sheet": str(out_dir / "exact_query_tracks_3d_contact_sheet.png") if has_tracks else None,
            "visibility_reason_gif": str(out_dir / "visibility_reason_tracks.gif") if has_tracks else None,
            "visibility_reason_contact_sheet": str(out_dir / "visibility_reason_contact_sheet.png") if has_tracks else None,
            "visibility_reason_timeline": str(out_dir / "visibility_reason_timeline.png") if has_tracks else None,
            "visibility_reason_counts": str(out_dir / "visibility_reason_counts.png") if has_tracks else None,
            "dense_3d_world_pointcloud_contact_sheet": str(out_dir / "exact_dense_3d_world_pointcloud_contact_sheet.png"),
            "dense_3d_camera_pointcloud_contact_sheet": str(out_dir / "exact_dense_3d_camera_pointcloud_contact_sheet.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO / "configs/mixture_all_10datasets_cos_planned.yaml"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scannet", "tartanair", "vkitti2", "mvssynth", "dynamic_replica"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="/data/zbf/openclaw/d4rt/tmp/exact_training_samples_latest")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--no-augs", action="store_true")
    parser.add_argument("--max-query-render", type=int, default=768)
    parser.add_argument("--max-3d-track-points", type=int, default=256)
    parser.add_argument("--max-static-focus-points", type=int, default=8)
    parser.add_argument("--max-visibility-reason-points", type=int, default=24)
    parser.add_argument("--visibility-display-scale", type=int, default=2)
    parser.add_argument("--visibility-crop-radius", type=int, default=52)
    parser.add_argument("--max-dense-points", type=int, default=20000)
    parser.add_argument("--dense-stride", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    reports = []
    failures = []
    for dataset_name in args.datasets:
        for offset in range(args.samples_per_dataset):
            sample_index = args.start_index + offset
            try:
                report = run_one_dataset(dataset_name, config, args, sample_index)
                reports.append(report)
                print(json.dumps({
                    "dataset": dataset_name,
                    "sample_index": sample_index,
                    "sequence": report["sequence_name"],
                    "mask_2d": report["mask_2d"],
                    "mask_3d": report["mask_3d"],
                    "has_temporal_supervision": report["has_temporal_supervision"],
                    "projected_stats": report["projected_3d_stats"],
                    "summary": str(Path(report["outputs"]["source_queries"]).parent / "summary.json"),
                }, ensure_ascii=False))
            except Exception as exc:
                failures.append({"dataset": dataset_name, "sample_index": sample_index, "error": repr(exc)})
                print(json.dumps(failures[-1], ensure_ascii=False), file=sys.stderr)

    aggregate = {
        "config": args.config,
        "out_dir": args.out_dir,
        "reports": reports,
        "failures": failures,
    }
    out_path = Path(args.out_dir) / "summary_all.json"
    out_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_all": str(out_path), "num_reports": len(reports), "num_failures": len(failures)}, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
