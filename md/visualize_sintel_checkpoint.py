#!/usr/bin/env python3
"""Dataset-aware Sintel visualization for D4RT checkpoints.

This complements ``visualize_video_checkpoint.py`` by using Sintel camera
intrinsics/extrinsics. It can therefore output canonical point clouds and camera
trajectory/intrinsics visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_sintel_checkpoint import (  # noqa: E402
    SINTEL_14_SEQUENCES,
    SintelScene,
    autocast_context,
    build_frame_indices,
    encode_video,
    estimate_relative_pose,
)
from utils.camera import compute_relative_pose_error, sim3_alignment  # noqa: E402
from utils.metrics import compute_pose_auc, compute_pose_metrics  # noqa: E402
from visualize_dynamic_replica_checkpoint import (  # noqa: E402
    compute_dense_canonical_sequence,
    compute_dense_pred_reference_sequence,
    load_model,
    plot_dense_gt_static,
    plot_dense_canonical_static,
    plot_dense_pred_reference_static,
    plot_dense_pred_world_static,
    prepare_dense_gt_depth_sequence,
    save_point_cloud_ply_allow_empty,
    select_device,
    write_dense_gt_gif,
    write_dense_canonical_gif,
    write_dense_pred_reference_gif,
    write_dense_pred_world_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Sintel checkpoint outputs with camera geometry.")
    parser.add_argument("--root", type=str, default="/data3/dataset/sintel")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-variant", type=str, default="auto", choices=("auto", "base", "large", "huge", "giant"))
    parser.add_argument("--patch-provider", type=str, default="sampled_highres")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--scene-names", type=str, default="")
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--sintel-pass", type=str, default="final", choices=("clean", "final"))
    parser.add_argument("--reference-frame", type=int, default=24)
    parser.add_argument("--dense-stride", type=int, default=4)
    parser.add_argument("--dense-gt-depth-stride", type=int, default=3)
    parser.add_argument("--dense-gt-max-points", type=int, default=0)
    parser.add_argument("--depth-gt-max-depth", type=float, default=100.0)
    parser.add_argument("--dense-vis-threshold", type=float, default=0.3)
    parser.add_argument("--dense-confidence-percentile", type=float, default=10.0)
    parser.add_argument("--dense-query-depth-percentile", type=float, default=100.0)
    parser.add_argument("--dense-query-batch-size", type=int, default=4096)
    parser.add_argument("--pose-grid-h", type=int, default=6)
    parser.add_argument("--pose-grid-w", type=int, default=6)
    parser.add_argument("--pose-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def select_scenes(root: Path, args: argparse.Namespace) -> list[str]:
    if args.scene_names:
        scenes = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.all_scenes:
        scenes = sorted(p.name for p in (root / "training" / args.sintel_pass).iterdir() if p.is_dir())
    else:
        scenes = list(SINTEL_14_SEQUENCES)
    if args.num_scenes > 0:
        scenes = scenes[: args.num_scenes]
    return scenes


def apply_sim3_to_poses(pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
    aligned = pred_poses.clone()
    rotation, translation, scale = sim3_alignment(pred_poses, gt_poses)
    aligned[:, :3, :3] = pred_poses[:, :3, :3] @ rotation.T
    centers = pred_poses[:, :3, 3]
    aligned[:, :3, 3] = scale * (centers @ rotation.T) + translation
    return aligned


def estimate_camera_trajectory(
    model: torch.nn.Module,
    result,
    args: argparse.Namespace,
    device: torch.device,
    num_real_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    encoder_features, frames_bcthw, transform_metadata = encode_video(model, result, device, args.patch_provider)

    gt_w2c = torch.from_numpy(result.extrinsics[:num_real_frames]).float()
    gt_c2w = torch.linalg.inv(gt_w2c)
    ref_w2c = gt_w2c[args.reference_frame]

    pred_c2w_list: list[torch.Tensor] = []
    rot_errors: list[torch.Tensor] = []
    trans_errors: list[torch.Tensor] = []
    for frame_idx in range(num_real_frames):
        if frame_idx == args.reference_frame:
            pred_w2c_rel = torch.eye(4, dtype=torch.float32)
        else:
            pred_w2c_rel = estimate_relative_pose(
                model,
                encoder_features,
                frames_bcthw,
                frame_i=args.reference_frame,
                frame_j=frame_idx,
                grid_h=args.pose_grid_h,
                grid_w=args.pose_grid_w,
                confidence_threshold=args.pose_confidence_threshold,
                device=device,
                transform_metadata=transform_metadata,
            )
            gt_w2c_rel = gt_w2c[frame_idx] @ torch.linalg.inv(ref_w2c)
            rel_errors = compute_relative_pose_error(
                pred_w2c_rel[:3, :3],
                pred_w2c_rel[:3, 3],
                gt_w2c_rel[:3, :3],
                gt_w2c_rel[:3, 3],
            )
            rot_errors.append(rel_errors["rotation_error"])
            trans_errors.append(rel_errors["translation_error"])
        pred_c2w_list.append(torch.linalg.inv(pred_w2c_rel))

    pred_c2w = torch.stack(pred_c2w_list)
    aligned_pred = apply_sim3_to_poses(pred_c2w, gt_c2w)
    pose_metrics = compute_pose_metrics(pred_c2w, gt_c2w, align=True)
    if rot_errors:
        auc = compute_pose_auc(torch.stack(rot_errors), torch.stack(trans_errors), threshold=30.0)
    else:
        auc = torch.tensor(float("nan"))

    metrics = {
        "pose_ate": float(pose_metrics["ate"].item()),
        "pose_rpe_trans": float(pose_metrics["rpe_trans"].item()),
        "pose_rpe_rot": float(pose_metrics["rpe_rot"].item()),
        "pose_auc_30": float(auc.item()),
        "pose_num_pairs": float(len(rot_errors)),
    }
    return gt_c2w.numpy(), aligned_pred.numpy(), metrics


def _set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    pts = points[np.isfinite(points).all(axis=-1)]
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_camera_frustum(ax, pose_c2w: np.ndarray, color: str, scale: float) -> None:
    center = pose_c2w[:3, 3]
    R = pose_c2w[:3, :3]
    corners_cam = np.array(
        [
            [-0.6, -0.4, 1.0],
            [0.6, -0.4, 1.0],
            [0.6, 0.4, 1.0],
            [-0.6, 0.4, 1.0],
        ],
        dtype=np.float32,
    ) * scale
    corners = (R @ corners_cam.T).T + center[None]
    for corner in corners:
        ax.plot([center[0], corner[0]], [center[1], corner[1]], [center[2], corner[2]], color=color, linewidth=0.7, alpha=0.55)
    loop = np.concatenate([corners, corners[:1]], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=0.9, alpha=0.75)


def plot_camera_trajectory(
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    gt_centers = gt_c2w[:, :3, 3]
    pred_centers = pred_c2w[:, :3, 3]
    all_centers = np.concatenate([gt_centers, pred_centers], axis=0)
    travel = np.linalg.norm(np.diff(gt_centers, axis=0), axis=-1).sum()
    frustum_scale = max(float(travel) / max(len(gt_centers), 1), 1e-3) * 2.0

    fig = plt.figure(figsize=(8.0, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color="#1f77b4", linewidth=2.0, label="GT")
    ax.plot(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], color="#d62728", linewidth=2.0, label="Pred aligned")
    step = max(len(gt_c2w) // 6, 1)
    for idx in range(0, len(gt_c2w), step):
        draw_camera_frustum(ax, gt_c2w[idx], "#1f77b4", frustum_scale)
        draw_camera_frustum(ax, pred_c2w[idx], "#d62728", frustum_scale)
    ax.scatter(gt_centers[[0, -1], 0], gt_centers[[0, -1], 1], gt_centers[[0, -1], 2], c=["#2ca02c", "#1f77b4"], s=40)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_axes(ax, all_centers)
    ax.legend(loc="best")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_intrinsics(intrinsics: np.ndarray, output_path: Path, title: str) -> dict[str, float]:
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    frames = np.arange(len(intrinsics))

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), constrained_layout=True, sharex=True)
    axes[0].plot(frames, fx, label="fx")
    axes[0].plot(frames, fy, label="fy")
    axes[0].set_ylabel("focal length")
    axes[0].legend()
    axes[1].plot(frames, cx, label="cx")
    axes[1].plot(frames, cy, label="cy")
    axes[1].set_xlabel("clip frame")
    axes[1].set_ylabel("principal point")
    axes[1].legend()
    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "fx_mean": float(np.mean(fx)),
        "fy_mean": float(np.mean(fy)),
        "cx_mean": float(np.mean(cx)),
        "cy_mean": float(np.mean(cy)),
        "fx_std": float(np.std(fx)),
        "fy_std": float(np.std(fy)),
    }


def visualize_scene(
    scene_name: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    scene = SintelScene(Path(args.root), scene_name, sintel_pass=args.sintel_pass)
    frame_indices, num_real_frames = build_frame_indices(scene.num_frames, args.num_frames, args.frame_stride, args.start_frame)
    result = scene.load_clip(frame_indices, args.resolution)
    result.sequence_name = scene_name
    result.metadata["dataset_name"] = "sintel"
    result.metadata["sequence_name"] = scene_name

    reference_frame = args.reference_frame
    if reference_frame < 0:
        reference_frame = len(result.images) // 2
    if reference_frame >= num_real_frames:
        reference_frame = max(num_real_frames - 1, 0)
    if not (0 <= reference_frame < len(result.images)):
        raise ValueError(f"reference_frame={reference_frame} out of range for {scene_name} ({len(result.images)} clip frames)")

    scene_dir = out_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    dense_pred = compute_dense_pred_reference_sequence(
        model=model,
        result=result,
        reference_frame=reference_frame,
        stride=args.dense_stride,
        vis_threshold=args.dense_vis_threshold,
        batch_size=args.dense_query_batch_size,
        device=device,
        flip_y=False,
        patch_provider=args.patch_provider,
        depth_percentile=100.0,
        query_depth_percentile=args.dense_query_depth_percentile,
        confidence_percentile=args.dense_confidence_percentile,
    )
    pred_static = scene_dir / "pred_dense_reference_static.png"
    pred_gif = scene_dir / "pred_dense_reference.gif"
    pred_ply = scene_dir / f"pred_dense_reference_frame_{reference_frame:03d}.ply"
    dense_pred_metrics = plot_dense_pred_reference_static(result, dense_pred, pred_static)
    write_dense_pred_reference_gif(result, dense_pred, pred_gif, args.gif_fps)
    save_point_cloud_ply_allow_empty(pred_ply, dense_pred[reference_frame]["points_ref"], colors=dense_pred[reference_frame]["colors"])

    dense_gt = prepare_dense_gt_depth_sequence(
        result=result,
        max_points=args.dense_gt_max_points,
        seed=42 + abs(hash(scene_name)) % 100000,
        flip_y=False,
        stride=args.dense_gt_depth_stride,
        max_depth=args.depth_gt_max_depth,
    )
    gt_static = scene_dir / "gt_dense_dynamic_world_static.png"
    gt_gif = scene_dir / "gt_dense_dynamic_world.gif"
    gt_ply = scene_dir / f"gt_dense_world_frame_{reference_frame:03d}.ply"
    dense_gt_metrics = plot_dense_gt_static(result, dense_gt, gt_static)
    write_dense_gt_gif(result, dense_gt, gt_gif, args.gif_fps)
    save_point_cloud_ply_allow_empty(
        gt_ply,
        dense_gt[reference_frame]["points_world"],
        colors=dense_gt[reference_frame]["colors"],
    )

    pred_world_static = scene_dir / "pred_dense_world_static.png"
    pred_world_gif = scene_dir / "pred_dense_world.gif"
    pred_world_ply = scene_dir / f"pred_dense_world_frame_{reference_frame:03d}.ply"
    dense_pred_world_metrics = plot_dense_pred_world_static(result, dense_pred, dense_gt, pred_world_static)
    write_dense_pred_world_gif(result, dense_pred, pred_world_gif, args.gif_fps, reference_frame_id=reference_frame)
    save_point_cloud_ply_allow_empty(
        pred_world_ply,
        dense_pred[reference_frame]["points_world"],
        colors=dense_pred[reference_frame]["colors"],
    )

    canonical = compute_dense_canonical_sequence(
        model=model,
        result=result,
        reference_frame=reference_frame,
        stride=args.dense_stride,
        vis_threshold=args.dense_vis_threshold,
        batch_size=args.dense_query_batch_size,
        device=device,
        flip_y=False,
        patch_provider=args.patch_provider,
        confidence_percentile=args.dense_confidence_percentile,
    )
    canonical_static = scene_dir / "canonical_static.png"
    canonical_gif = scene_dir / "canonical.gif"
    canonical_ply = scene_dir / f"canonical_frame_{reference_frame:03d}.ply"
    canonical_metrics = plot_dense_canonical_static(result, canonical, canonical_static)
    write_dense_canonical_gif(result, canonical, canonical_gif, args.gif_fps)
    save_point_cloud_ply_allow_empty(canonical_ply, canonical[reference_frame]["points_ref"], colors=canonical[reference_frame]["colors"])

    old_reference_frame = args.reference_frame
    args.reference_frame = reference_frame
    try:
        gt_c2w, pred_c2w, pose_metrics = estimate_camera_trajectory(model, result, args, device, num_real_frames)
    finally:
        args.reference_frame = old_reference_frame
    camera_path = scene_dir / "camera_trajectory_gt_pred.png"
    intrinsics_path = scene_dir / "camera_intrinsics.png"
    plot_camera_trajectory(gt_c2w, pred_c2w, camera_path, f"{scene_name} camera trajectory (GT vs predicted, Sim3 aligned)")
    intrinsics_metrics = plot_intrinsics(result.intrinsics[:num_real_frames], intrinsics_path, f"{scene_name} camera intrinsics")

    summary = {
        "scene_name": scene_name,
        "num_frames": int(scene.num_frames),
        "num_real_frames": int(num_real_frames),
        "frame_indices": [int(i) for i in frame_indices],
        "reference_frame": int(reference_frame),
        "dense_stride": int(args.dense_stride),
        "dense_gt_depth_stride": int(args.dense_gt_depth_stride),
        "depth_gt_max_depth": float(args.depth_gt_max_depth),
        **dense_gt_metrics,
        **dense_pred_metrics,
        **dense_pred_world_metrics,
        **canonical_metrics,
        **pose_metrics,
        **intrinsics_metrics,
        "artifacts": {
            "gt_dense_dynamic_world_static": str(gt_static),
            "gt_dense_dynamic_world_gif": str(gt_gif),
            "gt_dense_world_reference_ply": str(gt_ply),
            "pred_dense_reference_static": str(pred_static),
            "pred_dense_reference_gif": str(pred_gif),
            "pred_dense_reference_ply": str(pred_ply),
            "pred_dense_world_static": str(pred_world_static),
            "pred_dense_world_gif": str(pred_world_gif),
            "pred_dense_world_ply": str(pred_world_ply),
            "canonical_static": str(canonical_static),
            "canonical_gif": str(canonical_gif),
            "canonical_ply": str(canonical_ply),
            "camera_trajectory_gt_pred": str(camera_path),
            "camera_intrinsics": str(intrinsics_path),
        },
    }
    with (scene_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    model = load_model(args, device)
    scenes = select_scenes(Path(args.root), args)

    summaries: list[dict[str, Any]] = []
    for idx, scene_name in enumerate(scenes):
        print(f"[sintel_vis] {idx + 1}/{len(scenes)} {scene_name}", flush=True)
        summaries.append(visualize_scene(scene_name, model, args, device, out_dir))

    metric_keys = [
        "dense_pred_mean_visible_points",
        "dense_pred_world_mean_visible",
        "dense_gt_mean_selected_points",
        "canonical_mean_visible_points",
        "pose_ate",
        "pose_rpe_trans",
        "pose_rpe_rot",
        "pose_auc_30",
        "fx_mean",
        "fy_mean",
    ]
    mean_metrics: dict[str, float] = {}
    for key in metric_keys:
        values = [float(s[key]) for s in summaries if key in s and np.isfinite(float(s[key]))]
        if values:
            mean_metrics[key] = float(np.mean(values))

    final = {
        "checkpoint": args.checkpoint,
        "model_variant": args.model_variant,
        "patch_provider": args.patch_provider,
        "root": args.root,
        "sintel_pass": args.sintel_pass,
        "num_scenes": len(summaries),
        "mean_metrics": mean_metrics,
        "scenes": summaries,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": str(summary_path), "mean_metrics": mean_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
