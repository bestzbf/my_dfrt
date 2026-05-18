#!/usr/bin/env python3
"""Camera-only Sintel evaluation for D4RT checkpoints.

This script follows the camera task protocol described by the D4RT query
interface:

- Extrinsics: query q_i=(u, v, i, i, i) and q_j=(u, v, i, i, j), then solve the
  rigid transform between the two point sets with Umeyama.
- Intrinsics: query q_i=(u, v, i, i, i), assume normalized principal point
  (0.5, 0.5), and recover focal lengths from the pinhole equations.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_sintel_checkpoint import (  # noqa: E402
    SINTEL_14_SEQUENCES,
    SintelScene,
    autocast_context,
    build_frame_indices,
    build_transform_metadata,
    load_model,
    select_device,
    to_patch_frames_tensor,
    to_video_tensor,
)
from utils.camera import compute_relative_pose_error, umeyama_alignment, umeyama_ransac  # noqa: E402
from utils.metrics import compute_pose_auc, compute_pose_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sintel camera extrinsics/intrinsics only.")
    parser.add_argument("--root", type=str, default="/data3/dataset/sintel")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-variant", type=str, default="large", choices=("base", "large", "huge", "giant"))
    parser.add_argument("--videomae-model", type=str, default=None)
    parser.add_argument("--patch-provider", type=str, default="sampled_highres")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--sintel-pass", type=str, default="final", choices=("clean", "final"))
    parser.add_argument("--use-14-subset", action="store_true", default=False)
    parser.add_argument("--all-scenes", action="store_true", default=False)
    parser.add_argument("--scene-names", type=str, default="")
    parser.add_argument("--num-scenes", type=int, default=0)

    parser.add_argument("--pose-mode", type=str, default="adjacent", choices=("reference", "adjacent"))
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--pose-grid-h", type=int, default=16)
    parser.add_argument("--pose-grid-w", type=int, default=16)
    parser.add_argument("--pose-margin", type=float, default=0.1)
    parser.add_argument("--pose-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--pose-confidence-quantile", type=float, default=0.4)
    parser.add_argument("--pose-weight-mode", type=str, default="product", choices=("product", "mean", "min", "none"))
    parser.add_argument("--pose-solver", type=str, default="umeyama", choices=("umeyama", "ransac"))
    parser.add_argument("--pose-ransac-iters", type=int, default=256)
    parser.add_argument("--pose-inlier-thresh", type=float, default=0.1)
    parser.add_argument("--pose-min-points", type=int, default=8)

    parser.add_argument("--intrinsics-grid-h", type=int, default=16)
    parser.add_argument("--intrinsics-grid-w", type=int, default=16)
    parser.add_argument("--intrinsics-margin", type=float, default=0.1)
    parser.add_argument("--intrinsics-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--intrinsics-confidence-quantile", type=float, default=0.2)
    parser.add_argument("--intrinsics-frame-stride", type=int, default=1)
    parser.add_argument("--intrinsics-min-points", type=int, default=8)
    parser.add_argument("--principal-point-x", type=float, default=0.5)
    parser.add_argument("--principal-point-y", type=float, default=0.5)

    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def autocast_for(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def encode_video_camera(model: torch.nn.Module, result, device: torch.device, patch_provider: str):
    video, aspect_ratio = to_video_tensor(result, device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.inference_mode(), autocast_for(device):
        encoder_features = model.encode(video, aspect_ratio)
    frames_bcthw = model._prepare_query_frames(video, patch_frames)
    return encoder_features, frames_bcthw, transform_metadata


def make_grid(h: int, w: int, margin: float, device: torch.device) -> torch.Tensor:
    u = torch.linspace(margin, 1.0 - margin, w, device=device)
    v = torch.linspace(margin, 1.0 - margin, h, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


def decode_points(
    model: torch.nn.Module,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    coords: torch.Tensor,
    t_src_value: int,
    t_tgt_value: int,
    t_cam_value: int,
    query_batch_size: int,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_list: list[torch.Tensor] = []
    conf_list: list[torch.Tensor] = []
    n = coords.shape[0]
    for start in range(0, n, query_batch_size):
        end = min(start + query_batch_size, n)
        chunk = coords[start:end].unsqueeze(0)
        count = end - start
        t_src = torch.full((1, count), t_src_value, device=device, dtype=torch.long)
        t_tgt = torch.full((1, count), t_tgt_value, device=device, dtype=torch.long)
        t_cam = torch.full((1, count), t_cam_value, device=device, dtype=torch.long)
        with torch.inference_mode(), autocast_for(device):
            outputs = model.decode(
                encoder_features,
                frames_bcthw,
                chunk,
                t_src,
                t_tgt,
                t_cam,
                transform_metadata=transform_metadata,
            )
        points_list.append(outputs["pos_3d"].squeeze(0).detach().float().cpu())
        conf_list.append(outputs["confidence_weight"].squeeze(0).squeeze(-1).detach().float().cpu())
    return torch.cat(points_list, dim=0), torch.cat(conf_list, dim=0)


def combine_confidence(conf_i: torch.Tensor, conf_j: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "product":
        return conf_i * conf_j
    if mode == "mean":
        return 0.5 * (conf_i + conf_j)
    if mode == "min":
        return torch.minimum(conf_i, conf_j)
    return torch.ones_like(conf_i)


def confidence_mask(
    weights: torch.Tensor,
    finite_mask: torch.Tensor,
    threshold: float,
    quantile: float,
) -> torch.Tensor:
    valid = finite_mask & torch.isfinite(weights)
    if threshold > 0:
        valid &= weights >= threshold
    if quantile > 0 and valid.any():
        q = min(max(float(quantile), 0.0), 1.0)
        thresh = torch.quantile(weights[valid], q)
        valid &= weights >= thresh
    return valid


def estimate_relative_pose_paper(
    model: torch.nn.Module,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    frame_i: int,
    frame_j: int,
    coords: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    points_i, conf_i = decode_points(
        model, encoder_features, frames_bcthw, coords,
        t_src_value=frame_i, t_tgt_value=frame_i, t_cam_value=frame_i,
        query_batch_size=args.query_batch_size, device=device,
        transform_metadata=transform_metadata,
    )
    points_j, conf_j = decode_points(
        model, encoder_features, frames_bcthw, coords,
        t_src_value=frame_i, t_tgt_value=frame_i, t_cam_value=frame_j,
        query_batch_size=args.query_batch_size, device=device,
        transform_metadata=transform_metadata,
    )

    weights = combine_confidence(conf_i, conf_j, args.pose_weight_mode)
    finite = torch.isfinite(points_i).all(dim=-1) & torch.isfinite(points_j).all(dim=-1)
    valid = confidence_mask(weights, finite, args.pose_confidence_threshold, args.pose_confidence_quantile)
    num_valid = int(valid.sum().item())
    if num_valid < args.pose_min_points:
        return torch.eye(4, dtype=torch.float32), {
            "pose_valid_points": float(num_valid),
            "pose_inlier_points": 0.0,
            "pose_failed_identity": 1.0,
        }

    src = points_i[valid]
    tgt = points_j[valid]
    w = weights[valid] if args.pose_weight_mode != "none" else None

    if args.pose_solver == "ransac":
        R, t, _, inliers = umeyama_ransac(
            src,
            tgt,
            weights=w,
            inlier_thresh=args.pose_inlier_thresh,
            max_iters=args.pose_ransac_iters,
            min_inliers=args.pose_min_points,
        )
        inlier_points = float(inliers.sum().item())
        failed = 1.0 if inlier_points < args.pose_min_points else 0.0
    else:
        R, t, _ = umeyama_alignment(src, tgt, weights=w, with_scale=False)
        inlier_points = float(num_valid)
        failed = 0.0

    pose_i_to_j = torch.eye(4, dtype=torch.float32)
    pose_i_to_j[:3, :3] = R.cpu()
    pose_i_to_j[:3, 3] = t.cpu()
    return pose_i_to_j, {
        "pose_valid_points": float(num_valid),
        "pose_inlier_points": inlier_points,
        "pose_failed_identity": failed,
    }


def compute_pose_for_clip(
    model: torch.nn.Module,
    result,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
    num_real_frames: int,
) -> dict[str, float]:
    coords = make_grid(args.pose_grid_h, args.pose_grid_w, args.pose_margin, device)
    gt_w2c = torch.from_numpy(result.extrinsics[:num_real_frames]).float()
    gt_c2w = torch.linalg.inv(gt_w2c)

    pred_w2c_rel_list: list[torch.Tensor] = []
    rot_errors: list[torch.Tensor] = []
    trans_errors: list[torch.Tensor] = []
    valid_counts: list[float] = []
    inlier_counts: list[float] = []
    failed_count = 0.0

    if args.pose_mode == "reference":
        ref = args.reference_frame
        if not (0 <= ref < num_real_frames):
            raise ValueError(f"reference_frame={ref} out of range for {num_real_frames} real frames")
        ref_w2c = gt_w2c[ref]
        for frame_idx in range(num_real_frames):
            if frame_idx == ref:
                pred_rel = torch.eye(4, dtype=torch.float32)
            else:
                pred_rel, stats = estimate_relative_pose_paper(
                    model, encoder_features, frames_bcthw,
                    frame_i=ref, frame_j=frame_idx, coords=coords,
                    args=args, device=device, transform_metadata=transform_metadata,
                )
                gt_rel = gt_w2c[frame_idx] @ torch.linalg.inv(ref_w2c)
                rel_errors = compute_relative_pose_error(
                    pred_rel[:3, :3], pred_rel[:3, 3],
                    gt_rel[:3, :3], gt_rel[:3, 3],
                )
                rot_errors.append(rel_errors["rotation_error"])
                trans_errors.append(rel_errors["translation_error"])
                valid_counts.append(stats["pose_valid_points"])
                inlier_counts.append(stats["pose_inlier_points"])
                failed_count += stats["pose_failed_identity"]
            pred_w2c_rel_list.append(pred_rel)
    else:
        current = torch.eye(4, dtype=torch.float32)
        pred_w2c_rel_list.append(current.clone())
        for frame_idx in range(1, num_real_frames):
            pred_step, stats = estimate_relative_pose_paper(
                model, encoder_features, frames_bcthw,
                frame_i=frame_idx - 1, frame_j=frame_idx, coords=coords,
                args=args, device=device, transform_metadata=transform_metadata,
            )
            current = pred_step @ current
            gt_step = gt_w2c[frame_idx] @ torch.linalg.inv(gt_w2c[frame_idx - 1])
            rel_errors = compute_relative_pose_error(
                pred_step[:3, :3], pred_step[:3, 3],
                gt_step[:3, :3], gt_step[:3, 3],
            )
            rot_errors.append(rel_errors["rotation_error"])
            trans_errors.append(rel_errors["translation_error"])
            valid_counts.append(stats["pose_valid_points"])
            inlier_counts.append(stats["pose_inlier_points"])
            failed_count += stats["pose_failed_identity"]
            pred_w2c_rel_list.append(current.clone())

    pred_c2w = torch.linalg.inv(torch.stack(pred_w2c_rel_list, dim=0))
    pose_metrics = compute_pose_metrics(pred_c2w, gt_c2w[:num_real_frames], align=True)
    if rot_errors:
        pose_auc_30 = compute_pose_auc(torch.stack(rot_errors), torch.stack(trans_errors), threshold=30.0)
        pair_rot = torch.stack(rot_errors).mean()
        pair_trans = torch.stack(trans_errors).mean()
    else:
        pose_auc_30 = torch.tensor(float("nan"))
        pair_rot = torch.tensor(float("nan"))
        pair_trans = torch.tensor(float("nan"))

    return {
        "pose_ate": float(pose_metrics["ate"].item()),
        "pose_rpe_trans": float(pose_metrics["rpe_trans"].item()),
        "pose_rpe_rot": float(pose_metrics["rpe_rot"].item()),
        "pose_auc_30": float(pose_auc_30.item() * 100.0),
        "pose_auc_30_frac": float(pose_auc_30.item()),
        "pose_pair_rot_error": float(pair_rot.item()),
        "pose_pair_trans_error": float(pair_trans.item()),
        "pose_num_pairs": float(len(rot_errors)),
        "pose_valid_points_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "pose_inlier_points_mean": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "pose_failed_identity_pairs": float(failed_count),
    }


def estimate_intrinsics_for_frame(
    model: torch.nn.Module,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    frame_idx: int,
    coords: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    points, conf = decode_points(
        model, encoder_features, frames_bcthw, coords,
        t_src_value=frame_idx, t_tgt_value=frame_idx, t_cam_value=frame_idx,
        query_batch_size=args.query_batch_size, device=device,
        transform_metadata=transform_metadata,
    )
    u = coords.detach().cpu()[:, 0]
    v = coords.detach().cpu()[:, 1]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    finite = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(conf)
        & (z > 1e-6)
        & (x.abs() > 1e-6)
        & (y.abs() > 1e-6)
    )
    valid = confidence_mask(conf, finite, args.intrinsics_confidence_threshold, args.intrinsics_confidence_quantile)
    fx_values = z[valid] * (u[valid] - args.principal_point_x) / x[valid]
    fy_values = z[valid] * (v[valid] - args.principal_point_y) / y[valid]
    positive = torch.isfinite(fx_values) & torch.isfinite(fy_values) & (fx_values > 0.0) & (fy_values > 0.0)
    fx_values = fx_values[positive]
    fy_values = fy_values[positive]

    if fx_values.numel() < args.intrinsics_min_points:
        return {
            "intrinsics_fx_norm_pred": float("nan"),
            "intrinsics_fy_norm_pred": float("nan"),
            "intrinsics_valid_points": float(fx_values.numel()),
        }
    return {
        "intrinsics_fx_norm_pred": float(torch.median(fx_values).item()),
        "intrinsics_fy_norm_pred": float(torch.median(fy_values).item()),
        "intrinsics_valid_points": float(fx_values.numel()),
    }


def compute_intrinsics_for_clip(
    model: torch.nn.Module,
    result,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
    num_real_frames: int,
) -> dict[str, float]:
    coords = make_grid(args.intrinsics_grid_h, args.intrinsics_grid_w, args.intrinsics_margin, device)
    pred_fx: list[float] = []
    pred_fy: list[float] = []
    gt_fx: list[float] = []
    gt_fy: list[float] = []
    valid_counts: list[float] = []
    denom = float(max(result.img_size - 1, 1))

    for frame_idx in range(0, num_real_frames, max(args.intrinsics_frame_stride, 1)):
        stats = estimate_intrinsics_for_frame(
            model, encoder_features, frames_bcthw, frame_idx, coords,
            args=args, device=device, transform_metadata=transform_metadata,
        )
        fx = stats["intrinsics_fx_norm_pred"]
        fy = stats["intrinsics_fy_norm_pred"]
        valid_counts.append(stats["intrinsics_valid_points"])
        if not (math.isfinite(fx) and math.isfinite(fy)):
            continue
        pred_fx.append(fx)
        pred_fy.append(fy)
        gt_fx.append(float(result.intrinsics[frame_idx, 0, 0]) / denom)
        gt_fy.append(float(result.intrinsics[frame_idx, 1, 1]) / denom)

    if not pred_fx:
        return {
            "intrinsics_num_frames": 0.0,
            "intrinsics_valid_points_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
            "intrinsics_fx_abs_rel": float("nan"),
            "intrinsics_fy_abs_rel": float("nan"),
            "intrinsics_focal_abs_rel": float("nan"),
        }

    pfx = np.asarray(pred_fx, dtype=np.float64)
    pfy = np.asarray(pred_fy, dtype=np.float64)
    gfx = np.asarray(gt_fx, dtype=np.float64)
    gfy = np.asarray(gt_fy, dtype=np.float64)
    fx_abs_rel = np.mean(np.abs(pfx - gfx) / np.maximum(np.abs(gfx), 1e-8))
    fy_abs_rel = np.mean(np.abs(pfy - gfy) / np.maximum(np.abs(gfy), 1e-8))
    focal_abs_rel = 0.5 * (fx_abs_rel + fy_abs_rel)
    return {
        "intrinsics_num_frames": float(len(pred_fx)),
        "intrinsics_valid_points_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "intrinsics_fx_norm_pred_mean": float(np.mean(pfx)),
        "intrinsics_fy_norm_pred_mean": float(np.mean(pfy)),
        "intrinsics_fx_norm_gt_mean": float(np.mean(gfx)),
        "intrinsics_fy_norm_gt_mean": float(np.mean(gfy)),
        "intrinsics_fx_abs_rel": float(fx_abs_rel),
        "intrinsics_fy_abs_rel": float(fy_abs_rel),
        "intrinsics_focal_abs_rel": float(focal_abs_rel),
    }


def evaluate_scene(
    root: Path,
    model: torch.nn.Module,
    scene_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    scene = SintelScene(root, scene_name, sintel_pass=args.sintel_pass)
    frame_indices, num_real_frames = build_frame_indices(
        scene.num_frames, args.num_frames, args.frame_stride, args.start_frame
    )
    result = scene.load_clip(frame_indices, args.resolution)
    encoder_features, frames_bcthw, transform_metadata = encode_video_camera(
        model, result, device, args.patch_provider
    )
    pose_metrics = compute_pose_for_clip(
        model, result, encoder_features, frames_bcthw,
        args=args, device=device, transform_metadata=transform_metadata,
        num_real_frames=num_real_frames,
    )
    intrinsics_metrics = compute_intrinsics_for_clip(
        model, result, encoder_features, frames_bcthw,
        args=args, device=device, transform_metadata=transform_metadata,
        num_real_frames=num_real_frames,
    )
    return {
        "scene_name": scene_name,
        "num_frames": scene.num_frames,
        "num_real_frames": num_real_frames,
        "clip_len": len(frame_indices),
        "frame_indices": frame_indices,
        **pose_metrics,
        **intrinsics_metrics,
    }


def summarize_scene_metrics(scene_summaries: list[dict[str, Any]]) -> dict[str, float]:
    if not scene_summaries:
        return {}
    skip = {"scene_name", "frame_indices", "num_frames", "clip_len"}
    keys = [k for k in scene_summaries[0] if k not in skip]
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for item in scene_summaries:
            try:
                value = float(item[key])
            except Exception:
                continue
            if math.isfinite(value):
                vals.append(value)
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def select_scenes(root: Path, args: argparse.Namespace) -> list[str]:
    if args.scene_names:
        scenes = [s.strip() for s in args.scene_names.split(",") if s.strip()]
    elif args.all_scenes:
        scenes = sorted(d.name for d in (root / "training" / args.sintel_pass).iterdir() if d.is_dir())
    elif args.use_14_subset:
        scenes = list(SINTEL_14_SEQUENCES)
    else:
        scenes = list(SINTEL_14_SEQUENCES)
    if args.num_scenes > 0:
        scenes = scenes[: args.num_scenes]
    return scenes


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    scenes = select_scenes(root, args)
    if not scenes:
        raise RuntimeError("No Sintel scenes selected.")

    print(f"[camera_eval] device={device}")
    print(f"[camera_eval] checkpoint={args.checkpoint}")
    print(f"[camera_eval] scenes={len(scenes)} pose_mode={args.pose_mode} solver={args.pose_solver}")
    print(f"[camera_eval] scenes: {scenes}")
    model, device = load_model(args, device)

    scene_summaries: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, str]] = []
    for idx, scene_name in enumerate(scenes):
        print(f"\n[camera_eval] scene {idx + 1}/{len(scenes)}: {scene_name}", flush=True)
        try:
            summary = evaluate_scene(root, model, scene_name, args, device)
        except Exception as exc:  # noqa: BLE001
            failure = {"scene_name": scene_name, "error": repr(exc)}
            failed_scenes.append(failure)
            print(json.dumps({"scene_name": scene_name, "status": "failed", "error": repr(exc)}, ensure_ascii=False), flush=True)
            continue
        scene_summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "frame_indices"}, ensure_ascii=False), flush=True)

    final = {
        "root": str(root),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "sintel_pass": args.sintel_pass,
        "args": vars(args),
        "protocol": {
            "task": "camera_pose_intrinsics_tuned",
            "paper_pose_subset": (not args.all_scenes) and (not args.scene_names) and args.num_scenes <= 0,
            "pose_mode": args.pose_mode,
            "pose_solver": args.pose_solver,
            "pose_grid_h": args.pose_grid_h,
            "pose_grid_w": args.pose_grid_w,
            "pose_margin": args.pose_margin,
            "pose_confidence_threshold": args.pose_confidence_threshold,
            "pose_confidence_quantile": args.pose_confidence_quantile,
            "pose_weight_mode": args.pose_weight_mode,
            "pose_ransac_iters": args.pose_ransac_iters,
            "pose_inlier_thresh": args.pose_inlier_thresh,
            "pose_min_points": args.pose_min_points,
            "intrinsics_confidence_quantile": args.intrinsics_confidence_quantile,
        },
        "num_requested_scenes": len(scenes),
        "num_scenes": len(scene_summaries),
        "num_failed_scenes": len(failed_scenes),
        "scene_summaries": scene_summaries,
        "failed_scenes": failed_scenes,
        "mean_metrics": summarize_scene_metrics(scene_summaries),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    mean = final["mean_metrics"]
    print("\n" + "=" * 60)
    print(f"[camera_eval] DONE - {len(scene_summaries)} scenes, {len(failed_scenes)} failed")
    print(
        "[camera_eval] Pose: "
        f"ATE={mean.get('pose_ate', float('nan')):.6f} "
        f"RPE-T={mean.get('pose_rpe_trans', float('nan')):.6f} "
        f"RPE-R={mean.get('pose_rpe_rot', float('nan')):.6f} "
        f"AUC@30={mean.get('pose_auc_30', float('nan')):.6f}"
    )
    print(
        "[camera_eval] Intrinsics: "
        f"focal AbsRel={mean.get('intrinsics_focal_abs_rel', float('nan')):.6f} "
        f"fx={mean.get('intrinsics_fx_abs_rel', float('nan')):.6f} "
        f"fy={mean.get('intrinsics_fy_abs_rel', float('nan')):.6f}"
    )
    print(f"[camera_eval] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
