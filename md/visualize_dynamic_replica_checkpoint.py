#!/usr/bin/env python3
"""为 checkpoint 在验证集样本上生成 2D / 3D 可视化结果。"""

from __future__ import annotations

import argparse
import imageio.v2 as imageio
import json
import math
import random
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.factory import create_training_dataset
from models import create_d4rt
from utils.visualization import save_point_cloud_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在 validation 样本上生成 checkpoint 的可视化结果。"
            "输出内容包括 2D 的 GT/Pred/Error 对比图、稀疏 3D GT-vs-Pred 对比图，"
            "以及稠密 GT 动态点云渲染。"
        )
    )
    parser.add_argument("--config", type=str, required=True, help="数据集配置 YAML。脚本会从这里读取 validation split。")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径。支持键 'model' 或 'model_state_dict'。")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录。所有可视化文件和 summary.json 都会写到这里。")
    parser.add_argument(
        "--patch-provider",
        type=str,
        default="sampled_resized",
        help=(
            "推理时 decoder 使用的 patch provider。"
            "Dynamic Replica 通常用 'sampled_resized'，"
            "PointOdyssey high-res checkpoint 通常用 'sampled_highres'。"
        ),
    )
    parser.add_argument("--resolution", type=int, default=256, help="模型输入分辨率，必须和 checkpoint / config 设置一致。")
    parser.add_argument("--num-frames", type=int, default=48, help="clip 长度，必须和 checkpoint 设置一致。")
    parser.add_argument("--num-samples", type=int, default=3, help="生成多少个 validation 样本。")
    parser.add_argument("--start-index", type=int, default=0, help="从哪个 validation index 开始找样本。")
    parser.add_argument("--max-search", type=int, default=200, help="最多向后扫描多少个 validation index 来凑够样本。")
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=-1,
        help="query 采样参考帧。>=0 表示显式帧号；<0 表示使用 clip 中间帧。",
    )
    parser.add_argument("--num-points", type=int, default=1024, help="稀疏 GT-vs-Pred 评估时采样多少个 tracked query 点。")
    parser.add_argument(
        "--num-display-points",
        type=int,
        default=180,
        help="2D 对比静态图和 GIF 中实际绘制多少个稀疏 query 点。",
    )
    parser.add_argument("--gif-fps", type=int, default=8, help="输出 GIF 的播放帧率。")
    parser.add_argument(
        "--dense-gt-max-points",
        type=int,
        default=0,
        help="稠密 GT 点云每帧最多渲染多少个点。0 表示使用该帧按 dense_gt_point_source 选出的全部点。",
    )
    parser.add_argument(
        "--dense-gt-point-source",
        type=str,
        default="visible",
        choices=("visible", "all_finite"),
        help=(
            "稠密 GT 3D 点云每帧用哪一类点。"
            "'visible' 只画当前帧可见点；"
            "'all_finite' 画当前帧所有有限 3D 点。"
        ),
    )
    parser.add_argument(
        "--dense-pred-point-cloud-stride",
        type=int,
        default=2,
        help="稠密预测点云的采样步长。越小越密，越慢。",
    )
    parser.add_argument(
        "--dense-pred-vis-threshold",
        type=float,
        default=0.5,
        help="稠密预测点云保留点的 visibility 阈值。",
    )
    parser.add_argument(
        "--dense-pred-query-batch-size",
        type=int,
        default=4096,
        help="稠密预测点云分块解码时每批 query 数。",
    )
    parser.add_argument("--max-depth", type=float, default=0.0, help="3D 可视化中 mask 掉 Z > max_depth 的远距离点。0 表示不过滤。")
    parser.add_argument(
        "--dense-pred-depth-percentile",
        type=float,
        default=100.0,
        help="稠密预测点云每帧按到原点距离过滤，只保留最近的 N%% 点。默认 100 不过滤，设为 80 则 mask 掉最远 20%%。",
    )
    parser.add_argument(
        "--dense-pred-query-depth-percentile",
        type=float,
        default=50.0,
        help=(
            "在 reference frame 上对格点预测深度（Z轴），只保留最近 N%% 的格点作为 query。"
            "默认 50，即只取前景/近处一半的格点；设为 100 则不过滤。"
        ),
    )
    parser.add_argument("--flip-y-axis", action="store_true", help="对所有 3D 输出和导出的 PLY 执行 Y 轴翻转。")
    parser.add_argument("--seed", type=int, default=42, help="用于点采样复现的随机种子。")
    parser.add_argument("--device", type=str, default="cuda", help="期望使用的设备，可选 'cuda' 或 'cpu'；不可用时会自动回退。")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"), help="使用训练集还是验证集。默认 val。")
    parser.add_argument("--allow-no-tracks", action="store_true", help="允许加载没有 tracks 的样本（用于 CO3Dv2 等静态数据集）。")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        raise KeyError(f"Unsupported checkpoint format in {checkpoint_path}")
    model.load_state_dict(state, strict=True)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = create_d4rt(
        variant="base",
        decoder_depth=6,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=(2, 16, 16),
        query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base",
        patch_provider=args.patch_provider,
    ).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()
    return model


def load_val_dataset(config_path: str, split: str = "val"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return create_training_dataset(config, split=split)


def load_transform_result(dataset, index: int, allow_no_tracks: bool = False):
    sample_index = index
    if dataset.reshuffle_each_epoch:
        sample_index = dataset.current_epoch * dataset.epoch_size + index
    rng = random.Random(dataset.seed + sample_index)

    last_error = None
    for attempt in range(10):
        dataset_idx, sequence_name, frame_indices = dataset.mixture_sampler.sample(rng)
        adapter = dataset.adapters[dataset_idx]
        try:
            clip = adapter.load_clip(sequence_name, frame_indices)
            result = dataset.transform(clip, rng=rng)
            if not allow_no_tracks and not bool(result.metadata.get("has_tracks", False)):
                raise RuntimeError("sample has no tracks")
            return result, dataset_idx, sequence_name, frame_indices
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            rng = random.Random(dataset.seed + index + attempt + 1)
    raise RuntimeError(f"Failed to load valid sample at index {index}: {last_error}")


def find_samples(dataset, start_index: int, count: int, max_search: int, allow_no_tracks: bool = False):
    found: list[dict[str, Any]] = []
    checked = 0
    idx = start_index
    while len(found) < count and checked < max_search:
        try:
            result, dataset_idx, sequence_name, frame_indices = load_transform_result(dataset, idx, allow_no_tracks=allow_no_tracks)
            found.append(
                {
                    "sample_index": idx,
                    "dataset_idx": dataset_idx,
                    "sequence_name": sequence_name,
                    "frame_indices": frame_indices,
                    "result": result,
                }
            )
        except Exception:
            pass
        idx += 1
        checked += 1
    if len(found) < count:
        raise RuntimeError(f"Only found {len(found)} trackable samples within {max_search} val indices")
    return found


def sample_query_points(
    result,
    num_points: int,
    seed: int,
    reference_frame: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajs_2d = result.trajs_2d
    trajs_3d_world = result.trajs_3d_world
    visibs = result.visibs
    if trajs_2d is None or trajs_3d_world is None or visibs is None:
        raise ValueError("result does not contain trajectories")

    if reference_frame < 0:
        t_ref = len(result.images) // 2
    else:
        t_ref = reference_frame
    if t_ref < 0 or t_ref >= len(result.images):
        raise ValueError(
            f"reference_frame={reference_frame} is out of range for clip length {len(result.images)}"
        )
    valid = visibs[t_ref].astype(bool)
    valid &= np.isfinite(trajs_2d[t_ref]).all(axis=-1)
    valid &= np.isfinite(trajs_3d_world[t_ref]).all(axis=-1)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise RuntimeError("no visible points in reference frame")

    rng = np.random.default_rng(seed)
    if len(indices) > num_points:
        indices = rng.choice(indices, size=num_points, replace=False)

    norm = np.array(
        [max(result.crop.crop_w - 1, 1), max(result.crop.crop_h - 1, 1)],
        dtype=np.float32,
    )
    query_points = trajs_2d[t_ref, indices] / norm[None]
    query_points = np.clip(query_points, 0.0, 1.0)
    query_frames = np.full((len(indices),), t_ref, dtype=np.int64)
    return t_ref, indices, query_points.astype(np.float32), query_frames, norm


def to_video_tensor(result, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    video_np = np.stack(result.images, axis=0).astype(np.float32)
    video = torch.from_numpy(video_np).unsqueeze(0).to(device)
    aspect_ratio = torch.tensor(
        [[float(result.crop.crop_w) / max(float(result.crop.crop_h), 1.0)]],
        dtype=torch.float32,
        device=device,
    )
    return video, aspect_ratio


def to_patch_frames_tensor(result, device: torch.device) -> torch.Tensor | None:
    cropped_images = getattr(result, "cropped_images", None)
    if not cropped_images:
        return None

    crop_h, crop_w = cropped_images[0].shape[:2]
    for frame_idx, image in enumerate(cropped_images):
        if image.shape[:2] != (crop_h, crop_w):
            raise ValueError(
                f"inconsistent cropped_images shape at frame {frame_idx}: "
                f"{image.shape[:2]} vs expected {(crop_h, crop_w)}"
            )

    if crop_h == result.img_size and crop_w == result.img_size:
        return None

    patch_np = np.stack(cropped_images, axis=0).astype(np.float32)
    return torch.from_numpy(patch_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def build_transform_metadata(result, device: torch.device) -> dict[str, torch.Tensor]:
    crop = result.crop
    return {
        "canonical_space": torch.tensor([0], dtype=torch.long, device=device),
        "original_hw": torch.tensor(
            [[float(result.original_h), float(result.original_w)]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_offset_xy": torch.tensor(
            [[float(crop.x0), float(crop.y0)]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_size_hw": torch.tensor(
            [[float(crop.crop_h), float(crop.crop_w)]],
            dtype=torch.float32,
            device=device,
        ),
        "resized_hw": torch.tensor(
            [[float(result.img_size), float(result.img_size)]],
            dtype=torch.float32,
            device=device,
        ),
    }


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_world.astype(np.float32), ones], axis=-1)
    points_cam = (extrinsic.astype(np.float32) @ points_h.T).T[:, :3]
    return points_cam.astype(np.float32)


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """将相机坐标系点转换到世界坐标系。extrinsic 是 world→camera 的 4×4 矩阵。"""
    inv_ext = np.linalg.inv(extrinsic.astype(np.float64)).astype(np.float32)
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_cam.astype(np.float32), ones], axis=-1)
    return (inv_ext @ points_h.T).T[:, :3]


def normalized_to_pixels(coords_norm: np.ndarray, size: int) -> np.ndarray:
    return coords_norm * float(max(size - 1, 1))


def point_colors_from_frame(
    image: np.ndarray,
    points_px: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]
    colors = np.full((points_px.shape[0], image.shape[2]), 0.65, dtype=np.float32)
    valid = np.isfinite(points_px).all(axis=-1)
    if not np.any(valid):
        return colors

    xs = np.round(points_px[valid, 0]).astype(np.int32)
    ys = np.round(points_px[valid, 1]).astype(np.int32)
    inbounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if np.any(inbounds):
        colors_valid = colors[valid]
        colors_valid[inbounds] = image[ys[inbounds], xs[inbounds]].astype(np.float32)
        colors[valid] = colors_valid
    return colors


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[..., :3].copy()


def maybe_flip_y(points: np.ndarray, flip_y: bool) -> np.ndarray:
    if not flip_y:
        return points
    out = points.copy()
    out[:, 1] *= -1.0
    return out


def save_point_cloud_ply_allow_empty(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    if colors is not None and len(points) > 0:
        save_point_cloud_ply(str(output_path), points, colors=colors)
        return
    save_point_cloud_ply(str(output_path), points, colors=None)


def compute_track_predictions(
    model: torch.nn.Module,
    result,
    point_indices: np.ndarray,
    query_points: np.ndarray,
    query_frames: np.ndarray,
    device: torch.device,
    patch_provider: str,
) -> dict[str, Any]:
    video, aspect_ratio = to_video_tensor(result, device)
    qp = torch.from_numpy(query_points).unsqueeze(0).to(device)
    qf = torch.from_numpy(query_frames).unsqueeze(0).to(device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model.predict_point_tracks(
                video=video,
                query_points=qp,
                query_frames=qf,
                aspect_ratio=aspect_ratio,
                patch_frames=patch_frames,
                transform_metadata=transform_metadata,
            )

    pred_2d = outputs["tracks_2d"][0].detach().float().cpu().numpy()
    pred_3d = outputs["tracks_3d"][0].detach().float().cpu().numpy()
    pred_vis = outputs["visibility"][0].detach().float().cpu().numpy()

    gt_xy_crop = result.trajs_2d[:, point_indices].transpose(1, 0, 2).astype(np.float32)
    gt_vis = result.visibs[:, point_indices].transpose(1, 0).astype(bool)
    gt_3d_world = result.trajs_3d_world[:, point_indices].transpose(1, 0, 2).astype(np.float32)

    T = gt_3d_world.shape[1]
    gt_3d_cam = np.stack(
        [world_to_camera(gt_3d_world[:, t], result.extrinsics[t]) for t in range(T)],
        axis=1,
    )

    size = result.img_size
    gt_2d_norm = gt_xy_crop / np.array(
        [max(result.crop.crop_w - 1, 1), max(result.crop.crop_h - 1, 1)],
        dtype=np.float32,
    )[None, None]
    pred_2d_px = normalized_to_pixels(pred_2d, size)
    gt_2d_px = normalized_to_pixels(gt_2d_norm, size)

    vis_mask = gt_vis & np.isfinite(gt_2d_px).all(axis=-1) & np.isfinite(pred_2d_px).all(axis=-1)
    e2d = np.linalg.norm(pred_2d_px - gt_2d_px, axis=-1)
    e2d = np.where(vis_mask, e2d, np.nan)

    mask3d = gt_vis & np.isfinite(gt_3d_cam).all(axis=-1) & np.isfinite(pred_3d).all(axis=-1)
    e3d = np.linalg.norm(pred_3d - gt_3d_cam, axis=-1)
    e3d = np.where(mask3d, e3d, np.nan)

    src_frame = int(query_frames[0])
    src_points_px = gt_2d_px[:, src_frame]
    colors = point_colors_from_frame(result.images[src_frame], src_points_px)

    return {
        "pred_2d_norm": pred_2d,
        "pred_2d_px": pred_2d_px,
        "pred_3d_cam": pred_3d,
        "pred_vis": pred_vis,
        "gt_2d_px": gt_2d_px,
        "gt_vis": gt_vis,
        "gt_3d_cam": gt_3d_cam,
        "err_2d_px": e2d,
        "err_3d": e3d,
        "colors": colors,
    }


def subsample_for_display(
    point_indices: np.ndarray,
    gt_vis: np.ndarray,
    num_display_points: int,
    seed: int,
) -> np.ndarray:
    visible_any = np.flatnonzero(gt_vis.any(axis=1))
    if len(visible_any) == 0:
        visible_any = np.arange(len(point_indices))
    rng = np.random.default_rng(seed)
    if len(visible_any) > num_display_points:
        return np.sort(rng.choice(visible_any, size=num_display_points, replace=False))
    return visible_any


def make_frame_ids(total_frames: int, num_frames: int) -> list[int]:
    frame_ids = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    return list(dict.fromkeys(frame_ids.tolist()))


def _error_vmax(err: np.ndarray, fallback: float = 10.0) -> float:
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return fallback
    return max(float(np.percentile(finite, 95)), 1.0)


def plot_2d_compare_static(
    result,
    pred: dict[str, Any],
    display_ids: np.ndarray,
    output_path: Path,
) -> None:
    T = len(result.images)
    frame_ids = make_frame_ids(T, num_frames=5)
    ncols = len(frame_ids)

    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(4.2 * ncols, 11.0),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    point_colors = pred["colors"][display_ids]
    vmax = _error_vmax(pred["err_2d_px"][display_ids])
    scat = None

    row_labels = ["GT", "Pred", "Error"]
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=12)

    for col, frame_id in enumerate(frame_ids):
        image = result.images[frame_id]
        gt_vis = pred["gt_vis"][display_ids, frame_id]
        pd_vis = pred["pred_vis"][display_ids, frame_id] > 0.5
        both_vis = gt_vis & pd_vis

        gt_pts = pred["gt_2d_px"][display_ids, frame_id]
        pd_pts = pred["pred_2d_px"][display_ids, frame_id]
        err = pred["err_2d_px"][display_ids, frame_id]

        ax_gt = axes[0, col]
        ax_pd = axes[1, col]
        ax_er = axes[2, col]
        for ax in (ax_gt, ax_pd, ax_er):
            ax.imshow(image)
            ax.axis("off")

        # 画轨迹线（从第0帧到当前帧）
        for pid in range(len(display_ids)):
            color = point_colors[pid]
            trail_gt = pred["gt_2d_px"][display_ids[pid], :frame_id + 1]
            trail_pd = pred["pred_2d_px"][display_ids[pid], :frame_id + 1]
            trail_gt_vis = pred["gt_vis"][display_ids[pid], :frame_id + 1]
            trail_pd_vis = pred["pred_vis"][display_ids[pid], :frame_id + 1] > 0.5
            if trail_gt_vis.sum() >= 2:
                ax_gt.plot(trail_gt[trail_gt_vis, 0], trail_gt[trail_gt_vis, 1],
                           color=color, linewidth=0.8, alpha=0.5)
            if trail_pd_vis.sum() >= 2:
                ax_pd.plot(trail_pd[trail_pd_vis, 0], trail_pd[trail_pd_vis, 1],
                           color=color, linewidth=0.8, alpha=0.5)

        if np.any(gt_vis):
            ax_gt.scatter(
                gt_pts[gt_vis, 0],
                gt_pts[gt_vis, 1],
                s=28,
                c=point_colors[gt_vis],
                alpha=1.0,
                linewidths=0.6,
                edgecolors="white",
            )

        if np.any(pd_vis):
            ax_pd.scatter(
                pd_pts[pd_vis, 0],
                pd_pts[pd_vis, 1],
                s=28,
                c=point_colors[pd_vis],
                alpha=1.0,
                linewidths=0.6,
                edgecolors="white",
            )

        scat = None
        if np.any(both_vis):
            for gt_pt, pd_pt in zip(gt_pts[both_vis], pd_pts[both_vis], strict=False):
                ax_er.plot(
                    [gt_pt[0], pd_pt[0]],
                    [gt_pt[1], pd_pt[1]],
                    color="#ffffff",
                    linewidth=0.45,
                    alpha=0.35,
                )
            ax_er.scatter(
                gt_pts[both_vis, 0],
                gt_pts[both_vis, 1],
                s=22,
                facecolors="none",
                edgecolors="#00ff95",
                linewidths=0.9,
            )
            scat = ax_er.scatter(
                pd_pts[both_vis, 0],
                pd_pts[both_vis, 1],
                s=12,
                c=np.clip(err[both_vis], 0.0, vmax),
                cmap="turbo",
                vmin=0.0,
                vmax=vmax,
                alpha=0.95,
            )

        ax_gt.set_title(f"frame {frame_id}\nGT vis={int(gt_vis.sum())}", fontsize=10)
        ax_pd.set_title(f"frame {frame_id}\nPred vis={int(pd_vis.sum())}", fontsize=10)
        frame_err = err[both_vis]
        finite_frame_err = frame_err[np.isfinite(frame_err)]
        frame_mean = float(np.mean(finite_frame_err)) if finite_frame_err.size > 0 else float("nan")
        ax_er.set_title(f"frame {frame_id}\nmean err={frame_mean:.2f}px", fontsize=10)

    if scat is not None:
        cbar = fig.colorbar(scat, ax=axes.ravel().tolist(), fraction=0.016, pad=0.01)
        cbar.set_label("2D error (px)")
    fig.suptitle(
        f"{result.sequence_name}  2D track comparison\n"
        f"displayed points={len(display_ids)}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_2d_compare_gif(
    result,
    pred: dict[str, Any],
    display_ids: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    point_colors = pred["colors"][display_ids]
    vmax = _error_vmax(pred["err_2d_px"][display_ids])
    frames_rgb: list[np.ndarray] = []

    for frame_id in range(len(result.images)):
        image = result.images[frame_id]
        gt_vis = pred["gt_vis"][display_ids, frame_id]
        pd_vis = pred["pred_vis"][display_ids, frame_id] > 0.5
        both_vis = gt_vis & pd_vis
        gt_pts = pred["gt_2d_px"][display_ids, frame_id]
        pd_pts = pred["pred_2d_px"][display_ids, frame_id]
        err = pred["err_2d_px"][display_ids, frame_id]

        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
        titles = ["GT", "Pred", "Error"]
        for ax, title in zip(axes, titles, strict=True):
            ax.imshow(image)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        # 画轨迹线（从第0帧到当前帧）
        for i, pid in enumerate(range(len(display_ids))):
            color = point_colors[pid]
            trail_gt = pred["gt_2d_px"][display_ids[pid], :frame_id + 1]
            trail_pd = pred["pred_2d_px"][display_ids[pid], :frame_id + 1]
            trail_gt_vis = pred["gt_vis"][display_ids[pid], :frame_id + 1]
            trail_pd_vis = pred["pred_vis"][display_ids[pid], :frame_id + 1] > 0.5
            if trail_gt_vis.sum() >= 2:
                axes[0].plot(trail_gt[trail_gt_vis, 0], trail_gt[trail_gt_vis, 1],
                             color=color, linewidth=0.8, alpha=0.5)
            if trail_pd_vis.sum() >= 2:
                axes[1].plot(trail_pd[trail_pd_vis, 0], trail_pd[trail_pd_vis, 1],
                             color=color, linewidth=0.8, alpha=0.5)

        if np.any(gt_vis):
            axes[0].scatter(
                gt_pts[gt_vis, 0],
                gt_pts[gt_vis, 1],
                s=28,
                c=point_colors[gt_vis],
                alpha=1.0,
                linewidths=0.6,
                edgecolors="white",
            )

        if np.any(pd_vis):
            axes[1].scatter(
                pd_pts[pd_vis, 0],
                pd_pts[pd_vis, 1],
                s=28,
                c=point_colors[pd_vis],
                alpha=1.0,
                linewidths=0.6,
                edgecolors="white",
            )

        if np.any(both_vis):
            for gt_pt, pd_pt in zip(gt_pts[both_vis], pd_pts[both_vis], strict=False):
                axes[2].plot(
                    [gt_pt[0], pd_pt[0]],
                    [gt_pt[1], pd_pt[1]],
                    color="#ffffff",
                    linewidth=0.45,
                    alpha=0.35,
                )
            axes[2].scatter(
                gt_pts[both_vis, 0],
                gt_pts[both_vis, 1],
                s=22,
                facecolors="none",
                edgecolors="#00ff95",
                linewidths=0.9,
            )
            scat = axes[2].scatter(
                pd_pts[both_vis, 0],
                pd_pts[both_vis, 1],
                s=12,
                c=np.clip(err[both_vis], 0.0, vmax),
                cmap="turbo",
                vmin=0.0,
                vmax=vmax,
                alpha=0.95,
            )
            cbar = fig.colorbar(scat, ax=axes, fraction=0.018, pad=0.01)
            cbar.set_label("2D error (px)")

        frame_err = err[both_vis]
        finite_frame_err = frame_err[np.isfinite(frame_err)]
        frame_mean = float(np.mean(finite_frame_err)) if finite_frame_err.size > 0 else float("nan")
        fig.suptitle(
            f"{result.sequence_name}  frame {frame_id}  "
            f"GT vis={int(gt_vis.sum())}  Pred vis={int(pd_vis.sum())}  mean err={frame_mean:.2f}px",
            fontsize=13,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def prepare_dense_gt_sequence(
    result,
    max_points: int,
    seed: int,
    flip_y: bool,
    point_source: str,
    outlier_percentile: float = 95.0,
) -> list[dict[str, Any]]:
    norm = np.array(
        [max(result.crop.crop_w - 1, 1), max(result.crop.crop_h - 1, 1)],
        dtype=np.float32,
    )
    dense_frames: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    # 跨帧统计 XYZ 范围，用 percentile 裁剪离群点阈值
    # 避免少量极端值（如 PointOdyssey Y 轴 -878）把整个点云压缩到不可见
    if outlier_percentile < 100.0:
        all_finite = result.trajs_3d_world[np.isfinite(result.trajs_3d_world).all(axis=-1)]
        if len(all_finite) > 0:
            lo = np.percentile(all_finite, 100.0 - outlier_percentile, axis=0)
            hi = np.percentile(all_finite, outlier_percentile, axis=0)
        else:
            lo = np.full(3, -np.inf, dtype=np.float32)
            hi = np.full(3, np.inf, dtype=np.float32)
    else:
        lo = np.full(3, -np.inf, dtype=np.float32)
        hi = np.full(3, np.inf, dtype=np.float32)

    for frame_id in range(len(result.images)):
        finite_3d = np.isfinite(result.trajs_3d_world[frame_id]).all(axis=-1)
        finite_2d = np.isfinite(result.trajs_2d[frame_id]).all(axis=-1)
        if point_source == "visible":
            selected = result.visibs[frame_id].astype(bool) & finite_3d & finite_2d
        elif point_source == "all_finite":
            selected = finite_3d
        else:
            raise ValueError(f"Unsupported dense GT point source: {point_source}")

        # 过滤离群点
        if outlier_percentile < 100.0:
            pts_all = result.trajs_3d_world[frame_id]
            in_range = np.all((pts_all >= lo) & (pts_all <= hi), axis=-1)
            selected = selected & in_range

        indices = np.flatnonzero(selected)
        total_selected = int(len(indices))
        if max_points > 0 and len(indices) > max_points:
            indices = rng.choice(indices, size=max_points, replace=False)
            indices = np.sort(indices)

        points_world = result.trajs_3d_world[frame_id, indices].astype(np.float32)
        points_world = maybe_flip_y(points_world, flip_y)
        coords_norm = result.trajs_2d[frame_id, indices].astype(np.float32) / norm[None]
        points_px = normalized_to_pixels(coords_norm, result.img_size)
        colors = point_colors_from_frame(result.images[frame_id], points_px)
        dense_frames.append(
            {
                "frame_id": frame_id,
                "points_world": points_world,
                "colors": colors,
                "selected_points": total_selected,
                "rendered_points": int(len(points_world)),
            }
        )
    return dense_frames


def compute_dense_pred_reference_sequence(
    model: torch.nn.Module,
    result,
    reference_frame: int,
    stride: int,
    vis_threshold: float,
    batch_size: int,
    device: torch.device,
    flip_y: bool,
    patch_provider: str,
    depth_percentile: float = 100.0,
    query_depth_percentile: float = 50.0,
) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError(f"dense-pred-point-cloud-stride must be positive, got {stride}")
    if batch_size <= 0:
        raise ValueError(f"dense-pred-query-batch-size must be positive, got {batch_size}")

    video, aspect_ratio = to_video_tensor(result, device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoder_features = model.encode(video, aspect_ratio)

    frames_bcthw = model._prepare_query_frames(video, patch_frames)
    total_frames = len(result.images)
    height, width = result.images[0].shape[:2]

    x_pixels = np.arange(0, width, stride, dtype=np.int32)
    y_pixels = np.arange(0, height, stride, dtype=np.int32)
    if len(x_pixels) == 0 or len(y_pixels) == 0:
        raise RuntimeError(f"stride={stride} produced empty dense grid for resolution {width}x{height}")

    grid_x, grid_y = np.meshgrid(x_pixels, y_pixels, indexing="xy")
    coords_px = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    coords_norm = coords_px.astype(np.float32) / np.array(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=np.float32,
    )[None]
    coords_torch = torch.from_numpy(coords_norm).to(device=device, dtype=torch.float32)
    reference_colors = result.images[reference_frame][coords_px[:, 1], coords_px[:, 0]].astype(np.float32)

    # ── reference frame query 深度过滤 ──────────────────────────────────────
    # 先对 reference frame 自身 decode 一次（t_src=t_tgt=reference_frame），
    # 用预测的 Z 轴深度选出最近 query_depth_percentile% 的格点，
    # 后续所有帧只对这批格点 decode，既减少推理量又聚焦近处。
    if query_depth_percentile < 100.0:
        ref_depth_points: list[np.ndarray] = []
        for start in range(0, len(coords_norm), batch_size):
            end = min(start + batch_size, len(coords_norm))
            coords_chunk = coords_torch[start:end].unsqueeze(0)
            chunk_size = end - start
            t_ref_tensor = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    ref_out = model.decode(
                        encoder_features,
                        frames_bcthw,
                        coords_chunk,
                        t_ref_tensor,
                        t_ref_tensor,
                        t_ref_tensor,
                        transform_metadata=transform_metadata,
                    )
            ref_depth_points.append(ref_out["pos_3d"][0].detach().float().cpu().numpy())
        ref_points_all = np.concatenate(ref_depth_points, axis=0)   # [N, 3]，参考帧相机坐标系
        finite_mask = np.isfinite(ref_points_all).all(axis=-1)
        ref_z = np.where(finite_mask, ref_points_all[:, 2], np.inf)
        depth_thresh = np.percentile(ref_z[finite_mask], query_depth_percentile) if finite_mask.any() else np.inf
        query_keep = finite_mask & (ref_z <= depth_thresh)
        coords_torch = coords_torch[query_keep]
        reference_colors = reference_colors[query_keep]
        coords_px = coords_px[query_keep]
    # ────────────────────────────────────────────────────────────────────────

    dense_frames: list[dict[str, Any]] = []
    total_points = int(coords_torch.shape[0])

    for frame_id in range(total_frames):
        points_list: list[np.ndarray] = []
        vis_list: list[np.ndarray] = []

        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            coords_chunk = coords_torch[start:end].unsqueeze(0)
            chunk_size = end - start
            t_src = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)
            t_tgt = torch.full((1, chunk_size), frame_id, device=device, dtype=torch.long)
            t_cam = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model.decode(
                        encoder_features,
                        frames_bcthw,
                        coords_chunk,
                        t_src,
                        t_tgt,
                        t_cam,
                        transform_metadata=transform_metadata,
                    )

            points_list.append(outputs["pos_3d"][0].detach().float().cpu().numpy())
            vis_list.append(torch.sigmoid(outputs["visibility"][0].squeeze(-1)).detach().float().cpu().numpy())

        points_ref = np.concatenate(points_list, axis=0)
        visibility = np.concatenate(vis_list, axis=0)
        keep = visibility > vis_threshold
        keep &= np.isfinite(points_ref).all(axis=-1)
        if depth_percentile < 100.0 and keep.sum() > 0:
            dists = np.linalg.norm(points_ref[keep], axis=-1)
            threshold = np.percentile(dists, depth_percentile)
            keep_indices = np.flatnonzero(keep)
            keep[keep_indices[dists > threshold]] = False

        # 将参考帧相机坐标系下的预测点转换到世界坐标系，与 GT dense world 可比
        ref_extrinsic = result.extrinsics[reference_frame]
        points_world_pred = camera_to_world(points_ref, ref_extrinsic)

        dense_frames.append(
            {
                "frame_id": frame_id,
                "points_ref": maybe_flip_y(points_ref[keep], flip_y),
                "points_world": maybe_flip_y(points_world_pred[keep], flip_y),
                "colors": reference_colors[keep],
                "point_ids": np.flatnonzero(keep).astype(np.int32),
                "visible_points": int(keep.sum()),
                "total_points": total_points,
            }
        )

    return dense_frames


def compute_dense_canonical_sequence(
    model: torch.nn.Module,
    result,
    reference_frame: int,
    stride: int,
    vis_threshold: float,
    batch_size: int,
    device: torch.device,
    flip_y: bool,
    patch_provider: str,
) -> list[dict[str, Any]]:
    """每帧独立重建到参考坐标系：t_src=t, t_tgt=t, t_cam=reference_frame。

    静态背景在参考坐标系下不动，动态物体移动，相机运动被消除。
    颜色取自当前帧像素（而非参考帧），因此颜色也会随帧变化。
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    video, aspect_ratio = to_video_tensor(result, device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoder_features = model.encode(video, aspect_ratio)

    frames_bcthw = model._prepare_query_frames(video, patch_frames)
    total_frames = len(result.images)
    height, width = result.images[0].shape[:2]

    x_pixels = np.arange(0, width, stride, dtype=np.int32)
    y_pixels = np.arange(0, height, stride, dtype=np.int32)
    if len(x_pixels) == 0 or len(y_pixels) == 0:
        raise RuntimeError(f"stride={stride} produced empty grid for {width}x{height}")
    grid_x, grid_y = np.meshgrid(x_pixels, y_pixels, indexing="xy")
    coords_px = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    coords_norm = coords_px.astype(np.float32) / np.array(
        [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
    )[None]
    coords_torch = torch.from_numpy(coords_norm).to(device=device, dtype=torch.float32)
    total_points = len(coords_norm)

    dense_frames: list[dict[str, Any]] = []
    for frame_id in range(total_frames):
        # 颜色取自当前帧
        frame_colors = result.images[frame_id][coords_px[:, 1], coords_px[:, 0]].astype(np.float32)

        points_list: list[np.ndarray] = []
        vis_list: list[np.ndarray] = []
        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            coords_chunk = coords_torch[start:end].unsqueeze(0)
            chunk_size = end - start
            # 关键：t_src=frame_id, t_tgt=frame_id, t_cam=reference_frame
            t_src = torch.full((1, chunk_size), frame_id, device=device, dtype=torch.long)
            t_tgt = torch.full((1, chunk_size), frame_id, device=device, dtype=torch.long)
            t_cam = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model.decode(
                        encoder_features,
                        frames_bcthw,
                        coords_chunk,
                        t_src,
                        t_tgt,
                        t_cam,
                        transform_metadata=transform_metadata,
                    )
            points_list.append(outputs["pos_3d"][0].detach().float().cpu().numpy())
            vis_list.append(torch.sigmoid(outputs["visibility"][0].squeeze(-1)).detach().float().cpu().numpy())

        points_ref = np.concatenate(points_list, axis=0)
        visibility = np.concatenate(vis_list, axis=0)
        keep = (visibility > vis_threshold) & np.isfinite(points_ref).all(axis=-1)

        dense_frames.append({
            "frame_id": frame_id,
            "points_ref": maybe_flip_y(points_ref[keep], flip_y),
            "colors": frame_colors[keep],
            "visible_points": int(keep.sum()),
            "total_points": total_points,
        })

    return dense_frames


def dense_axis_limits(
    dense_frames: list[dict[str, Any]],
    points_key: str = "points_world",
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """用 percentile 裁剪计算轴范围，避免极端离群点把正常点压缩到不可见。"""
    valid_clouds = [frame[points_key] for frame in dense_frames if len(frame[points_key]) > 0]
    if not valid_clouds:
        return np.zeros(3, dtype=np.float32), 1.0
    pts = np.concatenate(valid_clouds, axis=0)
    lo = np.percentile(pts, 100.0 - percentile, axis=0)
    hi = np.percentile(pts, percentile, axis=0)
    center = ((lo + hi) / 2.0).astype(np.float32)
    radius = max(float((hi - lo).max()) / 2.0, 1e-3)
    return center, radius


def set_axes_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_dense_gt_static(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, float]:
    frame_ids = make_frame_ids(len(dense_frames), num_frames=6)
    center, radius = dense_axis_limits(dense_frames)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    for plot_idx, frame_id in enumerate(frame_ids, start=1):
        ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
        frame = dense_frames[frame_id]
        ax.scatter(
            frame["points_world"][:, 0],
            frame["points_world"][:, 1],
            frame["points_world"][:, 2],
            c=frame["colors"],
            s=0.45,
            alpha=0.9,
        )
        ax.view_init(elev=20, azim=-62)
        ax.set_title(
            f"frame {frame_id}\n"
            f"selected={frame['selected_points']} rendered={frame['rendered_points']}",
            fontsize=10,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)

    fig.suptitle(f"{result.sequence_name}  GT dense dynamic point cloud (world frame)", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_counts = np.array([frame["selected_points"] for frame in dense_frames], dtype=np.float32)
    return {
        "dense_gt_mean_selected_points": float(np.mean(selected_counts)),
        "dense_gt_max_selected_points": float(np.max(selected_counts)),
    }


def write_dense_gt_gif(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
) -> None:
    center, radius = dense_axis_limits(dense_frames)
    frames_rgb: list[np.ndarray] = []

    for frame in dense_frames:
        fig = plt.figure(figsize=(6.4, 5.6), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            frame["points_world"][:, 0],
            frame["points_world"][:, 1],
            frame["points_world"][:, 2],
            c=frame["colors"],
            s=0.45,
            alpha=0.9,
        )
        ax.view_init(elev=20, azim=-62)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)
        fig.suptitle(
            f"{result.sequence_name}  GT dense point cloud  frame {frame['frame_id']}\n"
            f"selected={frame['selected_points']} rendered={frame['rendered_points']}",
            fontsize=13,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def plot_dense_pred_reference_static(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, float]:
    frame_ids = make_frame_ids(len(dense_frames), num_frames=6)
    center, radius = dense_axis_limits(dense_frames, points_key="points_ref")

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    for plot_idx, frame_id in enumerate(frame_ids, start=1):
        ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
        frame = dense_frames[frame_id]
        ax.scatter(
            frame["points_ref"][:, 0],
            frame["points_ref"][:, 1],
            frame["points_ref"][:, 2],
            c=frame["colors"],
            s=0.45,
            alpha=0.9,
        )
        ax.view_init(elev=20, azim=-62)
        ax.set_title(
            f"frame {frame_id}\n"
            f"visible={frame['visible_points']} / {frame['total_points']}",
            fontsize=10,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)

    fig.suptitle(
        f"{result.sequence_name}  Pred dense reference point cloud",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    visible_counts = np.array([frame["visible_points"] for frame in dense_frames], dtype=np.float32)
    return {
        "dense_pred_mean_visible_points": float(np.mean(visible_counts)),
        "dense_pred_max_visible_points": float(np.max(visible_counts)),
    }


def write_dense_pred_reference_gif(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
) -> None:
    center, radius = dense_axis_limits(dense_frames, points_key="points_ref")
    frames_rgb: list[np.ndarray] = []

    for frame in dense_frames:
        fig = plt.figure(figsize=(6.4, 5.6), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            frame["points_ref"][:, 0],
            frame["points_ref"][:, 1],
            frame["points_ref"][:, 2],
            c=frame["colors"],
            s=0.45,
            alpha=0.9,
        )
        ax.view_init(elev=20, azim=-62)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)
        fig.suptitle(
            f"{result.sequence_name}  Pred dense reference point cloud  frame {frame['frame_id']}\n"
            f"visible={frame['visible_points']} / {frame['total_points']}",
            fontsize=13,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def plot_dense_pred_world_static(
    result,
    dense_pred_frames: list[dict[str, Any]],
    dense_gt_frames: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, float]:
    """GT vs Pred 并排对比，3 个视角（正面/侧面/俯视），4 帧。"""
    frame_ids = make_frame_ids(len(dense_pred_frames), num_frames=4)

    # 合并两者点云计算统一坐标轴范围，保证 GT 和 Pred 坐标轴对齐
    combined = [{"points_world": f["points_world"]} for f in dense_pred_frames + dense_gt_frames if len(f["points_world"]) > 0]
    center, radius = dense_axis_limits(combined, points_key="points_world")

    VIEWS = [(20, -62), (0, 0), (90, 0)]
    VIEW_LABELS = ["front (elev=20)", "side (elev=0)", "top (elev=90)"]
    # 行布局：GT(front), Pred(front), Pred(side), Pred(top)
    nrows = 4
    ncols = len(frame_ids)
    row_labels = [f"GT {VIEW_LABELS[0]}", f"Pred {VIEW_LABELS[0]}", f"Pred {VIEW_LABELS[1]}", f"Pred {VIEW_LABELS[2]}"]

    fig = plt.figure(figsize=(4.5 * ncols, 5.0 * nrows), constrained_layout=True)

    for col_idx, frame_id in enumerate(frame_ids):
        pred_frame = dense_pred_frames[frame_id]
        gt_frame = dense_gt_frames[frame_id]

        for row_idx in range(nrows):
            ax = fig.add_subplot(nrows, ncols, row_idx * ncols + col_idx + 1, projection="3d")

            if row_idx == 0:
                pts = gt_frame["points_world"]
                cols = gt_frame["colors"]
                elev, azim = VIEWS[0]
                info = f"sel={gt_frame['selected_points']}"
            else:
                pts = pred_frame["points_world"]
                cols = pred_frame["colors"]
                elev, azim = VIEWS[row_idx - 1]
                info = f"vis={pred_frame['visible_points']}/{pred_frame['total_points']}"

            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.5, alpha=0.9)
            ax.view_init(elev=elev, azim=azim)
            set_axes_limits(ax, center, radius)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)

            title_row = f"{row_labels[row_idx]}\n" if col_idx == 0 else ""
            ax.set_title(f"{title_row}frame {frame_id}  {info}", fontsize=9)

    fig.suptitle(f"{result.sequence_name}  GT vs Pred dense world point cloud", fontsize=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    visible_counts = np.array([f["visible_points"] for f in dense_pred_frames], dtype=np.float32)
    return {
        "dense_pred_world_mean_visible": float(np.mean(visible_counts)),
        "dense_pred_world_max_visible": float(np.max(visible_counts)),
    }


def write_dense_pred_world_gif(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
    reference_frame_id: int = -1,
) -> None:
    """位移颜色编码的动态 GIF：动态点高亮（plasma colormap），视角缓慢旋转。"""
    center, radius = dense_axis_limits(dense_frames, points_key="points_world")
    total_frames = len(dense_frames)

    if reference_frame_id < 0:
        reference_frame_id = total_frames // 2

    # 建立 reference frame 的 point_id -> world_pos 映射
    ref_data = dense_frames[reference_frame_id]
    ref_pos_map: dict[int, np.ndarray] = {
        int(pid): pw for pid, pw in zip(ref_data["point_ids"], ref_data["points_world"])
    }

    # 预计算所有帧的位移，统一 vmax
    all_disps: list[np.ndarray] = []
    for frame in dense_frames:
        pids = frame["point_ids"]
        pts = frame["points_world"]
        disps = np.zeros(len(pids), dtype=np.float32)
        for i, pid in enumerate(pids):
            ref_pt = ref_pos_map.get(int(pid))
            if ref_pt is not None:
                disps[i] = float(np.linalg.norm(pts[i] - ref_pt))
        all_disps.append(disps)

    all_disps_cat = np.concatenate(all_disps) if all_disps else np.array([1.0], dtype=np.float32)
    finite_disps = all_disps_cat[np.isfinite(all_disps_cat) & (all_disps_cat > 0)]
    vmax_disp = float(np.percentile(finite_disps, 95)) if len(finite_disps) > 0 else 1.0
    vmax_disp = max(vmax_disp, 1e-4)

    cmap = plt.cm.plasma
    frames_rgb: list[np.ndarray] = []

    for frame_idx, (frame, disps) in enumerate(zip(dense_frames, all_disps)):
        # 视角缓慢旋转一圈
        azim = -62 + frame_idx * 360.0 / max(total_frames, 1)

        fig = plt.figure(figsize=(7.0, 6.0), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        pts = frame["points_world"]
        if len(pts) > 0:
            norm_disp = np.clip(disps / vmax_disp, 0.0, 1.0)
            scatter_colors = cmap(norm_disp)[:, :3]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=scatter_colors, s=1.5, alpha=0.9)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=vmax_disp))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.08, shrink=0.6)
            cbar.set_label("displacement (m)", fontsize=9)

        ax.view_init(elev=20, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)
        fig.suptitle(
            f"{result.sequence_name}  Pred dense world  frame {frame['frame_id']}\n"
            f"vis={frame['visible_points']}/{frame['total_points']}  color=displacement from ref={reference_frame_id}",
            fontsize=11,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def plot_dense_canonical_static(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, float]:
    """canonical 重建静态图：每帧独立重建到参考坐标系，3 个视角，4 帧。"""
    frame_ids = make_frame_ids(len(dense_frames), num_frames=4)
    center, radius = dense_axis_limits(dense_frames, points_key="points_ref")

    VIEWS = [(20, -62), (0, 0), (90, 0)]
    VIEW_LABELS = ["front (elev=20)", "side (elev=0)", "top (elev=90)"]
    nrows = len(VIEWS)
    ncols = len(frame_ids)

    fig = plt.figure(figsize=(4.5 * ncols, 5.0 * nrows), constrained_layout=True)
    for col_idx, frame_id in enumerate(frame_ids):
        frame = dense_frames[frame_id]
        pts = frame["points_ref"]
        cols = frame["colors"]
        for row_idx, (elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(nrows, ncols, row_idx * ncols + col_idx + 1, projection="3d")
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.5, alpha=0.9)
            ax.view_init(elev=elev, azim=azim)
            set_axes_limits(ax, center, radius)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            row_label = f"{VIEW_LABELS[row_idx]}\n" if col_idx == 0 else ""
            ax.set_title(f"{row_label}frame {frame_id}  vis={frame['visible_points']}/{frame['total_points']}", fontsize=9)

    fig.suptitle(
        f"{result.sequence_name}  Canonical reconstruction (t_src=t, t_cam=ref)\n"
        f"static background fixed, dynamic objects move, camera motion removed",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    visible_counts = np.array([f["visible_points"] for f in dense_frames], dtype=np.float32)
    return {
        "canonical_mean_visible_points": float(np.mean(visible_counts)),
        "canonical_max_visible_points": float(np.max(visible_counts)),
    }


def write_dense_canonical_gif(
    result,
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
) -> None:
    """canonical 重建 GIF：静态背景不动，动态物体移动，相机运动消除。"""
    center, radius = dense_axis_limits(dense_frames, points_key="points_ref")
    frames_rgb: list[np.ndarray] = []

    for frame in dense_frames:
        fig = plt.figure(figsize=(6.4, 5.6), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        pts = frame["points_ref"]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=frame["colors"], s=1.5, alpha=0.9)
        ax.view_init(elev=20, azim=-62)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)
        fig.suptitle(
            f"{result.sequence_name}  Canonical  frame {frame['frame_id']}\n"
            f"vis={frame['visible_points']}/{frame['total_points']}  (static=fixed, dynamic=moves)",
            fontsize=11,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def plot_2d_overlay(
    result,
    pred: dict[str, Any],
    t_ref: int,
    display_ids: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    T = len(result.images)
    frame_ids = sorted({0, max(0, t_ref - 8), t_ref, min(T - 1, t_ref + 8), T - 1})
    frame_ids = list(dict.fromkeys(frame_ids))

    fig, axes = plt.subplots(
        1,
        len(frame_ids),
        figsize=(4.4 * len(frame_ids), 4.8),
        constrained_layout=True,
    )
    if len(frame_ids) == 1:
        axes = [axes]

    all_err = pred["err_2d_px"][display_ids]
    finite_err = all_err[np.isfinite(all_err)]
    vmax = float(np.nanpercentile(finite_err, 95)) if finite_err.size > 0 else 10.0
    vmax = max(vmax, 1.0)
    scat = None

    for ax, frame_id in zip(axes, frame_ids, strict=True):
        image = result.images[frame_id]
        ax.imshow(image)
        gt_vis = pred["gt_vis"][display_ids, frame_id]
        pred_vis = pred["pred_vis"][display_ids, frame_id] > 0.5
        mask = gt_vis & pred_vis

        gt_pts = pred["gt_2d_px"][display_ids, frame_id]
        pd_pts = pred["pred_2d_px"][display_ids, frame_id]
        err = pred["err_2d_px"][display_ids, frame_id]

        if np.any(mask):
            ax.scatter(
                gt_pts[mask, 0],
                gt_pts[mask, 1],
                s=22,
                facecolors="none",
                edgecolors="#00e676",
                linewidths=0.9,
                label="GT",
            )
            scat = ax.scatter(
                pd_pts[mask, 0],
                pd_pts[mask, 1],
                s=12,
                c=np.clip(err[mask], 0.0, vmax),
                cmap="turbo",
                vmin=0.0,
                vmax=vmax,
                alpha=0.9,
                label="Pred",
            )
            for gt_pt, pd_pt in zip(gt_pts[mask], pd_pts[mask], strict=False):
                ax.plot(
                    [gt_pt[0], pd_pt[0]],
                    [gt_pt[1], pd_pt[1]],
                    color="#ffffff",
                    linewidth=0.4,
                    alpha=0.45,
                )
        ax.set_title(
            f"frame {frame_id}\n"
            f"GT vis {int(gt_vis.sum())}  pred vis {int(pred_vis.sum())}",
            fontsize=10,
        )
        ax.axis("off")

    title = (
        f"{result.sequence_name}  2D tracks\n"
        f"source frame={t_ref}, displayed points={len(display_ids)}"
    )
    fig.suptitle(title, fontsize=13)
    if scat is not None:
        cbar = fig.colorbar(scat, ax=axes, fraction=0.018, pad=0.02)
        cbar.set_label("2D error (px)")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    finite = pred["err_2d_px"][np.isfinite(pred["err_2d_px"])]
    if finite.size == 0:
        return {"mean_2d_px_err": math.nan, "median_2d_px_err": math.nan, "p90_2d_px_err": math.nan}
    return {
        "mean_2d_px_err": float(np.mean(finite)),
        "median_2d_px_err": float(np.median(finite)),
        "p90_2d_px_err": float(np.percentile(finite, 90)),
    }


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    pts = points[np.isfinite(points).all(axis=-1)]
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_3d_scatter(
    result,
    pred: dict[str, Any],
    t_ref: int,
    output_path: Path,
    pred_ply_path: Path,
    gt_ply_path: Path,
    flip_y: bool = False,
    max_depth: float = 0.0,
) -> dict[str, float]:
    mask = pred["gt_vis"][:, t_ref]
    mask &= np.isfinite(pred["gt_3d_cam"][:, t_ref]).all(axis=-1)
    mask &= np.isfinite(pred["pred_3d_cam"][:, t_ref]).all(axis=-1)
    if max_depth > 0.0:
        mask &= pred["gt_3d_cam"][:, t_ref, 2] <= max_depth

    gt_pts = pred["gt_3d_cam"][mask, t_ref]
    pd_pts = pred["pred_3d_cam"][mask, t_ref]
    colors = pred["colors"][mask]
    err = np.linalg.norm(pd_pts - gt_pts, axis=-1) if len(gt_pts) > 0 else np.empty((0,), dtype=np.float32)

    if len(gt_pts) == 0:
        raise RuntimeError(f"no valid 3D points at reference frame {t_ref}")

    gt_pts_vis = maybe_flip_y(gt_pts, flip_y)
    pd_pts_vis = maybe_flip_y(pd_pts, flip_y)
    all_pts = np.concatenate([gt_pts_vis, pd_pts_vis], axis=0)

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pd = fig.add_subplot(1, 2, 2, projection="3d")

    ax_gt.scatter(gt_pts_vis[:, 0], gt_pts_vis[:, 1], gt_pts_vis[:, 2], c=colors, s=3.0, alpha=0.9)
    ax_gt.set_title(f"GT 3D @ frame {t_ref}", fontsize=11)

    scat = ax_pd.scatter(
        pd_pts_vis[:, 0],
        pd_pts_vis[:, 1],
        pd_pts_vis[:, 2],
        c=np.clip(err, 0.0, max(float(np.percentile(err, 95)), 1e-3)),
        cmap="turbo",
        s=3.0,
        alpha=0.9,
    )
    ax_pd.set_title(f"Pred 3D @ frame {t_ref}", fontsize=11)

    for ax in (ax_gt, ax_pd):
        ax.view_init(elev=24, azim=-62)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_equal_3d_axes(ax, all_pts)

    fig.suptitle(
        f"{result.sequence_name}  3D camera-frame scatter\n"
        f"points={len(gt_pts)}  ref frame={t_ref}",
        fontsize=13,
    )
    cbar = fig.colorbar(scat, ax=[ax_gt, ax_pd], fraction=0.02, pad=0.03)
    cbar.set_label("3D error")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    save_point_cloud_ply(str(pred_ply_path), pd_pts_vis, colors=colors)
    save_point_cloud_ply(str(gt_ply_path), gt_pts_vis, colors=colors)

    return {
        "num_3d_points": int(len(gt_pts)),
        "mean_3d_euc_err": float(np.mean(err)),
        "median_3d_euc_err": float(np.median(err)),
        "p90_3d_euc_err": float(np.percentile(err, 90)),
    }


def summarize_visibility(pred: dict[str, Any]) -> dict[str, float]:
    gt = pred["gt_vis"].astype(bool)
    pd = pred["pred_vis"] > 0.5
    total = gt.size
    return {
        "vis_acc": float((gt == pd).sum() / max(total, 1)),
        "gt_vis_ratio": float(gt.mean()),
        "pred_vis_ratio": float(pd.mean()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    dataset = load_val_dataset(args.config, split=args.split)
    model = load_model(args, device)

    samples = find_samples(
        dataset=dataset,
        start_index=args.start_index,
        count=args.num_samples,
        max_search=args.max_search,
        allow_no_tracks=args.allow_no_tracks,
    )

    summary: dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "patch_provider": args.patch_provider,
        "dense_gt_point_source": args.dense_gt_point_source,
        "dense_pred_point_cloud_stride": args.dense_pred_point_cloud_stride,
        "dense_pred_vis_threshold": args.dense_pred_vis_threshold,
        "dense_pred_query_batch_size": args.dense_pred_query_batch_size,
        "device": str(device),
        "num_samples": len(samples),
        "samples": [],
    }

    for sample_rank, sample in enumerate(samples):
        result = sample["result"]
        sample_index = int(sample["sample_index"])
        sample_dir = out_dir / f"sample_{sample_rank:02d}_idx{sample_index}_{result.sequence_name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        has_tracks = bool(result.metadata.get("has_tracks", False)) and result.trajs_2d is not None

        if has_tracks:
            try:
                t_ref, point_indices, query_points, query_frames, _ = sample_query_points(
                    result=result,
                    num_points=args.num_points,
                    seed=args.seed + sample_index,
                    reference_frame=args.reference_frame,
                )
            except RuntimeError:
                has_tracks = False
        if has_tracks:
            pred = compute_track_predictions(
                model=model,
                result=result,
                point_indices=point_indices,
                query_points=query_points,
                query_frames=query_frames,
                device=device,
                patch_provider=args.patch_provider,
            )
            display_ids = subsample_for_display(
                point_indices=point_indices,
                gt_vis=pred["gt_vis"],
                num_display_points=args.num_display_points,
                seed=args.seed + 17 * sample_index,
            )
        else:
            t_ref = len(result.images) // 2 if args.reference_frame < 0 else args.reference_frame
            pred = None
            display_ids = None

        overlay_path = sample_dir / "tracks_2d_overlay.png"
        compare_static_path = sample_dir / "tracks_2d_compare_static.png"
        compare_gif_path = sample_dir / "tracks_2d_compare.gif"
        scatter_path = sample_dir / "tracks_3d_scatter.png"
        pred_ply_path = sample_dir / "pred_3d_points.ply"
        gt_ply_path = sample_dir / "gt_3d_points.ply"
        dense_gt_static_path = sample_dir / "gt_dense_dynamic_world_static.png"
        dense_gt_gif_path = sample_dir / "gt_dense_dynamic_world.gif"
        dense_gt_ref_ply_path = sample_dir / f"gt_dense_world_frame_{t_ref:03d}.ply"
        dense_pred_static_path = sample_dir / "pred_dense_reference_static.png"
        dense_pred_gif_path = sample_dir / "pred_dense_reference.gif"
        dense_pred_ref_ply_path = sample_dir / f"pred_dense_reference_frame_{t_ref:03d}.ply"
        dense_pred_world_static_path = sample_dir / "pred_dense_world_static.png"
        dense_pred_world_gif_path = sample_dir / "pred_dense_world.gif"
        dense_pred_world_ref_ply_path = sample_dir / f"pred_dense_world_frame_{t_ref:03d}.ply"
        canonical_static_path = sample_dir / "canonical_static.png"
        canonical_gif_path = sample_dir / "canonical.gif"
        canonical_ref_ply_path = sample_dir / f"canonical_frame_{t_ref:03d}.ply"

        if has_tracks:
            metrics_2d = plot_2d_overlay(
                result=result,
                pred=pred,
                t_ref=t_ref,
                display_ids=display_ids,
                output_path=overlay_path,
            )
            plot_2d_compare_static(
                result=result,
                pred=pred,
                display_ids=display_ids,
                output_path=compare_static_path,
            )
            write_2d_compare_gif(
                result=result,
                pred=pred,
                display_ids=display_ids,
                output_path=compare_gif_path,
                fps=args.gif_fps,
            )
            metrics_3d = plot_3d_scatter(
                result=result,
                pred=pred,
                t_ref=t_ref,
                output_path=scatter_path,
                pred_ply_path=pred_ply_path,
                gt_ply_path=gt_ply_path,
                flip_y=args.flip_y_axis,
                max_depth=args.max_depth,
            )
        else:
            metrics_2d = {"mean_2d_px_err": math.nan, "median_2d_px_err": math.nan, "p90_2d_px_err": math.nan}
            metrics_3d = {"num_3d_points": 0, "mean_3d_euc_err": math.nan, "median_3d_euc_err": math.nan, "p90_3d_euc_err": math.nan}
        dense_gt_frames = prepare_dense_gt_sequence(
            result=result,
            max_points=args.dense_gt_max_points,
            seed=args.seed + 101 * sample_index,
            flip_y=args.flip_y_axis,
            point_source=args.dense_gt_point_source,
        )
        dense_gt_metrics = plot_dense_gt_static(
            result=result,
            dense_frames=dense_gt_frames,
            output_path=dense_gt_static_path,
        )
        write_dense_gt_gif(
            result=result,
            dense_frames=dense_gt_frames,
            output_path=dense_gt_gif_path,
            fps=args.gif_fps,
        )
        dense_ref_frame = dense_gt_frames[t_ref]
        save_point_cloud_ply_allow_empty(
            dense_gt_ref_ply_path,
            dense_ref_frame["points_world"],
            colors=dense_ref_frame["colors"],
        )
        dense_pred_frames = compute_dense_pred_reference_sequence(
            model=model,
            result=result,
            reference_frame=t_ref,
            stride=args.dense_pred_point_cloud_stride,
            vis_threshold=args.dense_pred_vis_threshold,
            batch_size=args.dense_pred_query_batch_size,
            device=device,
            flip_y=args.flip_y_axis,
            patch_provider=args.patch_provider,
            depth_percentile=args.dense_pred_depth_percentile,
            query_depth_percentile=args.dense_pred_query_depth_percentile,
        )
        dense_pred_metrics = plot_dense_pred_reference_static(
            result=result,
            dense_frames=dense_pred_frames,
            output_path=dense_pred_static_path,
        )
        write_dense_pred_reference_gif(
            result=result,
            dense_frames=dense_pred_frames,
            output_path=dense_pred_gif_path,
            fps=args.gif_fps,
        )
        dense_pred_ref_frame = dense_pred_frames[t_ref]
        save_point_cloud_ply_allow_empty(
            dense_pred_ref_ply_path,
            dense_pred_ref_frame["points_ref"],
            colors=dense_pred_ref_frame["colors"],
        )
        # 世界坐标系下的稠密预测动态点云：query 取自 reference_frame，3D 轨迹反映真实空间运动
        dense_pred_world_metrics = plot_dense_pred_world_static(
            result=result,
            dense_pred_frames=dense_pred_frames,
            dense_gt_frames=dense_gt_frames,
            output_path=dense_pred_world_static_path,
        )
        write_dense_pred_world_gif(
            result=result,
            dense_frames=dense_pred_frames,
            output_path=dense_pred_world_gif_path,
            fps=args.gif_fps,
            reference_frame_id=t_ref,
        )
        dense_pred_world_ref_frame = dense_pred_frames[t_ref]
        save_point_cloud_ply_allow_empty(
            dense_pred_world_ref_ply_path,
            dense_pred_world_ref_frame["points_world"],
            colors=dense_pred_world_ref_frame["colors"],
        )
        # canonical 重建：每帧独立重建到参考坐标系，静态背景不动，动态物体移动
        canonical_frames = compute_dense_canonical_sequence(
            model=model,
            result=result,
            reference_frame=t_ref,
            stride=args.dense_pred_point_cloud_stride,
            vis_threshold=args.dense_pred_vis_threshold,
            batch_size=args.dense_pred_query_batch_size,
            device=device,
            flip_y=args.flip_y_axis,
            patch_provider=args.patch_provider,
        )
        canonical_metrics = plot_dense_canonical_static(
            result=result,
            dense_frames=canonical_frames,
            output_path=canonical_static_path,
        )
        write_dense_canonical_gif(
            result=result,
            dense_frames=canonical_frames,
            output_path=canonical_gif_path,
            fps=args.gif_fps,
        )
        canonical_ref_frame = canonical_frames[t_ref]
        save_point_cloud_ply_allow_empty(
            canonical_ref_ply_path,
            canonical_ref_frame["points_ref"],
            colors=canonical_ref_frame["colors"],
        )
        vis_metrics = summarize_visibility(pred) if has_tracks else {"vis_acc": math.nan, "gt_vis_ratio": math.nan, "pred_vis_ratio": math.nan}

        artifacts = {
            "gt_dense_dynamic_world_static": str(dense_gt_static_path),
            "gt_dense_dynamic_world_gif": str(dense_gt_gif_path),
            "gt_dense_world_reference_ply": str(dense_gt_ref_ply_path),
            "pred_dense_reference_static": str(dense_pred_static_path),
            "pred_dense_reference_gif": str(dense_pred_gif_path),
            "pred_dense_reference_ply": str(dense_pred_ref_ply_path),
            "pred_dense_world_static": str(dense_pred_world_static_path),
            "pred_dense_world_gif": str(dense_pred_world_gif_path),
            "pred_dense_world_ply": str(dense_pred_world_ref_ply_path),
            "canonical_static": str(canonical_static_path),
            "canonical_gif": str(canonical_gif_path),
            "canonical_ply": str(canonical_ref_ply_path),
        }
        if has_tracks:
            artifacts.update({
                "tracks_2d_overlay": str(overlay_path),
                "tracks_2d_compare_static": str(compare_static_path),
                "tracks_2d_compare_gif": str(compare_gif_path),
                "tracks_3d_scatter": str(scatter_path),
                "pred_3d_points_ply": str(pred_ply_path),
                "gt_3d_points_ply": str(gt_ply_path),
            })

        sample_summary = {
            "sample_rank": sample_rank,
            "sample_index": sample_index,
            "sequence_name": result.sequence_name,
            "frame_indices": list(map(int, sample["frame_indices"])),
            "reference_frame": int(t_ref),
            "dense_gt_point_source": args.dense_gt_point_source,
            "has_tracks": has_tracks,
            "num_query_points": int(len(point_indices)) if has_tracks else 0,
            "num_display_points": int(len(display_ids)) if has_tracks else 0,
            **metrics_2d,
            **metrics_3d,
            **dense_gt_metrics,
            **dense_pred_metrics,
            **dense_pred_world_metrics,
            **canonical_metrics,
            **vis_metrics,
            "artifacts": artifacts,
        }
        summary["samples"].append(sample_summary)
        print(json.dumps(sample_summary, ensure_ascii=False))

    mean_keys = [
        "mean_2d_px_err",
        "median_2d_px_err",
        "p90_2d_px_err",
        "mean_3d_euc_err",
        "median_3d_euc_err",
        "p90_3d_euc_err",
        "dense_gt_mean_selected_points",
        "dense_gt_max_selected_points",
        "dense_pred_mean_visible_points",
        "dense_pred_max_visible_points",
        "dense_pred_world_mean_visible",
        "dense_pred_world_max_visible",
        "canonical_mean_visible_points",
        "canonical_max_visible_points",
        "vis_acc",
        "gt_vis_ratio",
        "pred_vis_ratio",
    ]
    summary["mean_metrics"] = {
        key: float(np.mean([s[key] for s in summary["samples"]]))
        for key in mean_keys
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": str(summary_path), "mean_metrics": summary["mean_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
