#!/usr/bin/env python3
"""通用 checkpoint 深度可视化：在模型输入分辨率下预测深度，输出 RGB/GT/Pred 对比。"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.factory import create_training_dataset
from models import create_d4rt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=48)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--patch-provider", default="sampled_resized")
    p.add_argument("--stride", type=int, default=2, help="深度图采样步长（在256x256下），stride=2→128x128深度图")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-search", type=int, default=500)
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(args, device):
    model = create_d4rt(
        variant="base", decoder_depth=6,
        img_size=args.resolution, temporal_size=args.num_frames,
        patch_size=(2, 16, 16), query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base",
        patch_provider=args.patch_provider,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model") or ckpt.get("model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def to_video_tensor(result, device):
    video_np = np.stack(result.images, axis=0).astype(np.float32)
    video = torch.from_numpy(video_np).unsqueeze(0).to(device)
    crop = getattr(result, "crop", None)
    if crop is not None:
        width = float(crop.crop_w)
        height = float(crop.crop_h)
    else:
        height, width = result.images[0].shape[:2]
        width = float(width)
        height = float(height)
    aspect_ratio = torch.tensor([[width / max(height, 1.0)]], dtype=torch.float32, device=device)
    return video, aspect_ratio


def to_patch_frames_tensor(result, device):
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


def build_transform_metadata(result, device):
    crop = getattr(result, "crop", None)
    if crop is None:
        crop_h, crop_w = result.images[0].shape[:2]
        crop_x0 = 0.0
        crop_y0 = 0.0
    else:
        crop_h = float(crop.crop_h)
        crop_w = float(crop.crop_w)
        crop_x0 = float(crop.x0)
        crop_y0 = float(crop.y0)

    return {
        "canonical_space": torch.tensor([0], dtype=torch.long, device=device),
        "original_hw": torch.tensor(
            [[float(result.original_h), float(result.original_w)]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_offset_xy": torch.tensor(
            [[crop_x0, crop_y0]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_size_hw": torch.tensor(
            [[crop_h, crop_w]],
            dtype=torch.float32,
            device=device,
        ),
        "resized_hw": torch.tensor(
            [[float(result.img_size), float(result.img_size)]],
            dtype=torch.float32,
            device=device,
        ),
    }


def encode_video(model, result, device, patch_provider):
    video, aspect_ratio = to_video_tensor(result, device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enc = model.encode(video, aspect_ratio)
    frames_bcthw = model._prepare_query_frames(video, patch_frames)
    return enc, frames_bcthw, transform_metadata


def predict_depth_frame(model, enc, frames_bcthw, frame_id, stride, S, device, transform_metadata=None):
    """在 S×S 坐标系下预测深度，返回 (S/stride, S/stride) 深度图。"""
    xs = np.arange(0, S, stride, dtype=np.int32)
    ys = np.arange(0, S, stride, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    coords_norm = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32) / max(S - 1, 1)
    coords_t = torch.from_numpy(coords_norm).to(device)
    N = len(coords_norm)

    depth_vals = []
    for start in range(0, N, 4096):
        end = min(start + 4096, N)
        chunk = coords_t[start:end].unsqueeze(0)
        sz = end - start
        t = torch.full((1, sz), frame_id, device=device, dtype=torch.long)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model.decode(
                    enc,
                    frames_bcthw,
                    chunk,
                    t,
                    t,
                    t,
                    transform_metadata=transform_metadata,
                )
        depth_vals.append(out["pos_3d"][0, :, 2].detach().float().cpu().numpy())

    return np.concatenate(depth_vals).reshape(len(ys), len(xs))


def align_depth_scale_shift(pred, gt_valid_mask, gt_values):
    """最小二乘求解 pred_aligned = s * pred + t，使其与 gt 对齐。"""
    pred_valid = pred[gt_valid_mask]
    if len(pred_valid) < 2:
        return 1.0, 0.0
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    s, t = np.linalg.lstsq(A, gt_values, rcond=None)[0]
    return float(s), float(t)


def make_depth_comparison(
    result,
    model,
    enc,
    frames_bcthw,
    stride,
    S,
    device,
    frame_ids,
    output_path,
    transform_metadata=None,
):
    """生成 RGB / GT深度 / 预测深度 / 对齐误差 并排静态图。"""
    ncols = len(frame_ids)
    pred_depths = {
        fid: predict_depth_frame(
            model,
            enc,
            frames_bcthw,
            fid,
            stride,
            S,
            device,
            transform_metadata=transform_metadata,
        )
        for fid in frame_ids
    }
    all_finite = np.concatenate([d[np.isfinite(d)] for d in pred_depths.values()])
    vmin = float(np.percentile(all_finite, 2)) if len(all_finite) > 0 else 0.0
    vmax = float(np.percentile(all_finite, 98)) if len(all_finite) > 0 else 10.0

    depths = getattr(result, "depths", None)
    has_gt = depths is not None and len(depths) > 0 and depths[0] is not None
    nrows = 4 if has_gt else 2
    row_labels = ["RGB (256×256)", "GT Depth", "Pred Depth", "Aligned Error"] if has_gt else ["RGB (256×256)", "Pred Depth"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), constrained_layout=True)
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    for row in range(nrows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11)

    for col, fid in enumerate(frame_ids):
        rgb = cv2.resize(result.images[fid], (S, S), interpolation=cv2.INTER_LINEAR)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"frame {fid}", fontsize=10)
        axes[0, col].axis("off")

        if has_gt:
            gt = np.array(depths[fid], dtype=np.float32) if fid < len(depths) else np.zeros((S, S), dtype=np.float32)
            gt_masked = np.where(gt > 0, gt, np.nan)
            im = axes[1, col].imshow(gt_masked, cmap="plasma", vmin=vmin, vmax=vmax)
            axes[1, col].axis("off")
            fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        pred = pred_depths[fid]
        im2 = axes[2 if has_gt else 1, col].imshow(pred, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[2 if has_gt else 1, col].axis("off")
        fig.colorbar(im2, ax=axes[2 if has_gt else 1, col], fraction=0.046, pad=0.04)

        if has_gt:
            gt_full = np.array(depths[fid], dtype=np.float32)
            ph, pw = pred.shape
            gt = cv2.resize(gt_full, (pw, ph), interpolation=cv2.INTER_NEAREST)
            mask = gt > 0
            if np.any(mask):
                s, t = align_depth_scale_shift(pred, mask, gt[mask])
                pred_aligned = s * pred + t
                error = np.abs(pred_aligned - gt)
                error_masked = np.where(mask, error, np.nan)
                im3 = axes[3, col].imshow(error_masked, cmap="hot", vmin=0, vmax=np.nanpercentile(error_masked, 95))
                axes[3, col].axis("off")
                fig.colorbar(im3, ax=axes[3, col], fraction=0.046, pad=0.04)
            else:
                axes[3, col].axis("off")

    fig.suptitle(f"{result.sequence_name}  Depth (Z in camera space)", fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_depth_gif(
    result,
    model,
    enc,
    frames_bcthw,
    stride,
    S,
    device,
    output_path,
    fps=8,
    transform_metadata=None,
):
    """生成每帧 RGB+深度 GIF。"""
    T = len(result.images)
    all_depths = [
        predict_depth_frame(
            model,
            enc,
            frames_bcthw,
            fid,
            stride,
            S,
            device,
            transform_metadata=transform_metadata,
        )
        for fid in range(T)
    ]
    all_finite = np.concatenate([d[np.isfinite(d)] for d in all_depths])
    vmin = float(np.percentile(all_finite, 2)) if len(all_finite) > 0 else 0.0
    vmax = float(np.percentile(all_finite, 98)) if len(all_finite) > 0 else 10.0

    frames_rgb = []
    for fid, depth in enumerate(all_depths):
        rgb = cv2.resize(result.images[fid], (S, S), interpolation=cv2.INTER_LINEAR)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB frame {fid}", fontsize=11)
        axes[0].axis("off")
        im = axes[1].imshow(depth, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Pred Depth frame {fid}", fontsize=11)
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Z (m)")
        fig.suptitle(result.sequence_name, fontsize=11)
        fig.canvas.draw()
        frames_rgb.append(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset = create_training_dataset(config, split=args.split)
    model = load_model(args, device)
    S = args.resolution  # 256

    found = 0
    idx = args.start_index
    while found < args.num_samples and idx < args.start_index + args.max_search:
        try:
            r = random.Random(dataset.seed + idx)
            dataset_idx, seq_name, frame_indices = dataset.mixture_sampler.sample(r)
            adapter = dataset.adapters[dataset_idx]
            clip = adapter.load_clip(seq_name, frame_indices)
            result = dataset.transform(clip, rng=r)

            sample_dir = out_dir / f"sample_{found:02d}_{result.sequence_name.replace('/', '_')}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            enc, frames_bcthw, transform_metadata = encode_video(
                model,
                result,
                device,
                args.patch_provider,
            )
            T = len(result.images)
            frame_ids = sorted({0, T//4, T//2, 3*T//4, T-1})

            make_depth_comparison(
                result,
                model,
                enc,
                frames_bcthw,
                args.stride,
                S,
                device,
                frame_ids,
                sample_dir / "depth_comparison.png",
                transform_metadata=transform_metadata,
            )
            make_depth_gif(
                result,
                model,
                enc,
                frames_bcthw,
                args.stride,
                S,
                device,
                sample_dir / "depth.gif",
                fps=args.gif_fps,
                transform_metadata=transform_metadata,
            )

            print(f"[{found}] {result.sequence_name} -> {sample_dir}")
            found += 1
        except Exception as e:
            print(f"  skip idx={idx}: {e}")
        idx += 1

    print(f"Done. {found} samples -> {out_dir}")


if __name__ == "__main__":
    main()
