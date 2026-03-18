import glob
import math
import os
import random
import re
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.patches import extract_local_patches


CANONICAL_QUERY_SPACE_CROP_NORMALIZED = 0


def crop_points_2d(points_2d, x0, y0):
    shifted = points_2d.copy().astype(np.float32)
    shifted[..., 0] -= float(x0)
    shifted[..., 1] -= float(y0)
    return shifted


def normalize_points_2d(points_2d, height, width):
    normalized = points_2d.copy().astype(np.float32)
    normalized[..., 0] /= max(float(width) - 1.0, 1.0)
    normalized[..., 1] /= max(float(height) - 1.0, 1.0)
    return normalized


def crop_intrinsics(intrinsics, x0, y0):
    cropped = intrinsics.copy().astype(np.float32)
    cropped[..., 0, 2] -= float(x0)
    cropped[..., 1, 2] -= float(y0)
    return cropped


def resize_intrinsics(intrinsics, src_h, src_w, dst_h, dst_w):
    resized = intrinsics.copy().astype(np.float32)
    scale_x = float(dst_w) / max(float(src_w), 1.0)
    scale_y = float(dst_h) / max(float(src_h), 1.0)
    resized[..., 0, 0] *= scale_x
    resized[..., 1, 1] *= scale_y
    resized[..., 0, 2] *= scale_x
    resized[..., 1, 2] *= scale_y
    return resized


def transform_intrinsics_for_crop_resize(intrinsics, x0, y0, crop_h, crop_w, dst_h, dst_w):
    intrinsics_crop = crop_intrinsics(intrinsics, x0=x0, y0=y0)
    intrinsics_resized = resize_intrinsics(
        intrinsics_crop,
        src_h=crop_h,
        src_w=crop_w,
        dst_h=dst_h,
        dst_w=dst_w,
    )
    return intrinsics_crop, intrinsics_resized


def decode_pointodyssey_depth(depth):
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / 65535.0 * 1000.0
    return depth.astype(np.float32)


def collate_fn(batch):
    """Collate successful dataset samples."""
    samples = [sample for sample, success in batch if success and sample]
    if not samples:
        return {}
    from torch.utils.data import default_collate

    filtered_samples = []
    for sample in samples:
        sample = dict(sample)
        if sample.get("local_patches") is None:
            sample.pop("local_patches", None)
        filtered_samples.append(sample)

    if any("video_query" in sample for sample in filtered_samples):
        if not all("video_query" in sample for sample in filtered_samples):
            raise ValueError("video_query must be present for all samples in a batch when enabled")
        max_h = max(sample["video_query"].shape[-2] for sample in filtered_samples)
        max_w = max(sample["video_query"].shape[-1] for sample in filtered_samples)
        for sample in filtered_samples:
            video_query = sample["video_query"]
            pad_h = max_h - video_query.shape[-2]
            pad_w = max_w - video_query.shape[-1]
            if pad_h > 0 or pad_w > 0:
                sample["video_query"] = F.pad(video_query, (0, pad_w, 0, pad_h))

    if all("local_patches" not in sample for sample in filtered_samples) and all(
        "video_query" not in sample for sample in filtered_samples
    ):
        return default_collate(filtered_samples)

    for sample in filtered_samples:
        if sample.get("video_query") is None:
            sample.pop("video_query", None)
    return default_collate(filtered_samples)


class PointOdysseyDataset(Dataset):
    def __init__(
        self,
        dataset_location="/home/zbf/Desktop/d4rt/d4rt-pytorch/data/pointodyssey",
        dset="train",
        use_augs=False,
        deterministic_sampling=False,
        S=48,
        N=32,
        strides=None,
        clip_step=2,
        quick=False,
        verbose=False,
        img_size=256,
        num_queries=2048,
        patch_size=9,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        cache_boundaries=True,
        sequence_name=None,
        query_mode="full",
        use_motion_boundaries=True,
        precompute_local_patches=True,
        return_query_video=False,
        static_scene_frame_idx: Optional[int] = None,
    ):
        self.dataset_location = dataset_location
        self.dset = dset
        self.use_augs = use_augs
        # Keep deterministic sample replay as an explicit opt-in. Disabling train-time
        # augmentations for overfit/debug runs should not freeze clip/query resampling.
        self.deterministic_sampling = deterministic_sampling
        self.S = S
        self.N = N
        self.strides = None if strides is None else list(strides)
        self.clip_step = clip_step
        self.quick = quick
        self.verbose = verbose
        self.img_size = img_size
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.cache_boundaries = cache_boundaries
        self.sequence_name = sequence_name
        self.query_mode = query_mode
        self.use_motion_boundaries = use_motion_boundaries
        self.precompute_local_patches = precompute_local_patches
        self.return_query_video = return_query_video
        self.static_scene_frame_idx = static_scene_frame_idx
        if self.query_mode not in {"full", "target_cam", "same_frame"}:
            raise ValueError(f"Unsupported query_mode: {self.query_mode}")

        self.root = os.path.join(dataset_location, dset)
        if not os.path.exists(self.root):
            print(f"Warning: Dataset root {self.root} does not exist!")
            self.dirs = []
        else:
            self.dirs = sorted(
                d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            )
            if sequence_name is not None:
                self.dirs = [d for d in self.dirs if d == sequence_name]
            if quick:
                self.dirs = self.dirs[:1]

        if verbose:
            print(f"Found {len(self.dirs)} sequences in {self.root}")

    def _build_frame_file_map(self, files):
        frame_map = {}
        for path in files:
            match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
            if match is None:
                continue
            frame_map[int(match.group(1))] = path
        return frame_map

    def _resolve_frame_path(self, frame_idx, files, frame_map):
        if frame_idx in frame_map:
            return frame_map[frame_idx]
        if 0 <= frame_idx < len(files):
            return files[frame_idx]
        raise FileNotFoundError(f"Missing file for frame index {frame_idx}")

    def __len__(self):
        return len(self.dirs)

    def _get_rngs(self, index):
        if self.deterministic_sampling:
            return random.Random(index), np.random.default_rng(index)
        return random, np.random

    def _load_rgb(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_depth(self, path):
        if path.endswith(".npy"):
            depth = np.load(path)
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(path)
        return decode_pointodyssey_depth(depth)

    def _load_normal(self, path):
        if path.endswith(".npy"):
            normal = np.load(path).astype(np.float32)
        else:
            normal = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if normal is None:
                raise FileNotFoundError(path)
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normal = normal * 2.0 - 1.0
        return normal

    def _load_normal_or_mask(self, path, fallback_shape):
        try:
            return self._load_normal(path), True
        except (FileNotFoundError, OSError, ValueError):
            height, width = fallback_shape
            return np.zeros((height, width, 3), dtype=np.float32), False

    def _sample_stride(self, total_frames, py_rng):
        if self.strides is None:
            max_stride = max(1, total_frames // max(1, self.S))
            # 论文设定：随机 stride
            # 即使 use_augs=False (如验证或过拟合实验)，也应该随机采样 stride
            # 限制 stride 在 1-4 之间 (经验值，防止运动过大)
            limit = min(4, max_stride)
            return py_rng.randint(1, limit) # randint(a, b) 包含 b

        valid_strides = [stride for stride in self.strides if total_frames >= self.S * stride]
        if not valid_strides:
            return 1
        # 即使指定了 stride 列表，也应该随机选择（如果列表有多个）
        return py_rng.choice(valid_strides)

    def _sample_crop(self, height, width, py_rng):
        if not self.use_augs:
            # 验证/过拟合模式：固定中心裁剪
            crop_size = min(height, width)
            x0 = (width - crop_size) // 2
            y0 = (height - crop_size) // 2
            return x0, y0, crop_size, crop_size

        area = height * width
        min_ratio = 3.0 / 4.0
        max_ratio = 4.0 / 3.0

        for _ in range(10):
            target_area = py_rng.uniform(0.3, 1.0) * area
            aspect = math.exp(py_rng.uniform(math.log(min_ratio), math.log(max_ratio)))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            if 0 < crop_w <= width and 0 < crop_h <= height:
                x0 = py_rng.randint(0, width - crop_w)
                y0 = py_rng.randint(0, height - crop_h)
                break
        else:
            crop_w = width
            crop_h = height
            x0 = 0
            y0 = 0

        if py_rng.random() < 0.05:
            zoom = py_rng.uniform(0.7, 0.95)
            zoom_w = max(1, int(round(crop_w * zoom)))
            zoom_h = max(1, int(round(crop_h * zoom)))
            x0 = x0 + max(0, (crop_w - zoom_w) // 2)
            y0 = y0 + max(0, (crop_h - zoom_h) // 2)
            crop_w = zoom_w
            crop_h = zoom_h

        return x0, y0, crop_w, crop_h

    def _apply_color_aug(self, rgb_frames, py_rng):
        if not self.use_augs:
            return rgb_frames

        brightness = py_rng.uniform(0.6, 1.4)
        contrast = py_rng.uniform(0.6, 1.4)
        saturation = py_rng.uniform(0.6, 1.4)
        hue_delta = py_rng.uniform(-0.1, 0.1) * 180.0
        apply_gray = py_rng.random() < 0.2
        apply_blur = py_rng.random() < 0.4
        blur_sigma = py_rng.uniform(0.1, 2.0)

        augmented = []
        for frame in rgb_frames:
            image = np.clip(frame * brightness, 0.0, 1.0)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast + mean, 0.0, 1.0)

            hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 255.0)
            hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            if apply_gray:
                gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
                image = np.repeat(gray[..., None], 3, axis=-1)

            if apply_blur:
                image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=blur_sigma, sigmaY=blur_sigma)

            augmented.append(np.clip(image, 0.0, 1.0))

        return augmented

    def _compute_boundary_mask(self, depth):
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sobel_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            return np.zeros_like(depth, dtype=bool)

        values = magnitude[valid]
        if values.size == 0 or float(np.max(values)) <= 0.0:
            return np.zeros_like(depth, dtype=bool)

        threshold = max(float(np.percentile(values, 85.0)), 1e-6)
        return valid & (magnitude >= threshold)

    def _compute_motion_boundary_mask(self, clip_trajs_2d, clip_valid, frame_idx, crop_h, crop_w):
        motion_map = np.zeros((crop_h, crop_w), dtype=np.float32)
        support_map = np.zeros((crop_h, crop_w), dtype=np.float32)
        points = clip_trajs_2d[frame_idx]
        current_valid = (clip_valid[frame_idx] > 0.5) & np.isfinite(points).all(axis=1)

        motion_mag = np.zeros((points.shape[0],), dtype=np.float32)
        motion_support = np.zeros((points.shape[0],), dtype=np.float32)

        if frame_idx + 1 < clip_trajs_2d.shape[0]:
            next_points = clip_trajs_2d[frame_idx + 1]
            next_valid = (clip_valid[frame_idx + 1] > 0.5) & np.isfinite(next_points).all(axis=1)
            pair_valid = current_valid & next_valid
            if np.any(pair_valid):
                motion_mag[pair_valid] += np.linalg.norm(next_points[pair_valid] - points[pair_valid], axis=1)
                motion_support[pair_valid] += 1.0

        if frame_idx > 0:
            prev_points = clip_trajs_2d[frame_idx - 1]
            prev_valid = (clip_valid[frame_idx - 1] > 0.5) & np.isfinite(prev_points).all(axis=1)
            pair_valid = current_valid & prev_valid
            if np.any(pair_valid):
                motion_mag[pair_valid] += np.linalg.norm(points[pair_valid] - prev_points[pair_valid], axis=1)
                motion_support[pair_valid] += 1.0

        motion_defined = motion_support > 0.0
        if not np.any(motion_defined):
            return np.zeros((crop_h, crop_w), dtype=bool)

        motion_mag[motion_defined] /= motion_support[motion_defined]
        src_xy = np.round(points[motion_defined]).astype(np.int32)
        src_xy[:, 0] = np.clip(src_xy[:, 0], 0, crop_w - 1)
        src_xy[:, 1] = np.clip(src_xy[:, 1], 0, crop_h - 1)
        for (x, y), value in zip(src_xy, motion_mag[motion_defined]):
            motion_map[y, x] += value
            support_map[y, x] += 1.0

        sampled = support_map > 0.0
        motion_map[sampled] /= support_map[sampled]
        motion_map = cv2.GaussianBlur(motion_map, ksize=(5, 5), sigmaX=0.0, sigmaY=0.0)
        support_mask = cv2.dilate(sampled.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        if not np.any(support_mask):
            return np.zeros((crop_h, crop_w), dtype=bool)

        sobel_x = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        support_values = magnitude[support_mask]
        if support_values.size == 0 or float(np.max(support_values)) <= 0.0:
            return np.zeros((crop_h, crop_w), dtype=bool)

        threshold = max(float(np.percentile(support_values, 85.0)), 1e-6)
        boundary = support_mask & (magnitude >= threshold)
        boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        return boundary


    def _build_transform_metadata(self, original_h, original_w, x0, y0, crop_h, crop_w):
        return {
            "canonical_space": torch.tensor(CANONICAL_QUERY_SPACE_CROP_NORMALIZED, dtype=torch.long),
            "original_hw": torch.tensor([float(original_h), float(original_w)], dtype=torch.float32),
            "crop_offset_xy": torch.tensor([float(x0), float(y0)], dtype=torch.float32),
            "crop_size_hw": torch.tensor([float(crop_h), float(crop_w)], dtype=torch.float32),
            "resized_hw": torch.tensor([float(self.img_size), float(self.img_size)], dtype=torch.float32),
        }

    def extract_patches(self, video, coords, patch_size=None):
        """
        Args:
            video: (S, H, W, 3) float32 in [0, 1]
            coords: (N, 3) with normalized (u, v, t_idx) in canonical crop-normalized space
        """
        patch_size = self.patch_size if patch_size is None else patch_size
        if torch.is_tensor(video):
            video_tensor = video.to(dtype=torch.float32)
        else:
            video_tensor = torch.from_numpy(video).to(dtype=torch.float32)
        if torch.is_tensor(coords):
            coords_tensor = coords.to(dtype=torch.float32)
        else:
            coords_tensor = torch.from_numpy(coords).to(dtype=torch.float32)

        frames_btchw = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        coords_uv = coords_tensor[:, :2].unsqueeze(0)
        t_src = coords_tensor[:, 2].round().long().unsqueeze(0)
        patches = extract_local_patches(frames_btchw, coords_uv, t_src, patch_size=patch_size)
        return patches.squeeze(0)

    def _sample_query_data(
        self,
        clip_trajs_2d,
        clip_world_3d,
        clip_valid,
        clip_vis,
        clip_extrinsics,
        crop_h,
        crop_w,
        normal_frames,
        normal_valid_frames,
        depth_frames,
        py_rng,
        np_rng,
    ):
        """
        为当前 clip 采样 query，并构造监督信号。

        这里返回的 supervision 会在 collate 后变成 batch["targets"]，供 loss 直接使用。

        坐标系约定:
        1. clip_trajs_2d:
           形状 (S, N, 2)，表示 crop 之后、resize 之前的 2D 像素坐标。
           也就是说，x/y 仍然是 crop 图像上的像素位置，而不是 [0, 1] 归一化坐标。
        2. coords_uv / targets["pos_2d"]:
           都会被归一化到当前 crop 空间，即除以 (crop_w - 1, crop_h - 1) 后得到 [0, 1]
           附近的值。注意如果目标点跑出 crop，pos_2d 仍然可能落在 [0, 1] 之外，但会由
           mask_2d 控制不参与 2D loss。
        3. clip_world_3d:
           形状 (S, N, 3)，是数据集原始提供的世界坐标系 3D 点。
        4. targets["pos_3d"] / targets["displacement"]:
           不是 world 坐标，而是先把 world 点通过 t_cam 指定的相机外参变换到相机坐标系
           后得到的监督。这样网络学的是“在某个 camera frame 下”的几何量。

        时间索引约定:
        1. t_src: query 来源帧。coords 表示这个 source frame 上的 2D 查询位置。
        2. t_tgt: 监督目标帧。pos_2d / visibility / normal 等主要对应这个帧。
        3. t_cam: 3D 监督所在的相机坐标系。pos_3d / displacement 都定义在这个相机系下。
        """
        num_frames, num_points, _ = clip_trajs_2d.shape

        # query 输入本身:
        # coords_uv: 每个 query 在 source frame 上的 2D 位置，归一化到 crop 空间 [0, 1]
        # t_src: 这个 query 从哪一帧发起
        # t_tgt: 要预测哪一帧上的结果
        # t_cam: 3D supervision 使用哪一帧的相机坐标系
        coords_uv = torch.zeros((self.num_queries, 2), dtype=torch.float32)
        t_src = torch.zeros((self.num_queries,), dtype=torch.long)
        t_tgt = torch.zeros((self.num_queries,), dtype=torch.long)
        t_cam = torch.zeros((self.num_queries,), dtype=torch.long)

        # 监督信号:
        # target_pos_2d: t_tgt 帧上的 2D 目标位置，归一化到 crop 空间
        # target_pos_3d: t_tgt 时刻该点在 t_cam 相机坐标系下的 3D 位置
        # target_vis: t_tgt 上该点是否“可见且在 crop 内”，用于 visibility 二分类监督
        # target_disp: 在同一个 t_cam 相机系下，目标点相对 source 点的 3D 位移
        # target_normal: t_tgt 帧该点的表面法向，来自 normal 图
        target_pos_2d = torch.zeros((self.num_queries, 2), dtype=torch.float32)
        target_pos_3d = torch.zeros((self.num_queries, 3), dtype=torch.float32)
        target_vis = torch.zeros((self.num_queries,), dtype=torch.float32)
        target_disp = torch.zeros((self.num_queries, 3), dtype=torch.float32)
        target_normal = torch.zeros((self.num_queries, 3), dtype=torch.float32)

        # 各种 supervision mask:
        # mask_3d: 该 query 是否拥有有效的 3D 位置监督
        # mask_2d: 该 query 是否拥有有效的 2D 位置监督
        # mask_disp: 该 query 是否拥有有效的 3D displacement 监督
        # mask_normal: 该 query 是否拥有有效的 normal 监督
        # mask_vis: 该 query 是否拥有有效的 visibility 监督
        #
        # 这些 mask 的作用是: target 张量即使始终有固定 shape，loss 也只会对“真的有效”
        # 的那部分 query 计算，避免因为点越界、不可见、法向缺失等情况把错误标签喂给模型。
        mask_3d = torch.zeros((self.num_queries,), dtype=torch.bool)
        mask_2d = torch.zeros((self.num_queries,), dtype=torch.bool)
        mask_disp = torch.zeros((self.num_queries,), dtype=torch.bool)
        mask_normal = torch.zeros((self.num_queries,), dtype=torch.bool)
        mask_vis = torch.zeros((self.num_queries,), dtype=torch.bool)

        # 下面这些不是直接训练主目标，而是“采样诊断信息 / 统计信息”:
        # source_is_boundary: source 点是否落在“任意边界”(深度边界或运动边界)上
        # source_is_depth_boundary: source 点是否落在深度边界上
        # source_is_motion_boundary: source 点是否落在运动边界上
        # point_indices: 这个 query 对应原始轨迹中的第几个点，便于可视化和调试
        source_is_boundary = torch.zeros((self.num_queries,), dtype=torch.bool)
        source_is_depth_boundary = torch.zeros((self.num_queries,), dtype=torch.bool)
        source_is_motion_boundary = torch.zeros((self.num_queries,), dtype=torch.bool)
        point_indices = torch.zeros((self.num_queries,), dtype=torch.long)

        # 为每一帧构造边界 mask，用于“边界点过采样”。
        # depth_boundary_masks 从 depth 梯度得到；
        # motion_boundary_masks 从轨迹运动幅值变化得到；
        # boundary_masks 取两者并集。
        depth_boundary_masks = [self._compute_boundary_mask(depth) for depth in depth_frames]
        if self.use_motion_boundaries:
            motion_boundary_masks = [
                self._compute_motion_boundary_mask(clip_trajs_2d, clip_valid, frame_idx, crop_h, crop_w)
                for frame_idx in range(num_frames)
            ]
        else:
            motion_boundary_masks = [
                np.zeros_like(depth_boundary_masks[frame_idx], dtype=bool)
                for frame_idx in range(num_frames)
            ]
        boundary_masks = [
            np.logical_or(depth_boundary_masks[frame_idx], motion_boundary_masks[frame_idx])
            for frame_idx in range(num_frames)
        ]
        valid_sources_by_frame = []
        boundary_sources_by_frame = []

        for frame_idx in range(num_frames):
            src_points = clip_trajs_2d[frame_idx]

            # source 点合法性的定义:
            # 1. 数据集标注里这个点在该帧有定义
            # 2. 该点在该帧可见
            # 3. 该点落在当前 crop 内
            # 4. 对应的 3D world 坐标是有限值
            src_defined = clip_valid[frame_idx] > 0.5
            src_visible = clip_vis[frame_idx] > 0.5
            src_in_bounds = (
                (src_points[:, 0] >= 0.0)
                & (src_points[:, 0] < crop_w)
                & (src_points[:, 1] >= 0.0)
                & (src_points[:, 1] < crop_h)
            )
            src_world_finite = np.isfinite(clip_world_3d[frame_idx]).all(axis=-1)
            valid_src = np.where(src_defined & src_visible & src_in_bounds & src_world_finite)[0]
            valid_sources_by_frame.append(valid_src)

            if len(valid_src) == 0 or self.boundary_ratio <= 0:
                boundary_sources_by_frame.append(np.empty((0,), dtype=np.int64))
                continue

            # 只在“已经合法”的 source 点里，再筛一遍哪些落在边界区域。
            # 这样后面做 boundary oversampling 时，不会采到非法点。
            src_xy = np.round(src_points[valid_src]).astype(np.int32)
            src_xy[:, 0] = np.clip(src_xy[:, 0], 0, crop_w - 1)
            src_xy[:, 1] = np.clip(src_xy[:, 1], 0, crop_h - 1)
            src_boundary = boundary_masks[frame_idx][src_xy[:, 1], src_xy[:, 0]]
            boundary_sources_by_frame.append(valid_src[src_boundary])

        valid_source_frames = [frame_idx for frame_idx, indices in enumerate(valid_sources_by_frame) if len(indices) > 0]
        if not valid_source_frames:
            raise RuntimeError("No valid source points available after augmentation")

        # D4RT Paper: Specialized Querying - Grid Sampling for Camera Pose/Intrinsics
        # We reserve a small portion of queries (e.g., 64) for grid sampling if needed for pose supervision
        # But per user request and paper "Method" section, this is usually for specific tasks.
        # However, the standard training mix includes everything.
        # To strictly follow the "Mix" description: "In a single batch... Group A... Group B".
        # The grid sampling for pose is described as "To derive relative camera pose... we create queries...".
        # It's often implemented as a separate task or mixed in.
        # Given the current structure, we implement the Temporal Sampling Mix (Group A/B) per query.

        for query_idx in range(self.num_queries):
            # 先随机选一个“存在合法 source 点”的 source frame。
            source_frame = py_rng.choice(valid_source_frames)

            if self.query_mode == "same_frame":
                # source / target / camera 全都取同一帧。
                # 常用于只关心单帧几何、去掉时序变化的实验。
                target_frame = source_frame
                camera_frame = source_frame
            elif self.query_mode == "target_cam":
                # 允许 source 和 target 不同，但 3D 监督总是在 target 对应的相机系下定义。
                target_frame = py_rng.randint(0, num_frames - 1)
                camera_frame = target_frame
            else:
                # D4RT Paper: Temporal Sampling - Pattern Mixing
                # Group A (40%): t_tgt = t_cam
                # Group B (60%): t_tgt != t_cam (random)
                if py_rng.random() < self.t_tgt_eq_t_cam_ratio:
                    target_frame = py_rng.randint(0, num_frames - 1)
                    camera_frame = target_frame
                else:
                    target_frame = py_rng.randint(0, num_frames - 1)
                    camera_frame = py_rng.randint(0, num_frames - 1)

            src_points = clip_trajs_2d[source_frame]
            valid_src = valid_sources_by_frame[source_frame]
            boundary_candidates = boundary_sources_by_frame[source_frame]

            # D4RT Paper: Spatial Sampling - Boundary Oversampling (30%)
            # 30% queries from boundaries (depth or motion), 70% random
            #
            # 这里采到的是“轨迹点索引 point_idx”，不是一个像素位置。
            # 后面 source/target 的 2D/3D 信息，都是围绕这个同一个轨迹点展开。
            if len(boundary_candidates) > 0 and py_rng.random() < self.boundary_ratio:
                point_idx = int(np_rng.choice(boundary_candidates))
            else:
                point_idx = int(np_rng.choice(valid_src))

            src_xy = clip_trajs_2d[source_frame, point_idx]
            tgt_xy = clip_trajs_2d[target_frame, point_idx]

            # source 点在 source_frame 上是否位于边界区域，仅用于统计/可视化/分析。
            src_px = int(np.clip(np.round(src_xy[0]), 0, crop_w - 1))
            src_py = int(np.clip(np.round(src_xy[1]), 0, crop_h - 1))
            source_is_boundary[query_idx] = bool(boundary_masks[source_frame][src_py, src_px])
            source_is_depth_boundary[query_idx] = bool(depth_boundary_masks[source_frame][src_py, src_px])
            source_is_motion_boundary[query_idx] = bool(motion_boundary_masks[source_frame][src_py, src_px])

            # target_defined: 数据集标注里，目标帧这个点是否存在
            # tgt_in_bounds: 目标点是否还落在 crop 范围内
            # vis_flag: 最终“2D 可监督”的定义，要求既有定义、又可见、又在 crop 内
            target_defined = bool(clip_valid[target_frame, point_idx] > 0.5)
            tgt_in_bounds = (
                tgt_xy[0] >= 0.0
                and tgt_xy[0] < crop_w
                and tgt_xy[1] >= 0.0
                and tgt_xy[1] < crop_h
            )
            vis_flag = bool(target_defined and clip_vis[target_frame, point_idx] > 0.5 and tgt_in_bounds)

            src_world = clip_world_3d[source_frame, point_idx]
            tgt_world = clip_world_3d[target_frame, point_idx]

            # 把 source / target 的 world 坐标都变换到同一个 camera_frame 相机坐标系下。
            # 这样定义有两个好处:
            # 1. pos_3d 和 displacement 都在统一坐标系里，监督更一致；
            # 2. 即使 source 和 target 来自不同时间，也能在同一个 camera 系里比较。
            src_cam_h = clip_extrinsics[camera_frame] @ np.concatenate([src_world, [1.0]], axis=0)
            tgt_cam_h = clip_extrinsics[camera_frame] @ np.concatenate([tgt_world, [1.0]], axis=0)
            src_cam = src_cam_h[:3].astype(np.float32)
            tgt_cam = tgt_cam_h[:3].astype(np.float32)

            # coords_uv 是模型输入 query 的 2D 位置，定义在 source_frame 上。
            coords_uv[query_idx, 0] = float(src_xy[0] / max(crop_w - 1, 1))
            coords_uv[query_idx, 1] = float(src_xy[1] / max(crop_h - 1, 1))
            t_src[query_idx] = source_frame
            t_tgt[query_idx] = target_frame
            t_cam[query_idx] = camera_frame

            # target_pos_2d 是监督目标在 target_frame 上的位置。
            # 即使目标点出界，这里仍会写入一个数值；是否参与 2D loss 由 mask_2d 决定。
            target_pos_2d[query_idx, 0] = float(tgt_xy[0] / max(crop_w - 1, 1))
            target_pos_2d[query_idx, 1] = float(tgt_xy[1] / max(crop_h - 1, 1))

            # target_pos_3d: 目标点在 t_cam 相机系下的 3D 位置。
            target_pos_3d[query_idx] = torch.from_numpy(tgt_cam)

            # visibility 是 0/1 浮点标签，监督网络预测“目标点是否可见”。
            target_vis[query_idx] = float(vis_flag)

            # displacement 也在 t_cam 相机系下定义，表示 target 相对 source 的 3D 位移。
            target_disp[query_idx] = torch.from_numpy(tgt_cam - src_cam)
            point_indices[query_idx] = point_idx

            # normal 监督来自 target_frame 的 normal 图。
            # 只有 target 可见，并且该帧 normal 图真的存在时，才写入法向值。
            if vis_flag and normal_valid_frames[target_frame]:
                x = int(np.clip(round(float(tgt_xy[0])), 0, crop_w - 1))
                y = int(np.clip(round(float(tgt_xy[1])), 0, crop_h - 1))
                target_normal[query_idx] = torch.from_numpy(normal_frames[target_frame][y, x].astype(np.float32))

            # 各个 mask 的含义:
            #
            # has_valid_3d:
            #   target 这个点在标注里存在，且 source/target 变换到相机系后的数值有限。
            #   用于 3D position loss 和 displacement loss。
            #
            # has_valid_normal:
            #   当前实现里用“法向数值有限”来判断 normal 是否可用。
            #   注意: target_normal 默认初始化为 0，因此如果某帧 normal 缺失，这里的判断会
            #   把 0 当成一个有限值。这个逻辑目前偏宽松，阅读时要留意这一点。
            has_valid_3d = bool(target_defined and np.isfinite(src_cam).all() and np.isfinite(tgt_cam).all())
            has_valid_normal = bool(vis_flag and torch.isfinite(target_normal[query_idx]).all().item())
            mask_3d[query_idx] = has_valid_3d
            mask_2d[query_idx] = vis_flag
            mask_vis[query_idx] = target_defined
            mask_disp[query_idx] = has_valid_3d
            mask_normal[query_idx] = has_valid_normal

        # 返回给训练代码的 targets 字典。
        #
        # batch 维拼起来以后，各字段的典型 shape 会变成:
        # pos_2d                -> float32 [B, Q, 2]
        # pos_3d                -> float32 [B, Q, 3]
        # visibility            -> float32 [B, Q]
        # displacement          -> float32 [B, Q, 3]
        # normal                -> float32 [B, Q, 3]
        # mask_*                -> bool    [B, Q]
        # source_is_*           -> bool    [B, Q]
        # point_indices         -> int64   [B, Q]
        targets = {
            # t_tgt 帧上的 2D 真值位置，归一化到 crop 空间。
            "pos_2d": target_pos_2d,
            # t_tgt 时刻该点在 t_cam 相机坐标系下的 3D 位置真值。
            "pos_3d": target_pos_3d,
            # 目标点是否“可见且在 crop 内”的 0/1 标签。
            "visibility": target_vis,
            # 在 t_cam 相机系下，target 相对 source 的 3D 位移。
            "displacement": target_disp,
            # target 帧上该点的表面法向量，来自 normal 图。
            "normal": target_normal,

            # 3D 位置监督是否有效。
            "mask_3d": mask_3d,
            # 2D 位置监督是否有效。只有可见且在 crop 内才为 True。
            "mask_2d": mask_2d,
            # visibility 监督是否有效。只要求该点在 target 帧有定义。
            "mask_vis": mask_vis,
            # displacement 监督是否有效。和 mask_3d 一致。
            "mask_disp": mask_disp,
            # normal 监督是否有效。
            "mask_normal": mask_normal,

            # source 点是否位于任意边界区域。
            "source_is_boundary": source_is_boundary,
            # source 点是否位于深度边界。
            "source_is_depth_boundary": source_is_depth_boundary,
            # source 点是否位于运动边界。
            "source_is_motion_boundary": source_is_motion_boundary,
            # query 对应的原始轨迹点索引，便于调试和可视化。
            "point_indices": point_indices,
        }

        return coords_uv, t_src, t_tgt, t_cam, targets

    def _resolve_static_scene_frame_idx(self, num_frames):
        if self.static_scene_frame_idx is None:
            return None

        frame_idx = int(self.static_scene_frame_idx)
        if frame_idx < 0:
            frame_idx += num_frames
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(
                f"static_scene_frame_idx={self.static_scene_frame_idx} is out of range for clip length {num_frames}"
            )
        return frame_idx

    def _apply_static_scene_degradation(
        self,
        frame_indices,
        rgb_frames,
        depth_frames,
        normal_frames,
        normal_valid_frames,
        clip_trajs_2d,
        clip_world_3d,
        clip_intrinsics,
        clip_valid,
        clip_vis,
        clip_extrinsics,
    ):
        anchor_idx = self._resolve_static_scene_frame_idx(len(rgb_frames))
        if anchor_idx is None:
            return (
                frame_indices,
                rgb_frames,
                depth_frames,
                normal_frames,
                normal_valid_frames,
                clip_trajs_2d,
                clip_world_3d,
                clip_intrinsics,
                clip_valid,
                clip_vis,
                clip_extrinsics,
            )

        num_frames = len(rgb_frames)
        anchor_frame_index = int(frame_indices[anchor_idx])
        anchor_rgb = rgb_frames[anchor_idx]
        anchor_depth = depth_frames[anchor_idx]
        anchor_normal = normal_frames[anchor_idx]
        anchor_normal_valid = bool(normal_valid_frames[anchor_idx])

        return (
            [anchor_frame_index] * num_frames,
            [anchor_rgb.copy() for _ in range(num_frames)],
            [anchor_depth.copy() for _ in range(num_frames)],
            [anchor_normal.copy() for _ in range(num_frames)],
            [anchor_normal_valid] * num_frames,
            np.repeat(clip_trajs_2d[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_world_3d[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_intrinsics[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_valid[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_vis[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_extrinsics[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
        )

    def __getitem__(self, index):
        seq_name = self.dirs[index]
        seq_path = os.path.join(self.root, seq_name)
        py_rng, np_rng = self._get_rngs(index)

        try:
            anno_path = os.path.join(seq_path, "anno.npz")
            if not os.path.exists(anno_path):
                npzs = glob.glob(os.path.join(seq_path, "*.npz"))
                if not npzs:
                    raise FileNotFoundError(f"No annotation found in {seq_path}")
                anno_path = npzs[0]

            anno = np.load(anno_path, allow_pickle=True)
            trajs_2d = anno["trajs_2d"]
            trajs_3d = anno["trajs_3d"]
            intrinsics = anno["intrinsics"]
            valids = anno["valids"]
            visibilities = anno["visibs"] if "visibs" in anno else valids
            extrinsics = anno["extrinsics"]
            total_frames = trajs_2d.shape[0]

            if (
                trajs_2d.ndim != 3
                or trajs_3d.ndim != 3
                or intrinsics.ndim != 3
                or valids.ndim != 2
                or visibilities.ndim != 2
            ):
                return {}, False

            stride = self._sample_stride(total_frames, py_rng)
            # Ensure the last frame index t_start + (S-1)*stride is within bounds [0, total_frames-1]
            max_safe_start = max(0, total_frames - 1 - (self.S - 1) * stride)
            # 论文设定：随机起始帧
            # 即使 use_augs=False，也要随机起始帧
            t_start = py_rng.randint(0, max_safe_start)
            frame_indices = [t_start + i * stride for i in range(self.S)]

            rgb_dir = os.path.join(seq_path, "rgbs")
            depth_dir = os.path.join(seq_path, "depths")
            normal_dir = os.path.join(seq_path, "normals")

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
            if not rgb_files:
                rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

            depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
            if not depth_files:
                depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

            normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.jpg")))
            if not normal_files:
                normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            if not normal_files:
                normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.npy")))

            rgb_file_map = self._build_frame_file_map(rgb_files)
            depth_file_map = self._build_frame_file_map(depth_files)
            normal_file_map = self._build_frame_file_map(normal_files)

            rgb_paths = [self._resolve_frame_path(t, rgb_files, rgb_file_map) for t in frame_indices]
            depth_paths = [self._resolve_frame_path(t, depth_files, depth_file_map) for t in frame_indices]
            normal_paths = [self._resolve_frame_path(t, normal_files, normal_file_map) for t in frame_indices]

            rgb_frames = [self._load_rgb(path) for path in rgb_paths]
            depth_frames = [self._load_depth(path) for path in depth_paths]
            normal_frames = []
            normal_valid_frames = []
            for frame_idx, path in enumerate(normal_paths):
                normal_frame, is_valid = self._load_normal_or_mask(
                    path,
                    rgb_frames[frame_idx].shape[:2],
                )
                normal_frames.append(normal_frame)
                normal_valid_frames.append(is_valid)

            clip_trajs_2d = trajs_2d[frame_indices].astype(np.float32)
            clip_world_3d = trajs_3d[frame_indices].astype(np.float32)
            clip_intrinsics = intrinsics[frame_indices].astype(np.float32)
            clip_valid = valids[frame_indices].astype(np.float32)
            clip_vis = visibilities[frame_indices].astype(np.float32)
            clip_extrinsics = extrinsics[frame_indices].astype(np.float32)

            (
                frame_indices,
                rgb_frames,
                depth_frames,
                normal_frames,
                normal_valid_frames,
                clip_trajs_2d,
                clip_world_3d,
                clip_intrinsics,
                clip_valid,
                clip_vis,
                clip_extrinsics,
            ) = self._apply_static_scene_degradation(
                frame_indices=frame_indices,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                normal_frames=normal_frames,
                normal_valid_frames=normal_valid_frames,
                clip_trajs_2d=clip_trajs_2d,
                clip_world_3d=clip_world_3d,
                clip_intrinsics=clip_intrinsics,
                clip_valid=clip_valid,
                clip_vis=clip_vis,
                clip_extrinsics=clip_extrinsics,
            )

            original_h, original_w = rgb_frames[0].shape[:2]
            x0, y0, crop_w, crop_h = self._sample_crop(original_h, original_w, py_rng)

            rgb_frames = [frame[y0:y0 + crop_h, x0:x0 + crop_w] for frame in rgb_frames]
            depth_frames = [depth[y0:y0 + crop_h, x0:x0 + crop_w] for depth in depth_frames]
            normal_frames = [normal[y0:y0 + crop_h, x0:x0 + crop_w] for normal in normal_frames]
            rgb_frames = self._apply_color_aug(rgb_frames, py_rng)

            clip_trajs_2d = crop_points_2d(clip_trajs_2d, x0=x0, y0=y0)
            in_bounds = (
                (clip_trajs_2d[..., 0] >= 0.0)
                & (clip_trajs_2d[..., 0] < crop_w)
                & (clip_trajs_2d[..., 1] >= 0.0)
                & (clip_trajs_2d[..., 1] < crop_h)
            )
            clip_vis = clip_vis * in_bounds.astype(np.float32)
            clip_intrinsics_crop, clip_intrinsics_resized = transform_intrinsics_for_crop_resize(
                clip_intrinsics,
                x0=x0,
                y0=y0,
                crop_h=crop_h,
                crop_w=crop_w,
                dst_h=self.img_size,
                dst_w=self.img_size,
            )

            query_video = None
            if self.return_query_video:
                query_video = torch.stack(
                    [torch.from_numpy(frame).permute(2, 0, 1) for frame in rgb_frames],
                    dim=0,
                )

            resized_video = torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32)
            resized_depths = torch.zeros((self.S, 1, self.img_size, self.img_size), dtype=torch.float32)
            resized_normals = torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32)

            for frame_idx in range(self.S):
                rgb_resized = cv2.resize(rgb_frames[frame_idx], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                depth_resized = cv2.resize(depth_frames[frame_idx], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                normal_resized = cv2.resize(normal_frames[frame_idx], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                resized_video[frame_idx] = torch.from_numpy(rgb_resized).permute(2, 0, 1)
                resized_depths[frame_idx] = torch.from_numpy(depth_resized).unsqueeze(0)
                resized_normals[frame_idx] = torch.from_numpy(normal_resized).permute(2, 0, 1)

            coords_uv, t_src, t_tgt, t_cam, targets = self._sample_query_data(
                clip_trajs_2d=clip_trajs_2d,
                clip_world_3d=clip_world_3d,
                clip_valid=clip_valid,
                clip_vis=clip_vis,
                clip_extrinsics=clip_extrinsics,
                crop_h=crop_h,
                crop_w=crop_w,
                normal_frames=normal_frames,
                normal_valid_frames=normal_valid_frames,
                depth_frames=depth_frames,
                py_rng=py_rng,
                np_rng=np_rng,
            )

            if bool(targets["mask_normal"].any()):
                normal_indices = torch.where(targets["mask_normal"])[0]
                target_uv = targets["pos_2d"][normal_indices]
                target_frames = t_tgt[normal_indices]
                x = torch.clamp(torch.round(target_uv[:, 0] * (self.img_size - 1)).long(), 0, self.img_size - 1)
                y = torch.clamp(torch.round(target_uv[:, 1] * (self.img_size - 1)).long(), 0, self.img_size - 1)
                targets["normal"][normal_indices] = resized_normals[target_frames, :, y, x]

            local_patches = None
            if self.precompute_local_patches:
                patch_queries = torch.cat(
                    [coords_uv, t_src.unsqueeze(-1).to(torch.float32)],
                    dim=-1,
                )
                local_patches = self.extract_patches(
                    resized_video.permute(0, 2, 3, 1),
                    patch_queries,
                )

            transform_metadata = self._build_transform_metadata(
                original_h=original_h,
                original_w=original_w,
                x0=x0,
                y0=y0,
                crop_h=crop_h,
                crop_w=crop_w,
            )

            sample = {
                "video": resized_video,
                "coords": coords_uv,
                "t_src": t_src,
                "t_tgt": t_tgt,
                "t_cam": t_cam,
                "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
                "intrinsics_original": torch.from_numpy(clip_intrinsics),
                "intrinsics_crop": torch.from_numpy(clip_intrinsics_crop),
                "intrinsics": torch.from_numpy(clip_intrinsics_resized),
                "extrinsics": torch.from_numpy(clip_extrinsics),
                "depths": resized_depths,
                "normals": resized_normals,
                "transform_metadata": transform_metadata,
                "aspect_ratio": torch.tensor(
                    [float(crop_w) / max(float(crop_h), 1.0)],
                    dtype=torch.float32,
                ),
                "targets": targets,
            }
            if local_patches is not None:
                sample["local_patches"] = local_patches
            if query_video is not None:
                sample["video_query"] = query_video
            return sample, True

        except Exception as exc:
            print(f"Error loading sequence {seq_name}: {exc}")
            import traceback
            traceback.print_exc()
            return {}, False
