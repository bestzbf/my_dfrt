"""
D4RT Query Builder  —  第 3 层（D4RT Query Builder）

职责：
  - 接受 TransformResult（来自 GeometryTransformPipeline）
  - 采样 t_src / t_tgt / t_cam（含 40% t_tgt=t_cam）
  - 采样 num_queries 个 source queries（30% boundary oversampling）
  - 构造所有 targets 和 mask_* 字段
  - 提取 local RGB patches

不做的事：
  - 不读文件
  - 不做 crop / resize
  - 不决定混合采样权重
  - 不对 dataset_name 做 if/else 分叉

has_tracks=False 的 clip（ScanNet、Co3D 等）：
  - mask_2d / mask_disp / mask_vis 全部为 False
  - 仍可产出 mask_3d（通过 depth 反投影），由调用方决定是否启用
  - 该路径由 metadata['has_tracks'] 控制，query builder 本身不感知 dataset 名
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from datasets.transforms import TransformResult
from utils.patches import extract_local_patches


# ---------------------------------------------------------------------------
# Boundary mask helpers
# ---------------------------------------------------------------------------

def _depth_boundary_mask(depth: np.ndarray) -> np.ndarray:
    """
    Bool mask [H,W]：深度边缘（Sobel 梯度 >= 85th percentile）。
    depth: [H,W] float32，无效值为 nan/inf/0。
    """
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros_like(d, dtype=bool)
    vals = mag[valid]
    if vals.size == 0 or float(vals.max()) <= 0.0:
        return np.zeros_like(d, dtype=bool)
    thr = max(float(np.percentile(vals, 85.0)), 1e-6)
    return valid & (mag >= thr)


def _motion_boundary_mask(
    trajs_2d: np.ndarray,   # [T,N,2]
    valids: np.ndarray,     # [T,N] bool
    frame_idx: int,
    crop_h: int,
    crop_w: int,
    temporal_step: int = 1,
) -> np.ndarray:
    """
    Bool mask [H,W]：运动边缘，由相邻帧轨迹速度场的 Sobel 导出。
    仅当 trajs_2d 存在时调用。
    """
    motion_map = np.zeros((crop_h, crop_w), dtype=np.float32)
    support_map = np.zeros((crop_h, crop_w), dtype=np.float32)

    pts = trajs_2d[frame_idx]                                   # [N,2]
    cur_valid = valids[frame_idx] & np.isfinite(pts).all(-1)

    mag = np.zeros(pts.shape[0], dtype=np.float32)
    sup = np.zeros(pts.shape[0], dtype=np.float32)

    for nbr_idx in [frame_idx + temporal_step, frame_idx - temporal_step]:
        if not (0 <= nbr_idx < trajs_2d.shape[0]):
            continue
        nbr = trajs_2d[nbr_idx]
        nbr_valid = valids[nbr_idx] & np.isfinite(nbr).all(-1)
        pair = cur_valid & nbr_valid
        if pair.any():
            mag[pair] += np.linalg.norm(nbr[pair] - pts[pair], axis=1)
            sup[pair] += 1.0

    defined = sup > 0.0
    if not defined.any():
        return np.zeros((crop_h, crop_w), dtype=bool)

    mag[defined] /= sup[defined]
    xy = np.round(pts[defined]).astype(np.int32)
    if xy.size == 0:
        return np.zeros((crop_h, crop_w), dtype=bool)
    xx = np.clip(xy[:, 0], 0, crop_w - 1)
    yy = np.clip(xy[:, 1], 0, crop_h - 1)
    flat_idx    = yy * crop_w + xx
    motion_map  = np.bincount(flat_idx, weights=mag[defined],
                              minlength=crop_h * crop_w).reshape(crop_h, crop_w).astype(np.float32)
    support_map = np.bincount(flat_idx,
                              minlength=crop_h * crop_w).reshape(crop_h, crop_w).astype(np.float32)

    filled = support_map > 0.0
    motion_map[filled] /= support_map[filled]
    motion_map = cv2.GaussianBlur(motion_map, (5, 5), sigmaX=0.0)
    support_mask = (
        cv2.dilate(filled.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    if not support_mask.any():
        return np.zeros((crop_h, crop_w), dtype=bool)

    sx2 = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
    sy2 = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = np.sqrt(sx2 ** 2 + sy2 ** 2)
    sup_vals = mag2[support_mask]
    if sup_vals.size == 0 or float(sup_vals.max()) <= 0.0:
        return np.zeros((crop_h, crop_w), dtype=bool)

    thr = max(float(np.percentile(sup_vals, 85.0)), 1e-6)
    boundary = support_mask & (mag2 >= thr)
    boundary = (
        cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    return boundary


# ---------------------------------------------------------------------------
# Query builder output
# ---------------------------------------------------------------------------

@dataclass
class QuerySample:
    """最终训练样本，由 D4RTQueryBuilder 产出。"""

    # ---- video ----
    video: torch.Tensor             # [S,3,H,W]  float32  [0,1]
    # Cropped frames at original crop resolution (before resize): [S,3,crop_h,crop_w]
    # Set to None when all frames already match img_size.
    highres_video: Optional[torch.Tensor]
    depths: Optional[torch.Tensor]  # [S,1,H,W]  float32  | None
    normals: Optional[torch.Tensor] # [S,3,H,W]  float32  | None

    # ---- query indices ----
    coords: torch.Tensor            # [Q,2]  归一化到 [0,1]（crop 坐标系）
    t_src: torch.Tensor             # [Q]    long
    t_tgt: torch.Tensor             # [Q]    long
    t_cam: torch.Tensor             # [Q]    long

    # ---- camera ----
    intrinsics: torch.Tensor        # [S,3,3]
    extrinsics: torch.Tensor        # [S,4,4]

    # ---- supervision targets ----
    targets: dict                   # see _build_empty_targets for keys

    # ---- optional patches ----
    local_patches: Optional[torch.Tensor]   # [Q,3,P,P]  | None

    # ---- transform metadata (for sampled_highres patch provider) ----
    transform_metadata: dict

    # ---- aspect ratio ----
    aspect_ratio: torch.Tensor              # [1] actual input view width / height before square resize

    # ---- metadata ----
    dataset_name: str
    sequence_name: str
    metadata: dict


def _build_empty_targets(num_queries: int) -> dict:
    Q = num_queries
    return {
        "pos_2d":                   torch.zeros(Q, 2),
        "pos_3d":                   torch.zeros(Q, 3),
        "visibility":               torch.zeros(Q),
        "displacement":             torch.zeros(Q, 3),
        "normal":                   torch.zeros(Q, 3),
        "mask_3d":                  torch.zeros(Q, dtype=torch.bool),
        "mask_2d":                  torch.zeros(Q, dtype=torch.bool),
        "mask_vis":                 torch.zeros(Q, dtype=torch.bool),
        "mask_disp":                torch.zeros(Q, dtype=torch.bool),
        "mask_normal":              torch.zeros(Q, dtype=torch.bool),
        "source_is_boundary":       torch.zeros(Q, dtype=torch.bool),
        "source_is_depth_boundary": torch.zeros(Q, dtype=torch.bool),
        "source_is_motion_boundary":torch.zeros(Q, dtype=torch.bool),
        "point_indices":            torch.zeros(Q, dtype=torch.long),
        # Semantic marker for static-reprojection supervision on has_tracks=False data.
        "is_static_reprojection":   torch.zeros(Q, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Per-dataset camera-space Z thresholds for mask_3d gating
# ---------------------------------------------------------------------------

# Camera-space depth upper bound per dataset name.
# Points with Z > threshold get mask_3d=False and are excluded from 3D loss.
# Thresholds derived from empirical Z-distribution survey (~30 sequences each,
# 5 frames per sequence, 2000 points per frame, camera-space Z values).
_DATASET_Z_MAX: dict[str, float] = {
    # 实测相机坐标系 Z 分布（各 ~30 sequences，每帧 2000 点抽样）：
    #   pointodyssey:   p50=2.99  p99=8.48  p99.9=31.76  max=517  → 12 覆盖 >99.7%
    #   kubric:         p50=8.66  p99=49.11 p99.9=49.90  max=50.8 → 55 覆盖全部
    #   dynamic_replica:p50=2.92  p99=7.33  p99.9=8.35   max=9.08 → 12 覆盖全部
    #   scannet:        metric metres, 室内场景，实测 max ≈ 2.18 m → 5 安全
    #   co3dv2:         任意坐标单位，14% 有效点 Z>20 → 50 保守上限
    #   blendedmvs:     rendered depth, 场景尺度差异极大（室内 ~1m 到室外 ~468m）
    #                   35% 场景 max>20m, p99 ≈ 448m, max=468m → 500 覆盖全部
    "pointodyssey":    12.0,
    "kubric":          55.0,
    "dynamic_replica": 12.0,
    "scannet":          5.0,
    "co3dv2":          50.0,
    "blendedmvs":     500.0,
}
_DEFAULT_Z_MAX = 20.0   # fallback for unknown datasets


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class D4RTQueryBuilder:
    """
    将 TransformResult 转换成最终训练 sample。

    Args:
        num_queries:          每个 clip 采样的 query 数，论文设定 2048。
        boundary_ratio:       boundary oversampling 概率，论文设定 0.3。
        t_tgt_eq_t_cam_ratio: t_tgt = t_cam 的概率，论文设定 0.4。
        patch_size:           local patch 边长（奇数），默认 9。
        use_motion_boundaries:是否计算运动边界 mask（需要 has_tracks=True）。
        precompute_patches:   是否预计算 local patches。
        precompute_from_highres: 是否从 highres_video 提取 patches（需要 precompute_patches=True）。
        dataset_z_max:        per-dataset camera-space Z upper bound for mask_3d.
                              Keys are dataset names; missing keys fall back to
                              _DATASET_Z_MAX then _DEFAULT_Z_MAX.
    """

    def __init__(
        self,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        patch_size: int = 9,
        use_motion_boundaries: bool = True,
        precompute_patches: bool = True,
        precompute_from_highres: bool = False,
        allow_track_fallback: bool = False,
        dataset_z_max: Optional[dict] = None,
    ) -> None:
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.patch_size = patch_size
        self.use_motion_boundaries = use_motion_boundaries
        self.precompute_patches = precompute_patches
        self.precompute_from_highres = precompute_from_highres
        self.allow_track_fallback = allow_track_fallback
        self.dataset_z_max: dict[str, float] = dataset_z_max or {}

        if precompute_from_highres and not precompute_patches:
            raise ValueError("precompute_from_highres requires precompute_patches=True")

    def _get_z_max(self, dataset_name: str) -> float:
        """Return the camera-space Z upper bound for mask_3d gating."""
        if dataset_name in self.dataset_z_max:
            return float(self.dataset_z_max[dataset_name])
        return _DATASET_Z_MAX.get(dataset_name, _DEFAULT_Z_MAX)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        result: TransformResult,
        py_rng: Optional[_random_module.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> QuerySample:
        if py_rng is None:
            py_rng = _random_module.Random()
        if np_rng is None:
            # Keep numpy-side sampling deterministic with the caller-provided
            # python RNG so the same dataset index maps to the same sample.
            np_rng = np.random.default_rng(py_rng.randint(0, 2**32 - 1))

        has_tracks = bool(result.metadata.get("has_tracks", False))
        T = len(result.images)
        cw, ch = result.crop.crop_w, result.crop.crop_h

        sample_metadata = dict(result.metadata)
        sample_metadata["has_temporal_supervision"] = has_tracks
        sample_metadata["query_semantics"] = (
            "full_temporal" if has_tracks else "static_reconstruction"
        )

        # ---- 转 tensor（video / depth / normal） ----
        video = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1) for img in result.images], dim=0
        )  # [T,3,H,W]

        depths_t: Optional[torch.Tensor] = None
        if result.depths is not None:
            depths_t = torch.stack(
                [torch.from_numpy(d).unsqueeze(0) for d in result.depths], dim=0
            )  # [T,1,H,W]

        normals_t: Optional[torch.Tensor] = None
        if result.normals is not None:
            normals_t = torch.stack(
                [torch.from_numpy(n).permute(2, 0, 1) for n in result.normals], dim=0
            )  # [T,3,H,W]

        # Build highres_video from cropped frames (before resize). Only keep it
        # when crop resolution differs from img_size; otherwise the resized path
        # already carries the same information.
        highres_video: Optional[torch.Tensor] = None
        if getattr(result, "cropped_images", None) is not None:
            ch_actual = result.cropped_images[0].shape[0]
            cw_actual = result.cropped_images[0].shape[1]
            for i, img in enumerate(result.cropped_images):
                if img.shape[0] != ch_actual or img.shape[1] != cw_actual:
                    raise ValueError(
                        f"Frame {i} has inconsistent crop size {img.shape[:2]} vs expected "
                        f"({ch_actual}, {cw_actual})."
                    )
            if ch_actual != result.img_size or cw_actual != result.img_size:
                highres_video = torch.stack(
                    [torch.from_numpy(img).permute(2, 0, 1) for img in result.cropped_images], dim=0
                )

        # ---- boundary masks（在 crop 分辨率下计算） ----
        depth_boundary: list[np.ndarray] = []
        if result.depths is not None:
            # depths 已是 resize 后的图；在 img_size 分辨率下算边界
            depth_boundary = [_depth_boundary_mask(d) for d in result.depths]
        else:
            S = result.img_size
            depth_boundary = [np.zeros((S, S), dtype=bool) for _ in range(T)]

        motion_boundary: list[np.ndarray] = []
        if has_tracks and self.use_motion_boundaries and result.trajs_2d is not None:
            # trajs_2d 在 crop 坐标系，motion boundary 也在 crop 坐标系
            # → resize 到 img_size 以和 depth_boundary 对齐
            S = result.img_size
            for fi in range(T):
                mb_crop = _motion_boundary_mask(
                    result.trajs_2d,
                    result.valids.astype(bool) if result.valids is not None
                    else np.ones(result.trajs_2d.shape[:2], dtype=bool),
                    frame_idx=fi,
                    crop_h=ch,
                    crop_w=cw,
                )
                # resize to img_size
                mb_resized = cv2.resize(
                    mb_crop.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                motion_boundary.append(mb_resized)
        else:
            S = result.img_size
            motion_boundary = [np.zeros((S, S), dtype=bool) for _ in range(T)]

        boundary = [
            np.logical_or(depth_boundary[fi], motion_boundary[fi]) for fi in range(T)
        ]

        # ---- 采样 queries ----
        if has_tracks:
            try:
                coords, t_src, t_tgt, t_cam, targets = self._sample_with_tracks(
                    result, depth_boundary, motion_boundary, boundary, py_rng, np_rng
                )
            except RuntimeError:
                if not self.allow_track_fallback:
                    raise
                coords, t_src, t_tgt, t_cam, targets = self._sample_no_tracks(
                    result, depth_boundary, boundary, py_rng, np_rng
                )
                sample_metadata["has_temporal_supervision"] = False
                sample_metadata["query_semantics"] = "tracks_missing_fallback"
        else:
            coords, t_src, t_tgt, t_cam, targets = self._sample_no_tracks(
                result, depth_boundary, boundary, py_rng, np_rng
            )

        # ---- local patches ----
        local_patches: Optional[torch.Tensor] = None
        if self.precompute_patches:
            patch_queries = torch.cat(
                [coords, t_src.unsqueeze(-1).float()], dim=-1
            )  # [Q,3]
            if self.precompute_from_highres and highres_video is not None:
                highres_hwc = highres_video.permute(0, 2, 3, 1)  # [T,H,W,3]
                local_patches = self._extract_patches(highres_hwc, patch_queries)
            else:
                video_hwc = video.permute(0, 2, 3, 1)   # [T,H,W,3]
                local_patches = self._extract_patches(video_hwc, patch_queries)

        crop = result.crop
        transform_metadata = {
            "canonical_space": torch.tensor(0, dtype=torch.long),
            "original_hw": torch.tensor(
                [float(result.original_h), float(result.original_w)], dtype=torch.float32
            ),
            "crop_offset_xy": torch.tensor(
                [float(crop.x0), float(crop.y0)], dtype=torch.float32
            ),
            "crop_size_hw": torch.tensor(
                [float(crop.crop_h), float(crop.crop_w)], dtype=torch.float32
            ),
            "resized_hw": torch.tensor(
                [float(result.img_size), float(result.img_size)], dtype=torch.float32
            ),
        }

        return QuerySample(
            video=video,
            highres_video=highres_video,
            depths=depths_t,
            normals=normals_t,
            coords=coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            intrinsics=torch.from_numpy(result.intrinsics),
            extrinsics=torch.from_numpy(result.extrinsics),
            targets=targets,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
            aspect_ratio=torch.tensor(
                [float(crop.crop_w) / max(float(crop.crop_h), 1.0)],
                dtype=torch.float32,
            ),
            dataset_name=result.dataset_name,
            sequence_name=result.sequence_name,
            metadata=sample_metadata,
        )

    # ------------------------------------------------------------------
    # has_tracks=True 路径（PointOdyssey / Kubric / DynamicReplica）
    # ------------------------------------------------------------------

    def _sample_with_tracks(
        self,
        result: TransformResult,
        depth_boundary: list[np.ndarray],
        motion_boundary: list[np.ndarray],
        boundary: list[np.ndarray],
        py_rng: _random_module.Random,
        np_rng: np.random.Generator,
    ) -> tuple:
        trajs_2d    = result.trajs_2d       # [T,N,2]  crop coords
        trajs_3d    = result.trajs_3d_world # [T,N,3]
        valids      = result.valids.astype(np.float32)  # [T,N]
        visibs      = result.visibs.astype(np.float32)  # [T,N]
        extrinsics  = result.extrinsics     # [T,4,4]
        T, N        = trajs_2d.shape[:2]
        cw, ch      = result.crop.crop_w, result.crop.crop_h
        S           = result.img_size

        targets = _build_empty_targets(self.num_queries)

        # ---- 预计算每帧有效 source 点 ----
        valid_by_frame: list[np.ndarray] = []
        boundary_by_frame: list[np.ndarray] = []

        for fi in range(T):
            pts = trajs_2d[fi]                                  # [N,2]
            in_bounds = (
                (pts[:, 0] >= 0) & (pts[:, 0] < cw) &
                (pts[:, 1] >= 0) & (pts[:, 1] < ch)
            )
            world_finite = np.isfinite(trajs_3d[fi]).all(-1)

            # 保留相机前方、深度图一致且重投影一致的点，避免明显离群轨迹污染采样池。
            E_fi = extrinsics[fi]  # [4,4]
            world_h = np.concatenate([trajs_3d[fi].astype(np.float32),
                                     np.ones((N, 1), dtype=np.float32)], axis=-1)  # [N,4]
            cam_coords = (E_fi @ world_h.T).T[:, :3]  # [N,3]
            depth_z = cam_coords[:, 2]
            depth_valid = (depth_z > 1e-3) & (depth_z < 1500) & np.isfinite(depth_z)

            depth_map_valid = np.ones(trajs_3d[fi].shape[0], dtype=bool)
            if result.depths is not None:
                sx, sy = S / cw, S / ch
                # Only index the resized depth map for finite in-bounds points.
                # Extremely large off-screen coords can still be finite, and
                # casting them to int32 would raise RuntimeWarning before the
                # later in_bounds mask filters them out.
                pts_depth_ok = np.isfinite(pts).all(-1) & in_bounds
                d_px = np.zeros(trajs_3d[fi].shape[0], dtype=np.float32)
                if pts_depth_ok.any():
                    pts_img = pts[pts_depth_ok] * np.array([sx, sy], dtype=np.float32)
                    xi = np.clip(np.round(pts_img[:, 0]), 0, S - 1).astype(np.int32)
                    yi = np.clip(np.round(pts_img[:, 1]), 0, S - 1).astype(np.int32)
                    d_px[pts_depth_ok] = result.depths[fi][yi, xi]
                depth_map_valid = pts_depth_ok & (d_px > 1e-3) & np.isfinite(d_px)

            K_fi = result.intrinsics[fi]
            proj_x = cam_coords[:, 0] / (depth_z + 1e-8) * K_fi[0, 0] + K_fi[0, 2]
            proj_y = cam_coords[:, 1] / (depth_z + 1e-8) * K_fi[1, 1] + K_fi[1, 2]
            sx, sy = S / cw, S / ch
            with np.errstate(over='ignore'):
                reproj_err = np.sqrt(
                    (proj_x - pts[:, 0] * sx) ** 2 +
                    (proj_y - pts[:, 1] * sy) ** 2
                )
            reproj_valid = (reproj_err < 3.0) | ~depth_valid

            valid_src = np.where(
                (valids[fi] > 0.5)
                & (visibs[fi] > 0.5)
                & in_bounds
                & world_finite
                & depth_valid
                & depth_map_valid
                & reproj_valid
            )[0]
            valid_by_frame.append(valid_src)

            if len(valid_src) == 0 or self.boundary_ratio <= 0:
                boundary_by_frame.append(np.empty(0, dtype=np.int64))
                continue

            # boundary mask は img_size 解像度、trajs は crop 解像度
            # → trajs を img_size へスケール
            sx = S / cw
            sy = S / ch
            xy_img = np.round(pts[valid_src] * np.array([sx, sy])).astype(np.int32)
            xy_img[:, 0] = np.clip(xy_img[:, 0], 0, S - 1)
            xy_img[:, 1] = np.clip(xy_img[:, 1], 0, S - 1)
            on_boundary = boundary[fi][xy_img[:, 1], xy_img[:, 0]]
            boundary_by_frame.append(valid_src[on_boundary])

        valid_frames = [fi for fi, v in enumerate(valid_by_frame) if len(v) > 0]
        if not valid_frames:
            raise RuntimeError(
                f"[{result.dataset_name}/{result.sequence_name}] "
                "No valid source points after transform."
            )

        coords   = torch.zeros(self.num_queries, 2)
        t_src_t  = torch.zeros(self.num_queries, dtype=torch.long)
        t_tgt_t  = torch.zeros(self.num_queries, dtype=torch.long)
        t_cam_t  = torch.zeros(self.num_queries, dtype=torch.long)

        # ---- 向量化采样（替换 num_queries 次 Python 循环） ----
        N = self.num_queries
        valid_frames_arr = np.array(valid_frames)

        # 1. 批量随机帧采样
        # PAPER-ALIGNED: t_src/t_tgt/t_cam sampled independently at random.
        # Enforce t_tgt=t_cam with prob t_tgt_eq_t_cam_ratio (paper: 0.4).
        # Previously forced t_tgt=t_src=t_cam for 40% of queries, which
        # collapsed tracking queries into depth-only self-reprojection.
        src_frames = np_rng.choice(valid_frames_arr, size=N)
        tgt_frames = np_rng.integers(0, T, size=N)
        cam_frames = np_rng.integers(0, T, size=N)
        force_eq   = np_rng.random(size=N) < self.t_tgt_eq_t_cam_ratio
        cam_frames = np.where(force_eq, tgt_frames, cam_frames)
        use_bnd    = np_rng.random(size=N) < self.boundary_ratio

        # 2. 按 source frame 分组选点（~S 次迭代而非 N 次）
        pt_indices = np.zeros(N, dtype=np.int64)
        for sf in np.unique(src_frames):
            mask   = src_frames == sf
            q_idxs = np.where(mask)[0]
            bnd_c  = boundary_by_frame[sf]
            val_c  = valid_by_frame[sf]
            wants  = use_bnd[q_idxs] & (len(bnd_c) > 0)
            if wants.any():
                pt_indices[q_idxs[wants]]  = np_rng.choice(bnd_c, size=int(wants.sum()))
            pt_indices[q_idxs[~wants]] = np_rng.choice(val_c, size=int((~wants).sum()))

        # 3. 批量 fancy indexing
        src_xy = trajs_2d[src_frames, pt_indices]   # [N,2]
        tgt_xy = trajs_2d[tgt_frames, pt_indices]   # [N,2]
        src_w3 = trajs_3d[src_frames, pt_indices]   # [N,3]
        tgt_w3 = trajs_3d[tgt_frames, pt_indices]   # [N,3]

        # 4. einsum 批量相机变换
        ones     = np.ones((N, 1), dtype=np.float32)
        src_w3_h = np.concatenate([src_w3.astype(np.float32), ones], axis=-1)  # [N,4]
        tgt_w3_h = np.concatenate([tgt_w3.astype(np.float32), ones], axis=-1)  # [N,4]
        E_batch  = extrinsics[cam_frames]                                        # [N,4,4]
        src_cam  = np.einsum('nij,nj->ni', E_batch, src_w3_h)[:, :3]           # [N,3]
        tgt_cam  = np.einsum('nij,nj->ni', E_batch, tgt_w3_h)[:, :3]           # [N,3]

        # 5. 向量化 validity / boundary 查表
        tgt_defined  = valids[tgt_frames, pt_indices] > 0.5
        tgt_inbounds = (
            (tgt_xy[:, 0] >= 0) & (tgt_xy[:, 0] < cw) &
            (tgt_xy[:, 1] >= 0) & (tgt_xy[:, 1] < ch)
        )
        vis_flag     = tgt_defined & (visibs[tgt_frames, pt_indices] > 0.5) & tgt_inbounds

        # 仅确保点在相机前方且深度量级合理。
        # 上限按数据集查表（见 _DATASET_Z_MAX）：
        #   pointodyssey/kubric/dynamic_replica: 20 m（离群点 Z>20 不参与 loss）
        #   scannet: 5 m（实测最大 2.18 m）
        #   co3dv2: 50（任意坐标单位，14% 有效点 Z>20）
        z_max = self._get_z_max(result.dataset_name)
        src_depth_valid = (src_cam[:, 2] > 1e-3) & (src_cam[:, 2] < z_max) & np.isfinite(src_cam[:, 2])
        tgt_depth_valid = (tgt_cam[:, 2] > 1e-3) & (tgt_cam[:, 2] < z_max) & np.isfinite(tgt_cam[:, 2])
        has_valid_3d = (tgt_defined & np.isfinite(src_cam).all(-1) & np.isfinite(tgt_cam).all(-1)
                       & src_depth_valid & tgt_depth_valid)

        sx, sy  = S / cw, S / ch
        src_ix  = np.clip(np.round(src_xy[:, 0] * sx).astype(np.int32), 0, S - 1)
        src_iy  = np.clip(np.round(src_xy[:, 1] * sy).astype(np.int32), 0, S - 1)

        bnd_stack  = np.stack(boundary,        axis=0)   # [T,S,S]
        dbnd_stack = np.stack(depth_boundary,  axis=0)   # [T,S,S]
        mbnd_stack = np.stack(motion_boundary, axis=0)   # [T,S,S]
        src_is_bnd  = bnd_stack [src_frames, src_iy, src_ix]
        src_is_dbnd = dbnd_stack[src_frames, src_iy, src_ix]
        src_is_mbnd = mbnd_stack[src_frames, src_iy, src_ix]

        # 6. 一次性写入输出 tensor
        norm    = np.array([max(cw - 1, 1), max(ch - 1, 1)], dtype=np.float32)
        coords  = torch.from_numpy((src_xy / norm).astype(np.float32))
        t_src_t = torch.from_numpy(src_frames.astype(np.int64))
        t_tgt_t = torch.from_numpy(tgt_frames.astype(np.int64))
        t_cam_t = torch.from_numpy(cam_frames.astype(np.int64))

        targets["pos_2d"][:]                     = torch.from_numpy((tgt_xy / norm).astype(np.float32))
        targets["pos_3d"][:]                     = torch.from_numpy(tgt_cam.astype(np.float32))
        targets["visibility"][:]                 = torch.from_numpy(vis_flag.astype(np.float32))
        targets["displacement"][:]               = torch.from_numpy((tgt_cam - src_cam).astype(np.float32))
        targets["mask_3d"][:]                    = torch.from_numpy(has_valid_3d)
        targets["mask_2d"][:]                    = torch.from_numpy(vis_flag)
        targets["mask_vis"][:]                   = torch.from_numpy(tgt_defined)
        targets["mask_disp"][:]                  = torch.from_numpy(has_valid_3d)
        targets["source_is_boundary"][:]         = torch.from_numpy(src_is_bnd)
        targets["source_is_depth_boundary"][:]   = torch.from_numpy(src_is_dbnd)
        targets["source_is_motion_boundary"][:] = torch.from_numpy(src_is_mbnd)
        targets["point_indices"][:]              = torch.from_numpy(pt_indices)
        # is_static_reprojection: marks queries with no temporal component
        # (t_tgt == t_src). In the paper-aligned sampling this arises naturally
        # when t_tgt happens to equal t_src, not from a forced 40% split.
        targets["is_static_reprojection"][:]     = torch.from_numpy(tgt_frames == src_frames)

        # ---- normal（向量化批量查表） ----
        # 只有 t_cam == t_tgt 时，normal 才与目标 3D 监督处于同一相机坐标系。
        if result.normals is not None and result.normal_valids is not None:
            normals_arr = np.stack(result.normals, axis=0)             # [T,S,S,3]
            valid_nf    = np.array(result.normal_valids, dtype=bool)   # [T]
            cam_eq_tgt  = cam_frames == tgt_frames
            active      = vis_flag & valid_nf[tgt_frames] & cam_eq_tgt
            if active.any():
                qi  = np.where(active)[0]
                uv  = targets["pos_2d"][qi].numpy()                    # [M,2]
                nx  = np.clip(np.round(uv[:, 0] * (S - 1)).astype(np.int32), 0, S - 1)
                ny  = np.clip(np.round(uv[:, 1] * (S - 1)).astype(np.int32), 0, S - 1)
                nv  = normals_arr[tgt_frames[qi], ny, nx]              # [M,3]
                nv_t = torch.from_numpy(nv.astype(np.float32))
                valid_normal = torch.isfinite(nv_t).all(-1) & (torch.linalg.norm(nv_t, dim=-1) > 1e-6)
                targets["normal"][qi] = torch.where(valid_normal.unsqueeze(-1), nv_t, torch.zeros_like(nv_t))
                targets["mask_normal"][qi] = valid_normal

        return coords, t_src_t, t_tgt_t, t_cam_t, targets

    # ------------------------------------------------------------------
    # has_tracks=False 路径（ScanNet / Co3D / BlendedMVS 等）
    # 只有 depth + pose → 单帧静态假设下产出 pos_3d
    # mask_2d / mask_disp / mask_vis 全部 False
    # ------------------------------------------------------------------

    def _sample_no_tracks(
        self,
        result: TransformResult,
        depth_boundary: list[np.ndarray],
        boundary: list[np.ndarray],
        py_rng: _random_module.Random,
        np_rng: np.random.Generator,
    ) -> tuple:
        T  = len(result.images)
        S  = result.img_size
        # Keep no-tracks coordinates in the same crop-normalized convention as
        # the tracks path instead of resized-image coordinates.
        cw = result.crop.crop_w if result.crop is not None else S
        ch = result.crop.crop_h if result.crop is not None else S
        K  = result.intrinsics   # [T,3,3]
        E  = result.extrinsics   # [T,4,4]
        z_max = self._get_z_max(result.dataset_name)

        targets = _build_empty_targets(self.num_queries)

        has_depth = result.depths is not None

        # 预计算每帧有效（有深度）的像素池
        valid_pixels_by_frame: list[np.ndarray] = []
        boundary_pixels_by_frame: list[np.ndarray] = []

        # depth / boundary live on the resized SxS plane; coords live on the
        # crop plane. We convert between them explicitly to keep supervision
        # semantics aligned with the tracks path.
        scale_x = cw / max(S, 1)
        scale_y = ch / max(S, 1)

        for fi in range(T):
            if not has_depth:
                # 没有 depth：在 crop 平面随机采像素，mask_3d 全 False
                ys, xs = np.mgrid[0:ch, 0:cw]
                all_px = np.stack([xs.ravel(), ys.ravel()], axis=1)   # [crop_h*crop_w,2]
                valid_pixels_by_frame.append(all_px)
                boundary_pixels_by_frame.append(np.empty((0, 2), dtype=np.int32))
                continue

            depth = result.depths[fi]       # [S,S] resized plane
            valid_mask = np.isfinite(depth) & (depth > 1e-3) & (depth < z_max)
            ys_r, xs_r = np.where(valid_mask)
            if len(ys_r) == 0:
                ys_c, xs_c = np.mgrid[0:ch, 0:cw]
                ys_c, xs_c = ys_c.ravel(), xs_c.ravel()
            else:
                xs_c = np.clip(np.round(xs_r * scale_x).astype(np.int32), 0, cw - 1)
                ys_c = np.clip(np.round(ys_r * scale_y).astype(np.int32), 0, ch - 1)
            px = np.stack([xs_c, ys_c], axis=1).astype(np.int32)       # [M,2] crop coords
            valid_pixels_by_frame.append(px)

            if self.boundary_ratio > 0:
                bpx_mask = depth_boundary[fi][ys_r, xs_r]
                boundary_pixels_by_frame.append(px[bpx_mask])
            else:
                boundary_pixels_by_frame.append(np.empty((0, 2), dtype=np.int32))

        valid_frames = list(range(T))   # 所有帧都可采

        coords   = torch.zeros(self.num_queries, 2)
        t_src_t  = torch.zeros(self.num_queries, dtype=torch.long)
        t_tgt_t  = torch.zeros(self.num_queries, dtype=torch.long)
        t_cam_t  = torch.zeros(self.num_queries, dtype=torch.long)

        for qi in range(self.num_queries):
            src_fi = py_rng.choice(valid_frames)

            # PAPER-ALIGNED: t_src/t_tgt/t_cam sampled independently at random.
            # Enforce t_tgt=t_cam with prob t_tgt_eq_t_cam_ratio (paper: 0.4).
            # In static scenes the world point doesn't move, so we can compute
            # both pos_3d (in t_cam frame) and pos_2d (in t_tgt frame) for
            # arbitrary t_tgt — no need to force t_tgt=t_src.
            tgt_fi = py_rng.randint(0, T - 1)
            cam_fi = py_rng.randint(0, T - 1)
            if py_rng.random() < self.t_tgt_eq_t_cam_ratio:
                cam_fi = tgt_fi

            vpx = valid_pixels_by_frame[src_fi]
            bpx = boundary_pixels_by_frame[src_fi]

            if len(bpx) > 0 and py_rng.random() < self.boundary_ratio:
                chosen = bpx[int(np_rng.integers(0, len(bpx)))]
            else:
                chosen = vpx[int(np_rng.integers(0, len(vpx)))]

            px_x, px_y = int(chosen[0]), int(chosen[1])

            # 坐标归一化到 crop 空间 [0,1]，与 tracks 路径一致。
            coords[qi, 0] = px_x / max(cw - 1, 1)
            coords[qi, 1] = px_y / max(ch - 1, 1)
            t_src_t[qi]   = src_fi
            t_tgt_t[qi]   = tgt_fi
            t_cam_t[qi]   = cam_fi

            # boundary/depth_boundary live on resized plane; remap crop coord.
            px_x_r = int(np.clip(round(px_x / scale_x), 0, S - 1))
            px_y_r = int(np.clip(round(px_y / scale_y), 0, S - 1))
            targets["source_is_boundary"][qi]        = bool(boundary[src_fi][px_y_r, px_x_r])
            targets["source_is_depth_boundary"][qi]  = bool(depth_boundary[src_fi][px_y_r, px_x_r])
            targets["point_indices"][qi]             = px_y * cw + px_x
            # is_static_reprojection: marks queries with no temporal component
            # (t_tgt == t_src), used for loss weighting analysis.
            targets["is_static_reprojection"][qi]    = (tgt_fi == src_fi)

            # pos_3d：从 src 帧 depth 反投影 → world → cam_fi 相机坐标系
            # pos_2d：投影 p_world 到 tgt_fi 图像平面（静态场景下 p_world 不变）
            if has_depth:
                depth_val = float(result.depths[src_fi][px_y_r, px_x_r])
                if depth_val > 1e-3 and depth_val < z_max and np.isfinite(depth_val):
                    K_src = K[src_fi]
                    fx, fy = K_src[0, 0], K_src[1, 1]
                    cx, cy = K_src[0, 2], K_src[1, 2]
                    # src 相机坐标
                    x_c = (px_x_r - cx) * depth_val / fx
                    y_c = (px_y_r - cy) * depth_val / fy
                    z_c = depth_val
                    p_src_cam = np.array([x_c, y_c, z_c, 1.0], dtype=np.float32)

                    # src cam → world
                    E_src = E[src_fi]
                    p_world = np.linalg.inv(E_src) @ p_src_cam

                    # world → cam_fi cam (for pos_3d)
                    E_cam = E[cam_fi]
                    p_cam = (E_cam @ p_world)[:3].astype(np.float32)

                    if np.isfinite(p_cam).all() and p_cam[2] > 1e-3 and p_cam[2] < z_max:
                        targets["pos_3d"][qi]  = torch.from_numpy(p_cam)
                        targets["mask_3d"][qi] = True

                        # pos_2d: only when t_tgt=t_cam (paper design).
                        # Project p_world to tgt_fi=cam_fi image plane.
                        if tgt_fi == cam_fi:
                            K_tgt = K[tgt_fi]
                            u = float(K_tgt[0,0] * p_cam[0] / p_cam[2] + K_tgt[0,2])
                            v = float(K_tgt[1,1] * p_cam[1] / p_cam[2] + K_tgt[1,2])
                            u_c = u * scale_x
                            v_c = v * scale_y
                            if 0 <= u_c < cw and 0 <= v_c < ch:
                                targets["pos_2d"][qi, 0] = u_c / max(cw - 1, 1)
                                targets["pos_2d"][qi, 1] = v_c / max(ch - 1, 1)
                                targets["mask_2d"][qi]   = True

        # mask_2d set above when t_tgt=t_cam and pos_2d in bounds; others remain False
        return coords, t_src_t, t_tgt_t, t_cam_t, targets

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _extract_patches(
        self,
        video_hwc: torch.Tensor,    # [T,H,W,3]
        patch_queries: torch.Tensor, # [Q,3] (u,v,t_src)  u/v in [0,1]
    ) -> torch.Tensor:
        frames_btchw = video_hwc.permute(0, 3, 1, 2).unsqueeze(0).float()  # [1,T,3,H,W]
        coords_uv    = patch_queries[:, :2].unsqueeze(0)             # [1,Q,2]
        t_src_idx    = patch_queries[:, 2].round().long().unsqueeze(0)  # [1,Q]
        patches = extract_local_patches(
            frames_btchw, coords_uv, t_src_idx, patch_size=self.patch_size
        )
        return patches.squeeze(0)   # [Q,3,P,P]
