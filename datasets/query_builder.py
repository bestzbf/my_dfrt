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
    np.add.at(motion_map, (yy, xx), mag[defined])
    np.add.at(support_map, (yy, xx), 1.0)

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
    }


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
    """

    def __init__(
        self,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        patch_size: int = 9,
        use_motion_boundaries: bool = True,
        precompute_patches: bool = True,
    ) -> None:
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.patch_size = patch_size
        self.use_motion_boundaries = use_motion_boundaries
        self.precompute_patches = precompute_patches

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
            np_rng = np.random.default_rng()

        has_tracks = bool(result.metadata.get("has_tracks", False))
        T = len(result.images)
        cw, ch = result.crop.crop_w, result.crop.crop_h

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
                # Fallback: all tracks invalid after transform, use depth-based sampling
                coords, t_src, t_tgt, t_cam, targets = self._sample_no_tracks(
                    result, depth_boundary, boundary, py_rng, np_rng
                )
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
            video_hwc = video.permute(0, 2, 3, 1)   # [T,H,W,3]
            local_patches = self._extract_patches(video_hwc, patch_queries)

        return QuerySample(
            video=video,
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
            dataset_name=result.dataset_name,
            sequence_name=result.sequence_name,
            metadata=result.metadata,
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
            valid_src = np.where(
                (valids[fi] > 0.5) & (visibs[fi] > 0.5) & in_bounds & world_finite
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

        for qi in range(self.num_queries):
            src_fi = py_rng.choice(valid_frames)

            # t_tgt / t_cam 采样（论文：40% t_tgt=t_cam）
            tgt_fi = py_rng.randint(0, T - 1)
            if py_rng.random() < self.t_tgt_eq_t_cam_ratio:
                cam_fi = tgt_fi
            else:
                cam_fi = py_rng.randint(0, T - 1)

            # boundary oversampling（30%）
            bcands = boundary_by_frame[src_fi]
            if len(bcands) > 0 and py_rng.random() < self.boundary_ratio:
                pt_idx = int(np_rng.choice(bcands))
            else:
                pt_idx = int(np_rng.choice(valid_by_frame[src_fi]))

            src_xy  = trajs_2d[src_fi, pt_idx]   # crop coords
            tgt_xy  = trajs_2d[tgt_fi, pt_idx]   # crop coords

            # source 在 img_size mask 里的位置（boundary 标注用）
            sx = S / cw;  sy = S / ch
            src_ix = int(np.clip(round(src_xy[0] * sx), 0, S - 1))
            src_iy = int(np.clip(round(src_xy[1] * sy), 0, S - 1))

            tgt_defined  = bool(valids[tgt_fi, pt_idx] > 0.5)
            tgt_inbounds = (0 <= tgt_xy[0] < cw) and (0 <= tgt_xy[1] < ch)
            vis_flag = bool(
                tgt_defined and visibs[tgt_fi, pt_idx] > 0.5 and tgt_inbounds
            )

            src_w3  = trajs_3d[src_fi, pt_idx]    # [3]
            tgt_w3  = trajs_3d[tgt_fi, pt_idx]    # [3]
            E_cam   = extrinsics[cam_fi]            # [4,4]

            src_cam_h = E_cam @ np.append(src_w3, 1.0)
            tgt_cam_h = E_cam @ np.append(tgt_w3, 1.0)
            src_cam   = src_cam_h[:3].astype(np.float32)
            tgt_cam   = tgt_cam_h[:3].astype(np.float32)

            has_valid_3d = bool(
                tgt_defined
                and np.isfinite(src_cam).all()
                and np.isfinite(tgt_cam).all()
            )

            # coords 归一化到 [0,1]（crop 空间）
            coords[qi, 0] = float(src_xy[0] / max(cw - 1, 1))
            coords[qi, 1] = float(src_xy[1] / max(ch - 1, 1))
            t_src_t[qi]   = src_fi
            t_tgt_t[qi]   = tgt_fi
            t_cam_t[qi]   = cam_fi

            targets["pos_2d"][qi, 0] = float(tgt_xy[0] / max(cw - 1, 1))
            targets["pos_2d"][qi, 1] = float(tgt_xy[1] / max(ch - 1, 1))
            targets["pos_3d"][qi]    = torch.from_numpy(tgt_cam)
            targets["visibility"][qi]= float(vis_flag)
            targets["displacement"][qi] = torch.from_numpy(tgt_cam - src_cam)

            targets["mask_3d"][qi]   = has_valid_3d
            targets["mask_2d"][qi]   = vis_flag
            targets["mask_vis"][qi]  = tgt_defined
            targets["mask_disp"][qi] = has_valid_3d

            targets["source_is_boundary"][qi]       = bool(boundary[src_fi][src_iy, src_ix])
            targets["source_is_depth_boundary"][qi] = bool(depth_boundary[src_fi][src_iy, src_ix])
            targets["source_is_motion_boundary"][qi]= bool(motion_boundary[src_fi][src_iy, src_ix])
            targets["point_indices"][qi]             = pt_idx

        # ---- normal（从 resized normal map 读像素） ----
        if result.normals is not None and result.normal_valids is not None:
            normals_np = [n for n in result.normals]    # list of [S,S,3]
            for qi in range(self.num_queries):
                if not bool(targets["mask_2d"][qi]):
                    continue
                tgt_fi = int(t_tgt_t[qi].item())
                if not result.normal_valids[tgt_fi]:
                    continue
                tgt_uv = targets["pos_2d"][qi]          # [0,1]
                nx = int(np.clip(round(float(tgt_uv[0]) * (S - 1)), 0, S - 1))
                ny = int(np.clip(round(float(tgt_uv[1]) * (S - 1)), 0, S - 1))
                nv = normals_np[tgt_fi][ny, nx]
                targets["normal"][qi] = torch.from_numpy(nv.astype(np.float32))
                targets["mask_normal"][qi] = bool(
                    torch.isfinite(targets["normal"][qi]).all()
                )

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
        K  = result.intrinsics   # [T,3,3]
        E  = result.extrinsics   # [T,4,4]

        targets = _build_empty_targets(self.num_queries)

        has_depth = result.depths is not None

        # 预计算每帧有效（有深度）的像素池
        valid_pixels_by_frame: list[np.ndarray] = []
        boundary_pixels_by_frame: list[np.ndarray] = []

        for fi in range(T):
            if not has_depth:
                # 没有 depth：随机采像素，mask_3d 全 False
                ys, xs = np.mgrid[0:S, 0:S]
                all_px = np.stack([xs.ravel(), ys.ravel()], axis=1)   # [S*S,2]
                valid_pixels_by_frame.append(all_px)
                boundary_pixels_by_frame.append(np.empty((0, 2), dtype=np.int32))
                continue

            depth = result.depths[fi]       # [S,S]
            valid_mask = np.isfinite(depth) & (depth > 0)
            ys, xs = np.where(valid_mask)
            if len(ys) == 0:
                # fallback：全像素
                ys, xs = np.mgrid[0:S, 0:S]
                ys, xs = ys.ravel(), xs.ravel()
            px = np.stack([xs, ys], axis=1).astype(np.int32)           # [M,2]
            valid_pixels_by_frame.append(px)

            if self.boundary_ratio > 0:
                bpx_mask = depth_boundary[fi][ys, xs]
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

            tgt_fi = py_rng.randint(0, T - 1)
            if py_rng.random() < self.t_tgt_eq_t_cam_ratio:
                cam_fi = tgt_fi
            else:
                cam_fi = py_rng.randint(0, T - 1)

            vpx = valid_pixels_by_frame[src_fi]
            bpx = boundary_pixels_by_frame[src_fi]

            if len(bpx) > 0 and py_rng.random() < self.boundary_ratio:
                chosen = bpx[int(np_rng.integers(0, len(bpx)))]
            else:
                chosen = vpx[int(np_rng.integers(0, len(vpx)))]

            px_x, px_y = int(chosen[0]), int(chosen[1])

            # 坐标归一化到 [0,1]
            coords[qi, 0] = px_x / max(S - 1, 1)
            coords[qi, 1] = px_y / max(S - 1, 1)
            t_src_t[qi]   = src_fi
            t_tgt_t[qi]   = tgt_fi
            t_cam_t[qi]   = cam_fi

            # boundary 标注
            targets["source_is_boundary"][qi]        = bool(boundary[src_fi][px_y, px_x])
            targets["source_is_depth_boundary"][qi]  = bool(depth_boundary[src_fi][px_y, px_x])
            targets["point_indices"][qi]              = px_y * S + px_x

            # pos_3d：从 src 帧 depth 反投影到 cam_fi 相机坐标系
            if has_depth:
                depth_val = float(result.depths[src_fi][px_y, px_x])
                if depth_val > 0 and np.isfinite(depth_val):
                    K_src = K[src_fi]
                    fx, fy = K_src[0, 0], K_src[1, 1]
                    cx, cy = K_src[0, 2], K_src[1, 2]
                    # src 相机坐标
                    x_c = (px_x - cx) * depth_val / fx
                    y_c = (px_y - cy) * depth_val / fy
                    z_c = depth_val
                    p_src_cam = np.array([x_c, y_c, z_c, 1.0], dtype=np.float32)

                    # src cam → world → cam_fi cam
                    E_src = E[src_fi]                       # w2c src
                    E_cam = E[cam_fi]                       # w2c cam_fi
                    # world = inv(E_src) @ p_src_cam
                    p_world = np.linalg.inv(E_src) @ p_src_cam
                    p_cam   = (E_cam @ p_world)[:3].astype(np.float32)

                    if np.isfinite(p_cam).all():
                        targets["pos_3d"][qi]  = torch.from_numpy(p_cam)
                        targets["mask_3d"][qi] = True

        # mask_2d / mask_vis / mask_disp / mask_normal 全部保持 False
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
