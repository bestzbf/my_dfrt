"""
Compute 2D/3D point tracks from depth maps and camera poses (strategy A).

Strategy A (static scene):
  - Sample N source points from the reference frame (frame 0) depth map.
  - Unproject them to 3D world coordinates.
  - Project into every other frame using that frame's pose.
  - Mark a point valid if it projects inside the image AND the depth
    at that pixel is consistent with the projected depth.

This is exact for static scenes; dynamic objects will accumulate invalid
marks, which is acceptable because the query builder already handles
sparse valid masks.

Usage:
    from computer.depth_to_tracks import compute_tracks

    result = compute_tracks(depths, intrinsics, extrinsics)
    # result['trajs_2d']       [T,N,2]
    # result['trajs_3d_world'] [T,N,3]
    # result['valids']         [T,N]
    # result['visibs']         [T,N]
"""

from __future__ import annotations

import cv2
import numpy as np


TRACK_SEMANTICS_VERSION = 2


# --------------------------------------------------------------------------- #
# Low-level geometry helpers
# --------------------------------------------------------------------------- #

def _unproject(
    uv: np.ndarray,      # [N,2]  (x=col, y=row) float32
    depth: np.ndarray,   # [N]    float32
    K: np.ndarray,       # [3,3]
) -> np.ndarray:         # [N,3]  camera-space XYZ
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (uv[:, 0] - cx) * depth / fx
    Y = (uv[:, 1] - cy) * depth / fy
    Z = depth
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def _project(
    pts_cam: np.ndarray,   # [N,3]  camera-space XYZ
    K: np.ndarray,         # [3,3]
) -> tuple[np.ndarray, np.ndarray]:   # uv [N,2], z [N]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = pts_cam[:, 2]
    # avoid division by zero for points behind camera
    safe_z = np.where(z > 0, z, 1.0)
    u = pts_cam[:, 0] / safe_z * fx + cx
    v = pts_cam[:, 1] / safe_z * fy + cy
    return np.stack([u, v], axis=-1).astype(np.float32), z.astype(np.float32)


def _c2w_from_w2c(E: np.ndarray) -> np.ndarray:
    """Invert a w2c [4,4] matrix analytically (numerically stable)."""
    R = E[:3, :3]   # [3,3]
    t = E[:3, 3]    # [3]
    R_inv = R.T
    t_inv = -R.T @ t
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_inv
    c2w[:3, 3]  = t_inv
    return c2w


def _depth_boundary_mask(depth: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Bool mask [H,W]: pixels at depth discontinuities."""
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    valid = (d > 0) & np.isfinite(d)
    if not valid.any():
        return np.zeros_like(d, dtype=bool)
    vals = mag[valid]
    if vals.size == 0 or float(vals.max()) <= 0:
        return np.zeros_like(d, dtype=bool)
    thr = max(float(np.percentile(vals, percentile)), 1e-6)
    return valid & (mag >= thr)


def _project_world_points_to_frames(
    points_world: np.ndarray,
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project static world points into all frames and separate defined/visible masks.

    Semantics:
      - ``valids``: the target 3D / visibility label is *defined* in this frame.
        This requires the point to project in-bounds, lie in front of the camera,
        and land on a pixel with a valid depth sample.
      - ``visibs``: the point is additionally *visible* under a depth-consistency
        check against the target depth map. Occluded points remain valid but are
        marked invisible.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    depths = [np.asarray(d, dtype=np.float32) for d in depths]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    T = len(depths)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"Expected points_world to have shape [N,3], got {tuple(points_world.shape)}"
        )
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[None], (T, 3, 3)).copy()

    H, W = depths[0].shape
    N = points_world.shape[0]
    trajs_2d = np.zeros((T, N, 2), dtype=np.float32)
    valids = np.zeros((T, N), dtype=bool)
    visibs = np.zeros((T, N), dtype=bool)

    ones = np.ones((N, 1), dtype=np.float32)
    points_world_h = np.concatenate([points_world, ones], axis=-1)

    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]
        depth_t = depths[t]

        points_cam_t = (E_t @ points_world_h.T).T[:, :3]
        uv_t, z_t = _project(points_cam_t, K_t)
        trajs_2d[t] = uv_t

        uv_finite = np.isfinite(uv_t).all(axis=-1)
        in_front = (z_t > 0) & np.isfinite(z_t) & (z_t < depth_max)
        in_bounds = (
            uv_finite
            & (uv_t[:, 0] >= 0)
            & (uv_t[:, 0] < W)
            & (uv_t[:, 1] >= 0)
            & (uv_t[:, 1] < H)
        )

        sampled_depth = np.zeros((N,), dtype=np.float32)
        sampled_depth_valid = np.zeros((N,), dtype=bool)
        sample_mask = in_bounds & in_front
        if sample_mask.any():
            uv_in = uv_t[sample_mask]
            px = np.clip(np.round(uv_in[:, 0]).astype(np.int32), 0, W - 1)
            py = np.clip(np.round(uv_in[:, 1]).astype(np.int32), 0, H - 1)
            sampled = depth_t[py, px]
            sampled_depth[sample_mask] = sampled
            sampled_depth_valid[sample_mask] = (
                (sampled > 0)
                & np.isfinite(sampled)
                & (sampled < depth_max)
            )

        defined_t = sample_mask & sampled_depth_valid
        depth_rel_err = np.full((N,), np.inf, dtype=np.float32)
        if defined_t.any():
            depth_rel_err[defined_t] = (
                np.abs(sampled_depth[defined_t] - z_t[defined_t])
                / np.maximum(z_t[defined_t], 1e-6)
            )
        visible_t = defined_t & (depth_rel_err < depth_consistency_thresh)

        valids[t] = defined_t
        visibs[t] = visible_t

    return trajs_2d, valids, visibs


def recompute_track_projection_masks(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    trajs_3d_world: np.ndarray,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
) -> dict[str, np.ndarray]:
    """Refresh projected 2D tracks and defined/visible masks from cached world points."""
    trajs_3d_world = np.asarray(trajs_3d_world, dtype=np.float32)
    if trajs_3d_world.ndim != 3 or trajs_3d_world.shape[-1] != 3:
        raise ValueError(
            f"Expected trajs_3d_world to have shape [T,N,3], got {tuple(trajs_3d_world.shape)}"
        )
    if trajs_3d_world.shape[0] == 0:
        return {
            "trajs_2d": np.zeros((0, 0, 2), dtype=np.float32),
            "valids": np.zeros((0, 0), dtype=bool),
            "visibs": np.zeros((0, 0), dtype=bool),
        }

    points_world = trajs_3d_world[0]
    trajs_2d, valids, visibs = _project_world_points_to_frames(
        points_world=points_world,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_consistency_thresh=depth_consistency_thresh,
        depth_max=depth_max,
    )
    return {
        "trajs_2d": trajs_2d,
        "valids": valids,
        "visibs": visibs,
    }


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #

def compute_tracks(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    num_points: int = 8000,
    boundary_ratio: float = 0.3,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
    rng_seed: int = 42,
) -> dict:
    """
    Compute 2D/3D point tracks for a static scene clip (strategy A).

    Args:
        depths:      List of T depth maps [H,W] float32. Invalid = 0/nan/inf.
        intrinsics:  [T,3,3] float32 (or [3,3] broadcast).
        extrinsics:  [T,4,4] float32 world-to-camera.
        num_points:  Number of source points N to sample.
        boundary_ratio:
            Fraction of N sampled near depth discontinuities (30%).
        depth_consistency_thresh:
            Relative depth error threshold for validity check (5%).
        rng_seed:    RNG seed for reproducibility.

    Returns:
        dict with keys:
            trajs_2d       [T,N,2]  float32  pixel coords (x=col, y=row)
            trajs_3d_world [T,N,3]  float32  world coords (same every frame)
            valids         [T,N]    bool     target defined (in-bounds, in-front,
                                             depth sample exists)
            visibs         [T,N]    bool     target additionally visible under
                                             depth-consistency check
            ref_frame      int
            num_points     int      actual N (may be < requested if depth sparse)
            track_semantics_version int     cache semantics version
    """
    rng = np.random.default_rng(rng_seed)

    depths = [np.asarray(d, dtype=np.float32) for d in depths]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    T = len(depths)
    H, W = depths[0].shape

    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[None], (T, 3, 3)).copy()

    # ------------------------------------------------------------------ #
    # 1. Select reference frame (most valid depth pixels)                  #
    # ------------------------------------------------------------------ #
    valid_counts = [int(((d > 0) & np.isfinite(d) & (d < depth_max)).sum()) for d in depths]
    ref = int(np.argmax(valid_counts))

    depth_ref = depths[ref]
    K_ref = intrinsics[ref]
    E_ref = extrinsics[ref]

    valid_ref = (depth_ref > 0) & np.isfinite(depth_ref) & (depth_ref < depth_max)

    # ------------------------------------------------------------------ #
    # 2. Sample N source points from reference frame                       #
    # ------------------------------------------------------------------ #
    n_boundary = int(num_points * boundary_ratio)
    n_uniform  = num_points - n_boundary

    # boundary candidates
    bnd_mask = _depth_boundary_mask(depth_ref) & valid_ref
    bnd_ys, bnd_xs = np.where(bnd_mask)

    # uniform candidates
    uni_ys, uni_xs = np.where(valid_ref)

    # sample boundary points
    if len(bnd_ys) >= n_boundary:
        idx = rng.choice(len(bnd_ys), n_boundary, replace=False)
        sel_bnd_y = bnd_ys[idx]
        sel_bnd_x = bnd_xs[idx]
    else:
        sel_bnd_y = bnd_ys
        sel_bnd_x = bnd_xs

    # sample uniform points (avoid already-selected boundary points if possible)
    n_uni_needed = num_points - len(sel_bnd_y)
    if len(uni_ys) >= n_uni_needed:
        idx = rng.choice(len(uni_ys), n_uni_needed, replace=False)
        sel_uni_y = uni_ys[idx]
        sel_uni_x = uni_xs[idx]
    else:
        # not enough valid pixels — use all and allow replacement for remainder
        sel_uni_y = uni_ys
        sel_uni_x = uni_xs
        short = n_uni_needed - len(uni_ys)
        if short > 0 and len(uni_ys) > 0:
            idx = rng.choice(len(uni_ys), short, replace=True)
            sel_uni_y = np.concatenate([sel_uni_y, uni_ys[idx]])
            sel_uni_x = np.concatenate([sel_uni_x, uni_xs[idx]])

    src_y = np.concatenate([sel_bnd_y, sel_uni_y]).astype(np.int32)
    src_x = np.concatenate([sel_bnd_x, sel_uni_x]).astype(np.int32)
    N = len(src_y)   # actual number of points

    src_uv = np.stack([src_x, src_y], axis=-1).astype(np.float32)   # [N,2]
    src_d  = depth_ref[src_y, src_x].astype(np.float32)              # [N]

    # ------------------------------------------------------------------ #
    # 3. Unproject to world coordinates                                    #
    # ------------------------------------------------------------------ #
    P_cam_ref = _unproject(src_uv, src_d, K_ref)   # [N,3]

    c2w_ref = _c2w_from_w2c(E_ref)                  # [4,4]
    ones    = np.ones((N, 1), dtype=np.float32)
    P_hom   = np.concatenate([P_cam_ref, ones], axis=-1)  # [N,4]
    P_world = (c2w_ref @ P_hom.T).T[:, :3]           # [N,3]

    # ------------------------------------------------------------------ #
    # 4. Project into every frame                                          #
    # ------------------------------------------------------------------ #
    trajs_2d, valids, visibs = _project_world_points_to_frames(
        points_world=P_world,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_consistency_thresh=depth_consistency_thresh,
        depth_max=depth_max,
    )
    trajs_3d_world = np.broadcast_to(P_world[None], (T, N, 3)).copy()

    return {
        "trajs_2d":       trajs_2d,
        "trajs_3d_world": trajs_3d_world,
        "valids":         valids,
        "visibs":         visibs,
        "ref_frame":      ref,
        "num_points":     N,
        "track_semantics_version": TRACK_SEMANTICS_VERSION,
    }
