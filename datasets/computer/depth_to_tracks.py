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
from collections.abc import Sequence


TRACK_SEMANTICS_VERSION = 3
SOURCE_DEPTH_SAMPLING_MODE_IDS = {
    "uniform": 0,
    "log_balanced": 1,
    "linear_balanced": 2,
}


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


def _select_ref_frames(
    valid_counts: np.ndarray,
    num_segments: int,
    strategy: str,
) -> list[int]:
    """Split the clip into segments and choose one reference frame per segment."""
    T = len(valid_counts)
    refs: list[int] = []
    num_segments = max(int(num_segments), 1)
    for seg in range(num_segments):
        start = seg * T // num_segments
        end = (seg + 1) * T // num_segments
        if start >= end:
            continue
        if strategy == "first":
            seg_valid = np.where(valid_counts[start:end] > 0)[0]
            ref = start + int(seg_valid[0]) if len(seg_valid) > 0 else start
        elif strategy == "max_valid":
            ref = start + int(np.argmax(valid_counts[start:end]))
        elif strategy == "middle":
            ref = (start + end) // 2
        else:
            raise ValueError(f"Unknown ref_frame_strategy: {strategy!r}")
        refs.append(int(ref))
    return refs


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


def _normalise_source_depth_sampling(mode: str) -> str:
    mode = str(mode or "uniform").strip().lower()
    aliases = {
        "balanced": "log_balanced",
        "log": "log_balanced",
        "linear": "linear_balanced",
    }
    mode = aliases.get(mode, mode)
    if mode not in SOURCE_DEPTH_SAMPLING_MODE_IDS:
        raise ValueError(
            f"Unknown source_depth_sampling={mode!r}; expected one of "
            f"{sorted(SOURCE_DEPTH_SAMPLING_MODE_IDS)}"
        )
    return mode


def _normalise_depth_bin_edges(edges: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if edges is None:
        return None
    arr = np.asarray(list(edges), dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    arr = np.unique(arr)
    if arr.size < 2:
        return None
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"source_depth_bin_edges must be strictly increasing, got {arr.tolist()}")
    return arr


def _normalise_depth_bin_weights(weights: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if weights is None:
        return None
    arr = np.asarray(list(weights), dtype=np.float32).reshape(-1)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    if arr.size == 0 or float(arr.sum()) <= 0.0:
        return None
    return arr


def _dynamic_depth_bin_edges(
    depth_values: np.ndarray,
    depth_max: float,
    source_depth_sampling: str,
    source_depth_num_bins: int,
) -> np.ndarray | None:
    vals = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 1e-6)]
    if vals.size == 0:
        return None
    num_bins = max(1, int(source_depth_num_bins))
    if num_bins <= 1:
        return None
    lo = float(vals.min())
    hi = min(float(depth_max), float(vals.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    if source_depth_sampling == "linear_balanced":
        return np.linspace(lo, hi, num_bins + 1, dtype=np.float32)
    lo = max(lo, 1e-3)
    if hi <= lo * 1.001:
        return None
    return np.geomspace(lo, hi, num_bins + 1).astype(np.float32)


def _sample_candidate_pixels(
    ys: np.ndarray,
    xs: np.ndarray,
    n: int,
    rng: np.random.Generator,
    replace_short: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or len(ys) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    if len(ys) >= n:
        idx = rng.choice(len(ys), n, replace=False)
        return ys[idx].astype(np.int32), xs[idx].astype(np.int32)
    if not replace_short:
        return ys.astype(np.int32), xs.astype(np.int32)

    short = n - len(ys)
    extra_idx = rng.choice(len(ys), short, replace=True) if short > 0 else np.empty(0, dtype=np.int64)
    out_y = np.concatenate([ys, ys[extra_idx]]) if short > 0 else ys
    out_x = np.concatenate([xs, xs[extra_idx]]) if short > 0 else xs
    return out_y.astype(np.int32), out_x.astype(np.int32)


def _sample_candidate_pixels_depth_balanced(
    depth: np.ndarray,
    mask: np.ndarray,
    n: int,
    rng: np.random.Generator,
    depth_max: float,
    source_depth_sampling: str,
    source_depth_bin_edges: np.ndarray | None,
    source_depth_bin_weights: np.ndarray | None,
    source_depth_num_bins: int,
    replace_short: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if n <= 0 or len(ys) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    depth_values = depth[ys, xs].astype(np.float32, copy=False)
    edges = source_depth_bin_edges
    if edges is None:
        edges = _dynamic_depth_bin_edges(
            depth_values,
            depth_max=depth_max,
            source_depth_sampling=source_depth_sampling,
            source_depth_num_bins=source_depth_num_bins,
        )
    if edges is None or len(edges) < 2:
        return _sample_candidate_pixels(ys, xs, n, rng, replace_short=replace_short)

    # Values outside the configured edge range are folded into the nearest bin.
    bin_ids = np.digitize(depth_values, edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    nonempty_bins = [int(b) for b in np.unique(bin_ids)]
    if not nonempty_bins:
        return _sample_candidate_pixels(ys, xs, n, rng, replace_short=replace_short)

    if source_depth_bin_weights is not None:
        if source_depth_bin_weights.size != len(edges) - 1:
            raise ValueError(
                "source_depth_bin_weights length must match number of depth bins "
                f"({len(edges) - 1}), got {source_depth_bin_weights.size}"
            )
        weights = source_depth_bin_weights[np.asarray(nonempty_bins, dtype=np.int64)].astype(np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(nonempty_bins), dtype=np.float64)
        raw_quota = weights / float(weights.sum()) * float(n)
        quotas = np.floor(raw_quota).astype(np.int64)
        remainder = int(n - quotas.sum())
        if remainder > 0:
            order = np.argsort(-(raw_quota - quotas))
            quotas[order[:remainder]] += 1
    else:
        quota_base = n // len(nonempty_bins)
        remainder = n - quota_base * len(nonempty_bins)
        quotas = np.array(
            [quota_base + (1 if out_i < remainder else 0) for out_i in range(len(nonempty_bins))],
            dtype=np.int64,
        )

    selected: list[np.ndarray] = []
    selected_mask = np.zeros(len(ys), dtype=bool)
    remaining = 0

    candidate_ids = np.arange(len(ys), dtype=np.int64)
    for out_i, bin_id in enumerate(nonempty_bins):
        quota = int(quotas[out_i])
        if quota <= 0:
            continue
        bin_candidates = candidate_ids[bin_ids == bin_id]
        take = min(quota, len(bin_candidates))
        if take > 0:
            chosen = rng.choice(bin_candidates, take, replace=False)
            selected.append(chosen)
            selected_mask[chosen] = True
        remaining += quota - take

    if remaining > 0:
        leftovers = candidate_ids[~selected_mask]
        take = min(remaining, len(leftovers))
        if take > 0:
            chosen = rng.choice(leftovers, take, replace=False)
            selected.append(chosen)
            selected_mask[chosen] = True
            remaining -= take

    if remaining > 0 and replace_short:
        chosen = rng.choice(candidate_ids, remaining, replace=True)
        selected.append(chosen)

    if not selected:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    idx = np.concatenate(selected)
    return ys[idx].astype(np.int32), xs[idx].astype(np.int32)


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
    num_ref_segments: int = 1,
    ref_frame_strategy: str = "max_valid",
    source_depth_sampling: str = "uniform",
    source_depth_bin_edges: Sequence[float] | np.ndarray | None = None,
    source_depth_bin_weights: Sequence[float] | np.ndarray | None = None,
    source_depth_num_bins: int = 7,
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
        source_depth_sampling:
            ``uniform`` preserves the legacy pixel-uniform sampler.
            ``log_balanced`` / ``linear_balanced`` split candidate source
            pixels into metric-depth bins and sample roughly equal counts from
            each nonempty bin, separately for boundary and non-boundary quotas.
        source_depth_bin_edges:
            Optional explicit bin edges in metres.  When provided, values
            outside the range are folded into the nearest edge bin.
        source_depth_bin_weights:
            Optional per-bin weights.  The length must equal the number of
            bins, i.e. ``len(source_depth_bin_edges) - 1`` for explicit edges.
            Larger weights oversample that depth range.
        source_depth_num_bins:
            Number of dynamic bins when explicit edges are not provided.

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
    # 1. Select reference frame(s)                                         #
    # ------------------------------------------------------------------ #
    valid_counts = np.array([int(((d > 0) & np.isfinite(d) & (d < depth_max)).sum()) for d in depths])
    ref_frames = _select_ref_frames(valid_counts, num_ref_segments, ref_frame_strategy)

    if not ref_frames:
        ref_frames = [int(np.argmax(valid_counts))]

    source_depth_sampling = _normalise_source_depth_sampling(source_depth_sampling)
    source_depth_bin_edges_arr = _normalise_depth_bin_edges(source_depth_bin_edges)
    source_depth_bin_weights_arr = _normalise_depth_bin_weights(source_depth_bin_weights)
    if source_depth_bin_weights_arr is not None:
        expected_bins = (
            len(source_depth_bin_edges_arr) - 1
            if source_depth_bin_edges_arr is not None
            else int(source_depth_num_bins)
        )
        if source_depth_bin_weights_arr.size != expected_bins:
            raise ValueError(
                "source_depth_bin_weights length must match depth bin count "
                f"({expected_bins}), got {source_depth_bin_weights_arr.size}"
            )

    # ------------------------------------------------------------------ #
    # 2. Sample source points from each reference frame                    #
    # ------------------------------------------------------------------ #
    points_per_seg = num_points // len(ref_frames)
    remainder = num_points - points_per_seg * len(ref_frames)

    all_P_world: list[np.ndarray] = []

    for seg_i, ref in enumerate(ref_frames):
        n_pts = points_per_seg + (1 if seg_i < remainder else 0)
        if n_pts <= 0:
            continue

        depth_ref = depths[ref]
        K_ref = intrinsics[ref]
        E_ref = extrinsics[ref]

        valid_ref = (depth_ref > 0) & np.isfinite(depth_ref) & (depth_ref < depth_max)

        n_boundary = int(n_pts * boundary_ratio)

        bnd_mask = _depth_boundary_mask(depth_ref) & valid_ref
        if source_depth_sampling == "uniform":
            bnd_ys, bnd_xs = np.where(bnd_mask)
            uni_ys, uni_xs = np.where(valid_ref)
            sel_bnd_y, sel_bnd_x = _sample_candidate_pixels(
                bnd_ys,
                bnd_xs,
                n_boundary,
                rng,
                replace_short=False,
            )
            n_uni_needed = n_pts - len(sel_bnd_y)
            sel_uni_y, sel_uni_x = _sample_candidate_pixels(
                uni_ys,
                uni_xs,
                n_uni_needed,
                rng,
            )
        else:
            sel_bnd_y, sel_bnd_x = _sample_candidate_pixels_depth_balanced(
                depth_ref,
                bnd_mask,
                n_boundary,
                rng,
                depth_max=depth_max,
                source_depth_sampling=source_depth_sampling,
                source_depth_bin_edges=source_depth_bin_edges_arr,
                source_depth_bin_weights=source_depth_bin_weights_arr,
                source_depth_num_bins=source_depth_num_bins,
                replace_short=False,
            )
            n_uni_needed = n_pts - len(sel_bnd_y)
            sel_uni_y, sel_uni_x = _sample_candidate_pixels_depth_balanced(
                depth_ref,
                valid_ref,
                n_uni_needed,
                rng,
                depth_max=depth_max,
                source_depth_sampling=source_depth_sampling,
                source_depth_bin_edges=source_depth_bin_edges_arr,
                source_depth_bin_weights=source_depth_bin_weights_arr,
                source_depth_num_bins=source_depth_num_bins,
            )

        src_y = np.concatenate([sel_bnd_y, sel_uni_y]).astype(np.int32)
        src_x = np.concatenate([sel_bnd_x, sel_uni_x]).astype(np.int32)
        src_uv = np.stack([src_x, src_y], axis=-1).astype(np.float32)
        src_d = depth_ref[src_y, src_x].astype(np.float32)

        P_cam_ref = _unproject(src_uv, src_d, K_ref)
        c2w_ref = _c2w_from_w2c(E_ref)
        ones = np.ones((len(src_y), 1), dtype=np.float32)
        P_hom = np.concatenate([P_cam_ref, ones], axis=-1)
        P_world_seg = (c2w_ref @ P_hom.T).T[:, :3]
        all_P_world.append(P_world_seg)

    if not all_P_world:
        raise RuntimeError("No valid source points could be sampled for precomputation")

    P_world = np.concatenate(all_P_world, axis=0)
    N = P_world.shape[0]

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
        "ref_frame":      int(ref_frames[0]),
        "ref_frames":     np.array(ref_frames, dtype=np.int32),
        "num_points":     N,
        "track_semantics_version": TRACK_SEMANTICS_VERSION,
        "num_ref_segments": int(num_ref_segments),
        "track_depth_max": float(depth_max),
        "source_depth_sampling_mode": SOURCE_DEPTH_SAMPLING_MODE_IDS[source_depth_sampling],
        "source_depth_bin_edges": (
            source_depth_bin_edges_arr.astype(np.float32)
            if source_depth_bin_edges_arr is not None
            else np.empty(0, dtype=np.float32)
        ),
        "source_depth_bin_weights": (
            source_depth_bin_weights_arr.astype(np.float32)
            if source_depth_bin_weights_arr is not None
            else np.empty(0, dtype=np.float32)
        ),
        "source_depth_num_bins": int(source_depth_num_bins),
    }
