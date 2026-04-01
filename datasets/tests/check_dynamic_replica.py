#!/usr/bin/env python3
"""Validation script for DynamicReplicaAdapter.

Checks performed
----------------
1.  Adapter construction and sequence enumeration.
2.  UnifiedClip schema: shapes, dtypes, None-fields, metadata keys.
3.  Image content: uint8 RGB, value range [0, 255].
4.  Depth content: float32, finite, non-negative, at least some positive values.
5.  Intrinsics: shape (T,3,3), finite, positive focal lengths,
    principal point inside image.
6.  Extrinsics: shape (T,4,4), finite, rotation near-orthonormal,
    last row == [0,0,0,1].
7.  Reprojection consistency: unproject depth pixels to world and
    re-project; check reprojection error < 1 px.
8.  Trajectory fields (left camera, when load_trajectories=True):
    shapes [T,N,2], [T,N,3], [T,N]; traj_2d in image bounds.
9.  Boundary frame indices: first frame (idx=0) and last frame (idx=N-1).
10. Right-camera sequences: trajectory fields are None.
11. per-sequence adapter.sanity_check() return dict.
12. Error-handling: bad sequence name, out-of-range indices.

Usage
-----
# Quick check on 3 sequences per camera side:
python check_dynamic_replica.py --data-root /data1/d4rt/datasets/Dynamic_Replica \\
    --split train --max-seqs 3

# Check a single sequence by name:
python check_dynamic_replica.py --data-root /data1/d4rt/datasets/Dynamic_Replica \\
    --split train --sequence 009850-3_obj_source_left

# Full split scan (slow, ~967 seqs):
python check_dynamic_replica.py --data-root /data1/d4rt/datasets/Dynamic_Replica \\
    --split train --max-seqs 0

# Skip trajectory loading (faster):
python check_dynamic_replica.py --data-root /data1/d4rt/datasets/Dynamic_Replica \\
    --split train --no-trajectories --max-seqs 5
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adapters.dynamic_replica import DynamicReplicaAdapter  # noqa: E402
from datasets.adapters.base import UnifiedClip                       # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate DynamicReplicaAdapter against the D4RT unified schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="Root directory of the Dynamic_Replica dataset.")
    p.add_argument("--split", default="train",
                   choices=["train", "valid", "test"],
                   help="Which split to validate.")
    p.add_argument("--sequence", default=None,
                   help="Restrict to a single sequence directory name.")
    p.add_argument("--max-seqs", type=int, default=3,
                   help="Max sequences to check (0 = all).")
    p.add_argument("--clip-len", type=int, default=8,
                   help="Number of frames to load per clip.")
    p.add_argument("--reproj-pixels", type=int, default=512,
                   help="Pixels to sample per frame for reprojection check.")
    p.add_argument("--reproj-max-frames", type=int, default=4,
                   help="Frames to use in reprojection check.")
    p.add_argument("--reproj-err-threshold", type=float, default=1.0,
                   help="Max allowed mean reprojection error in pixels.")
    p.add_argument("--no-trajectories", action="store_true",
                   help="Disable trajectory loading (faster).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-check detail instead of only [OK]/[FAIL] lines.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checker helper
# ---------------------------------------------------------------------------

class _Checker:
    """Accumulates pass / fail messages for one sequence."""

    def __init__(self, seq_name: str, verbose: bool) -> None:
        self.seq_name = seq_name
        self.verbose = verbose
        self.errors: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, msg: str) -> bool:
        if not condition:
            self.errors.append(msg)
        elif self.verbose:
            self.notes.append(f"  ok  {msg}")
        return condition

    def warn(self, msg: str) -> None:
        self.notes.append(f"  warn {msg}")

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        tag = "[OK]  " if self.ok else "[FAIL]"
        line = f"{tag} {self.seq_name}"
        if not self.ok:
            line += "\n" + "\n".join(f"       x {e}" for e in self.errors)
        return line


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_schema(c: _Checker, clip: UnifiedClip, T: int) -> None:
    """Check all UnifiedClip fields for shape / dtype / None correctness."""

    c.require(clip.dataset_name == "dynamic_replica",
              f"dataset_name == 'dynamic_replica' (got {clip.dataset_name!r})")
    c.require(isinstance(clip.sequence_name, str) and clip.sequence_name,
              "sequence_name is non-empty string")
    c.require(clip.num_frames == T, f"num_frames == {T}")

    # images
    c.require(isinstance(clip.images, list) and len(clip.images) == T,
              f"images is list of length {T}")
    if clip.images:
        img0 = clip.images[0]
        c.require(isinstance(img0, np.ndarray), "images[0] is np.ndarray")
        c.require(img0.ndim == 3 and img0.shape[2] == 3,
                  f"images[0] shape is (H,W,3): {img0.shape}")
        c.require(img0.dtype == np.uint8,
                  f"images[0] dtype is uint8: {img0.dtype}")

    # depths
    c.require(isinstance(clip.depths, list) and len(clip.depths) == T,
              f"depths is list of length {T}")
    if clip.depths:
        dep0 = clip.depths[0]
        c.require(isinstance(dep0, np.ndarray), "depths[0] is np.ndarray")
        c.require(dep0.ndim == 2, f"depths[0] is 2-D: {dep0.shape}")
        c.require(dep0.dtype == np.float32,
                  f"depths[0] dtype is float32: {dep0.dtype}")

    # image/depth size consistency
    if clip.images and clip.depths:
        ih, iw = clip.images[0].shape[:2]
        dh, dw = clip.depths[0].shape[:2]
        c.require((ih, iw) == (dh, dw),
                  f"image and depth share spatial dims ({ih},{iw}) vs ({dh},{dw})")

    # normals must be None
    c.require(clip.normals is None, "normals is None (not provided by Dynamic_Replica)")

    # intrinsics
    c.require(
        isinstance(clip.intrinsics, np.ndarray) and clip.intrinsics.shape == (T, 3, 3),
        f"intrinsics.shape == ({T},3,3): {clip.intrinsics.shape}",
    )

    # extrinsics
    c.require(
        isinstance(clip.extrinsics, np.ndarray) and clip.extrinsics.shape == (T, 4, 4),
        f"extrinsics.shape == ({T},4,4): {clip.extrinsics.shape}",
    )

    # frame_paths
    c.require(
        isinstance(clip.frame_paths, list) and len(clip.frame_paths) == T,
        f"frame_paths is list of length {T}",
    )

    # metadata keys
    required_meta = {
        "has_depth", "has_normals", "has_tracks", "has_visibility",
        "extrinsics_convention", "depth_unit",
    }
    if isinstance(clip.metadata, dict):
        missing = required_meta - clip.metadata.keys()
        c.require(not missing, f"metadata has required keys (missing: {missing})")
        c.require(clip.metadata.get("has_depth") is True, "metadata.has_depth == True")
        c.require(clip.metadata.get("has_normals") is False, "metadata.has_normals == False")
        c.require(clip.metadata.get("extrinsics_convention") == "w2c",
                  "metadata.extrinsics_convention == 'w2c'")


def check_image_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, img in enumerate(clip.images):
        c.require(np.isfinite(img.astype(np.float32)).all(),
                  f"images[{t}] all finite")
        c.require(img.min() >= 0 and img.max() <= 255,
                  f"images[{t}] values in [0,255]: min={img.min()}, max={img.max()}")
        c.require(img.max() > 0, f"images[{t}] is not all-black")


def check_depth_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, dep in enumerate(clip.depths):
        c.require(np.isfinite(dep).all(), f"depths[{t}] all finite")
        c.require((dep >= 0).all(), f"depths[{t}] all non-negative")
        n_positive = int((dep > 0).sum())
        c.require(n_positive > 0, f"depths[{t}] has at least one positive value")
        if n_positive > 0 and c.verbose:
            valid = dep[dep > 0]
            c.notes.append(
                f"  info depths[{t}]: min={valid.min():.5f}, max={valid.max():.5f}, "
                f"coverage={n_positive/dep.size:.1%}"
            )


def check_intrinsics(c: _Checker, clip: UnifiedClip) -> None:
    K = clip.intrinsics
    c.require(np.isfinite(K).all(), "intrinsics all finite")

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    c.require(np.all(fx > 0) and np.all(fy > 0),
              f"focal lengths positive (fx={fx[0]:.1f}, fy={fy[0]:.1f})")

    skew = K[:, 0, 1]
    if not np.allclose(skew, 0.0, atol=1e-3):
        c.warn(f"non-zero skew: max|skew|={np.abs(skew).max():.4f}")

    bottom = K[:, 2, :]
    c.require(np.allclose(bottom, [[0.0, 0.0, 1.0]], atol=1e-5),
              "intrinsics bottom row == [0,0,1]")

    if clip.images:
        H, W = clip.images[0].shape[:2]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        c.require(
            np.all(cx > 0) and np.all(cx < W) and np.all(cy > 0) and np.all(cy < H),
            f"principal point inside image ({W}x{H}): "
            f"cx={cx[0]:.1f}, cy={cy[0]:.1f}",
        )


def check_extrinsics(c: _Checker, clip: UnifiedClip) -> None:
    E = clip.extrinsics

    c.require(np.isfinite(E).all(), "extrinsics all finite")

    bottom = E[:, 3, :]
    c.require(np.allclose(bottom, [[0.0, 0.0, 0.0, 1.0]], atol=1e-5),
              "extrinsics last row == [0,0,0,1]")

    R = E[:, :3, :3].astype(np.float64)
    RRt = R @ np.transpose(R, (0, 2, 1))
    ortho_err = np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2))
    c.require(float(ortho_err.max()) < 1e-4,
              f"rotation matrices near-orthonormal (max ||RR^T-I||_F={ortho_err.max():.2e})")

    dets = np.linalg.det(R)
    c.require(np.all(np.abs(dets - 1.0) < 1e-4),
              f"rotation matrices det≈+1 (range [{dets.min():.6f}, {dets.max():.6f}])")


def check_reprojection(
    c: _Checker,
    clip: UnifiedClip,
    n_pixels: int,
    max_frames: int,
    err_threshold: float,
    rng: np.random.Generator,
) -> None:
    """Unproject valid depth pixels to world, re-project, check error < threshold."""
    T_use = min(clip.num_frames, max_frames)
    frame_sel = np.unique(np.linspace(0, clip.num_frames - 1, T_use, dtype=int))
    all_errs: list[float] = []

    for t in frame_sel:
        dep = clip.depths[t].astype(np.float64)
        K = clip.intrinsics[t].astype(np.float64)
        E = clip.extrinsics[t].astype(np.float64)

        valid = np.isfinite(dep) & (dep > 0)
        ys, xs = np.where(valid)
        if ys.size == 0:
            c.warn(f"reproj frame {t}: no valid depth, skipping")
            continue

        n = min(n_pixels, ys.size)
        idx = rng.choice(ys.size, size=n, replace=False)
        ys_s, xs_s = ys[idx], xs[idx]
        ds = dep[ys_s, xs_s]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_c = (xs_s - cx) / fx * ds
        y_c = (ys_s - cy) / fy * ds
        p_cam = np.stack([x_c, y_c, ds, np.ones_like(ds)], axis=1)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            c.errors.append(f"reproj frame {t}: E is singular")
            continue

        p_world = (E_inv @ p_cam.T).T
        p_cam2 = (E @ p_world.T).T[:, :3]
        z2 = p_cam2[:, 2]
        pos_z = np.abs(z2) > 1e-8
        if not np.any(pos_z):
            c.warn(f"reproj frame {t}: all reprojected z <= 0, skipping")
            continue

        u2 = K[0, 0] * p_cam2[pos_z, 0] / z2[pos_z] + K[0, 2]
        v2 = K[1, 1] * p_cam2[pos_z, 1] / z2[pos_z] + K[1, 2]
        errs = np.hypot(u2 - xs_s[pos_z], v2 - ys_s[pos_z])
        all_errs.extend(errs.tolist())

    if not all_errs:
        c.errors.append("reprojection: no valid samples found")
        return

    arr = np.array(all_errs)
    mean_err = float(arr.mean())
    p95_err = float(np.percentile(arr, 95))
    c.require(mean_err < err_threshold,
              f"reproj mean={mean_err:.4f}px < {err_threshold}px "
              f"(p95={p95_err:.4f}px, n={arr.size})")
    if c.verbose:
        c.notes.append(
            f"  info reproj: mean={mean_err:.5f}px, p95={p95_err:.5f}px, n={arr.size}"
        )


def check_trajectories(c: _Checker, clip: UnifiedClip) -> None:
    """Validate trajectory fields when they are present."""
    T = clip.num_frames

    if clip.trajs_2d is None:
        c.require(clip.trajs_3d_world is None,
                  "trajs_3d_world is None (consistent with trajs_2d=None)")
        c.require(clip.visibs is None,
                  "visibs is None (consistent with trajs_2d=None)")
        c.require(clip.valids is None,
                  "valids is None (consistent with trajs_2d=None)")
        return

    N = clip.trajs_2d.shape[1]

    # shape checks
    c.require(clip.trajs_2d.shape == (T, N, 2),
              f"trajs_2d.shape == ({T},{N},2): {clip.trajs_2d.shape}")
    c.require(
        clip.trajs_3d_world is not None and clip.trajs_3d_world.shape == (T, N, 3),
        f"trajs_3d_world.shape == ({T},{N},3)",
    )
    c.require(
        clip.visibs is not None and clip.visibs.shape == (T, N),
        f"visibs.shape == ({T},{N})",
    )
    c.require(
        clip.valids is not None and clip.valids.shape == (T, N),
        f"valids.shape == ({T},{N})",
    )

    # dtype checks
    c.require(clip.trajs_2d.dtype == np.float32,
              f"trajs_2d dtype float32: {clip.trajs_2d.dtype}")
    if clip.trajs_3d_world is not None:
        c.require(clip.trajs_3d_world.dtype == np.float32,
                  f"trajs_3d_world dtype float32: {clip.trajs_3d_world.dtype}")

    # finite checks
    c.require(np.isfinite(clip.trajs_2d).all(), "trajs_2d all finite")
    if clip.trajs_3d_world is not None:
        c.require(np.isfinite(clip.trajs_3d_world).all(), "trajs_3d_world all finite")

    # traj_2d should be within image bounds for visible points
    if clip.images and clip.visibs is not None:
        H, W = clip.images[0].shape[:2]
        vis = clip.visibs  # (T, N)
        if vis.any():
            uv_vis = clip.trajs_2d[vis]
            in_bounds = (
                (uv_vis[:, 0] >= 0) & (uv_vis[:, 0] < W)
                & (uv_vis[:, 1] >= 0) & (uv_vis[:, 1] < H)
            )
            frac_in = float(in_bounds.mean())
            c.require(frac_in >= 0.5,
                      f"visible trajs_2d >= 50% in image bounds: {frac_in:.1%}")
            if c.verbose:
                c.notes.append(
                    f"  info trajs: N={N}, vis_ratio={vis.mean():.3f}, "
                    f"in_bounds={frac_in:.3f}"
                )

    # valids should be consistent with visibs
    if clip.valids is not None and clip.visibs is not None:
        c.require(
            (clip.valids == clip.visibs).all(),
            "valids == visibs (Dynamic_Replica: valid iff visible)",
        )


def check_right_camera_no_tracks(c: _Checker, clip: UnifiedClip) -> None:
    """Right-camera clips must have all trajectory fields as None."""
    camera_name = (clip.metadata or {}).get("camera_name", "unknown")
    if camera_name != "right":
        return

    c.require(clip.trajs_2d is None,
              "trajs_2d is None for right-camera sequence (no trajectories)")
    c.require(clip.trajs_3d_world is None,
              "trajs_3d_world is None for right-camera sequence")
    c.require(clip.visibs is None, "visibs is None for right-camera sequence")
    c.require(clip.valids is None, "valids is None for right-camera sequence")


def check_boundary_frames(
    c: _Checker,
    adapter: DynamicReplicaAdapter,
    seq_name: str,
) -> None:
    """Load a 2-frame clip with first and last indices."""
    info = adapter.get_sequence_info(seq_name)
    last = info["num_frames"] - 1
    try:
        clip = adapter.load_clip(seq_name, [0, last])
        c.require(clip.num_frames == 2, "boundary clip has 2 frames")
    except Exception as exc:
        c.errors.append(f"boundary frame load failed: {exc}")


def check_error_handling(
    c: _Checker,
    adapter: DynamicReplicaAdapter,
    seq_name: str,
) -> None:
    """Verify expected exceptions for invalid inputs."""
    info = adapter.get_sequence_info(seq_name)
    N = info["num_frames"]

    raised = False
    try:
        adapter.load_clip(seq_name, [N])
    except (IndexError, ValueError):
        raised = True
    c.require(raised, "out-of-range index raises IndexError/ValueError")

    raised = False
    try:
        adapter.load_clip(seq_name, [])
    except (ValueError, IndexError):
        raised = True
    c.require(raised, "empty frame_indices raises ValueError/IndexError")

    raised = False
    try:
        adapter.get_sequence_info("__nonexistent__")
    except (KeyError, FileNotFoundError, ValueError):
        raised = True
    c.require(raised, "unknown sequence_name raises KeyError/FileNotFoundError")


def check_sanity_check_return(
    c: _Checker,
    adapter: DynamicReplicaAdapter,
    seq_name: str,
) -> None:
    """Call adapter.sanity_check() and validate the return structure."""
    try:
        result = adapter.sanity_check(seq_name)
    except Exception as exc:
        c.errors.append(f"sanity_check() raised: {exc}")
        return

    c.require(isinstance(result, dict), "sanity_check returns dict")
    for key in ("dataset_name", "sequence_name", "ok", "messages"):
        c.require(key in result, f"sanity_check result has key '{key}'")

    if isinstance(result, dict):
        c.require(result.get("ok") is True,
                  f"sanity_check ok=True (messages={result.get('messages')})")
        c.require(isinstance(result.get("messages"), list),
                  "sanity_check messages is a list")


# ---------------------------------------------------------------------------
# Per-sequence orchestration
# ---------------------------------------------------------------------------

def validate_sequence(
    adapter: DynamicReplicaAdapter,
    seq_name: str,
    clip_len: int,
    reproj_pixels: int,
    reproj_max_frames: int,
    reproj_err_threshold: float,
    seed: int,
    verbose: bool,
) -> _Checker:
    c = _Checker(seq_name, verbose)
    rng = np.random.default_rng(seed)

    info = adapter.get_sequence_info(seq_name)
    T = min(clip_len, info["num_frames"])
    frame_indices = np.unique(
        np.linspace(0, info["num_frames"] - 1, T, dtype=int)
    ).tolist()

    try:
        clip = adapter.load_clip(seq_name, frame_indices)
    except Exception as exc:
        c.errors.append(f"load_clip raised: {exc}")
        if verbose:
            traceback.print_exc()
        return c

    T_actual = len(frame_indices)

    check_schema(c, clip, T_actual)
    check_image_content(c, clip)
    check_depth_content(c, clip)
    check_intrinsics(c, clip)
    check_extrinsics(c, clip)
    check_reprojection(
        c, clip,
        n_pixels=reproj_pixels,
        max_frames=reproj_max_frames,
        err_threshold=reproj_err_threshold,
        rng=rng,
    )
    check_trajectories(c, clip)
    check_right_camera_no_tracks(c, clip)
    check_boundary_frames(c, adapter, seq_name)
    check_error_handling(c, adapter, seq_name)
    check_sanity_check_return(c, adapter, seq_name)

    if verbose and c.notes:
        for note in c.notes:
            print(note)

    return c


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> tuple[int, int]:
    print(f"\n{'='*60}")
    print(f"  DynamicReplica adapter check — split: {args.split}")
    print(f"  load_trajectories: {not args.no_trajectories}")
    print(f"{'='*60}")

    try:
        adapter = DynamicReplicaAdapter(
            root=args.data_root,
            split=args.split,
            load_trajectories=not args.no_trajectories,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[FAIL] Adapter construction failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 0, 1

    seqs = adapter.list_sequences()
    print(f"  Total sequences in split: {len(seqs)}")

    if args.sequence:
        if args.sequence not in seqs:
            print(f"[FAIL] --sequence {args.sequence!r} not found.", file=sys.stderr)
            return 0, 1
        seqs = [args.sequence]
        print(f"  Checking only: {args.sequence}")
    elif args.max_seqs > 0 and len(seqs) > args.max_seqs:
        step = max(1, len(seqs) // args.max_seqs)
        seqs = seqs[::step][: args.max_seqs]
        print(f"  Checking {len(seqs)} sequences (--max-seqs={args.max_seqs})")
    else:
        print(f"  Checking all {len(seqs)} sequences")

    # Report camera breakdown of selected sequences
    left_seqs = [s for s in seqs if s.endswith("_source_left")]
    right_seqs = [s for s in seqs if s.endswith("_source_right")]
    print(f"  Selected: {len(left_seqs)} left, {len(right_seqs)} right")

    passed = 0
    for seq in seqs:
        checker = validate_sequence(
            adapter=adapter,
            seq_name=seq,
            clip_len=args.clip_len,
            reproj_pixels=args.reproj_pixels,
            reproj_max_frames=args.reproj_max_frames,
            reproj_err_threshold=args.reproj_err_threshold,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(checker.summary())
        if checker.ok:
            passed += 1

    total = len(seqs)
    print(f"\n  Result: {passed}/{total} passed")
    return passed, total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    passed, total = run(args)

    print()
    if total == 0:
        print("No sequences were checked.")
        raise SystemExit(1)

    if passed == total:
        print(f"All {total} check(s) passed.")
    else:
        failed = total - passed
        print(f"{failed}/{total} check(s) FAILED.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
