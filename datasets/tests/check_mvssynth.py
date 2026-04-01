#!/usr/bin/env python3
"""Validation script for MVSSynthAdapter.

Checks performed
----------------
1.  Adapter construction & sequence enumeration.
2.  UnifiedClip schema: shapes, dtypes, None-fields, metadata keys.
3.  Image content: uint8 RGB, non-black, value range [0, 255].
4.  Depth content: float32, finite, at least some positive values.
5.  Intrinsics: shape (T,3,3), finite, positive focal lengths, principal point
    inside image.
6.  Extrinsics: shape (T,4,4), finite, rotation near-orthonormal, det(R)≈±1,
    last row == [0,0,0,1].
7.  Reprojection cross-check: unproject depth pixels to world space with
    E^{-1} and re-project with K and E. Error must be < 1 px.
8.  Boundary frame indices: first and last frame load without error.
9.  Error-handling: out-of-range index, empty list, unknown sequence name.
10. Per-sequence adapter.sanity_check() return structure and result.

Usage
-----
# Quick: single sequence
python check_mvssynth.py --data-root /data2/d4rt/datasets/MVS-Synth/GTAV_1080 \\
    --sequence 0000 --verbose

# Multiple sequences
python check_mvssynth.py --data-root /data2/d4rt/datasets/MVS-Synth/GTAV_1080 \\
    --max-seqs 5

# Full scan (all 120 sequences)
python check_mvssynth.py --data-root /data2/d4rt/datasets/MVS-Synth/GTAV_1080 \\
    --max-seqs 0 --verbose
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

from datasets.adapters.mvssynth import MVSSynthAdapter  # noqa: E402
from datasets.adapters.base import UnifiedClip           # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate MVSSynthAdapter against the D4RT unified schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", required=True,
        help="Root directory of the MVS-Synth GTAV_1080 dataset.",
    )
    p.add_argument(
        "--sequence", default=None,
        help="Test a single sequence (e.g., '0000'). Overrides --max-seqs.",
    )
    p.add_argument(
        "--max-seqs", type=int, default=3,
        help="Max sequences to test (0 = all).",
    )
    p.add_argument(
        "--clip-len", type=int, default=8,
        help="Number of frames per clip for load_clip checks.",
    )
    p.add_argument(
        "--reproj-pixels", type=int, default=512,
        help="Number of pixels sampled per frame for reprojection checks.",
    )
    p.add_argument(
        "--reproj-max-frames", type=int, default=4,
        help="Frames per sequence used for reprojection.",
    )
    p.add_argument(
        "--reproj-err-threshold", type=float, default=1.0,
        help="Maximum allowed mean reprojection error (pixels).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show per-check detail.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-check helpers
# ---------------------------------------------------------------------------

class _Checker:
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
        return not self.errors

    def summary(self) -> str:
        tag = "[OK]  " if self.ok else "[FAIL]"
        line = f"{tag} {self.seq_name}"
        if not self.ok:
            line += "\n" + "\n".join(f"       ✗ {e}" for e in self.errors)
        return line


# ---------------------------------------------------------------------------

def check_schema(c: _Checker, clip: UnifiedClip, T: int) -> None:
    c.require(clip.dataset_name == "mvssynth",
              f"dataset_name='{clip.dataset_name}'=='mvssynth'")
    c.require(isinstance(clip.sequence_name, str) and clip.sequence_name,
              "sequence_name non-empty str")
    c.require(clip.num_frames == T,
              f"clip.num_frames={clip.num_frames}=={T}")

    # images
    c.require(isinstance(clip.images, list) and len(clip.images) == T,
              f"images is list[{T}]")
    if clip.images:
        img0 = clip.images[0]
        c.require(isinstance(img0, np.ndarray) and img0.dtype == np.uint8,
                  f"images[0] dtype uint8: {img0.dtype}")
        c.require(img0.ndim == 3 and img0.shape[2] == 3,
                  f"images[0] shape (H,W,3): {img0.shape}")

    # depths
    c.require(isinstance(clip.depths, list) and len(clip.depths) == T,
              f"depths is list[{T}]")
    if clip.depths:
        d0 = clip.depths[0]
        c.require(isinstance(d0, np.ndarray) and d0.dtype == np.float32,
                  f"depths[0] dtype float32: {d0.dtype}")
        c.require(d0.ndim == 2, f"depths[0] is 2-D: {d0.shape}")

    # depth-image spatial dimensions match
    if clip.images and clip.depths:
        ih, iw = clip.images[0].shape[:2]
        dh, dw = clip.depths[0].shape[:2]
        c.require((ih, iw) == (dh, dw),
                  f"image/depth share (H,W)=({ih},{iw}) vs ({dh},{dw})")

    # None-fields
    for attr in ("normals", "trajs_2d", "trajs_3d_world", "valids", "visibs"):
        c.require(getattr(clip, attr) is None, f"{attr} is None")

    # intrinsics / extrinsics shapes
    c.require(
        isinstance(clip.intrinsics, np.ndarray)
        and clip.intrinsics.shape == (T, 3, 3),
        f"intrinsics.shape==({T},3,3): {clip.intrinsics.shape}",
    )
    c.require(
        isinstance(clip.extrinsics, np.ndarray)
        and clip.extrinsics.shape == (T, 4, 4),
        f"extrinsics.shape==({T},4,4): {clip.extrinsics.shape}",
    )

    # frame_paths
    c.require(isinstance(clip.frame_paths, list) and len(clip.frame_paths) == T,
              f"frame_paths is list[{T}]")

    # metadata flags
    if isinstance(clip.metadata, dict):
        required = {"has_depth", "has_normals", "has_tracks", "extrinsics_convention"}
        missing = required - clip.metadata.keys()
        c.require(not missing, f"metadata keys present (missing: {missing})")
        c.require(clip.metadata.get("has_depth") is True, "metadata.has_depth==True")
        c.require(clip.metadata.get("has_normals") is False, "metadata.has_normals==False")
        c.require(clip.metadata.get("has_tracks") is False, "metadata.has_tracks==False")
        c.require(clip.metadata.get("extrinsics_convention") == "w2c",
                  "metadata.extrinsics_convention=='w2c'")


def check_image_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, img in enumerate(clip.images):
        c.require(np.isfinite(img.astype(np.float32)).all(),
                  f"images[{t}] finite")
        c.require(0 <= int(img.min()) and int(img.max()) <= 255,
                  f"images[{t}] range [0,255]: [{img.min()},{img.max()}]")
        c.require(int(img.max()) > 0,
                  f"images[{t}] not all-black")


def check_depth_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, dep in enumerate(clip.depths):
        c.require(np.isfinite(dep).all(), f"depths[{t}] finite")
        c.require((dep >= 0).all(), f"depths[{t}] no negative values")
        n_valid = int((dep > 0).sum())
        if n_valid == 0:
            c.warn(f"depths[{t}] has NO positive pixels")
        elif c.verbose:
            total = dep.size
            valid_dep = dep[dep > 0]
            c.notes.append(
                f"  info depths[{t}]: min={valid_dep.min():.2f}, "
                f"max={valid_dep.max():.2f}, coverage={n_valid/total:.1%}"
            )


def check_intrinsics(c: _Checker, clip: UnifiedClip) -> None:
    K = clip.intrinsics
    c.require(np.isfinite(K).all(), "intrinsics finite")

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    c.require(np.all(fx > 0) and np.all(fy > 0),
              f"focal lengths positive (fx∈[{fx.min():.1f},{fx.max():.1f}])")

    # Bottom row == [0, 0, 1]
    c.require(np.allclose(K[:, 2, :], [0, 0, 1], atol=1e-5),
              "intrinsics bottom row==[0,0,1]")

    # principal point inside image
    if clip.images:
        H, W = clip.images[0].shape[:2]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        c.require(
            np.all(cx > 0) and np.all(cx < W) and np.all(cy > 0) and np.all(cy < H),
            f"principal point inside image ({W}×{H}): "
            f"cx∈[{cx.min():.1f},{cx.max():.1f}], cy∈[{cy.min():.1f},{cy.max():.1f}]",
        )


def check_extrinsics(c: _Checker, clip: UnifiedClip) -> None:
    E = clip.extrinsics
    c.require(np.isfinite(E).all(), "extrinsics finite")

    # Last row
    c.require(np.allclose(E[:, 3, :], [0, 0, 0, 1], atol=1e-5),
              "extrinsics last row==[0,0,0,1]")

    R = E[:, :3, :3].astype(np.float64)
    RRt = R @ np.transpose(R, (0, 2, 1))
    ortho_err = float(np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2)).max())
    c.require(ortho_err < 1e-4,
              f"rotation orthonormal (max ||RR^T-I||_F={ortho_err:.2e})")

    dets = np.linalg.det(R)
    # MVS-Synth has det(R) ≈ -1 (left-handed system)
    c.require(np.all(np.abs(np.abs(dets) - 1.0) < 1e-4),
              f"det(R)≈±1 (range [{dets.min():.6f},{dets.max():.6f}])")


def check_reproj(
    c: _Checker,
    clip: UnifiedClip,
    n_pixels: int,
    max_frames: int,
    err_threshold: float,
    rng: np.random.Generator,
) -> None:
    """Unproject depth pixels and re-project; verify consistency."""
    T_use = min(clip.num_frames, max_frames)
    frame_sel = np.unique(np.linspace(0, clip.num_frames - 1, T_use, dtype=int))

    all_errs: list[float] = []
    frames_checked = 0

    for t in frame_sel:
        dep = clip.depths[t].astype(np.float64)
        K = clip.intrinsics[t].astype(np.float64)
        E = clip.extrinsics[t].astype(np.float64)  # w2c

        ys, xs = np.where(dep > 0)
        if ys.size == 0:
            c.warn(f"reproj frame {t}: no valid depth, skipped")
            continue

        n = min(n_pixels, ys.size)
        idx = rng.choice(ys.size, size=n, replace=False)
        ys_s, xs_s, ds = ys[idx], xs[idx], dep[ys[idx], xs[idx]]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        xc = (xs_s - cx) / fx * ds
        yc = (ys_s - cy) / fy * ds
        pts_cam = np.stack([xc, yc, ds, np.ones_like(ds)], axis=1)  # (N,4)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            c.errors.append(f"reproj frame {t}: E is singular")
            continue

        pts_world = (E_inv @ pts_cam.T).T       # (N,4)
        pts_cam2 = (E @ pts_world.T).T[:, :3]  # (N,3)
        z2 = pts_cam2[:, 2]
        valid_z = np.abs(z2) > 1e-8

        u2 = fx * pts_cam2[valid_z, 0] / z2[valid_z] + cx
        v2 = fy * pts_cam2[valid_z, 1] / z2[valid_z] + cy
        errs = np.hypot(u2 - xs_s[valid_z], v2 - ys_s[valid_z])
        all_errs.extend(errs.tolist())
        frames_checked += 1

    if not all_errs:
        c.errors.append("reproj: no valid samples found")
        return

    arr = np.array(all_errs)
    mean_e, p95_e, max_e = arr.mean(), np.percentile(arr, 95), arr.max()
    c.require(
        mean_e < err_threshold,
        f"reproj mean={mean_e:.5f}px < {err_threshold}px "
        f"(p95={p95_e:.5f}px, max={max_e:.5f}px, n={arr.size}, frames={frames_checked})",
    )


def check_boundary_frames(c: _Checker, adapter: MVSSynthAdapter, seq_name: str) -> None:
    info = adapter.get_sequence_info(seq_name)
    last = info["num_frames"] - 1
    try:
        clip = adapter.load_clip(seq_name, [0, last])
        c.require(clip.num_frames == 2, "boundary clip has 2 frames")
    except Exception as exc:
        c.errors.append(f"boundary frame load failed: {exc}")


def check_error_handling(c: _Checker, adapter: MVSSynthAdapter, seq_name: str) -> None:
    N = adapter.get_sequence_info(seq_name)["num_frames"]

    raised = False
    try:
        adapter.load_clip(seq_name, [N])
    except (IndexError, ValueError):
        raised = True
    c.require(raised, "out-of-range index → IndexError/ValueError")

    raised = False
    try:
        adapter.load_clip(seq_name, [])
    except (ValueError, IndexError):
        raised = True
    c.require(raised, "empty frame_indices → ValueError/IndexError")

    raised = False
    try:
        adapter.get_sequence_info("__bad_seq__")
    except (KeyError, FileNotFoundError, ValueError):
        raised = True
    c.require(raised, "unknown sequence_name → KeyError/FileNotFoundError")


def check_sanity_check_return(
    c: _Checker, adapter: MVSSynthAdapter, seq_name: str
) -> None:
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


# ---------------------------------------------------------------------------
# Per-sequence orchestration
# ---------------------------------------------------------------------------

def validate_sequence(
    adapter: MVSSynthAdapter,
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
    check_reproj(
        c, clip,
        n_pixels=reproj_pixels,
        max_frames=reproj_max_frames,
        err_threshold=reproj_err_threshold,
        rng=rng,
    )
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

def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  MVS-Synth adapter check")
    print(f"{'='*60}")

    try:
        adapter = MVSSynthAdapter(
            root=args.data_root,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[FAIL] Adapter construction failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        raise SystemExit(1)

    all_seqs = adapter.list_sequences()
    print(f"  Total sequences: {len(all_seqs)}")

    # ── Sequence selection ────────────────────────────────────────────
    if args.sequence:
        if args.sequence not in all_seqs:
            print(f"[FAIL] --sequence '{args.sequence}' not found.", file=sys.stderr)
            raise SystemExit(1)
        seqs_to_check = [args.sequence]
        print(f"  Testing single sequence: {args.sequence}")
    elif args.max_seqs > 0 and len(all_seqs) > args.max_seqs:
        step = max(1, len(all_seqs) // args.max_seqs)
        seqs_to_check = all_seqs[::step][:args.max_seqs]
        print(f"  Testing {len(seqs_to_check)} sequences (--max-seqs={args.max_seqs})")
    else:
        seqs_to_check = all_seqs
        print(f"  Testing all {len(seqs_to_check)} sequences")

    # ── Per-sequence checks ───────────────────────────────────────────
    passed = 0
    for seq in seqs_to_check:
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

    total = len(seqs_to_check)
    print(f"\n  Result: {passed}/{total} passed")

    if passed < total:
        print(f"{total - passed}/{total} check(s) FAILED.", file=sys.stderr)
        raise SystemExit(1)
    else:
        print(f"All {total} check(s) passed.")


if __name__ == "__main__":
    main()
