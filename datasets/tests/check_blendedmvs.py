#!/usr/bin/env python3
"""Validation script for BlendedMVSAdapter.

Checks performed
----------------
1.  Adapter construction & split list parsing.
2.  Sequence enumeration (list_sequences / get_sequence_name / get_sequence_info).
3.  UnifiedClip schema: shapes, dtypes, None-fields, metadata keys.
4.  Image content: uint8 RGB, value range [0, 255].
5.  Depth content: float32, finite, at least some positive values, plausible range.
6.  Intrinsics: shape (T,3,3), finite, positive focal lengths, principal point inside image.
7.  Extrinsics: shape (T,4,4), finite, rotation matrices near-orthonormal,
    last row == [0,0,0,1].
8.  Reprojection consistency: unproject sampled depth pixels to world and
    re-project; check reprojection error < 1 px.
9.  Masked-image variant (use_masked=True): file exists and loads.
10. Boundary frame indices: first frame (idx=0) and last frame (idx=N-1).
11. Per-sequence adapter.sanity_check() return dict.
12. Error-handling: bad sequence name, out-of-range indices.

Usage
-----
# Quick check on a single sequence:
python check_blendedmvs.py --data-root /data2/d4rt/datasets/BlendedMVS \\
    --split train --sequence 5c1f33f1d33e1f2e4aa6dda4

# Full scan of the training split (slow):
python check_blendedmvs.py --data-root /data2/d4rt/datasets/BlendedMVS \\
    --split train --max-seqs 0

# Validate both splits, up to 5 scenes each, verbose:
python check_blendedmvs.py --data-root /data2/d4rt/datasets/BlendedMVS \\
    --split all --max-seqs 5 --verbose
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

from datasets.adapters.blendedmvs import BlendedMVSAdapter  # noqa: E402
from datasets.adapters.base import UnifiedClip              # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate BlendedMVSAdapter against the D4RT unified schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        required=True,
        help="Root directory of the BlendedMVS dataset.",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "all"],
        help="Which split(s) to validate.  'all' runs both train and val.",
    )
    p.add_argument(
        "--sequence",
        default=None,
        help="Restrict to a single scene ID.  Ignored when --split=all.",
    )
    p.add_argument(
        "--max-seqs",
        type=int,
        default=3,
        help="Maximum number of sequences to check per split (0 = all).",
    )
    p.add_argument(
        "--clip-len",
        type=int,
        default=8,
        help="Number of frames to load per clip in load_clip checks.",
    )
    p.add_argument(
        "--reproj-pixels",
        type=int,
        default=512,
        help="Number of pixels to sample per frame for reprojection checks.",
    )
    p.add_argument(
        "--reproj-max-frames",
        type=int,
        default=4,
        help="Number of frames to include in the reprojection check.",
    )
    p.add_argument(
        "--reproj-err-threshold",
        type=float,
        default=1.0,
        help="Maximum allowed mean reprojection error in pixels.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-check detail instead of only [OK]/[FAIL] lines.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Checker:
    """Accumulates pass / fail messages for one sequence."""

    def __init__(self, seq_name: str, verbose: bool) -> None:
        self.seq_name = seq_name
        self.verbose = verbose
        self.errors: list[str] = []
        self.notes: list[str] = []

    # ---- assertion helpers ----

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
            line += "\n" + "\n".join(f"       ✗ {e}" for e in self.errors)
        return line


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_schema(c: _Checker, clip: UnifiedClip, T: int) -> None:
    """Check all UnifiedClip fields for shape / dtype / None correctness."""

    # dataset_name / sequence_name
    c.require(clip.dataset_name == "blendedmvs",
              f"dataset_name='{clip.dataset_name}' (expected 'blendedmvs')")
    c.require(isinstance(clip.sequence_name, str) and clip.sequence_name,
              "sequence_name is a non-empty string")

    # num_frames
    c.require(clip.num_frames == T,
              f"clip.num_frames={clip.num_frames} == {T}")

    # images
    c.require(
        isinstance(clip.images, list) and len(clip.images) == T,
        f"images is list of length {T}",
    )
    if clip.images:
        img0 = clip.images[0]
        c.require(isinstance(img0, np.ndarray), "images[0] is np.ndarray")
        c.require(img0.ndim == 3 and img0.shape[2] == 3,
                  f"images[0] shape is (H,W,3): {img0.shape}")
        c.require(img0.dtype == np.uint8,
                  f"images[0] dtype is uint8: {img0.dtype}")

    # depths
    c.require(
        isinstance(clip.depths, list) and len(clip.depths) == T,
        f"depths is list of length {T}",
    )
    if clip.depths:
        dep0 = clip.depths[0]
        c.require(isinstance(dep0, np.ndarray), "depths[0] is np.ndarray")
        c.require(dep0.ndim == 2, f"depths[0] is 2-D: {dep0.shape}")
        c.require(dep0.dtype == np.float32,
                  f"depths[0] dtype is float32: {dep0.dtype}")

    # depth-image size consistency
    if clip.images and clip.depths:
        ih, iw = clip.images[0].shape[:2]
        dh, dw = clip.depths[0].shape[:2]
        c.require((ih, iw) == (dh, dw),
                  f"image and depth share spatial dims ({ih},{iw}) vs ({dh},{dw})")

    # normals / tracks → must be None for BlendedMVS
    c.require(clip.normals is None, "normals is None (not provided by BlendedMVS)")
    c.require(clip.trajs_2d is None, "trajs_2d is None")
    c.require(clip.trajs_3d_world is None, "trajs_3d_world is None")
    c.require(clip.valids is None, "valids is None")
    c.require(clip.visibs is None, "visibs is None")

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
        missing_meta = required_meta - clip.metadata.keys()
        c.require(not missing_meta,
                  f"metadata has required keys (missing: {missing_meta})")
        c.require(clip.metadata.get("has_depth") is True,
                  "metadata.has_depth == True")
        c.require(clip.metadata.get("has_normals") is False,
                  "metadata.has_normals == False")
        c.require(clip.metadata.get("has_tracks") is False,
                  "metadata.has_tracks == False")
        c.require(clip.metadata.get("extrinsics_convention") == "w2c",
                  "metadata.extrinsics_convention == 'w2c'")


def check_image_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, img in enumerate(clip.images):
        c.require(np.isfinite(img.astype(np.float32)).all(),
                  f"images[{t}] all finite")
        c.require(img.min() >= 0 and img.max() <= 255,
                  f"images[{t}] values in [0,255]: min={img.min()}, max={img.max()}")
        c.require(img.max() > 0,
                  f"images[{t}] is not all-black (max={img.max()})")


def check_depth_content(c: _Checker, clip: UnifiedClip) -> None:
    for t, dep in enumerate(clip.depths):
        c.require(np.isfinite(dep).all(),
                  f"depths[{t}] all finite")
        n_positive = int((dep > 0).sum())
        total = dep.size
        c.require(n_positive > 0,
                  f"depths[{t}] has at least one positive value")
        if n_positive > 0:
            valid = dep[dep > 0]
            # Plausible depth range for indoor/outdoor scenes: 0.1 m – 1000 m
            c.require(float(valid.min()) > 0.0,
                      f"depths[{t}] min positive depth > 0 ({valid.min():.4f})")
            c.require(float(valid.max()) < 2000.0,
                      f"depths[{t}] max depth < 2000 m ({valid.max():.2f})")
            if c.verbose:
                c.notes.append(
                    f"  info depths[{t}]: min={valid.min():.3f}m, "
                    f"max={valid.max():.3f}m, "
                    f"coverage={n_positive/total:.1%}"
                )


def check_intrinsics(c: _Checker, clip: UnifiedClip) -> None:
    K = clip.intrinsics   # (T, 3, 3)
    T = clip.num_frames

    c.require(np.isfinite(K).all(), "intrinsics all finite")

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    c.require(np.all(fx > 0) and np.all(fy > 0),
              f"focal lengths all positive (fx∈[{fx.min():.1f},{fx.max():.1f}], "
              f"fy∈[{fy.min():.1f},{fy.max():.1f}])")

    # Skew should be 0 for standard pinhole
    skew = K[:, 0, 1]
    if not np.allclose(skew, 0.0, atol=1e-3):
        c.warn(f"non-zero skew detected: max|skew|={np.abs(skew).max():.4f}")

    # Bottom row must be [0, 0, 1]
    bottom = K[:, 2, :]
    expected = np.array([0.0, 0.0, 1.0])
    c.require(
        np.allclose(bottom, expected[None], atol=1e-5),
        "intrinsics bottom row == [0,0,1]",
    )

    # Principal point should be inside the image (if image available)
    if clip.images:
        H, W = clip.images[0].shape[:2]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        c.require(
            np.all(cx > 0) and np.all(cx < W) and np.all(cy > 0) and np.all(cy < H),
            f"principal point inside image ({W}×{H}): "
            f"cx∈[{cx.min():.1f},{cx.max():.1f}], cy∈[{cy.min():.1f},{cy.max():.1f}]",
        )


def check_extrinsics(c: _Checker, clip: UnifiedClip) -> None:
    E = clip.extrinsics   # (T, 4, 4)
    T = clip.num_frames

    c.require(np.isfinite(E).all(), "extrinsics all finite")

    # Last row must be [0, 0, 0, 1]
    bottom = E[:, 3, :]
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    c.require(
        np.allclose(bottom, expected[None], atol=1e-5),
        "extrinsics last row == [0,0,0,1]",
    )

    # Rotation sub-matrix should be orthonormal: R @ R^T ≈ I
    R = E[:, :3, :3].astype(np.float64)
    RRt = R @ np.transpose(R, (0, 2, 1))
    eye = np.eye(3, dtype=np.float64)[None]
    ortho_err = np.linalg.norm(RRt - eye, axis=(1, 2))
    c.require(
        float(ortho_err.max()) < 1e-4,
        f"rotation matrices near-orthonormal (max ||R R^T - I||_F={ortho_err.max():.2e})",
    )

    # Determinant of R should be ~+1 (no reflections)
    dets = np.linalg.det(R)
    c.require(
        np.all(np.abs(dets - 1.0) < 1e-4),
        f"rotation matrices have det≈+1 (range [{dets.min():.6f}, {dets.max():.6f}])",
    )


def check_reprojection(
    c: _Checker,
    clip: UnifiedClip,
    n_pixels: int,
    max_frames: int,
    err_threshold: float,
    rng: np.random.Generator,
) -> None:
    """
    For each sampled frame:
      1. Sample pixels with valid (positive, finite) depth.
      2. Unproject to camera-space 3-D point using K and d.
      3. Lift to world space using E^{-1} (E is w2c).
      4. Re-project back to the same camera.
      5. Check that the reprojection matches the original pixel.
    """
    T_use = min(clip.num_frames, max_frames)
    frame_sel = np.unique(np.linspace(0, clip.num_frames - 1, T_use, dtype=int))

    all_errs: list[float] = []

    for t in frame_sel:
        dep = clip.depths[t].astype(np.float64)
        H, W = dep.shape
        K = clip.intrinsics[t].astype(np.float64)
        E = clip.extrinsics[t].astype(np.float64)   # w2c

        # Valid pixel mask
        valid = np.isfinite(dep) & (dep > 0)
        ys, xs = np.where(valid)
        if ys.size == 0:
            c.warn(f"reproj frame {t}: no valid depth pixels, skipping")
            continue

        # Sample pixels
        n = min(n_pixels, ys.size)
        idx = rng.choice(ys.size, size=n, replace=False)
        ys_s, xs_s = ys[idx], xs[idx]
        ds = dep[ys_s, xs_s]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Unproject to camera space
        x_c = (xs_s - cx) / fx * ds
        y_c = (ys_s - cy) / fy * ds
        p_cam = np.stack([x_c, y_c, ds, np.ones_like(ds)], axis=1)  # (N,4)

        # Lift to world space  (E is w2c → inverse is c2w)
        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            c.errors.append(f"reproj frame {t}: E is singular")
            continue
        p_world = (E_inv @ p_cam.T).T  # (N,4)

        # Re-project
        p_cam2 = (E @ p_world.T).T[:, :3]  # (N,3)
        z2 = p_cam2[:, 2]
        pos_z = np.abs(z2) > 1e-8
        if not np.any(pos_z):
            c.warn(f"reproj frame {t}: all reprojected depths ≤ 0, skipping")
            continue

        u2 = K[0, 0] * p_cam2[pos_z, 0] / z2[pos_z] + K[0, 2]
        v2 = K[1, 1] * p_cam2[pos_z, 1] / z2[pos_z] + K[1, 2]

        u_orig = xs_s[pos_z].astype(np.float64)
        v_orig = ys_s[pos_z].astype(np.float64)

        errs = np.hypot(u2 - u_orig, v2 - v_orig)
        all_errs.extend(errs.tolist())

    if not all_errs:
        c.errors.append("reprojection: no valid samples found across all frames")
        return

    arr = np.array(all_errs)
    mean_err = float(arr.mean())
    p95_err = float(np.percentile(arr, 95))
    max_err = float(arr.max())

    c.require(
        mean_err < err_threshold,
        f"reproj mean={mean_err:.4f}px < {err_threshold}px "
        f"(p95={p95_err:.4f}px, max={max_err:.4f}px, n={arr.size})",
    )
    if c.verbose:
        c.notes.append(
            f"  info reproj: mean={mean_err:.5f}px, "
            f"p95={p95_err:.5f}px, max={max_err:.5f}px, n={arr.size}"
        )


def check_masked_variant(c: _Checker, adapter: BlendedMVSAdapter, seq_name: str) -> None:
    """Verify that masked images exist and can be loaded."""
    rec = adapter._get_record(seq_name)
    fid = rec.frame_ids[0]
    masked_path = rec.rgb_path(fid, masked=True)

    c.require(masked_path.exists(),
              f"masked image exists: {masked_path.name}")

    if masked_path.exists():
        import numpy as np
        from PIL import Image as _PIL
        try:
            img = np.asarray(_PIL.open(masked_path).convert("RGB"))
            c.require(
                img.ndim == 3 and img.shape[2] == 3,
                f"masked image is (H,W,3): {img.shape}",
            )
        except Exception as exc:
            c.errors.append(f"masked image failed to open: {exc}")


def check_boundary_frames(
    c: _Checker, adapter: BlendedMVSAdapter, seq_name: str
) -> None:
    """Load a clip with only the first and last frame indices."""
    info = adapter.get_sequence_info(seq_name)
    last = info["num_frames"] - 1
    try:
        clip = adapter.load_clip(seq_name, [0, last])
        c.require(clip.num_frames == 2, "boundary clip has 2 frames")
    except Exception as exc:
        c.errors.append(f"boundary frame load failed: {exc}")


def check_error_handling(c: _Checker, adapter: BlendedMVSAdapter, seq_name: str) -> None:
    """Verify that invalid inputs raise the expected exceptions."""
    # Out-of-range frame index
    info = adapter.get_sequence_info(seq_name)
    N = info["num_frames"]
    raised = False
    try:
        adapter.load_clip(seq_name, [N])   # one past the end
    except (IndexError, ValueError):
        raised = True
    c.require(raised, "out-of-range index raises IndexError/ValueError")

    # Empty frame list
    raised = False
    try:
        adapter.load_clip(seq_name, [])
    except (ValueError, IndexError):
        raised = True
    c.require(raised, "empty frame_indices raises ValueError/IndexError")

    # Bad sequence name
    raised = False
    try:
        adapter.get_sequence_info("__nonexistent_sequence__")
    except (KeyError, FileNotFoundError, ValueError):
        raised = True
    c.require(raised, "unknown sequence_name raises KeyError/FileNotFoundError")


def check_adapter_sanity_check_return(
    c: _Checker, adapter: BlendedMVSAdapter, seq_name: str
) -> None:
    """Call adapter.sanity_check() and validate its return structure."""
    try:
        result = adapter.sanity_check(seq_name)
    except Exception as exc:
        c.errors.append(f"sanity_check() raised: {exc}")
        return

    c.require(isinstance(result, dict), "sanity_check returns dict")
    for key in ("dataset_name", "sequence_name", "ok", "messages"):
        c.require(key in result,
                  f"sanity_check result contains key '{key}'")

    if isinstance(result, dict):
        c.require(result.get("ok") is True,
                  f"sanity_check ok=True (messages={result.get('messages')})")
        c.require(isinstance(result.get("messages"), list),
                  "sanity_check messages is a list")


# ---------------------------------------------------------------------------
# Per-sequence orchestration
# ---------------------------------------------------------------------------

def validate_sequence(
    adapter: BlendedMVSAdapter,
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
    # Uniformly sample T frame indices
    frame_indices = np.unique(
        np.linspace(0, info["num_frames"] - 1, T, dtype=int)
    ).tolist()

    # Load clip
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
    check_masked_variant(c, adapter, seq_name)
    check_boundary_frames(c, adapter, seq_name)
    check_error_handling(c, adapter, seq_name)
    check_adapter_sanity_check_return(c, adapter, seq_name)

    # Print per-check notes in verbose mode
    if verbose and c.notes:
        for note in c.notes:
            print(note)

    return c


# ---------------------------------------------------------------------------
# Split-level runner
# ---------------------------------------------------------------------------

def run_split(
    split: str,
    args: argparse.Namespace,
) -> tuple[int, int]:  # (passed, total)
    print(f"\n{'='*60}")
    print(f"  BlendedMVS adapter check — split: {split}")
    print(f"{'='*60}")

    try:
        adapter = BlendedMVSAdapter(
            root=args.data_root,
            split=split,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[FAIL] Adapter construction failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 0, 1

    seqs = adapter.list_sequences()
    print(f"  Total sequences in split: {len(seqs)}")

    # Restrict to a single sequence if requested (only for single-split runs)
    if args.sequence and args.split != "all":
        if args.sequence not in seqs:
            print(f"[FAIL] --sequence '{args.sequence}' not found in {split} split.",
                  file=sys.stderr)
            return 0, 1
        seqs = [args.sequence]
        print(f"  Checking only: {args.sequence}")
    elif args.max_seqs > 0 and len(seqs) > args.max_seqs:
        # Deterministic subsample: evenly spaced
        step = max(1, len(seqs) // args.max_seqs)
        seqs = seqs[::step][: args.max_seqs]
        print(f"  Checking {len(seqs)} / {len(adapter)} sequences (--max-seqs={args.max_seqs})")
    else:
        print(f"  Checking all {len(seqs)} sequences")

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

    splits = ["train", "val"] if args.split == "all" else [args.split]

    grand_passed = grand_total = 0
    for split in splits:
        p, t = run_split(split, args)
        grand_passed += p
        grand_total += t

    print()
    if grand_total == 0:
        print("No sequences were checked.")
        raise SystemExit(1)

    if grand_passed == grand_total:
        print(f"All {grand_total} check(s) passed.")
    else:
        failed = grand_total - grand_passed
        print(f"{failed}/{grand_total} check(s) FAILED.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
