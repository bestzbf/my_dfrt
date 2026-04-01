from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip, load_precomputed_fast


# ---------------------------------------------------------------------------
# Sequence index record
# ---------------------------------------------------------------------------

@dataclass
class _Co3DSequenceRecord:
    """Index entry for one Co3Dv2 sequence."""

    category: str
    sequence_name: str
    sequence_dir: Path
    # Frame numbers included in the chosen subset/split (sorted ascending).
    frame_numbers: list[int]

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def uid(self) -> str:
        """Globally unique sequence ID: '<category>/<sequence_name>'."""
        return f"{self.category}/{self.sequence_name}"


# ---------------------------------------------------------------------------
# Camera conversion helpers
# ---------------------------------------------------------------------------

def _ndc_to_pinhole(
    focal_length: list[float],
    principal_point: list[float],
    image_wh: list[int],
) -> np.ndarray:
    """Convert Co3D NDC camera intrinsics to a standard 3×3 pinhole matrix.

    Co3D stores intrinsics in PyTorch3D NDC (normalised device coordinates)
    with ``intrinsics_format="ndc_isotropic"``:

    - ``focal_length`` is normalised by ``min(W, H) / 2``
    - ``principal_point`` has (0, 0) at the image centre, **x pointing left,
      y pointing up** (opposite to the OpenCV pixel convention)

    We convert to OpenCV pixel-space intrinsics (origin at top-left corner,
    x pointing right, y pointing down).

    Args:
        focal_length:     [fx_ndc, fy_ndc]
        principal_point:  [px_ndc, py_ndc]   (NDC, (0,0) = centre)
        image_wh:         [width, height]

    Returns:
        K: (3, 3) float32 pinhole intrinsics in pixel coordinates.
    """
    # Co3D annotation stores image.size as [H, W] (height first).
    H, W = float(image_wh[0]), float(image_wh[1])
    half_s = min(H, W) / 2.0

    fx = focal_length[0] * half_s
    fy = focal_length[1] * half_s

    # NDC pp sign convention: pp_ndc_x = (W/2 - cx_px) / half_s
    #  → cx_px = W/2 - pp_ndc_x * half_s
    cx = W / 2.0 - principal_point[0] * half_s
    cy = H / 2.0 - principal_point[1] * half_s

    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _p3d_to_opencv_extrinsics(
    R_p3d: list[list[float]],
    T_p3d: list[float],
) -> np.ndarray:
    """Convert PyTorch3D camera pose to a standard 4×4 OpenCV-style w2c matrix.

    PyTorch3D uses **row-vector** convention:
        p_cam_p3d = p_world @ R + T

    and its camera space has **+x left, +y up, +z into scene**, whereas OpenCV
    uses **+x right, +y down, +z into scene**.

    Conversion:
        - Transpose R to switch to column-vector convention.
        - Apply axis flip D = diag(−1, −1, 1) to correct +x and +y directions.

        R_cv = D @ R_p3d.T
        T_cv = D @ T_p3d

    The resulting 4×4 matrix satisfies
        p_cam_cv = E_w2c @ p_world_h   (homogeneous column vectors, OpenCV).

    Args:
        R_p3d:  3×3 rotation as nested list (row-vector convention).
        T_p3d:  3-element translation as list.

    Returns:
        E: (4, 4) float32 world-to-camera matrix (OpenCV convention).
    """
    R = np.array(R_p3d, dtype=np.float64)   # (3, 3)
    T = np.array(T_p3d, dtype=np.float64)   # (3,)

    D = np.diag([-1.0, -1.0, 1.0])
    R_cv = (D @ R.T).astype(np.float32)
    T_cv = (D @ T).astype(np.float32)

    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_cv
    E[:3, 3] = T_cv
    return E


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _load_jgz(path: Path) -> Any:
    """Load a gzip-compressed JSON file."""
    with gzip.open(path, "rb") as f:
        return json.load(f)


def _load_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    """Load a Co3D 16-bit PNG depth map and scale to scene units.

    Co3D depth maps store values as uint16 in [0, 65535].  The depth in
    scene units is::

        depth = raw_uint16 / 65535.0 * scale_adjustment

    The resulting values are in the same unit as the camera extrinsics
    (Co3D scene units, consistent with PyTorch3D world coordinates).

    Args:
        path:             Path to the 16-bit PNG depth file.
        scale_adjustment: Per-frame scale factor from frame annotations.

    Returns:
        depth: (H, W) float32 array.  Zero indicates invalid / missing depth.
    """
    raw = np.array(Image.open(path), dtype=np.float32)
    return raw / 65535.0 * float(scale_adjustment)


def _load_depth_mask(path: Path) -> np.ndarray:
    """Load a Co3D depth validity mask.

    Returns:
        mask: (H, W) bool array.  True = valid depth.
    """
    return np.array(Image.open(path), dtype=bool)


def _load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as uint8 (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# All 51 Co3Dv2 categories.
ALL_CATEGORIES: list[str] = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car",
    "carrot", "cellphone", "chair", "couch", "cup", "donut", "frisbee",
    "hairdryer", "handbag", "hotdog", "hydrant", "keyboard", "kite", "laptop",
    "microwave", "motorcycle", "mouse", "orange", "parkingmeter", "pizza",
    "plant", "remote", "sandwich", "skateboard", "stopsign", "suitcase",
    "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain",
    "toytruck", "tv", "umbrella", "vase", "wineglass",
]

# Maps subset_name → (set_lists filename suffix, split key inside that file).
_SUBSET_MAP: dict[str, tuple[str, str]] = {
    "fewview_train":    ("fewview_train",    "train"),
    "fewview_dev":      ("fewview_dev",      "val"),
    "fewview_test":     ("fewview_test",     "test"),
    "manyview_dev_0":   ("manyview_dev_0",   "val"),
    "manyview_dev_1":   ("manyview_dev_1",   "val"),
    "manyview_test_0":  ("manyview_test_0",  "test"),
}


class Co3Dv2Adapter(BaseAdapter):
    """Dataset adapter for Co3D version 2.

    Dataset layout (per category)::

        <root>/
            <category>/
                frame_annotations.jgz
                sequence_annotations.jgz
                set_lists/
                    set_lists_fewview_train.json   # {train/val/test: [[seq,frame,path], ...]}
                    set_lists_fewview_dev.json
                    ...
                <sequence_name>/
                    images/
                        frameXXXXXX.jpg
                    depths/
                        frameXXXXXX.jpg.geometric.png   (uint16, depth in scene units)
                    depth_masks/
                        frameXXXXXX.png                 (bool, valid depth)
                    masks/
                        frameXXXXXX.png                 (bool, object foreground)
                    pointcloud.ply

    Sequence ID
    -----------
    A sequence in Co3D corresponds to a video of a single object category.
    The unique ID used by this adapter is ``"<category>/<sequence_name>"``.

    Camera convention
    -----------------
    PyTorch3D cameras (row-vector convention, +x left, +y up) are converted to
    standard OpenCV extrinsics (column-vector convention, +x right, +y down,
    world-to-camera).

    Intrinsics
    ----------
    NDC intrinsics (``ndc_isotropic`` format) are converted to pixel-space
    pinhole intrinsics.

    Depth
    -----
    ``depth = raw_uint16 / 65535.0 * scale_adjustment``

    The depth mask is applied in-place (invalid pixels set to 0.0).
    Depth is in the same units as the camera extrinsics (Co3D scene units).

    Supervision availability
    ------------------------
    When a ``precomputed.npz`` or ``precomputed.h5`` file is present inside
    the sequence directory (i.e. ``<root>/<category>/<sequence>/precomputed.*``),
    all track-based supervision is enabled:

    - ``has_depth``      : True  (always)
    - ``has_normals``    : True  (from precomputed file)
    - ``has_tracks``     : True  (from precomputed file)
    - ``has_visibility`` : True  (from precomputed file)

    If no precomputed file is found:

    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : False
    - ``has_visibility`` : False
    """

    dataset_name: str = "co3dv2"

    def __init__(
        self,
        root: str,
        categories: Optional[list[str]] = None,
        subset_name: str = "fewview_train",
        split: str = "train",
        min_frames: int = 2,
        strict: bool = False,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the Co3Dv2 dataset.
        categories :
            List of category names to include.  ``None`` → all 51 categories.
        subset_name :
            Which subset to use.  One of:
            ``"fewview_train"``, ``"fewview_dev"``, ``"fewview_test"``,
            ``"manyview_dev_0"``, ``"manyview_dev_1"``, ``"manyview_test_0"``.
        split :
            Split key inside the set_lists file: ``"train"``, ``"val"``,
            or ``"test"``.  Defaults to ``"train"``.
        min_frames :
            Sequences with fewer than this many frames in the chosen split are
            skipped.
        strict :
            If *True*, raise on any error during index construction.
            If *False*, skip broken categories/sequences with a warning.
        verbose :
            Print a summary line after index construction.
        """
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Co3Dv2 root not found: {self.root}")

        if subset_name not in _SUBSET_MAP:
            raise ValueError(
                f"Unknown subset_name '{subset_name}'. "
                f"Valid options: {sorted(_SUBSET_MAP.keys())}"
            )

        self.categories: list[str] = (
            list(categories) if categories is not None else list(ALL_CATEGORIES)
        )
        self.subset_name = subset_name
        self.split = split
        self.min_frames = min_frames
        self.strict = strict
        self.verbose = verbose
        self.precompute_root = Path(precompute_root) if precompute_root else self.root

        # category → {(seq_name, frame_number): frame_annotation_dict}
        # Populated lazily on first access.
        self._frame_anno_cache: dict[str, dict[tuple[str, int], dict]] = {}

        self._records: list[_Co3DSequenceRecord] = self._build_index()
        self._uid_to_record: dict[str, _Co3DSequenceRecord] = {
            r.uid: r for r in self._records
        }

        if not self._records:
            raise RuntimeError(
                f"No valid Co3Dv2 sequences found. "
                f"categories={self.categories}, subset={subset_name}, split={split}"
            )

        if self.verbose:
            print(
                f"[Co3Dv2Adapter] subset={subset_name!r}, split={split!r}, "
                f"categories={len(self.categories)}, "
                f"sequences={len(self._records)}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.uid for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].uid

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        # Retrieve image size from the first frame's annotation.
        frame_anno = self._get_frame_anno(r.category, r.sequence_name, r.frame_numbers[0])
        H, W = frame_anno["image"]["size"]   # annotation stores [H, W]

        # Check whether precomputed tracks are available for this sequence.
        has_precomputed = False
        if self.precompute_root is not None:
            npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            h5_path  = npz_path.with_suffix(".h5")
            has_precomputed = h5_path.exists() or npz_path.exists()

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "subset_name": self.subset_name,
            "sequence_name": r.uid,
            "category": r.category,
            "sequence_root": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "height": H,
            "width": W,
            "has_depth": True,
            "has_normals": has_precomputed,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
            "has_trajs_3d_world": has_precomputed,
            "extrinsics_convention": "w2c",
            "depth_unit": "co3d_scene_units",
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        """Load a clip from a Co3Dv2 sequence.

        Parameters
        ----------
        sequence_name :
            Unique sequence ID in ``"<category>/<sequence_name>"`` format.
        frame_indices :
            Positions into the sequence's sorted frame list.  Must be within
            ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            If a ``precomputed.npz`` / ``.h5`` file is found next to the
            sequence directory, ``normals``, ``trajs_2d``, ``trajs_3d_world``,
            ``valids``, and ``visibs`` are populated from it and
            ``metadata["has_tracks"]`` is set to ``True``.
            Otherwise these fields are ``None`` and ``has_tracks`` is ``False``.
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        frame_paths: list[str] = []
        frame_numbers_clip: list[int] = []
        image_sizes: list[tuple[int, int]] = []

        for idx in frame_indices:
            fn = r.frame_numbers[idx]
            anno = self._get_frame_anno(r.category, r.sequence_name, fn)

            img_path = self.root / anno["image"]["path"]
            dep_path = self.root / anno["depth"]["path"]
            dep_mask_path = self.root / anno["depth"]["mask_path"]

            frame_paths.append(str(img_path))
            frame_numbers_clip.append(fn)

            images.append(_load_rgb(img_path))

            depth = _load_depth(dep_path, anno["depth"]["scale_adjustment"])
            if dep_mask_path.exists():
                dep_mask = _load_depth_mask(dep_mask_path)
                depth[~dep_mask] = 0.0
            depths.append(depth)

            H, W = anno["image"]["size"]   # annotation stores [H, W]
            image_sizes.append((H, W))

            K = _ndc_to_pinhole(
                anno["viewpoint"]["focal_length"],
                anno["viewpoint"]["principal_point"],
                anno["image"]["size"],
            )
            intrinsics_list.append(K)

            E = _p3d_to_opencv_extrinsics(
                anno["viewpoint"]["R"],
                anno["viewpoint"]["T"],
            )
            extrinsics_list.append(E)

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3) float32
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) float32

        # Load precomputed normals / tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        has_normals_out, has_tracks_out = False, False
        if self.precompute_root is not None:
            npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            cache = load_precomputed_fast(npz_path, frame_indices)
            if cache is not None:
                # load_precomputed_fast already reorders data to frame_indices order,
                # so we index with consecutive [0, 1, ..., len-1] here.
                T_clip = len(frame_indices)
                # normals: shape (T_clip, H, W, 3) → list of (H,W,3)
                normals_raw = cache["normals"]  # (T_clip, H, W, 3) float16
                normals_out = [
                    normals_raw[i].astype(np.float32) for i in range(T_clip)
                ]
                trajs_2d_out    = cache["trajs_2d"]       # (T_clip, N, 2)
                trajs_3d_out    = cache["trajs_3d_world"]  # (T_clip, N, 3)
                valids_out      = cache["valids"]          # (T_clip, N)
                visibs_out      = cache["visibs"]          # (T_clip, N)
                has_normals_out = True
                has_tracks_out  = True

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "category": r.category,
                "split": self.split,
                "subset_name": self.subset_name,
                "sequence_root": str(r.sequence_dir),
                "frame_indices": list(frame_indices),
                "frame_numbers": frame_numbers_clip,
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": list(image_sizes[0]) if image_sizes else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "depth_unit": "co3d_scene_units",
                "depth_note": (
                    "depth = raw_uint16 / 65535 * scale_adjustment; "
                    "consistent with camera extrinsics (Co3D scene units)"
                ),
            },
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Image, depth, and depth-mask files exist for the first 5 frames.
        2. Intrinsics shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite; at least some positive values per frame.
        5. Round-trip reprojection: unproject a valid depth pixel, re-project
           back, check error < 1 px.

        Returns
        -------
        dict with ``dataset_name``, ``sequence_name``, ``ok`` (bool),
        ``messages`` (list[str]).
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        # ── 1. File existence (first 5 frames) ──────────────────────────
        probe_count = min(5, r.num_frames)
        for i in range(probe_count):
            fn = r.frame_numbers[i]
            try:
                anno = self._get_frame_anno(r.category, r.sequence_name, fn)
            except KeyError as exc:
                ok = False
                msgs.append(f"frame {fn}: annotation lookup failed: {exc}")
                continue

            for key, subpath in [
                ("image", anno["image"]["path"]),
                ("depth", anno["depth"]["path"]),
                ("depth_mask", anno["depth"]["mask_path"]),
            ]:
                p = self.root / subpath
                if not p.exists():
                    ok = False
                    msgs.append(f"frame {fn}: missing {key} file: {p}")

        if not ok:
            return {"dataset_name": self.dataset_name, "sequence_name": sequence_name,
                    "ok": False, "messages": msgs}

        # ── Load probe clip ──────────────────────────────────────────────
        try:
            clip = self.load_clip(sequence_name, list(range(probe_count)))
        except Exception as exc:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs + [f"load_clip failed: {repr(exc)}"],
            }

        T = probe_count

        # ── 2. Intrinsics ───────────────────────────────────────────────
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(f"intrinsics shape {clip.intrinsics.shape} != ({T},3,3)")
        elif not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")
        else:
            fx = clip.intrinsics[:, 0, 0]
            fy = clip.intrinsics[:, 1, 1]
            if not (np.all(fx > 0) and np.all(fy > 0)):
                ok = False
                msgs.append(f"non-positive focal lengths: fx={fx}, fy={fy}")

        # ── 3. Extrinsics ───────────────────────────────────────────────
        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(f"extrinsics shape {clip.extrinsics.shape} != ({T},4,4)")
        elif not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")
        else:
            # Last row == [0,0,0,1]
            bottom = clip.extrinsics[:, 3, :]
            if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
                ok = False
                msgs.append("extrinsics last row != [0,0,0,1]")
            # Rotation orthonormality
            R = clip.extrinsics[:, :3, :3].astype(np.float64)
            RRt = R @ np.transpose(R, (0, 2, 1))
            err = float(np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2)).max())
            if err > 1e-4:
                ok = False
                msgs.append(f"rotation matrices non-orthonormal: max ||RR^T-I||_F={err:.2e}")

        # ── 4. Depth ────────────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")

        # ── 5. Reprojection ─────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)

            ys, xs = np.where(dep > 0)
            if ys.size == 0:
                msgs.append(f"reproj check skipped frame {t}: no valid depth")
                continue

            # Sample centre-ish pixel
            mid = ys.size // 2
            u, v = int(xs[mid]), int(ys[mid])
            d = float(dep[v, u])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            xc = (u - cx) / fx * d
            yc = (v - cy) / fy * d
            p_cam = np.array([xc, yc, d, 1.0])

            try:
                p_world = np.linalg.inv(E) @ p_cam
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_cam2 = E @ p_world
            p_2d = K @ p_cam2[:3]
            u2 = p_2d[0] / p_2d[2]
            v2 = p_2d[1] / p_2d[2]
            reproj_err = float(np.hypot(u2 - u, v2 - v))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reproj error frame {t}: ({u2:.2f},{v2:.2f}) vs ({u},{v}), "
                    f"err={reproj_err:.4f}px"
                )

        if ok and not msgs:
            msgs.append("all checks passed")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
            "num_frames": r.num_frames,
            "probe_frames": probe_count,
        }

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_Co3DSequenceRecord]:
        file_suffix, split_key = _SUBSET_MAP[self.subset_name]
        records: list[_Co3DSequenceRecord] = []
        skipped_cats: list[str] = []

        for cat in self.categories:
            cat_dir = self.root / cat
            if not cat_dir.is_dir():
                msg = f"category directory not found: {cat_dir}"
                if self.strict:
                    raise FileNotFoundError(msg)
                skipped_cats.append(f"{cat}: {msg}")
                continue

            set_lists_path = cat_dir / "set_lists" / f"set_lists_{file_suffix}.json"
            if not set_lists_path.exists():
                msg = f"set_lists file not found: {set_lists_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                skipped_cats.append(f"{cat}: {msg}")
                continue

            try:
                cat_records = self._index_category(
                    cat, cat_dir, set_lists_path, split_key
                )
                records.extend(cat_records)
            except Exception as exc:
                if self.strict:
                    raise
                skipped_cats.append(f"{cat}: {exc}")
                if self.verbose:
                    print(f"[Co3Dv2Adapter][WARN] skip category {cat}: {exc}")

        if self.verbose and skipped_cats:
            print(
                f"[Co3Dv2Adapter] skipped {len(skipped_cats)} categories "
                "(non-strict mode)"
            )

        return records

    def _index_category(
        self,
        cat: str,
        cat_dir: Path,
        set_lists_path: Path,
        split_key: str,
    ) -> list[_Co3DSequenceRecord]:
        with open(set_lists_path, "r") as f:
            set_lists = json.load(f)

        if split_key not in set_lists:
            raise KeyError(
                f"split key '{split_key}' not found in {set_lists_path}. "
                f"Available keys: {list(set_lists.keys())}"
            )

        # Group (sequence_name, frame_number) entries by sequence.
        seq_to_frames: dict[str, list[int]] = {}
        for seq_name, frame_number, _img_path in set_lists[split_key]:
            seq_to_frames.setdefault(seq_name, []).append(int(frame_number))

        records: list[_Co3DSequenceRecord] = []
        for seq_name, frame_numbers in seq_to_frames.items():
            frame_numbers_sorted = sorted(set(frame_numbers))
            if len(frame_numbers_sorted) < self.min_frames:
                continue

            seq_dir = cat_dir / seq_name
            if not seq_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(
                        f"sequence directory not found: {seq_dir}"
                    )
                continue

            records.append(
                _Co3DSequenceRecord(
                    category=cat,
                    sequence_name=seq_name,
                    sequence_dir=seq_dir,
                    frame_numbers=frame_numbers_sorted,
                )
            )

        return records

    # ------------------------------------------------------------------
    # Frame annotation lazy loader
    # ------------------------------------------------------------------

    def _ensure_frame_anno_loaded(self, category: str) -> None:
        """Load and cache frame_annotations.jgz for *category* if not already done."""
        if category in self._frame_anno_cache:
            return

        path = self.root / category / "frame_annotations.jgz"
        if not path.exists():
            raise FileNotFoundError(f"frame_annotations.jgz not found: {path}")

        annotations = _load_jgz(path)
        # Index by (sequence_name, frame_number) for O(1) lookup.
        index: dict[tuple[str, int], dict] = {}
        for entry in annotations:
            key = (entry["sequence_name"], int(entry["frame_number"]))
            index[key] = entry

        self._frame_anno_cache[category] = index

    def _get_frame_anno(
        self, category: str, sequence_name: str, frame_number: int
    ) -> dict:
        self._ensure_frame_anno_loaded(category)
        key = (sequence_name, int(frame_number))
        cache = self._frame_anno_cache[category]
        if key not in cache:
            raise KeyError(
                f"No annotation for ({category}/{sequence_name}, frame {frame_number})"
            )
        return cache[key]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _Co3DSequenceRecord:
        if sequence_name not in self._uid_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._uid_to_record[sequence_name]

    def _check_indices(
        self, frame_indices: list[int], num_frames: int, sequence_name: str
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
