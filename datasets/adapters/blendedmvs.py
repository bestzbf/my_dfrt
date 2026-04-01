from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip


# ---------------------------------------------------------------------------
# PFM I/O
# ---------------------------------------------------------------------------

def _read_pfm(path: Path) -> np.ndarray:
    """Read a PFM (Portable Float Map) file and return a float32 2-D array.

    PFM layout (text header + binary payload):
      Line 1 : "PF" (3-channel) or "Pf" (single-channel)
      Line 2 : "<width> <height>"
      Line 3 : "<scale>"   negative  → little-endian, positive → big-endian
      Payload : height × width × channels float32 values, stored bottom-up.

    BlendedMVS depth maps are single-channel Pf files.
    """
    with open(path, "rb") as f:
        magic = f.readline().decode("latin-1").strip()
        if magic not in ("PF", "Pf"):
            raise ValueError(f"Not a valid PFM file (magic='{magic}'): {path}")

        dims = f.readline().decode("latin-1").strip()
        width, height = map(int, dims.split())

        scale_raw = float(f.readline().decode("latin-1").strip())
        little_endian = scale_raw < 0
        scale = abs(scale_raw)

        channels = 3 if magic == "PF" else 1
        dtype = np.dtype("<f4" if little_endian else ">f4")
        data = np.frombuffer(f.read(), dtype=dtype)

    expected = height * width * channels
    if data.size != expected:
        raise ValueError(
            f"PFM payload size mismatch (got {data.size}, expected {expected}): {path}"
        )

    data = data.reshape((height, width, channels) if channels > 1 else (height, width))
    data = np.flipud(data)  # PFM stored bottom-up

    if scale != 1.0:
        data = data * scale

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Camera-file parser
# ---------------------------------------------------------------------------

def _parse_cam_file(path: Path) -> dict[str, Any]:
    """Parse a BlendedMVS / MVSNet-style camera parameter file.

    File format::

        extrinsic
        r00 r01 r02 t0
        r10 r11 r12 t1
        r20 r21 r22 t2
        0.0 0.0 0.0 1.0

        intrinsic
        fx  0   cx
        0   fy  cy
        0   0   1.0

        depth_min depth_interval depth_num depth_max

    The ``extrinsic`` matrix is the **world-to-camera** (w2c) transform, i.e.
    ``p_cam = E @ p_world_homogeneous``.

    Returns
    -------
    dict with keys:
        ``extrinsic``       : (4, 4) float32 world-to-camera matrix
        ``intrinsic``       : (3, 3) float32 camera intrinsic matrix
        ``depth_min``       : float
        ``depth_interval``  : float
        ``depth_num``       : float  (number of depth hypotheses / max range)
        ``depth_max``       : float
    """
    with open(path, "r") as f:
        # Filter out blank lines so indices are deterministic.
        lines = [ln.strip() for ln in f if ln.strip()]

    # lines[0] == "extrinsic"
    # lines[1..4] == 4 rows of the 4×4 matrix
    extrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(1, 5)],
        dtype=np.float32,
    )

    # lines[5] == "intrinsic"
    # lines[6..8] == 3 rows of the 3×3 matrix
    intrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(6, 9)],
        dtype=np.float32,
    )

    # lines[9] == depth params
    depth_params = [float(v) for v in lines[9].split()]

    return {
        "extrinsic": extrinsic,          # (4, 4)  w2c
        "intrinsic": intrinsic,          # (3, 3)
        "depth_min": depth_params[0],
        "depth_interval": depth_params[1],
        "depth_num": depth_params[2],
        "depth_max": depth_params[3],
    }


# ---------------------------------------------------------------------------
# Sequence record
# ---------------------------------------------------------------------------

class _SequenceRecord:
    """Index entry for a single BlendedMVS scene."""

    __slots__ = (
        "scene_id",
        "scene_dir",
        "frame_ids",     # list[str]  e.g. ["00000000", "00000001", ...]
        "num_frames",
    )

    def __init__(
        self,
        scene_id: str,
        scene_dir: Path,
        frame_ids: list[str],
    ) -> None:
        self.scene_id = scene_id
        self.scene_dir = scene_dir
        self.frame_ids = frame_ids
        self.num_frames = len(frame_ids)

    def rgb_path(self, frame_id: str, masked: bool = False) -> Path:
        suffix = "_masked.jpg" if masked else ".jpg"
        return self.scene_dir / "blended_images" / f"{frame_id}{suffix}"

    def depth_path(self, frame_id: str) -> Path:
        return self.scene_dir / "rendered_depth_maps" / f"{frame_id}.pfm"

    def cam_path(self, frame_id: str) -> Path:
        return self.scene_dir / "cams" / f"{frame_id}_cam.txt"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_CAM_FILENAME_RE = re.compile(r"^(\d+)_cam\.txt$")


class BlendedMVSAdapter(BaseAdapter):
    """Dataset adapter for BlendedMVS.

    Expected dataset layout::

        <root>/
            BlendedMVS_training.txt     # one scene ID per line (training split)
            validation_list.txt         # one scene ID per line (val split)
            <scene_id>/
                blended_images/
                    00000000.jpg
                    00000000_masked.jpg
                    00000001.jpg
                    ...
                cams/
                    00000000_cam.txt
                    00000001_cam.txt
                    ...
                rendered_depth_maps/
                    00000000.pfm
                    00000001.pfm
                    ...

    Frames are indexed by the camera files present in ``cams/``.
    Each frame maps to one RGB image, one depth map, and one camera file.

    Supervision availability
    ------------------------
    - ``has_depth``      : True   (rendered depth maps in PFM format)
    - ``has_normals``    : False  (not provided by BlendedMVS)
    - ``has_tracks``     : False  (no point-track annotations)
    - ``has_visibility`` : False  (no per-point visibility annotation)

    Extrinsics convention
    ---------------------
    BlendedMVS camera files store **world-to-camera** (w2c) transforms:
    ``p_cam = E @ p_world_homogeneous``.
    This matches the ``"w2c"`` convention used by the rest of the D4RT adapter
    family (see ScanNetAdapter).
    """

    dataset_name: str = "blendedmvs"

    # split-name → list file inside the dataset root
    _SPLIT_LIST_FILES: dict[str, str] = {
        "train": "BlendedMVS_training.txt",
        "val": "validation_list.txt",
        "valid": "validation_list.txt",
        "validation": "validation_list.txt",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        use_masked: bool = False,
        strict: bool = True,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the BlendedMVS dataset.
        split :
            One of ``"train"``, ``"val"`` / ``"valid"`` / ``"validation"``.
        use_masked :
            When *True*, load ``XXXXXXXX_masked.jpg`` instead of
            ``XXXXXXXX.jpg`` as the RGB input.
        strict :
            When *True*, raise on any scene that cannot be indexed.
            When *False*, skip broken scenes with a warning.
        verbose :
            Print summary information during construction.
        """
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"BlendedMVS root not found: {self.root}")

        split_key = split.lower()
        if split_key not in self._SPLIT_LIST_FILES:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Valid splits: {sorted(self._SPLIT_LIST_FILES.keys())}"
            )
        self.split = split_key
        self.use_masked = use_masked
        self.strict = strict
        self.verbose = verbose
        self.precompute_root = Path(precompute_root) if precompute_root else self.root

        self._records: list[_SequenceRecord] = self._build_index()
        self._name_to_record: dict[str, _SequenceRecord] = {
            r.scene_id: r for r in self._records
        }

        if len(self._records) == 0:
            raise RuntimeError(
                f"No valid BlendedMVS scenes found for split='{split}' under {self.root}"
            )

        if self.verbose:
            print(
                f"[BlendedMVSAdapter] split={self.split!r}, "
                f"num_sequences={len(self._records)}, "
                f"use_masked={self.use_masked}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.scene_id for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].scene_id

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        # Probe image size from first frame.
        h, w = self._read_image_size(r.rgb_path(r.frame_ids[0], self.use_masked))
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": r.scene_id,
            "sequence_root": str(r.scene_dir),
            "num_frames": r.num_frames,
            "height": h,
            "width": w,
            "has_depth": True,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
            "has_trajs_3d_world": False,
            "extrinsics_convention": "w2c",
            "depth_unit": "meters",
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        """Load a clip from a BlendedMVS scene.

        Parameters
        ----------
        sequence_name :
            Scene ID (24-character hex string).
        frame_indices :
            List of integer frame indices into the sorted camera file list for
            this scene.  Must be within ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals``, ``trajs_2d``, ``trajs_3d_world``, ``valids``, and
            ``visibs`` are all ``None`` (not available in BlendedMVS).
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        frame_paths: list[str] = []
        depth_ranges: list[tuple[float, float]] = []

        for idx in frame_indices:
            fid = r.frame_ids[idx]

            rgb_p = r.rgb_path(fid, self.use_masked)
            dep_p = r.depth_path(fid)
            cam_p = r.cam_path(fid)

            frame_paths.append(str(rgb_p))

            images.append(self._read_image(rgb_p))
            depths.append(_read_pfm(dep_p))

            cam = _parse_cam_file(cam_p)
            intrinsics_list.append(cam["intrinsic"])    # (3, 3) float32
            extrinsics_list.append(cam["extrinsic"])    # (4, 4) float32 w2c
            depth_ranges.append((cam["depth_min"], cam["depth_max"]))

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3) float32
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) float32 w2c

        # Load precomputed normals / tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        has_normals_out, has_tracks_out = False, False
        if self.precompute_root is not None:
            cache = self._load_precomputed(sequence_name, frame_indices)
            if cache is not None:
                normals_out     = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out    = cache["trajs_2d"]
                trajs_3d_out    = cache["trajs_3d_world"]
                valids_out      = cache["valids"]
                visibs_out      = cache["visibs"]
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
                "split": self.split,
                "sequence_root": str(r.scene_dir),
                "frame_indices": list(frame_indices),
                "frame_ids": [r.frame_ids[i] for i in frame_indices],
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": list(images[0].shape[:2]) if images else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
                "depth_ranges": depth_ranges,  # [(depth_min, depth_max), ...]
                "use_masked": self.use_masked,
            },
        )

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed data for frame_indices. Prefers .h5 over .npz."""
        from datasets.adapters.base import load_precomputed_fast
        path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = path.with_suffix('.h5')
        if not path.exists() and not h5_path.exists():
            return None
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, 'r') as f:
                n = f['trajs_2d'].shape[0] if 'trajs_2d' in f else 0
        else:
            n = int(np.load(path, allow_pickle=False)["num_frames"])
        if n < max(frame_indices) + 1:
            return None
        return load_precomputed_fast(path, frame_indices)

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Camera, image, and depth files exist for all frames.
        2. Intrinsics shape correct: (T, 3, 3).
        3. Extrinsics shape correct: (T, 4, 4).
        4. All intrinsic/extrinsic values are finite.
        5. Extrinsic rotation matrices are non-singular.
        6. Depth values are finite and contain at least some positive values.
        7. Round-trip reprojection: unproject center pixel with its depth,
           re-project, check reprojection error < 1 pixel (sampled from first
           5 frames).

        Returns
        -------
        dict with keys:
            ``dataset_name``, ``sequence_name``, ``ok`` (bool),
            ``messages`` (list[str])
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        # ── 1. File existence (all frames) ──────────────────────────────
        missing_rgb = []
        missing_dep = []
        missing_cam = []
        for fid in r.frame_ids:
            if not r.rgb_path(fid, self.use_masked).exists():
                missing_rgb.append(fid)
            if not r.depth_path(fid).exists():
                missing_dep.append(fid)
            if not r.cam_path(fid).exists():
                missing_cam.append(fid)

        for modal, lst in [("rgb", missing_rgb), ("depth", missing_dep), ("cam", missing_cam)]:
            if lst:
                ok = False
                msgs.append(
                    f"missing {modal} files ({len(lst)}): {lst[:5]}"
                    + (" ..." if len(lst) > 5 else "")
                )

        if not ok:
            return {"dataset_name": self.dataset_name, "sequence_name": sequence_name,
                    "ok": False, "messages": msgs}

        # ── Load a short probe clip (up to 5 frames) ────────────────────
        probe_count = min(5, r.num_frames)
        probe_indices = list(range(probe_count))
        try:
            clip = self.load_clip(sequence_name, probe_indices)
        except Exception as exc:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs + [f"load_clip failed: {repr(exc)}"],
            }

        T = probe_count

        # ── 2 & 3. Shape checks ─────────────────────────────────────────
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(f"intrinsics shape {clip.intrinsics.shape} != ({T}, 3, 3)")

        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(f"extrinsics shape {clip.extrinsics.shape} != ({T}, 4, 4)")

        # ── 4. Finite checks ─────────────────────────────────────────────
        if not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")

        if not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")

        # ── 5. Rotation non-singularity ──────────────────────────────────
        for t in range(T):
            det = float(np.linalg.det(clip.extrinsics[t, :3, :3]))
            if abs(det) < 1e-6:
                ok = False
                msgs.append(
                    f"extrinsics[{t}] rotation det ≈ {det:.2e} (nearly singular)"
                )

        # ── 6. Depth sanity ──────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")
            if (dep > 0).sum() == 0:
                ok = False
                msgs.append(f"depth[{t}] has no positive values")

        # ── 7. Round-trip reprojection ───────────────────────────────────
        for t in range(T):
            img = clip.images[t]
            H, W = img.shape[:2]
            K = clip.intrinsics[t]          # (3, 3) w2c
            E = clip.extrinsics[t]          # (4, 4) w2c
            dep = clip.depths[t]            # (H, W)

            cy_px, cx_px = H // 2, W // 2
            d = float(dep[cy_px, cx_px])

            if not (np.isfinite(d) and d > 0):
                msgs.append(
                    f"reprojection check skipped for frame {t}: "
                    "depth at image center is invalid"
                )
                continue

            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # Unproject center pixel to camera space.
            x_c = (cx_px - cx) / fx * d
            y_c = (cy_px - cy) / fy * d
            p_cam = np.array([x_c, y_c, d, 1.0], dtype=np.float64)

            # To world (E is w2c → invert).
            try:
                E_inv = np.linalg.inv(E.astype(np.float64))
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] is singular — cannot invert")
                continue

            p_world = E_inv @ p_cam
            # Back to camera.
            p_cam2 = E.astype(np.float64) @ p_world
            # Project.
            p_proj = K.astype(np.float64) @ p_cam2[:3]
            u = p_proj[0] / p_proj[2]
            v = p_proj[1] / p_proj[2]

            reproj_err = max(abs(u - cx_px), abs(v - cy_px))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reprojection error too large at frame {t}: "
                    f"({u:.2f}, {v:.2f}) vs center ({cx_px}, {cy_px}), "
                    f"max err={reproj_err:.4f} px"
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
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_SequenceRecord]:
        list_filename = self._SPLIT_LIST_FILES[self.split]
        list_path = self.root / list_filename

        if not list_path.exists():
            raise FileNotFoundError(
                f"Split list file not found: {list_path}\n"
                f"Expected a text file with one scene ID per line."
            )

        with open(list_path, "r") as f:
            scene_ids = [ln.strip() for ln in f if ln.strip()]

        if not scene_ids:
            raise RuntimeError(f"Split list is empty: {list_path}")

        records: list[_SequenceRecord] = []
        skipped: list[str] = []

        for scene_id in scene_ids:
            scene_dir = self.root / scene_id
            try:
                rec = self._index_scene(scene_id, scene_dir)
                if rec is not None:
                    records.append(rec)
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"{scene_id}: {exc}")
                if self.verbose:
                    print(f"[BlendedMVSAdapter][WARN] skip {scene_id}: {exc}")

        if self.verbose and skipped:
            print(f"[BlendedMVSAdapter] skipped {len(skipped)} scenes (non-strict mode)")

        return records

    def _index_scene(
        self, scene_id: str, scene_dir: Path
    ) -> Optional[_SequenceRecord]:
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

        cam_dir = scene_dir / "cams"
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Missing cams/ in scene: {scene_dir}")

        # Enumerate camera files; sort by frame index.
        cam_files = [
            p for p in cam_dir.iterdir()
            if p.is_file() and _CAM_FILENAME_RE.match(p.name)
        ]
        if not cam_files:
            raise RuntimeError(f"No camera files found in {cam_dir}")

        cam_files.sort(key=lambda p: int(_CAM_FILENAME_RE.match(p.name).group(1)))
        frame_ids = [_CAM_FILENAME_RE.match(p.name).group(1) for p in cam_files]

        return _SequenceRecord(
            scene_id=scene_id,
            scene_dir=scene_dir,
            frame_ids=frame_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _SequenceRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._name_to_record[sequence_name]

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

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        """Load an RGB image as uint8 (H, W, 3)."""
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def _read_image_size(path: Path) -> tuple[int, int]:
        img = Image.open(path)
        return img.height, img.width
