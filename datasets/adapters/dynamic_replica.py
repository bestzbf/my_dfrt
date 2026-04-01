from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip


# ---------------------------------------------------------------------------
# Camera conversion helpers (identical to Co3Dv2, shared PyTorch3D format)
# ---------------------------------------------------------------------------

def _ndc_to_pinhole(
    focal_length: list[float],
    principal_point: list[float],
    image_size: list[int],
) -> np.ndarray:
    """Convert PyTorch3D NDC intrinsics to a standard 3×3 pinhole matrix.

    Dynamic_Replica uses the same ``ndc_isotropic`` format as Co3Dv2:

    - ``focal_length`` normalised by ``min(W, H) / 2``
    - ``principal_point`` has (0, 0) at image centre, x pointing left,
      y pointing up (opposite to OpenCV pixel convention)

    Args:
        focal_length:    [fx_ndc, fy_ndc]
        principal_point: [px_ndc, py_ndc]   ((0, 0) = centre)
        image_size:      [H, W]  (annotation stores height first)

    Returns:
        K: (3, 3) float32 pinhole intrinsics in pixel coordinates.
    """
    H, W = float(image_size[0]), float(image_size[1])
    half_s = min(H, W) / 2.0

    fx = focal_length[0] * half_s
    fy = focal_length[1] * half_s

    # NDC pp sign: pp_ndc_x = (W/2 - cx_px) / half_s  →  cx_px = W/2 - pp_ndc_x * half_s
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
    """Convert PyTorch3D camera pose to a 4×4 OpenCV-style w2c matrix.

    PyTorch3D uses row-vector convention (+x left, +y up, +z into scene).
    We convert to OpenCV column-vector convention (+x right, +y down, +z forward).

        R_cv = diag(-1, -1, 1) @ R_p3d.T
        T_cv = diag(-1, -1, 1) @ T_p3d

    Returns:
        E: (4, 4) float32 world-to-camera matrix (OpenCV convention, w2c).
    """
    R = np.array(R_p3d, dtype=np.float64)
    T = np.array(T_p3d, dtype=np.float64)

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
    with gzip.open(path, "rb") as f:
        return json.load(f)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    """Load a 16-bit PNG depth map and scale to scene units.

    Encoding (same as Co3Dv2):
        depth = raw_uint16 / 65535.0 * scale_adjustment

    For Dynamic_Replica, scale_adjustment is always 1.0, giving depth values
    in PyTorch3D world units (consistent with the camera extrinsics).
    Pixels with raw=0 are invalid; depth is returned as 0.0 for those.

    Returns:
        (H, W) float32 depth array.  Zero indicates invalid / missing depth.
    """
    raw = np.array(Image.open(path), dtype=np.float32)
    return raw / 65535.0 * float(scale_adjustment)


def _load_trajectory_pth(path: Path) -> dict[str, np.ndarray]:
    """Load a per-frame trajectory .pth file (PyTorch zip-archive format).

    Each file stores data for the N consistently-indexed mesh vertices:
        traj_3d_world : (N, 3) float32  – 3D world positions at this frame
        traj_2d       : (N, 3) float32  – 2D pixel positions (cols 0–1 are u,v)
        verts_inds_vis: (N,)   bool     – per-vertex visibility flag

    Returns:
        dict with keys "traj_3d_world", "traj_2d", "verts_inds_vis".

    Raises:
        ImportError  if PyTorch is not installed.
        FileNotFoundError  if the .pth file does not exist.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load Dynamic_Replica trajectory files. "
            "Install it or set load_trajectories=False."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    return {
        "traj_3d_world": data["traj_3d_world"].numpy().astype(np.float32),  # (N, 3)
        "traj_2d": data["traj_2d"].numpy().astype(np.float32),              # (N, 3)
        "verts_inds_vis": data["verts_inds_vis"].numpy().astype(bool),      # (N,)
    }


# ---------------------------------------------------------------------------
# Sequence index record
# ---------------------------------------------------------------------------

@dataclass
class _DRSequenceRecord:
    """Index entry for one Dynamic_Replica sequence (= one camera track)."""

    sequence_name: str        # full dir name, e.g. "009850-3_obj_source_left"
    base_seq_name: str        # base scene, e.g. "009850-3_obj"
    camera_name: str          # "left" or "right"
    split: str                # "train", "valid", or "test"
    sequence_dir: Path        # absolute path to the sequence directory
    frame_numbers: list[int]  # sorted ascending list of annotation frame numbers
    has_trajectories: bool    # True only for "left" camera sequences

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DynamicReplicaAdapter(BaseAdapter):
    """Dataset adapter for the Dynamic_Replica dataset.

    Dataset layout::

        <root>/
            <split>/                              # train / valid / test
                frame_annotations_<split>.jgz    # per-frame camera + path annotations
                <base_seq>_source_left/
                    images/
                        <base_seq>_source_left-XXXX.png       (RGBA, uint8)
                    depths/
                        <base_seq>_source_left_XXXX.geometric.png  (uint16)
                    trajectories/
                        XXXXXX.pth                             (PyTorch zip)
                    masks/
                    ...
                <base_seq>_source_right/
                    images/ depths/ ...           (no trajectories)

    Sequence IDs
    ------------
    Each adapter sequence corresponds to one camera track (one sub-directory).
    The unique ID is the directory name, e.g. ``"009850-3_obj_source_left"``.

    Camera convention
    -----------------
    Annotations use PyTorch3D NDC intrinsics and row-vector extrinsics.
    These are converted to OpenCV pixel-space intrinsics and world-to-camera
    (w2c) extrinsics (same conversion as the Co3Dv2 adapter).

    Depth encoding
    --------------
    ``depth = raw_uint16 / 65535.0 * scale_adjustment``
    (scale_adjustment == 1.0 for all Dynamic_Replica frames)
    Depth is in PyTorch3D world units (consistent with camera extrinsics).
    Pixels with raw == 0 are invalid (depth returned as 0.0).

    Trajectory files
    ----------------
    Only left-camera sequences have trajectory data.  Each per-frame ``.pth``
    file stores the 3D world positions and 2D pixel positions of N=22,671
    consistently-indexed mesh vertices (``verts_inds_vis`` encodes visibility).

    Loading trajectories requires PyTorch.  Pass ``load_trajectories=False``
    to skip trajectory loading (all trajectory fields will be ``None``).

    Supervision availability
    ------------------------
    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : True  (left camera only; False if load_trajectories=False
                                   or right camera)
    - ``has_visibility`` : True  (when tracks are available)
    """

    dataset_name: str = "dynamic_replica"

    def __init__(
        self,
        root: str,
        split: str = "train",
        load_trajectories: bool = True,
        min_frames: int = 2,
        strict: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the Dynamic_Replica dataset.
        split :
            Which split to load: ``"train"``, ``"valid"``, or ``"test"``.
        load_trajectories :
            If True (default), load per-frame trajectory ``.pth`` files for
            left-camera sequences (requires PyTorch).  Set to False to skip
            trajectory loading for faster I/O.
        min_frames :
            Sequences with fewer than this many frames are skipped.
        strict :
            If True, raise on any indexing error.
            If False, skip broken sequences with a warning.
        verbose :
            Print a summary after index construction.
        """
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dynamic_Replica root not found: {self.root}")

        self.split = split
        self.split_root = self.root / split
        if not self.split_root.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_root}. "
                f"Valid splits: train, valid, test."
            )

        self.load_trajectories = load_trajectories
        self.min_frames = min_frames
        self.strict = strict
        self.verbose = verbose

        # Lazily loaded frame annotation index.
        # Maps (base_seq_name, camera_name, frame_number) -> annotation dict.
        self._anno_index: Optional[dict[tuple[str, str, int], dict]] = None

        self._records: list[_DRSequenceRecord] = self._build_index()
        self._name_to_record: dict[str, _DRSequenceRecord] = {
            r.sequence_name: r for r in self._records
        }

        if not self._records:
            raise RuntimeError(
                f"No valid Dynamic_Replica sequences found under {self.split_root}."
            )

        if self.verbose:
            n_left = sum(1 for r in self._records if r.camera_name == "left")
            n_right = sum(1 for r in self._records if r.camera_name == "right")
            print(
                f"[DynamicReplicaAdapter] split={split!r}, "
                f"sequences={len(self._records)} "
                f"(left={n_left}, right={n_right}), "
                f"load_trajectories={load_trajectories}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_name for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_name

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        self._ensure_anno_loaded()
        first_fn = r.frame_numbers[0]
        anno = self._get_frame_anno(r.base_seq_name, r.camera_name, first_fn)
        H, W = anno["image"]["size"]
        has_tracks = r.has_trajectories and self.load_trajectories
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": r.sequence_name,
            "base_seq_name": r.base_seq_name,
            "camera_name": r.camera_name,
            "sequence_dir": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "height": H,
            "width": W,
            "has_depth": True,
            "has_normals": False,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            "extrinsics_convention": "w2c",
            "depth_unit": "dynamic_replica_world_units",
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        """Load a clip from a Dynamic_Replica sequence.

        Parameters
        ----------
        sequence_name :
            Directory name, e.g. ``"009850-3_obj_source_left"``.
        frame_indices :
            Positions into the sequence's sorted frame list.
            Must be in ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals`` is always ``None``.
            ``trajs_2d``, ``trajs_3d_world``, ``visibs``, ``valids`` are set
            for left-camera sequences when ``load_trajectories=True``,
            otherwise ``None``.
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)
        self._ensure_anno_loaded()

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        frame_paths: list[str] = []
        frame_numbers_clip: list[int] = []
        traj_paths_clip: list[Optional[Path]] = []

        for idx in frame_indices:
            fn = r.frame_numbers[idx]
            anno = self._get_frame_anno(r.base_seq_name, r.camera_name, fn)

            img_path = self.split_root / anno["image"]["path"]
            dep_path = self.split_root / anno["depth"]["path"]
            scale_adj = float(anno["depth"]["scale_adjustment"])

            frame_paths.append(str(img_path))
            frame_numbers_clip.append(fn)

            images.append(_load_rgb(img_path))
            depths.append(_load_depth(dep_path, scale_adj))

            H_anno, W_anno = anno["image"]["size"]
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

            # Trajectory path (only left camera; might be None for frame 0 of right)
            traj_rel = anno["trajectories"].get("path")
            if traj_rel is not None:
                traj_paths_clip.append(self.split_root / traj_rel)
            else:
                traj_paths_clip.append(None)

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3) float32
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) float32

        # Load trajectories if enabled and available
        has_tracks = (
            self.load_trajectories
            and r.has_trajectories
            and all(p is not None for p in traj_paths_clip)
        )

        trajs_2d: Optional[np.ndarray] = None
        trajs_3d_world: Optional[np.ndarray] = None
        valids: Optional[np.ndarray] = None
        visibs: Optional[np.ndarray] = None

        if has_tracks:
            traj_3d_list: list[np.ndarray] = []
            traj_2d_list: list[np.ndarray] = []
            vis_list: list[np.ndarray] = []

            for p in traj_paths_clip:
                traj_data = _load_trajectory_pth(p)
                traj_3d_list.append(traj_data["traj_3d_world"])   # (N, 3)
                traj_2d_list.append(traj_data["traj_2d"][:, :2])  # (N, 2) – drop 3rd col
                vis_list.append(traj_data["verts_inds_vis"])       # (N,)

            trajs_3d_world = np.stack(traj_3d_list, axis=0)  # (T, N, 3)
            trajs_2d = np.stack(traj_2d_list, axis=0)        # (T, N, 2)
            visibs = np.stack(vis_list, axis=0)               # (T, N)
            valids = visibs.copy()                            # visible = valid

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "split": self.split,
                "base_seq_name": r.base_seq_name,
                "camera_name": r.camera_name,
                "sequence_dir": str(r.sequence_dir),
                "frame_indices": list(frame_indices),
                "frame_numbers": frame_numbers_clip,
                "num_frames_in_sequence": r.num_frames,
                "has_depth": True,
                "has_normals": False,
                "has_tracks": has_tracks,
                "has_visibility": has_tracks,
                "has_trajs_3d_world": has_tracks,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "depth_encoding": "raw_uint16 / 65535.0 * scale_adjustment",
                "depth_unit": "dynamic_replica_world_units",
            },
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Image and depth files exist for the first 5 frames.
        2. Intrinsics: shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics: shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite; at least some positive values per frame.
        5. Round-trip reprojection error < 1 px (depth → world → camera → pixel).
        6. If trajectories available: shapes [T, N, 2], [T, N, 3], [T, N];
           traj_2d coordinates inside image bounds.

        Returns
        -------
        dict with ``dataset_name``, ``sequence_name``, ``ok``, ``messages``.
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        self._ensure_anno_loaded()
        probe_count = min(5, r.num_frames)

        # ── 1. File existence ────────────────────────────────────────────
        for i in range(probe_count):
            fn = r.frame_numbers[i]
            try:
                anno = self._get_frame_anno(r.base_seq_name, r.camera_name, fn)
            except KeyError as exc:
                ok = False
                msgs.append(f"frame {fn}: annotation lookup failed: {exc}")
                continue

            for key, rel in [
                ("image", anno["image"]["path"]),
                ("depth", anno["depth"]["path"]),
            ]:
                p = self.split_root / rel
                if not p.exists():
                    ok = False
                    msgs.append(f"frame {fn}: missing {key} file: {p}")

        if not ok:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs,
            }

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
            bottom = clip.extrinsics[:, 3, :]
            if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
                ok = False
                msgs.append("extrinsics last row != [0,0,0,1]")
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
            elif not (dep >= 0).all():
                ok = False
                msgs.append(f"depth[{t}] contains negative values")

        # ── 5. Reprojection ─────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)

            ys, xs = np.where(dep > 0)
            if ys.size == 0:
                msgs.append(f"reproj check skipped frame {t}: no valid depth")
                continue

            mid = ys.size // 2
            u_px, v_px = int(xs[mid]), int(ys[mid])
            d = float(dep[v_px, u_px])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            xc = (u_px - cx) / fx * d
            yc = (v_px - cy) / fy * d
            p_cam = np.array([xc, yc, d, 1.0])

            try:
                p_world = np.linalg.inv(E) @ p_cam
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_cam2 = E @ p_world
            if abs(p_cam2[2]) < 1e-8:
                msgs.append(f"reproj frame {t}: reprojected depth ≈ 0, skipped")
                continue

            u2 = K[0, 0] * p_cam2[0] / p_cam2[2] + K[0, 2]
            v2 = K[1, 1] * p_cam2[1] / p_cam2[2] + K[1, 2]
            reproj_err = float(np.hypot(u2 - u_px, v2 - v_px))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reproj error frame {t}: err={reproj_err:.4f}px "
                    f"(reprojected ({u2:.2f},{v2:.2f}) vs original ({u_px},{v_px}))"
                )

        # ── 6. Trajectories (if loaded) ──────────────────────────────────
        if clip.trajs_2d is not None:
            N = clip.trajs_2d.shape[1]
            if clip.trajs_2d.shape != (T, N, 2):
                ok = False
                msgs.append(f"trajs_2d shape {clip.trajs_2d.shape} != ({T},{N},2)")
            if clip.trajs_3d_world is not None and clip.trajs_3d_world.shape != (T, N, 3):
                ok = False
                msgs.append(f"trajs_3d_world shape {clip.trajs_3d_world.shape} != ({T},{N},3)")
            if clip.visibs is not None and clip.visibs.shape != (T, N):
                ok = False
                msgs.append(f"visibs shape {clip.visibs.shape} != ({T},{N})")
            if clip.valids is not None and clip.valids.shape != (T, N):
                ok = False
                msgs.append(f"valids shape {clip.valids.shape} != ({T},{N})")

            # traj_2d should be in roughly image pixel range
            if clip.images:
                H_img, W_img = clip.images[0].shape[:2]
                vis_mask = clip.visibs if clip.visibs is not None else np.ones((T, N), bool)
                if vis_mask.any():
                    uv_vis = clip.trajs_2d[vis_mask]
                    in_bounds = (
                        (uv_vis[:, 0] >= 0) & (uv_vis[:, 0] < W_img)
                        & (uv_vis[:, 1] >= 0) & (uv_vis[:, 1] < H_img)
                    )
                    frac_in = float(in_bounds.mean())
                    if frac_in < 0.5:
                        ok = False
                        msgs.append(
                            f"trajs_2d visible points mostly out of bounds "
                            f"(frac_in_bounds={frac_in:.3f})"
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

    def _build_index(self) -> list[_DRSequenceRecord]:
        self._ensure_anno_loaded()

        records: list[_DRSequenceRecord] = []
        skipped: list[str] = []

        for seq_dir in sorted(self.split_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            try:
                rec = self._index_sequence(seq_dir)
                if rec is not None and rec.num_frames >= self.min_frames:
                    records.append(rec)
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"{seq_dir.name}: {exc}")
                if self.verbose:
                    print(f"[DynamicReplicaAdapter][WARN] skip {seq_dir.name}: {exc}")

        if self.verbose and skipped:
            print(
                f"[DynamicReplicaAdapter] skipped {len(skipped)} directories "
                "(non-strict mode)"
            )

        return records

    def _index_sequence(self, seq_dir: Path) -> Optional[_DRSequenceRecord]:
        """Parse one sequence directory and build a record."""
        name = seq_dir.name

        # Parse base name and camera side from directory name.
        # Expected pattern: "<base_name>_source_<camera>" e.g. "009850-3_obj_source_left"
        if "_source_" not in name:
            return None  # skip annotation files and unrecognised dirs

        base_name, _, camera = name.rpartition("_source_")
        if camera not in ("left", "right"):
            return None

        images_dir = seq_dir / "images"
        if not images_dir.exists():
            return None

        # Look up frame numbers from the annotation index.
        all_anno = self._anno_index
        frame_numbers = sorted(
            fn
            for (bsn, cam, fn), _ in all_anno.items()
            if bsn == base_name and cam == camera
        )

        if len(frame_numbers) == 0:
            return None

        # Check whether this sequence has trajectory files.
        has_traj = False
        if camera == "left":
            first_fn = frame_numbers[0]
            anno = all_anno.get((base_name, camera, first_fn))
            if anno is not None:
                traj_rel = anno["trajectories"].get("path")
                has_traj = traj_rel is not None

        return _DRSequenceRecord(
            sequence_name=name,
            base_seq_name=base_name,
            camera_name=camera,
            split=self.split,
            sequence_dir=seq_dir,
            frame_numbers=frame_numbers,
            has_trajectories=has_traj,
        )

    # ------------------------------------------------------------------
    # Annotation cache
    # ------------------------------------------------------------------

    def _ensure_anno_loaded(self) -> None:
        """Load and index the split annotation file on first call."""
        if self._anno_index is not None:
            return

        anno_file = self.split_root / f"frame_annotations_{self.split}.jgz"
        if not anno_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {anno_file}. "
                f"Expected at: <root>/<split>/frame_annotations_<split>.jgz"
            )

        raw = _load_jgz(anno_file)

        index: dict[tuple[str, str, int], dict] = {}
        for entry in raw:
            key = (
                entry["sequence_name"],
                entry["camera_name"],
                int(entry["frame_number"]),
            )
            index[key] = entry

        self._anno_index = index

    def _get_frame_anno(
        self,
        base_seq_name: str,
        camera_name: str,
        frame_number: int,
    ) -> dict:
        key = (base_seq_name, camera_name, int(frame_number))
        if key not in self._anno_index:
            raise KeyError(
                f"No annotation for ({base_seq_name!r}, {camera_name!r}, "
                f"frame {frame_number})"
            )
        return self._anno_index[key]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _DRSequenceRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._name_to_record[sequence_name]

    def _check_indices(
        self,
        frame_indices: list[int],
        num_frames: int,
        sequence_name: str,
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
