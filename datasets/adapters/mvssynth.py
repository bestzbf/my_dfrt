from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseAdapter, UnifiedClip


# ---------------------------------------------------------------------------
# EXR depth reader
# ---------------------------------------------------------------------------

def _read_exr_depth(path: Path) -> np.ndarray:
    """Read OpenEXR depth map and convert inf to 0.

    Requires opencv-python with OpenEXR support.
    Set OPENCV_IO_ENABLE_OPENEXR=1 environment variable if needed.

    Returns:
        depth: (H, W) float32 array. Invalid pixels (inf) are set to 0.
    """
    import cv2
    dep = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise RuntimeError(f"Failed to read EXR file: {path}")
    dep = dep.astype(np.float32)
    dep[~np.isfinite(dep)] = 0.0
    return dep


def _read_rgb(path: Path) -> np.ndarray:
    """Load RGB image as uint8 (H, W, 3)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sequence record
# ---------------------------------------------------------------------------

class _SequenceRecord:
    """Index entry for one MVS-Synth sequence."""

    __slots__ = ("sequence_id", "sequence_dir", "num_frames")

    def __init__(self, sequence_id: str, sequence_dir: Path, num_frames: int):
        self.sequence_id = sequence_id
        self.sequence_dir = sequence_dir
        self.num_frames = num_frames

    def image_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "images" / f"{frame_idx:04d}.png"

    def depth_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "depths" / f"{frame_idx:04d}.exr"

    def pose_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "poses" / f"{frame_idx:04d}.json"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MVSSynthAdapter(BaseAdapter):
    """Dataset adapter for MVS-Synth (GTAV_1080 variant).

    Expected dataset layout::

        <root>/
            num_images.json              # [100, 100, ..., 100] (120 sequences)
            0000/
                images/
                    0000.png, 0001.png, ..., 0099.png
                depths/
                    0000.exr, 0001.exr, ..., 0099.exr
                poses/
                    0000.json, 0001.json, ..., 0099.json
            0001/
                ...
            ...
            0119/
                ...

    Each pose JSON contains::

        {
            "f_x": 1157.84...,
            "f_y": 1157.84...,
            "c_x": 960,
            "c_y": 540,
            "extrinsic": [[4x4 matrix]]  # world-to-camera (w2c)
        }

    Depth maps are OpenEXR float32 with inf values for invalid/sky pixels.

    Supervision availability
    ------------------------
    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : False
    - ``has_visibility`` : False

    Extrinsics convention
    ---------------------
    MVS-Synth stores **world-to-camera** (w2c) transforms.
    Note: rotation matrices have det(R) ≈ -1 due to GTA V's left-handed
    coordinate system. This is preserved as-is for consistency with the
    source data.
    """

    dataset_name: str = "mvssynth"

    def __init__(
        self,
        root: str,
        split: str = "train",
        strict: bool = True,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        root :
            Root directory of the MVS-Synth GTAV_1080 dataset.
        split :
            Split name (ignored, MVS-Synth doesn't have splits).
        strict :
            If True, raise on any indexing error.
            If False, skip broken sequences with a warning.
        verbose :
            Print summary after index construction.
        """
        self.split = split  # Store but don't use
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"MVS-Synth root not found: {self.root}")

        self.strict = strict
        self.verbose = verbose
        self.precompute_root = Path(precompute_root) if precompute_root else self.root

        # Enable OpenEXR support in OpenCV
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

        self._records: list[_SequenceRecord] = self._build_index()
        self._name_to_record: dict[str, _SequenceRecord] = {
            r.sequence_id: r for r in self._records
        }

        if not self._records:
            raise RuntimeError(
                f"No valid MVS-Synth sequences found under {self.root}"
            )

        if self.verbose:
            print(
                f"[MVSSynthAdapter] root={self.root.name}, "
                f"sequences={len(self._records)}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_id for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_id

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)

        # Probe first frame for image size
        img_path = r.image_path(0)
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        H, W = img.shape[:2]

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": r.sequence_id,
            "sequence_root": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "height": H,
            "width": W,
            "has_depth": True,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
            "has_trajs_3d_world": False,
            "extrinsics_convention": "w2c",
            "depth_unit": "meters",
        }

    def load_clip(
        self, sequence_name: str, frame_indices: list[int]
    ) -> UnifiedClip:
        """Load a clip from an MVS-Synth sequence.

        Parameters
        ----------
        sequence_name :
            Sequence ID (e.g., "0000", "0042").
        frame_indices :
            Frame indices within [0, num_frames).

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals``, ``trajs_2d``, ``trajs_3d_world``, ``valids``,
            and ``visibs`` are all ``None``.
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        frame_paths: list[str] = []

        for idx in frame_indices:
            img_path = r.image_path(idx)
            dep_path = r.depth_path(idx)
            pose_path = r.pose_path(idx)

            frame_paths.append(str(img_path))

            images.append(_read_rgb(img_path))
            depths.append(_read_exr_depth(dep_path))

            with open(pose_path, "r") as f:
                pose = json.load(f)

            fx = float(pose["f_x"])
            fy = float(pose["f_y"])
            cx = float(pose["c_x"])
            cy = float(pose["c_y"])
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            intrinsics_list.append(K)

            E = np.array(pose["extrinsic"], dtype=np.float32)  # w2c
            extrinsics_list.append(E)

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3)
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) w2c

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
                "sequence_root": str(r.sequence_dir),
                "frame_indices": list(frame_indices),
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": [images[0].shape[0], images[0].shape[1]] if images else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
                "depth_note": (
                    "OpenEXR float32; inf values converted to 0 (invalid/sky)"
                ),
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
        1. Image, depth, and pose files exist for first 5 frames.
        2. Intrinsics shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite after inf→0 conversion; at least some
           positive values per frame.
        5. Round-trip reprojection: unproject center pixel, re-project,
           check error < 1 px.

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
            for key, path in [
                ("image", r.image_path(i)),
                ("depth", r.depth_path(i)),
                ("pose", r.pose_path(i)),
            ]:
                if not path.exists():
                    ok = False
                    msgs.append(f"frame {i}: missing {key} file: {path}")

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
            # Note: det(R) ≈ -1 is expected for MVS-Synth (left-handed system)
            dets = np.linalg.det(R)
            if not np.all(np.abs(np.abs(dets) - 1.0) < 1e-4):
                ok = False
                msgs.append(f"rotation det not ±1: range [{dets.min():.6f},{dets.max():.6f}]")

        # ── 4. Depth ────────────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")
            if (dep > 0).sum() == 0:
                ok = False
                msgs.append(f"depth[{t}] has no positive values")

        # ── 5. Reprojection ─────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)  # w2c

            ys, xs = np.where(dep > 0)
            if ys.size == 0:
                msgs.append(f"reproj check skipped frame {t}: no valid depth")
                continue

            # Sample center-ish pixel
            mid = ys.size // 2
            u, v = int(xs[mid]), int(ys[mid])
            d = float(dep[v, u])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            xc = (u - cx) / fx * d
            yc = (v - cy) / fy * d
            p_cam = np.array([xc, yc, d, 1.0])

            try:
                E_inv = np.linalg.inv(E)
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_world = E_inv @ p_cam
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
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_SequenceRecord]:
        # Check for num_images.json
        num_images_path = self.root / "num_images.json"
        if num_images_path.exists():
            with open(num_images_path, "r") as f:
                num_images_list = json.load(f)
        else:
            num_images_list = None

        records: list[_SequenceRecord] = []
        skipped: list[str] = []

        # Enumerate sequence directories (0000, 0001, ..., 0119)
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if not seq_dir.name.isdigit():
                continue

            try:
                rec = self._index_sequence(seq_dir, num_images_list)
                if rec is not None:
                    records.append(rec)
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"{seq_dir.name}: {exc}")
                if self.verbose:
                    print(f"[MVSSynthAdapter][WARN] skip {seq_dir.name}: {exc}")

        if self.verbose and skipped:
            print(
                f"[MVSSynthAdapter] skipped {len(skipped)} sequences "
                "(non-strict mode)"
            )

        return records

    def _index_sequence(
        self, seq_dir: Path, num_images_list: Optional[list[int]]
    ) -> Optional[_SequenceRecord]:
        seq_id = seq_dir.name

        # Check required subdirectories
        for subdir in ["images", "depths", "poses"]:
            if not (seq_dir / subdir).is_dir():
                raise FileNotFoundError(f"Missing {subdir}/ in {seq_dir}")

        # Determine num_frames
        if num_images_list is not None:
            seq_idx = int(seq_id)
            if seq_idx < len(num_images_list):
                num_frames = num_images_list[seq_idx]
            else:
                raise ValueError(
                    f"Sequence {seq_id} index out of range in num_images.json"
                )
        else:
            # Count pose files
            pose_files = list((seq_dir / "poses").glob("*.json"))
            num_frames = len(pose_files)

        if num_frames == 0:
            raise RuntimeError(f"No frames found in {seq_dir}")

        return _SequenceRecord(
            sequence_id=seq_id,
            sequence_dir=seq_dir,
            num_frames=num_frames,
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
