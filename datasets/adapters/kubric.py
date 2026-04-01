# from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip


class KubricAdapter(BaseAdapter):
    """
    Kubric adapter for extracted scene format.

    Expected layout:
        root/
          0000/
            frames/
            0000.npy
            0000_with_rank.npz
          0001/
            frames/
            0001.npy
            0001_with_rank.npz
          ...

    Notes:
    - ann["coords"]         : [N, T, 2]
    - ann["coords_depth"]   : [N, T]
    - ann["visibility"]     : [N, T]
    - ann["depth"]          : [T, H, W, 1]
    - rank["shared_intrinsics"] : [3, 3]
    - rank["extrinsics"]        : [T, 3, 4], confirmed world-to-camera (w2c)

    Adapter output unified schema:
    - trajs_2d         : [T, N, 2]
    - trajs_3d_world   : [T, N, 3]
    - visibs           : [T, N]
    - intrinsics       : [T, 3, 3]
    - extrinsics       : [T, 4, 4]   (w2c)
    """

    dataset_name = "kubric"

    def __init__(self, root: str, split: str = "train"):
        """
        Args:
            root: Root directory containing Kubric scenes
            split: Split name (ignored, Kubric doesn't have splits)
        """
        self.root = Path(root)
        self.split = split  # Store but don't use
        if not self.root.exists():
            raise FileNotFoundError(f"Kubric root not found: {self.root}")
        self.sequence_names = self._build_index()

    def __len__(self) -> int:
        return len(self.sequence_names)

    def list_sequences(self) -> list[str]:
        return self.sequence_names

    def get_sequence_name(self, index: int) -> str:
        return self.sequence_names[index]

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        scene_dir = self.root / sequence_name
        # Support nested structure
        if (scene_dir / sequence_name).is_dir():
            scene_dir = scene_dir / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        ann = np.load(scene_dir / f"{sequence_name}.npy", allow_pickle=True).item()
        rank = np.load(scene_dir / f"{sequence_name}_with_rank.npz", allow_pickle=True)

        coords = np.asarray(ann["coords"])          # [N,T,2]
        depth = np.asarray(ann["depth"])            # [T,H,W,1]
        frame_files = self._find_frame_files(scene_dir / "frames")

        info = {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "path": str(scene_dir),
            "num_frames": int(coords.shape[1]),
            "num_tracks": int(coords.shape[0]),
            "height": int(depth.shape[1]),
            "width": int(depth.shape[2]),
            "frame_file_count": len(frame_files),
            "has_depth": True,
            "has_normals": False,
            "has_tracks": True,
            "has_visibility": True,
            "has_trajs_3d_world": True,
            "intrinsics_shape": tuple(rank["shared_intrinsics"].shape),
            "extrinsics_shape": tuple(rank["extrinsics"].shape),
            "extrinsics_convention": "w2c",
        }
        return info

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        scene_dir = self.root / sequence_name
        # Support nested structure
        if (scene_dir / sequence_name).is_dir():
            scene_dir = scene_dir / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        ann_path = scene_dir / f"{sequence_name}.npy"
        rank_path = scene_dir / f"{sequence_name}_with_rank.npz"

        h5_path = ann_path.with_suffix('.h5')
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, 'r') as hf:
                T_total = hf['trajs_2d'].shape[0]
                idx = np.array(frame_indices)
                trajs_2d     = hf['trajs_2d'][idx]       # [T,N,2]
                coords_depth = hf['coords_depth'][idx]   # [T,N]
                visibs       = hf['visibility'][idx]     # [T,N]

            # depth still comes from per-frame .npy files in depths/
            depth_dir = scene_dir / "depths"
            depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
            if depth_files:
                depths = [np.load(depth_files[i]).astype(np.float32).squeeze() for i in frame_indices]
            else:
                depths = []

            ann = None
            seg_thw1 = None
            camera_raw = None
            metadata_raw = None
        else:
            ann = np.load(ann_path, allow_pickle=True).item()

            coords_nt2 = np.asarray(ann["coords"], dtype=np.float32)              # [N,T,2]
            coords_depth_nt = np.asarray(ann["coords_depth"], dtype=np.float32)   # [N,T]
            visibility_nt = np.asarray(ann["visibility"], dtype=bool)             # [N,T]

            trajs_2d     = np.transpose(coords_nt2[:, frame_indices, :], (1, 0, 2))
            coords_depth = np.transpose(coords_depth_nt[:, frame_indices], (1, 0))
            visibs       = np.transpose(visibility_nt[:, frame_indices], (1, 0))

            dense_depth_thw1 = np.asarray(ann["depth"], dtype=np.float32)
            dense_depth = dense_depth_thw1[frame_indices, ..., 0]
            depths = [dense_depth[i] for i in range(len(frame_indices))]

            seg_thw1 = np.asarray(ann["segmentations"]) if "segmentations" in ann else None
            camera_raw   = ann.get("camera", None)
            metadata_raw = ann.get("metadata", None)

        rank = np.load(rank_path, allow_pickle=True)

        K_shared = np.asarray(rank["shared_intrinsics"], dtype=np.float32)    # [3,3]
        extrinsics_t34 = np.asarray(rank["extrinsics"], dtype=np.float32)     # [T,3,4], w2c
        ranking_tt = np.asarray(rank["ranking"])

        if ann is not None:
            T_total = np.asarray(ann["coords"]).shape[1]
        # else T_total was already set from h5 header
        self._check_indices(frame_indices, T_total, sequence_name)

        frame_files = self._find_frame_files(scene_dir / "frames")
        if len(frame_files) != T_total:
            raise ValueError(
                f"[{sequence_name}] frame file count mismatch: "
                f"frames/ has {len(frame_files)}, annotations have {T_total}"
            )

        selected_frame_paths = [str(frame_files[i]) for i in frame_indices]
        images = [self._read_image(frame_files[i]) for i in frame_indices]

        # trajs_2d, coords_depth, visibs already set (both paths produce [T,N,*])

        intrinsics = np.repeat(K_shared[None, :, :], len(frame_indices), axis=0)   # [T,3,3]
        extrinsics = self._to_4x4(extrinsics_t34[frame_indices])                    # [T,4,4], w2c

        trajs_3d_world = self._backproject_tracks_to_world(
            trajs_2d=trajs_2d,
            coords_depth=coords_depth,
            intrinsics=intrinsics,
            extrinsics_w2c=extrinsics,
        )  # [T,N,3]

        valids = (
            np.isfinite(trajs_2d[..., 0])
            & np.isfinite(trajs_2d[..., 1])
            & np.isfinite(coords_depth)
            & (coords_depth > 0)
        )

        metadata = {
            "backend": "kubric_extracted",
            "has_depth": True,
            "has_normals": False,
            "has_tracks": True,
            "has_visibility": True,
            "has_trajs_3d_world": True,
            "num_frames_total": int(T_total),
            "num_frames_clip": int(len(frame_indices)),
            "coords_depth": coords_depth,  # [T,N], useful for debug / reprojection checks
            "segmentations": seg_thw1[frame_indices, ..., 0] if seg_thw1 is not None else None,
            "ranking": ranking_tt[np.ix_(frame_indices, frame_indices)],
            "camera_raw": camera_raw,
            "metadata_raw": metadata_raw,
            "extrinsics_convention": "w2c",
        }

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=selected_frame_paths,
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata=metadata,
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        scene_dir = self.root / sequence_name
        # Support nested structure
        if (scene_dir / sequence_name).is_dir():
            scene_dir = scene_dir / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        ann = np.load(scene_dir / f"{sequence_name}.npy", allow_pickle=True).item()
        rank = np.load(scene_dir / f"{sequence_name}_with_rank.npz", allow_pickle=True)

        msgs: list[str] = []
        ok = True

        coords = np.asarray(ann["coords"])
        coords_depth = np.asarray(ann["coords_depth"])
        visibility = np.asarray(ann["visibility"])
        depth = np.asarray(ann["depth"])
        extrinsics = np.asarray(rank["extrinsics"])
        K = np.asarray(rank["shared_intrinsics"])

        if coords.ndim != 3 or coords.shape[-1] != 2:
            ok = False
            msgs.append(f"coords shape invalid: {coords.shape}")

        if coords_depth.shape != coords.shape[:2]:
            ok = False
            msgs.append(
                f"coords_depth shape {coords_depth.shape} != coords[:2] {coords.shape[:2]}"
            )

        if visibility.shape != coords.shape[:2]:
            ok = False
            msgs.append(
                f"visibility shape {visibility.shape} != coords[:2] {coords.shape[:2]}"
            )

        if depth.ndim != 4 or depth.shape[-1] != 1:
            ok = False
            msgs.append(f"depth shape invalid: {depth.shape}")

        if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
            ok = False
            msgs.append(f"extrinsics shape invalid: {extrinsics.shape}")

        if K.shape != (3, 3):
            ok = False
            msgs.append(f"shared_intrinsics shape invalid: {K.shape}")

        frame_files = self._find_frame_files(scene_dir / "frames")
        if len(frame_files) != coords.shape[1]:
            ok = False
            msgs.append(
                f"frames/ count {len(frame_files)} != annotation num_frames {coords.shape[1]}"
            )

        # Confirm extrinsics convention using camera positions.
        # For w2c, camera center in world coordinates is: C = -R^T t
        if "camera" in ann and isinstance(ann["camera"], dict) and "positions" in ann["camera"]:
            pos = np.asarray(ann["camera"]["positions"], dtype=np.float32)  # [T,3]
            E = extrinsics.astype(np.float32)                               # [T,3,4]
            R = E[:, :, :3]
            t = E[:, :, 3]
            pos_from_w2c = -np.einsum("tji,tj->ti", R, t)
            err_w2c = np.linalg.norm(pos_from_w2c - pos, axis=1).mean()
            msgs.append(f"mean position error assuming w2c: {err_w2c:.6e}")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> list[str]:
        sequence_names: list[str] = []
        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue
            seq = p.name
            # Check both flat and nested structure
            # Flat: kubric/0001/0001.npy
            # Nested: kubric/0001/0001/0001.npy
            seq_dir = p / seq if (p / seq).is_dir() else p
            if (
                (seq_dir / f"{seq}.npy").exists()
                and (seq_dir / f"{seq}_with_rank.npz").exists()
                and (seq_dir / "frames").exists()
            ):
                sequence_names.append(seq)

        if len(sequence_names) == 0:
            raise RuntimeError(f"No valid Kubric scenes found under: {self.root}")
        return sequence_names

    def _check_scene_exists(self, scene_dir: Path, sequence_name: str) -> None:
        if not scene_dir.exists():
            raise FileNotFoundError(f"[{sequence_name}] scene dir not found: {scene_dir}")
        if not (scene_dir / f"{sequence_name}.npy").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing file: {scene_dir / f'{sequence_name}.npy'}")
        if not (scene_dir / f"{sequence_name}_with_rank.npz").exists():
            raise FileNotFoundError(
                f"[{sequence_name}] missing file: {scene_dir / f'{sequence_name}_with_rank.npz'}"
            )
        if not (scene_dir / "frames").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing frames dir: {scene_dir / 'frames'}")

    def _find_frame_files(self, frame_dir: Path) -> list[Path]:
        files: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(frame_dir.glob(ext))
        return sorted(files)

    def _read_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    def _check_indices(self, frame_indices: list[int], num_frames: int, sequence_name: str) -> None:
        if len(frame_indices) == 0:
            raise ValueError("frame_indices is empty")
        if min(frame_indices) < 0 or max(frame_indices) >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={min(frame_indices)}, max={max(frame_indices)}, num_frames={num_frames}"
            )

    def _to_4x4(self, extrinsics_t34: np.ndarray) -> np.ndarray:
        """
        Convert [T,3,4] -> [T,4,4].
        Input extrinsics are world-to-camera (w2c).
        """
        T = extrinsics_t34.shape[0]
        out = np.zeros((T, 4, 4), dtype=np.float32)
        out[:, :3, :4] = extrinsics_t34
        out[:, 3, 3] = 1.0
        return out

    def _backproject_tracks_to_world(
        self,
        trajs_2d: np.ndarray,        # [T,N,2]
        coords_depth: np.ndarray,    # [T,N]
        intrinsics: np.ndarray,      # [T,3,3]
        extrinsics_w2c: np.ndarray,  # [T,4,4]
    ) -> np.ndarray:
        """
        Reconstruct world-space track points from:
        - 2D pixel coords
        - per-track depth in camera z
        - K
        - w2c extrinsics

        world point = inv(w2c) @ (z * inv(K) @ [u,v,1])
        """
        T, N, _ = trajs_2d.shape
        trajs_3d_world = np.full((T, N, 3), np.nan, dtype=np.float32)

        for t in range(T):
            uv = trajs_2d[t]            # [N,2]
            z = coords_depth[t]         # [N]
            K = intrinsics[t]           # [3,3]
            w2c = extrinsics_w2c[t]     # [4,4]

            valid = (
                np.isfinite(uv[:, 0])
                & np.isfinite(uv[:, 1])
                & np.isfinite(z)
                & (z > 0)
            )
            if not np.any(valid):
                continue

            uv_valid = uv[valid]
            z_valid = z[valid]

            ones = np.ones((uv_valid.shape[0], 1), dtype=np.float32)
            pix = np.concatenate([uv_valid, ones], axis=-1)  # [M,3]

            K_inv = np.linalg.inv(K).astype(np.float32)
            rays = (K_inv @ pix.T).T                          # [M,3]
            pts_cam = rays * z_valid[:, None]                 # [M,3]

            pts_cam_h = np.concatenate(
                [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)],
                axis=-1,
            )  # [M,4]

            c2w = np.linalg.inv(w2c).astype(np.float32)
            pts_world_h = (c2w @ pts_cam_h.T).T               # [M,4]
            trajs_3d_world[t, valid] = pts_world_h[:, :3]

        return trajs_3d_world