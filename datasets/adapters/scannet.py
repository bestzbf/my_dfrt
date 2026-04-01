from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip


class ScanNetAdapter(BaseAdapter):
    """
    ScanNet adapter.

    Expected layout:
        root/
          scene0030_00/
            images/
            depths/
            metadata.json
          scene0046_00/
            ...

    Supported metadata patterns:
    - metadata["intrinsics"] can be:
        1) shared 3x3 matrix
        2) fx/fy/cx/cy-style dict
        3) per-frame dict keyed by frame id or filename, e.g. "002490.png" -> 3x3
        4) nested dicts such as color/rgb/image/depth/camera

    - metadata["extrinsics"] can be:
        1) shared / stacked [T,4,4] or [T,3,4]
        2) explicit keys like w2c / c2w / camera_from_world / world_from_camera
        3) per-frame dict keyed by frame id or filename, e.g. "002490.png" -> 4x4 / 3x4
        4) nested dict values such as {"pose": ...}, {"matrix": ...}

    Output convention:
    - extrinsics are always returned as world-to-camera (w2c), shape [T,4,4]

    Notes:
    - This adapter only reads existing modalities from ScanNet.
    - It does not fabricate tracks.
    """

    dataset_name = "scannet"

    def __init__(
        self,
        root: str,
        split: str = "train",
        depth_scale: float = 1000.0,
        default_pose_convention: str = "c2w",
        precompute_root: Optional[str] = None,
    ):
        """
        Args:
            root: Root directory containing ScanNet scenes
            split: Split name (ignored, ScanNet doesn't have standard splits)
            depth_scale: Scale factor for depth values
            default_pose_convention: Default pose convention
            precompute_root: Optional precomputed data root
        """
        self.split = split  # Store but don't use
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ScanNet root not found: {self.root}")

        if default_pose_convention not in ("c2w", "w2c"):
            raise ValueError("default_pose_convention must be 'c2w' or 'w2c'")

        self.depth_scale = float(depth_scale)
        self.default_pose_convention = default_pose_convention
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.sequence_names = self._build_index()

    def __len__(self) -> int:
        return len(self.sequence_names)

    def list_sequences(self) -> list[str]:
        return self.sequence_names

    def get_sequence_name(self, index: int) -> str:
        return self.sequence_names[index]

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        scene_dir = self.root / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        metadata = self._load_metadata(scene_dir / "metadata.json")
        image_files = self._find_image_files(scene_dir / "images")
        depth_files = self._find_depth_files(scene_dir / "depths")

        num_frames = self._infer_num_frames(metadata, image_files, depth_files)

        intrinsics, extrinsics_w2c, pose_src = self._extract_camera_params(
            metadata=metadata,
            image_files=image_files,
            num_frames=num_frames,
            sequence_name=sequence_name,
        )

        if len(image_files) > 0:
            H, W = self._read_image_size(image_files[0])
        else:
            H = metadata.get("height", None)
            W = metadata.get("width", None)

        # Dynamically check whether precomputed tracks exist for this sequence.
        has_precomputed = False
        if self.precompute_root is not None:
            npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            h5_path  = npz_path.with_suffix(".h5")
            has_precomputed = h5_path.exists() or npz_path.exists()

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "path": str(scene_dir),
            "num_frames": int(num_frames),
            "frame_file_count": len(image_files),
            "depth_file_count": len(depth_files),
            "height": int(H) if H is not None else None,
            "width": int(W) if W is not None else None,
            "has_depth": len(depth_files) > 0,
            "has_normals": has_precomputed,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
            "has_trajs_3d_world": has_precomputed,
            "intrinsics_shape": tuple(intrinsics.shape),
            "extrinsics_shape": tuple(extrinsics_w2c.shape),
            "extrinsics_convention": "w2c",
            "pose_source": pose_src,
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        scene_dir = self.root / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        image_files = self._find_image_files(scene_dir / "images")
        depth_files = self._find_depth_files(scene_dir / "depths")
        metadata = self._load_metadata(scene_dir / "metadata.json")

        if len(image_files) == 0:
            raise RuntimeError(f"[{sequence_name}] no images found under {scene_dir / 'images'}")
        if len(depth_files) == 0:
            raise RuntimeError(f"[{sequence_name}] no depths found under {scene_dir / 'depths'}")

        num_frames = self._infer_num_frames(metadata, image_files, depth_files)
        self._check_indices(frame_indices, num_frames, sequence_name)

        intrinsics_all, extrinsics_all_w2c, pose_src = self._extract_camera_params(
            metadata=metadata,
            image_files=image_files,
            num_frames=num_frames,
            sequence_name=sequence_name,
        )

        selected_frame_paths = [str(image_files[i]) for i in frame_indices]
        images = [self._read_image(image_files[i]) for i in frame_indices]
        depths = [self._read_depth(depth_files[i]) for i in frame_indices]

        intrinsics = intrinsics_all[frame_indices]       # [T,3,3]
        extrinsics = extrinsics_all_w2c[frame_indices]   # [T,4,4], w2c

        metadata_out = {
            "backend": "scannet_metadata_json",
            "has_depth": True,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
            "has_trajs_3d_world": False,
            "num_frames_total": int(num_frames),
            "num_frames_clip": int(len(frame_indices)),
            "depth_scale": self.depth_scale,
            "pad_pixel": metadata.get("pad_pixel", None),
            "depth_range": metadata.get("depth_range", None),
            "pose_source": pose_src,
            "extrinsics_convention": "w2c",
            "metadata_raw": metadata,
        }

        # Load precomputed normals / tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        if self.precompute_root is not None:
            cache = self._load_precomputed(sequence_name, frame_indices)
            if cache is not None:
                normals_out   = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out  = cache["trajs_2d"]
                trajs_3d_out  = cache["trajs_3d_world"]
                valids_out    = cache["valids"]
                visibs_out    = cache["visibs"]
                metadata_out["has_normals"] = True
                metadata_out["has_tracks"]  = True
                metadata_out["has_visibility"] = True
                metadata_out["has_trajs_3d_world"] = True

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=selected_frame_paths,
            images=images,
            depths=depths,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata=metadata_out,
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
        scene_dir = self.root / sequence_name
        self._check_scene_exists(scene_dir, sequence_name)

        msgs: list[str] = []
        ok = True

        image_files = self._find_image_files(scene_dir / "images")
        depth_files = self._find_depth_files(scene_dir / "depths")

        if len(image_files) == 0:
            ok = False
            msgs.append("images/ is empty")

        if len(depth_files) == 0:
            ok = False
            msgs.append("depths/ is empty")

        if len(image_files) != len(depth_files):
            msgs.append(
                f"warning: image count {len(image_files)} != depth count {len(depth_files)}; "
                f"adapter will use min(counts)"
            )

        metadata = self._load_metadata(scene_dir / "metadata.json")
        num_frames = self._infer_num_frames(metadata, image_files, depth_files)

        try:
            intrinsics, extrinsics_w2c, pose_src = self._extract_camera_params(
                metadata=metadata,
                image_files=image_files,
                num_frames=num_frames,
                sequence_name=sequence_name,
            )

            if intrinsics.shape != (num_frames, 3, 3):
                ok = False
                msgs.append(f"intrinsics shape invalid: {intrinsics.shape}")

            if extrinsics_w2c.shape != (num_frames, 4, 4):
                ok = False
                msgs.append(f"extrinsics shape invalid: {extrinsics_w2c.shape}")

            if not np.isfinite(intrinsics).all():
                ok = False
                msgs.append("intrinsics contains non-finite values")

            if not np.isfinite(extrinsics_w2c).all():
                ok = False
                msgs.append("extrinsics contains non-finite values")

            msgs.append(f"pose_source: {pose_src}")

        except Exception as e:
            ok = False
            msgs.append(f"camera extraction failed: {repr(e)}")

        if len(image_files) > 0:
            try:
                img = self._read_image(image_files[0])
                if img.ndim != 3 or img.shape[2] != 3:
                    ok = False
                    msgs.append(f"first image shape invalid: {img.shape}")
            except Exception as e:
                ok = False
                msgs.append(f"failed to read first image: {repr(e)}")

        if len(depth_files) > 0:
            try:
                dep = self._read_depth(depth_files[0])
                if dep.ndim != 2:
                    ok = False
                    msgs.append(f"first depth shape invalid: {dep.shape}")
                if not np.isfinite(dep).all():
                    ok = False
                    msgs.append("first depth contains non-finite values")
            except Exception as e:
                ok = False
                msgs.append(f"failed to read first depth: {repr(e)}")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
        }

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> list[str]:
        sequence_names: list[str] = []

        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue

            if (
                (p / "images").exists()
                and (p / "depths").exists()
                and (p / "metadata.json").exists()
            ):
                sequence_names.append(p.name)

        if len(sequence_names) == 0:
            raise RuntimeError(f"No valid ScanNet scenes found under: {self.root}")
        return sequence_names

    def _check_scene_exists(self, scene_dir: Path, sequence_name: str) -> None:
        if not scene_dir.exists():
            raise FileNotFoundError(f"[{sequence_name}] scene dir not found: {scene_dir}")
        if not (scene_dir / "images").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing images dir: {scene_dir / 'images'}")
        if not (scene_dir / "depths").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing depths dir: {scene_dir / 'depths'}")
        if not (scene_dir / "metadata.json").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing metadata.json: {scene_dir / 'metadata.json'}")
        
    def _check_indices(self, frame_indices: list[int], num_frames: int, sequence_name: str) -> None:
        if len(frame_indices) == 0:
            raise ValueError("frame_indices is empty")

        if min(frame_indices) < 0 or max(frame_indices) >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={min(frame_indices)}, max={max(frame_indices)}, num_frames={num_frames}"
            )

    # ------------------------------------------------------------------
    # file reading
    # ------------------------------------------------------------------

    def _find_image_files(self, image_dir: Path) -> list[Path]:
        files: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(image_dir.glob(ext))
        return sorted(files, key=self._natural_sort_key)

    def _find_depth_files(self, depth_dir: Path) -> list[Path]:
        files: list[Path] = []
        for ext in ("*.png", "*.npy"):
            files.extend(depth_dir.glob(ext))
        return sorted(files, key=self._natural_sort_key)

    def _read_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    def _read_image_size(self, path: Path) -> tuple[int, int]:
        img = Image.open(path)
        return img.height, img.width

    def _read_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            depth = np.load(path).astype(np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            return depth

        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]

        arr = arr.astype(np.float32)
        depth = arr / self.depth_scale
        return depth

    def _load_metadata(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_num_frames(
        self,
        metadata: dict[str, Any],
        image_files: list[Path],
        depth_files: list[Path],
    ) -> int:
        candidates = []

        if "sequence_length" in metadata:
            try:
                candidates.append(int(metadata["sequence_length"]))
            except Exception:
                pass

        if len(image_files) > 0:
            candidates.append(len(image_files))
        if len(depth_files) > 0:
            candidates.append(len(depth_files))

        if len(candidates) == 0:
            raise RuntimeError("Cannot infer num_frames from metadata or files")

        return min(candidates)

    # ------------------------------------------------------------------
    # camera extraction
    # ------------------------------------------------------------------

    def _extract_camera_params(
        self,
        metadata: dict[str, Any],
        image_files: list[Path],
        num_frames: int,
        sequence_name: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        intrinsics = self._extract_top_level_intrinsics(metadata, image_files, num_frames)
        extrinsics_w2c, pose_src = self._extract_top_level_extrinsics(metadata, image_files, num_frames)
        return intrinsics, extrinsics_w2c, pose_src

    def _extract_top_level_intrinsics(
        self,
        metadata: dict[str, Any],
        image_files: list[Path],
        num_frames: int,
    ) -> np.ndarray:
        """
        Returns:
            intrinsics: [T,3,3]

        Supports:
        1. metadata["intrinsics"] is already a matrix / array
        2. metadata["intrinsics"] is a dict with shared matrix
        3. metadata["intrinsics"] is a dict with fx/fy/cx/cy
        4. metadata["intrinsics"] contains nested dicts like color/rgb/image/camera/depth
        5. metadata["intrinsics"] is a per-frame dict keyed by frame id or filename
        """
        if "intrinsics" not in metadata:
            raise KeyError("metadata missing key: intrinsics")

        intr = metadata["intrinsics"]

        # Case A: already array-like
        if not isinstance(intr, dict):
            arr = np.asarray(intr, dtype=np.float32)
            if arr.shape == (3, 3):
                return np.repeat(arr[None, :, :], num_frames, axis=0)
            if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
                if arr.shape[0] < num_frames:
                    raise ValueError(f"intrinsics has only {arr.shape[0]} frames, expected {num_frames}")
                return arr[:num_frames]
            raise ValueError(f"Unsupported intrinsics array shape: {arr.shape}")

        # Case B: direct parse from current dict
        parsed = self._extract_intrinsics_from_nested_dict(intr, num_frames)
        if parsed is not None:
            return parsed

        # Case C: nested dict parse
        for subkey in ["color", "rgb", "image", "images", "camera", "depth"]:
            if subkey in intr and isinstance(intr[subkey], dict):
                parsed = self._extract_intrinsics_from_nested_dict(intr[subkey], num_frames)
                if parsed is not None:
                    return parsed

        # Case D: per-frame dict keyed by filename / frame id
        per_frame_mats = self._extract_matrices_by_image_order(
            container=intr,
            image_files=image_files,
            num_frames=num_frames,
            matrix_shape=(3, 3),
        )
        if per_frame_mats is not None:
            return per_frame_mats

        # Case E: generic per-frame dict
        per_frame_mats = self._extract_per_frame_matrices_from_dict(
            intr,
            num_frames=num_frames,
            matrix_shape=(3, 3),
        )
        if per_frame_mats is not None:
            return per_frame_mats

        raise KeyError(
            "Could not parse metadata['intrinsics']. "
            "Supported patterns: shared 3x3 matrix, fx/fy/cx/cy, nested color/rgb/image/camera dict, "
            "or per-frame dict keyed by frame id / filename."
        )

    def _extract_intrinsics_from_nested_dict(
        self,
        d: dict[str, Any],
        num_frames: int,
    ) -> Optional[np.ndarray]:
        """
        Try to parse intrinsics from a dict using common field names.
        """

        # shared 3x3 matrix candidates
        mat = self._extract_matrix_like_from_dict(
            d,
            candidate_keys=[
                "shared_intrinsics",
                "K",
                "matrix",
                "intrinsic_matrix",
                "camera_matrix",
                "intrinsics",
            ],
            allowed_shapes=[(3, 3)],
        )
        if mat is not None:
            return np.repeat(mat[None, :, :], num_frames, axis=0)

        # standard fx/fy/cx/cy
        fx = d.get("fx", d.get("f_x", d.get("focal_x", d.get("focal_length_x", None))))
        fy = d.get("fy", d.get("f_y", d.get("focal_y", d.get("focal_length_y", None))))
        cx = d.get("cx", d.get("c_x", d.get("principal_x", d.get("principal_point_x", None))))
        cy = d.get("cy", d.get("c_y", d.get("principal_y", d.get("principal_point_y", None))))
        if fx is not None and fy is not None and cx is not None and cy is not None:
            K = np.array(
                [
                    [float(fx), 0.0, float(cx)],
                    [0.0, float(fy), float(cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return np.repeat(K[None, :, :], num_frames, axis=0)

        # ScanNet-style color intrinsics
        fx = d.get("fx_color", None)
        fy = d.get("fy_color", None)
        cx = d.get("mx_color", d.get("cx_color", None))
        cy = d.get("my_color", d.get("cy_color", None))
        if fx is not None and fy is not None and cx is not None and cy is not None:
            K = np.array(
                [
                    [float(fx), 0.0, float(cx)],
                    [0.0, float(fy), float(cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return np.repeat(K[None, :, :], num_frames, axis=0)

        # ScanNet-style depth intrinsics
        fx = d.get("fx_depth", None)
        fy = d.get("fy_depth", None)
        cx = d.get("mx_depth", d.get("cx_depth", None))
        cy = d.get("my_depth", d.get("cy_depth", None))
        if fx is not None and fy is not None and cx is not None and cy is not None:
            K = np.array(
                [
                    [float(fx), 0.0, float(cx)],
                    [0.0, float(fy), float(cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return np.repeat(K[None, :, :], num_frames, axis=0)

        return None

    def _extract_top_level_extrinsics(
        self,
        metadata: dict[str, Any],
        image_files: list[Path],
        num_frames: int,
    ) -> tuple[np.ndarray, str]:
        """
        Returns:
            extrinsics_w2c: [T,4,4]
            pose_source: str
        """
        if "extrinsics" not in metadata:
            raise KeyError("metadata missing key: extrinsics")

        extr = metadata["extrinsics"]

        # Case A: already array-like
        if not isinstance(extr, dict):
            arr = np.asarray(extr, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] >= num_frames and arr.shape[-2:] in [(4, 4), (3, 4)]:
                arr = arr[:num_frames]
                arr44 = self._to_4x4(arr)
                pose_src = f"top_level:extrinsics({self.default_pose_convention})"
                return self._to_w2c(arr44, self.default_pose_convention), pose_src
            raise ValueError(f"Unsupported extrinsics array shape: {arr.shape}")

        # Case B: dict with explicit shared arrays
        explicit_candidates = [
            ("w2c", "w2c"),
            ("camera_from_world", "w2c"),
            ("c2w", "c2w"),
            ("world_from_camera", "c2w"),
            ("poses", self.default_pose_convention),
            ("matrices", self.default_pose_convention),
            ("extrinsics", self.default_pose_convention),
        ]

        for key, convention in explicit_candidates:
            if key not in extr:
                continue
            arr = np.asarray(extr[key], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] >= num_frames and arr.shape[-2:] in [(4, 4), (3, 4)]:
                arr = arr[:num_frames]
                arr44 = self._to_4x4(arr)
                pose_src = f"top_level:extrinsics.{key}({convention})"
                return self._to_w2c(arr44, convention), pose_src

        # Case C: per-frame dict keyed by filename / frame id
        mats, convs = self._extract_pose_matrices_by_image_order(
            container=extr,
            image_files=image_files,
            num_frames=num_frames,
        )
        if mats is not None:
            convention = convs[0] if len(convs) > 0 else self.default_pose_convention
            arr44 = self._to_4x4(mats)
            pose_src = f"top_level:extrinsics.by_image_order({convention})"
            return self._to_w2c(arr44, convention), pose_src

        # Case D: generic per-frame dict
        items = []
        for k, v in extr.items():
            frame_idx = self._parse_frame_index(k)
            if frame_idx is None:
                continue

            arr, convention = self._extract_pose_from_nested_value(v)
            if arr is None:
                continue

            items.append((frame_idx, arr, convention))

        if len(items) > 0:
            items.sort(key=lambda x: x[0])
            mats = []
            conventions = []
            for _, arr, convention in items[:num_frames]:
                mats.append(arr)
                conventions.append(convention)

            if len(mats) < num_frames:
                raise ValueError(f"extrinsics per-frame dict has only {len(mats)} matrices, expected {num_frames}")

            convention = conventions[0]
            arr = np.stack(mats, axis=0).astype(np.float32)
            arr44 = self._to_4x4(arr)
            pose_src = f"top_level:extrinsics.per_frame({convention})"
            return self._to_w2c(arr44, convention), pose_src

        raise KeyError(
            "Could not parse metadata['extrinsics']. "
            "Supported patterns: explicit w2c/c2w arrays, or per-frame dict keyed by frame id / filename."
        )

    # ------------------------------------------------------------------
    # dict parsing helpers
    # ------------------------------------------------------------------

    def _extract_matrices_by_image_order(
        self,
        container: dict[str, Any],
        image_files: list[Path],
        num_frames: int,
        matrix_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Match matrices using image filenames directly.
        Example:
            metadata["intrinsics"]["002490.png"] = [[...],[...],[...]]
        """
        mats = []

        for img_path in image_files[:num_frames]:
            candidates = [
                img_path.name,                # 002490.png
                img_path.stem,                # 002490
                str(img_path.name),
                str(img_path.stem),
            ]

            found = None
            for key in candidates:
                if key not in container:
                    continue

                v = container[key]
                if isinstance(v, dict):
                    arr = self._extract_matrix_like_from_dict(
                        v,
                        candidate_keys=["K", "matrix", "intrinsics", "intrinsic_matrix", "camera_matrix"],
                        allowed_shapes=[matrix_shape],
                    )
                    if arr is not None:
                        found = arr
                        break
                else:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape == matrix_shape:
                        found = arr
                        break

            if found is None:
                return None

            mats.append(found)

        return np.stack(mats, axis=0).astype(np.float32)

    def _extract_pose_matrices_by_image_order(
        self,
        container: dict[str, Any],
        image_files: list[Path],
        num_frames: int,
    ) -> tuple[Optional[np.ndarray], list[str]]:
        """
        Match pose matrices using image filenames directly.
        Example:
            metadata["extrinsics"]["002490.png"] = [[...],[...],[...],[...]]
            or
            metadata["extrinsics"]["002490.png"] = {"pose": ...}
        """
        mats = []
        conventions = []

        for img_path in image_files[:num_frames]:
            candidates = [
                img_path.name,
                img_path.stem,
                str(img_path.name),
                str(img_path.stem),
            ]

            found_arr = None
            found_conv = None

            for key in candidates:
                if key not in container:
                    continue

                arr, conv = self._extract_pose_from_nested_value(container[key])
                if arr is not None:
                    found_arr = arr
                    found_conv = conv
                    break

            if found_arr is None:
                return None, []

            mats.append(found_arr)
            conventions.append(found_conv or self.default_pose_convention)

        return np.stack(mats, axis=0).astype(np.float32), conventions

    def _extract_per_frame_matrices_from_dict(
        self,
        d: dict[str, Any],
        num_frames: int,
        matrix_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        items = []

        for k, v in d.items():
            frame_idx = self._parse_frame_index(k)
            if frame_idx is None:
                continue

            arr = None

            if isinstance(v, dict):
                arr = self._extract_matrix_like_from_dict(
                    v,
                    candidate_keys=[
                        "K",
                        "matrix",
                        "intrinsics",
                        "intrinsic_matrix",
                        "camera_matrix",
                    ],
                    allowed_shapes=[matrix_shape] if matrix_shape is not None else [(4, 4), (3, 4), (3, 3)],
                )
            else:
                arr0 = np.asarray(v, dtype=np.float32)
                if arr0.ndim == 2:
                    if matrix_shape is not None:
                        if arr0.shape == matrix_shape:
                            arr = arr0
                    else:
                        if arr0.shape in [(4, 4), (3, 4)]:
                            arr = arr0

            if arr is None:
                continue

            items.append((frame_idx, arr))

        if len(items) == 0:
            return None

        items.sort(key=lambda x: x[0])
        mats = [arr for _, arr in items[:num_frames]]

        if len(mats) < num_frames:
            raise ValueError(f"per-frame dict has only {len(mats)} matrices, expected {num_frames}")

        return np.stack(mats, axis=0).astype(np.float32)

    def _extract_pose_from_nested_value(
        self,
        v: Any,
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Parse one extrinsics value. Supports:
        - direct matrix [4,4] or [3,4]
        - dict with keys like w2c / c2w / pose / matrix / extrinsics
        """
        if not isinstance(v, dict):
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 2 and arr.shape in [(4, 4), (3, 4)]:
                return arr, self.default_pose_convention
            return None, None

        explicit = [
            ("w2c", "w2c"),
            ("camera_from_world", "w2c"),
            ("c2w", "c2w"),
            ("world_from_camera", "c2w"),
            ("pose", self.default_pose_convention),
            ("matrix", self.default_pose_convention),
            ("extrinsics", self.default_pose_convention),
        ]

        for key, convention in explicit:
            if key not in v:
                continue
            arr = np.asarray(v[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape in [(4, 4), (3, 4)]:
                return arr, convention

        return None, None

    def _extract_matrix_like_from_dict(
        self,
        d: dict[str, Any],
        candidate_keys: list[str],
        allowed_shapes: list[tuple[int, int]],
    ) -> Optional[np.ndarray]:
        for key in candidate_keys:
            if key not in d:
                continue
            arr = np.asarray(d[key], dtype=np.float32)
            if arr.ndim == 2 and tuple(arr.shape) in allowed_shapes:
                return arr
        return None

    def _parse_frame_index(self, key: Any) -> Optional[int]:
        """
        Accept examples:
            0
            "0"
            "00000"
            "002490.png"
            "frame_0"
            "frame-12"
            "img_23"
            "frame_002490.png"
        """
        if isinstance(key, int):
            return key

        if not isinstance(key, str):
            return None

        key = key.strip()
        stem = Path(key).stem  # 002490.png -> 002490

        if stem.isdigit():
            return int(stem)

        m = re.search(r"(\d+)$", stem)
        if m is not None:
            return int(m.group(1))

        # fallback: last digit group anywhere
        matches = re.findall(r"(\d+)", stem)
        if len(matches) > 0:
            return int(matches[-1])

        return None

    # ------------------------------------------------------------------
    # matrix helpers
    # ------------------------------------------------------------------

    def _to_4x4(self, arr: np.ndarray) -> np.ndarray:
        """
        Supports:
        - [4,4] -> [4,4]
        - [3,4] -> [4,4]
        - [T,4,4] -> [T,4,4]
        - [T,3,4] -> [T,4,4]
        """
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 2 and arr.shape == (4, 4):
            return arr

        if arr.ndim == 2 and arr.shape == (3, 4):
            out = np.zeros((4, 4), dtype=np.float32)
            out[:3, :4] = arr
            out[3, 3] = 1.0
            return out

        if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
            return arr

        if arr.ndim == 3 and arr.shape[-2:] == (3, 4):
            T = arr.shape[0]
            out = np.zeros((T, 4, 4), dtype=np.float32)
            out[:, :3, :4] = arr
            out[:, 3, 3] = 1.0
            return out

        raise ValueError(f"Cannot convert to 4x4 from shape: {arr.shape}")

    def _to_w2c(self, arr44: np.ndarray, convention: str) -> np.ndarray:
        """
        Convert pose matrices to world-to-camera.
        """
        if convention not in ("c2w", "w2c"):
            raise ValueError(f"Invalid pose convention: {convention}")

        if arr44.ndim == 2:
            return arr44 if convention == "w2c" else np.linalg.inv(arr44).astype(np.float32)

        if arr44.ndim == 3:
            if convention == "w2c":
                return arr44
            return np.linalg.inv(arr44).astype(np.float32)

        raise ValueError(f"Invalid pose array ndim: {arr44.ndim}")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def _natural_sort_key(self, path: Path):
        parts = re.split(r"(\d+)", path.name)
        out = []
        for p in parts:
            if p.isdigit():
                out.append(int(p))
            else:
                out.append(p)
        return out