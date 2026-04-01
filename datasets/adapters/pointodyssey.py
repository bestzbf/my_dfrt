from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr", ".npy"}


@dataclass
class SequenceRecord:
    dataset_name: str
    split: str
    sequence_name: str
    sequence_root: Path

    rgb_paths: list[Path]
    depth_paths: Optional[list[Path]]
    normal_paths: Optional[list[Path]]
    mask_paths: Optional[list[Path]]

    anno_path: Optional[Path]
    info_path: Optional[Path]
    scene_info_path: Optional[Path]

    num_frames: int
    image_size: Optional[tuple[int, int]]


class PointOdysseyAdapter(BaseAdapter):
    dataset_name: str = "pointodyssey"

    def __init__(
        self,
        root: str,
        split: str = "train",
        strict: bool = True,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.strict = strict
        self.verbose = verbose

        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(
                f"Split root not found: {split_root}\n"
                f"Expected root like /path/to/PointOdyssey and split like 'train'/'test'."
            )

        self.records: list[SequenceRecord] = self._build_index()
        self.name_to_record: dict[str, SequenceRecord] = {}
        for r in self.records:
            if r.sequence_name in self.name_to_record:
                raise ValueError(f"Duplicate sequence_name found: {r.sequence_name}")
            self.name_to_record[r.sequence_name] = r

        if len(self.records) == 0:
            raise RuntimeError(f"No valid PointOdyssey sequences found under {split_root}")

        self._anno_cache: dict[str, dict[str, Any]] = {}
        self._scene_info_cache: dict[str, Optional[dict[str, Any]]] = {}

        if self.verbose:
            print(
                f"[PointOdysseyAdapter] split={self.split}, "
                f"num_sequences={len(self.records)}, strict={self.strict}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_name for r in self.records]

    def get_sequence_name(self, index: int) -> str:
        return self.records[index].sequence_name

    def get_record(self, sequence_name: str) -> SequenceRecord:
        if sequence_name not in self.name_to_record:
            raise KeyError(f"Unknown sequence_name: {sequence_name}")
        return self.name_to_record[sequence_name]

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)
        return {
            "dataset_name": r.dataset_name,
            "split": r.split,
            "sequence_name": r.sequence_name,
            "sequence_root": str(r.sequence_root),
            "num_frames": r.num_frames,
            "image_size": r.image_size,
            "has_depth": r.depth_paths is not None,
            "has_normals": r.normal_paths is not None,
            "has_masks": r.mask_paths is not None,
            "has_anno": r.anno_path is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
            "has_tracks": r.anno_path is not None,
            "has_visibility": r.anno_path is not None,
            "has_trajs_3d_world": r.anno_path is not None,
        }

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)

        out: dict[str, Any] = {
            "dataset_name": r.dataset_name,
            "split": r.split,
            "sequence_name": r.sequence_name,
            "sequence_root": str(r.sequence_root),
            "num_frames": r.num_frames,
            "image_size": r.image_size,
            "rgb_count": len(r.rgb_paths),
            "depth_count": len(r.depth_paths) if r.depth_paths is not None else None,
            "normal_count": len(r.normal_paths) if r.normal_paths is not None else None,
            "mask_count": len(r.mask_paths) if r.mask_paths is not None else None,
            "has_anno": r.anno_path is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
        }

        if len(r.rgb_paths) != r.num_frames:
            out["error"] = f"rgb count {len(r.rgb_paths)} != num_frames {r.num_frames}"
            return out

        for modal_name, modal_paths in [
            ("depth", r.depth_paths),
            ("normal", r.normal_paths),
            ("mask", r.mask_paths),
        ]:
            if modal_paths is not None and len(modal_paths) != r.num_frames:
                out["error"] = (
                    f"{modal_name} count {len(modal_paths)} != num_frames {r.num_frames}"
                )
                return out

        if r.anno_path is not None:
            anno = self._load_anno(sequence_name)
            required = [
                "trajs_2d",
                "trajs_3d",
                "valids",
                "visibs",
                "intrinsics",
                "extrinsics",
            ]
            missing = [k for k in required if k not in anno]
            if missing:
                out["error"] = f"missing anno keys: {missing}"
                return out

            out["anno_shapes"] = {
                "trajs_2d": tuple(anno["trajs_2d"].shape),
                "trajs_3d": tuple(anno["trajs_3d"].shape),
                "valids": tuple(anno["valids"].shape),
                "visibs": tuple(anno["visibs"].shape),
                "intrinsics": tuple(anno["intrinsics"].shape),
                "extrinsics": tuple(anno["extrinsics"].shape),
            }

        out["ok"] = True
        return out

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        r = self.get_record(sequence_name)

        frame_indices_np = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices_np.ndim != 1:
            raise ValueError("frame_indices must be a 1D list/array of ints.")
        if len(frame_indices_np) == 0:
            raise ValueError("frame_indices must not be empty.")
        if frame_indices_np.min() < 0 or frame_indices_np.max() >= r.num_frames:
            raise IndexError(
                f"frame_indices out of range for sequence '{sequence_name}', "
                f"valid range = [0, {r.num_frames - 1}]"
            )

        anno = self._load_anno(sequence_name)
        scene_info = self._load_scene_info(sequence_name)

        images = [self._read_rgb(r.rgb_paths[i]) for i in frame_indices_np]
        depths = (
            [self._read_depth(r.depth_paths[i]) for i in frame_indices_np]
            if r.depth_paths is not None
            else None
        )
        normals = (
            [self._read_normal(r.normal_paths[i]) for i in frame_indices_np]
            if r.normal_paths is not None
            else None
        )

        trajs_2d = anno["trajs_2d"][frame_indices_np].astype(np.float32)          # [T,N,2]

        # Handle missing trajs_3d (some sequences have empty trajs_3d)
        trajs_3d_raw = anno["trajs_3d"]
        if trajs_3d_raw.ndim == 0 or trajs_3d_raw.shape[0] == 0:
            trajs_3d_world = None
        else:
            trajs_3d_world = trajs_3d_raw[frame_indices_np].astype(np.float32)    # [T,N,3]

        valids = anno["valids"][frame_indices_np].astype(bool)                    # [T,N]
        visibs = anno["visibs"][frame_indices_np].astype(bool)                    # [T,N]
        intrinsics = anno["intrinsics"][frame_indices_np].astype(np.float32)      # [T,3,3]
        extrinsics = anno["extrinsics"][frame_indices_np].astype(np.float32)      # [T,4,4], world_to_camera

        clip = UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=[str(r.rgb_paths[i]) for i in frame_indices_np],
            images=images,
            depths=depths,
            normals=normals,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "split": r.split,
                "sequence_root": str(r.sequence_root),
                "frame_indices": frame_indices_np.tolist(),
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": list(r.image_size) if r.image_size is not None else None,
                "has_depth": depths is not None,
                "has_normals": normals is not None,
                "has_tracks": trajs_3d_world is not None,
                "has_visibility": trajs_3d_world is not None,
                "has_trajs_3d_world": trajs_3d_world is not None,
                "has_masks": r.mask_paths is not None,
                "pose_convention": "world_to_camera",
                "trajs_3d_convention": "world",
                "intrinsics_convention": "pinhole",
                "normal_convention": "unknown",
                "depth_unit": "unknown",
                "raw_mask_semantics": "unknown",
                "scene_info_keys": list(scene_info.keys()) if scene_info is not None else None,
            },
        )
        return clip

    def probe_sequence_meta(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)
        out: dict[str, Any] = {
            "sequence_name": sequence_name,
            "anno": None,
            "info": None,
            "scene_info_json_keys": None,
        }

        if r.anno_path is not None:
            z = np.load(r.anno_path, allow_pickle=True)
            out["anno"] = {
                k: {
                    "shape": getattr(z[k], "shape", None),
                    "dtype": str(getattr(z[k], "dtype", None)),
                }
                for k in z.files
            }

        if r.info_path is not None:
            z = np.load(r.info_path, allow_pickle=True)
            out["info"] = {
                k: {
                    "shape": getattr(z[k], "shape", None),
                    "dtype": str(getattr(z[k], "dtype", None)),
                }
                for k in z.files
            }

        if r.scene_info_path is not None:
            with open(r.scene_info_path, "r") as f:
                meta = json.load(f)
            out["scene_info_json_keys"] = list(meta.keys())

        return out

    def _build_index(self) -> list[SequenceRecord]:
        split_root = self.root / self.split
        scene_dirs = sorted(
            [p for p in split_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

        records: list[SequenceRecord] = []
        skipped: list[str] = []

        for scene_dir in scene_dirs:
            try:
                rec = self._index_scene(scene_dir)
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                if self.strict:
                    raise
                skipped.append(f"{scene_dir.name}: {e}")
                if self.verbose:
                    print(f"[WARN] skip scene {scene_dir.name}: {e}")

        if self.verbose and len(skipped) > 0:
            print(f"[PointOdysseyAdapter] skipped {len(skipped)} scenes in non-strict mode.")

        return records

    def _index_scene(self, scene_dir: Path) -> Optional[SequenceRecord]:
        rgb_dir = scene_dir / "rgbs"
        if not rgb_dir.exists():
            return None

        rgb_files = self._sorted_files(rgb_dir)
        if len(rgb_files) == 0:
            return None

        depth_files = self._sorted_files(scene_dir / "depths") if (scene_dir / "depths").exists() else None
        normal_files = self._sorted_files(scene_dir / "normals") if (scene_dir / "normals").exists() else None
        mask_files = self._sorted_files(scene_dir / "masks") if (scene_dir / "masks").exists() else None

        rgb_map = self._build_frame_map(rgb_files, scene_dir=scene_dir, modal_name="rgbs")
        depth_map = (
            self._build_frame_map(depth_files, scene_dir=scene_dir, modal_name="depths")
            if depth_files is not None
            else None
        )
        normal_map = (
            self._build_frame_map(normal_files, scene_dir=scene_dir, modal_name="normals")
            if normal_files is not None
            else None
        )
        mask_map = (
            self._build_frame_map(mask_files, scene_dir=scene_dir, modal_name="masks")
            if mask_files is not None
            else None
        )

        frame_ids = sorted(rgb_map.keys(), key=lambda x: int(x))

        rgb_paths = [rgb_map[fid] for fid in frame_ids]
        depth_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir,
            modal_name="depths",
            rgb_map=rgb_map,
            modal_map=depth_map,
            frame_ids=frame_ids,
        )
        normal_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir,
            modal_name="normals",
            rgb_map=rgb_map,
            modal_map=normal_map,
            frame_ids=frame_ids,
        )
        mask_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir,
            modal_name="masks",
            rgb_map=rgb_map,
            modal_map=mask_map,
            frame_ids=frame_ids,
        )

        first = self._read_rgb(rgb_paths[0])
        image_size = (int(first.shape[0]), int(first.shape[1]))

        return SequenceRecord(
            dataset_name=self.dataset_name,
            split=self.split,
            sequence_name=scene_dir.name,
            sequence_root=scene_dir,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            normal_paths=normal_paths,
            mask_paths=mask_paths,
            anno_path=(scene_dir / "anno.npz") if (scene_dir / "anno.npz").exists() else None,
            info_path=(scene_dir / "info.npz") if (scene_dir / "info.npz").exists() else None,
            scene_info_path=(scene_dir / "scene_info.json") if (scene_dir / "scene_info.json").exists() else None,
            num_frames=len(rgb_paths),
            image_size=image_size,
        )

    def _align_optional_modal_paths(
        self,
        scene_dir: Path,
        modal_name: str,
        rgb_map: dict[str, Path],
        modal_map: Optional[dict[str, Path]],
        frame_ids: list[str],
    ) -> Optional[list[Path]]:
        if modal_map is None:
            return None

        missing = [fid for fid in frame_ids if fid not in modal_map]
        extra = [fid for fid in modal_map.keys() if fid not in rgb_map]

        if missing or extra:
            msg = (
                f"{scene_dir.name}: {modal_name} not aligned with rgb. "
                f"missing={missing[:10]} (count={len(missing)}), "
                f"extra={extra[:10]} (count={len(extra)})"
            )
            if self.strict:
                raise ValueError(msg)
            if self.verbose:
                print(f"[WARN] {msg}")
            return None

        return [modal_map[fid] for fid in frame_ids]

    def _load_anno(self, sequence_name: str) -> dict[str, Any]:
        if sequence_name in self._anno_cache:
            return self._anno_cache[sequence_name]

        r = self.get_record(sequence_name)
        if r.anno_path is None:
            raise FileNotFoundError(f"Missing anno.npz for sequence: {sequence_name}")

        h5_path = r.anno_path.with_suffix('.h5')
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, 'r') as f:
                anno = {k: f[k][()] for k in f.keys()}
        else:
            z = np.load(r.anno_path, allow_pickle=True)
            anno = {k: z[k] for k in z.files}

        self._anno_cache[sequence_name] = anno
        return anno

    def _load_scene_info(self, sequence_name: str) -> Optional[dict[str, Any]]:
        if sequence_name in self._scene_info_cache:
            return self._scene_info_cache[sequence_name]

        r = self.get_record(sequence_name)
        if r.scene_info_path is None:
            self._scene_info_cache[sequence_name] = None
            return None

        with open(r.scene_info_path, "r") as f:
            meta = json.load(f)

        self._scene_info_cache[sequence_name] = meta
        return meta

    def _sorted_files(self, d: Path) -> list[Path]:
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]

        def sort_key(p: Path):
            fid = self._try_frame_id_from_path(p)
            if fid is not None:
                return (0, int(fid), p.name)
            return (1, p.stem, p.name)

        return sorted(files, key=sort_key)

    def _build_frame_map(
        self,
        files: list[Path],
        scene_dir: Path,
        modal_name: str,
    ) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in files:
            fid = self._frame_id_from_path(p)
            if fid in out:
                raise ValueError(
                    f"{scene_dir.name}: duplicate frame id '{fid}' in {modal_name}: "
                    f"{out[fid].name} and {p.name}"
                )
            out[fid] = p
        return out

    def _try_frame_id_from_path(self, p: Path) -> Optional[str]:
        m = re.search(r"(\d+)$", p.stem)
        if m is None:
            return None
        return m.group(1)

    def _frame_id_from_path(self, p: Path) -> str:
        fid = self._try_frame_id_from_path(p)
        if fid is None:
            raise ValueError(f"Cannot parse frame id from filename: {p.name}")
        return fid

    def _read_rgb(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read RGB image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return arr.astype(np.float32)

        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Failed to read depth: {path}")
        return depth.astype(np.float32)

    def _read_normal(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return arr.astype(np.float32)

        normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise IOError(f"Failed to read normal: {path}")

        if normal.ndim == 2:
            return normal.astype(np.float32)

        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32)
        return normal