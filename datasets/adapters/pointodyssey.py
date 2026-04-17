from __future__ import annotations

import io
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy.lib.format as _npy_fmt  # for lightweight numpy header reads
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr", ".npy"}

FAST_ANNOTATION_DIRNAME = "anno_fast"
FAST_REQUIRED_ANNO_FILES = ("trajs_2d", "trajs_3d", "valids", "intrinsics", "extrinsics")
FAST_FRAME_MANIFEST_FILENAME = "frame_manifest.json"
FAST_ENCODED_META_FILENAME = "frame_pack_meta.json"
FAST_ENCODED_RGB_BIN = "rgb_frames.bin"
FAST_ENCODED_RGB_OFFSETS = "rgb_frames_offsets.npy"
FAST_ENCODED_DEPTH_BIN = "depth_frames.bin"
FAST_ENCODED_DEPTH_OFFSETS = "depth_frames_offsets.npy"
FAST_ENCODED_NORMAL_BIN = "normal_frames.bin"
FAST_ENCODED_NORMAL_OFFSETS = "normal_frames_offsets.npy"
FAST_ENCODED_NORMAL_VALIDS = "normal_frames_valids.npy"


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

    # Fast-loading paths (populated for PointOdyssey_fast sequences)
    fast_dir: Optional[Path] = None
    fast_anno_paths: Optional[dict[str, Path]] = None
    encoded_cache_paths: Optional[dict[str, Path]] = None


class PointOdysseyAdapter(BaseAdapter):
    dataset_name: str = "pointodyssey"

    def __init__(
        self,
        root: str,
        split: str = "train",
        strict: bool = True,
        verbose: bool = True,
        index_workers: int = 8,
        fast_root: Optional[str] = None,
        require_tracks: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.strict = strict
        self.verbose = verbose
        self.index_workers = index_workers
        self.fast_root = Path(fast_root) if fast_root is not None else None
        self.require_tracks = require_tracks

        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(
                f"Split root not found: {split_root}\n"
                f"Expected root like /path/to/PointOdyssey and split like 'train'/'test'."
            )

        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            cache_key = {
                "dataset": self.dataset_name,
                "split": split,
                "root": str(self.root.resolve()),
                "fast_root": str(self.fast_root.resolve()) if self.fast_root is not None else None,
                "require_tracks": self.require_tracks,
                "strict": self.strict,
                "cache_schema": 2,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{split}_{cache_suffix}.pkl"
            self.records: list[SequenceRecord] = load_or_build(self._build_index, _cache_path)
        else:
            self.records: list[SequenceRecord] = self._build_index()
        if self.require_tracks:
            original_count = len(self.records)
            self.records = [r for r in self.records if self._record_has_tracks(r)]
            if self.verbose:
                print(
                    f"[PointOdysseyAdapter] require_tracks=True kept "
                    f"{len(self.records)}/{original_count} sequences"
                )
        self.name_to_record: dict[str, SequenceRecord] = {}
        for r in self.records:
            if r.sequence_name in self.name_to_record:
                raise ValueError(f"Duplicate sequence_name found: {r.sequence_name}")
            self.name_to_record[r.sequence_name] = r

        if len(self.records) == 0:
            raise RuntimeError(f"No valid PointOdyssey sequences found under {split_root}")

        self._anno_cache: dict[str, dict[str, Any]] = {}
        self._scene_info_cache: dict[str, Optional[dict[str, Any]]] = {}
        # Cache for open memmap handles (avoids re-opening on every load_clip call)
        self._encoded_cache_store: dict[str, dict[str, Any]] = {}

        if self.verbose:
            fast_count = sum(1 for r in self.records if r.fast_anno_paths is not None)
            enc_count = sum(1 for r in self.records if r.encoded_cache_paths is not None)
            print(
                f"[PointOdysseyAdapter] split={self.split}, "
                f"num_sequences={len(self.records)}, strict={self.strict}, "
                f"fast_anno={fast_count}, encoded_cache={enc_count}"
            )

    def _record_has_tracks(self, record: SequenceRecord) -> bool:
        if record.fast_anno_paths is not None and "trajs_3d" in record.fast_anno_paths:
            trajs_3d = np.load(record.fast_anno_paths["trajs_3d"], mmap_mode="r")
            return not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)
        if record.anno_path is not None:
            anno = np.load(record.anno_path, allow_pickle=True)
            trajs_3d = anno["trajs_3d"]
            return not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)
        return False

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

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

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read from cached record."""
        return self.get_record(sequence_name).num_frames

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)
        has_tracks = self._record_has_tracks(r)
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
            "has_anno": r.anno_path is not None or r.fast_anno_paths is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            "fast_anno": r.fast_anno_paths is not None,
            "encoded_cache": r.encoded_cache_paths is not None,
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
            "has_anno": r.anno_path is not None or r.fast_anno_paths is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
            "fast_anno": r.fast_anno_paths is not None,
            "encoded_cache": r.encoded_cache_paths is not None,
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

        anno = self._load_anno(sequence_name)
        required = ["trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics"]
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

        if r.encoded_cache_paths is not None:
            # Fast path: decode frames from packed binary cache
            cache = self._get_encoded_cache(sequence_name, r)
            images = [self._decode_rgb_from_cache(cache, int(i)) for i in frame_indices_np]
            depths = [self._decode_depth_from_cache(cache, int(i)) for i in frame_indices_np]
            normals_raw = [self._decode_normal_from_cache(cache, int(i)) for i in frame_indices_np]
            # Replace None (invalid normal) with zeros matching the image shape
            h, w = images[0].shape[:2]
            normals = [
                n if n is not None else np.zeros((h, w, 3), dtype=np.float32)
                for n in normals_raw
            ]
        else:
            # Fallback: read individual frame files
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

        # PointOdyssey depth uses 16-bit visualization units:
        # depth_uint16 = depth_meters * 65535 / 1000.
        if depths is not None:
            depths = [d * (1000.0 / 65535.0) if d is not None else None for d in depths]

        trajs_2d = anno["trajs_2d"][frame_indices_np].astype(np.float32)          # [T,N,2]

        trajs_3d_raw = anno["trajs_3d"]
        if trajs_3d_raw.ndim == 0 or trajs_3d_raw.shape[0] == 0:
            trajs_3d_world = None
        else:
            trajs_3d_world = trajs_3d_raw[frame_indices_np].astype(np.float32)    # [T,N,3]

        valids = anno["valids"][frame_indices_np].astype(bool)                    # [T,N]
        visibs = anno["visibs"][frame_indices_np].astype(bool)                    # [T,N]
        intrinsics = anno["intrinsics"][frame_indices_np].astype(np.float32)      # [T,3,3]
        extrinsics = anno["extrinsics"][frame_indices_np].astype(np.float32)      # [T,4,4]

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
                "normal_convention": "blender_pass_decoded",
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

        if r.fast_anno_paths is not None:
            out["anno"] = {
                k: {
                    "shape": tuple(np.load(p, mmap_mode="r", allow_pickle=False).shape),
                    "dtype": str(np.load(p, mmap_mode="r", allow_pickle=False).dtype),
                }
                for k, p in r.fast_anno_paths.items()
            }
        elif r.anno_path is not None:
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

    # ------------------------------------------------------------------ #
    #  Index building                                                      #
    # ------------------------------------------------------------------ #

    def _build_index(self) -> list[SequenceRecord]:
        split_root = self.root / self.split
        scene_dirs = sorted(
            [p for p in split_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

        results: list[Optional[SequenceRecord]] = [None] * len(scene_dirs)
        skipped: list[str] = []

        def _index_one(args):
            i, scene_dir = args
            try:
                rec = self._index_scene(scene_dir)
                return i, rec, None
            except Exception as e:
                return i, None, (scene_dir.name, e)

        n_workers = min(self.index_workers, len(scene_dirs))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_index_one, (i, d))
                    for i, d in enumerate(scene_dirs)
                ]
                for fut in as_completed(futures):
                    i, rec, err = fut.result()
                    if err is not None:
                        name, e = err
                        if self.strict:
                            raise e
                        skipped.append(f"{name}: {e}")
                        if self.verbose:
                            print(f"[WARN] skip scene {name}: {e}")
                    elif rec is not None:
                        results[i] = rec
        else:
            for i, scene_dir in enumerate(scene_dirs):
                i, rec, err = _index_one((i, scene_dir))
                if err is not None:
                    name, e = err
                    if self.strict:
                        raise e
                    skipped.append(f"{name}: {e}")
                    if self.verbose:
                        print(f"[WARN] skip scene {name}: {e}")
                elif rec is not None:
                    results[i] = rec

        valid_records = [r for r in results if r is not None]

        if self.verbose and skipped:
            print(f"[PointOdysseyAdapter] skipped {len(skipped)} scenes in non-strict mode.")

        return valid_records

    def _index_scene(self, scene_dir: Path) -> Optional[SequenceRecord]:
        fast_dir = scene_dir / FAST_ANNOTATION_DIRNAME
        if fast_dir.is_dir():
            return self._index_scene_fast(scene_dir, fast_dir)

        if self.fast_root is not None:
            external_fast_dir = self.fast_root / self.split / scene_dir.name / FAST_ANNOTATION_DIRNAME
            if external_fast_dir.is_dir():
                return self._index_scene_fast(scene_dir, external_fast_dir)

        return self._index_scene_original(scene_dir)

    def _index_scene_fast(self, scene_dir: Path, fast_dir: Path) -> Optional[SequenceRecord]:
        """Index a PointOdyssey_fast sequence (all data in anno_fast/).

        Key optimizations vs naive approach:
        - Single os.listdir() replaces N individual stat() calls
        - Frame count from numpy file header (~128 bytes) — no mmap, no JSON parse
        - Path lists generated from known PointOdyssey naming convention
        - Image size from the tiny motion-boundary meta JSON
        """
        # Single directory scan → O(1) set lookups replace O(N) stat() calls
        present = {p.name for p in fast_dir.iterdir()}

        pack_meta_path = fast_dir / FAST_ENCODED_META_FILENAME

        # Frame count: read only the numpy file header (128–256 bytes, no data load)
        num_frames = 0
        trajs_npy_name = "trajs_2d.npy"
        if trajs_npy_name in present:
            with open(fast_dir / trajs_npy_name, "rb") as _f:
                try:
                    _ver = _npy_fmt.read_magic(_f)
                    _header_reader = (
                        _npy_fmt.read_array_header_1_0 if _ver == (1, 0)
                        else _npy_fmt.read_array_header_2_0
                    )
                    _shape, _, _ = _header_reader(_f)
                    num_frames = int(_shape[0])
                except Exception:
                    pass
        if num_frames == 0 and pack_meta_path.name in present:
            with open(pack_meta_path, "r") as _f:
                num_frames = int(json.load(_f).get("frames", 0))
        if num_frames == 0:
            return None

        # PointOdyssey canonical suffixes — avoids reading the large manifest JSON
        depth_suffix = ".png"
        normal_suffix = ".jpg"

        # Generate path lists from known naming convention (no I/O)
        # Generate path lists as strings — 5× cheaper than pathlib Path division,
        # since fast sequences load frames via encoded cache anyway (paths are labels only).
        _sd = str(scene_dir)
        rgb_paths = [f"{_sd}/rgbs/rgb_{i:05d}.jpg" for i in range(num_frames)]
        depth_paths = [f"{_sd}/depths/depth_{i:05d}{depth_suffix}" for i in range(num_frames)]
        normal_paths = [f"{_sd}/normals/normal_{i:05d}{normal_suffix}" for i in range(num_frames)]

        # Image size from the tiny motion-boundary meta JSON (has height/width)
        image_size: Optional[tuple[int, int]] = None
        for stride in (1, 2, 3, 4):
            mb_name = f"motion_boundary_stride_{stride:02d}_meta.json"
            if mb_name in present:
                with open(fast_dir / mb_name, "r") as f:
                    mb = json.load(f)
                image_size = (int(mb["height"]), int(mb["width"]))
                break

        # Fast annotation .npy files (loaded on demand with mmap)
        fast_anno_paths: dict[str, Path] = {}
        for key in FAST_REQUIRED_ANNO_FILES:
            fname = f"{key}.npy"
            if fname in present:
                fast_anno_paths[key] = fast_dir / fname
        if "visibs.npy" in present:
            fast_anno_paths["visibs"] = fast_dir / "visibs.npy"

        # Encoded frame cache (set membership, no stat() calls)
        encoded_cache_paths: Optional[dict[str, Path]] = None
        enc_names = {
            "rgb_bin_path": FAST_ENCODED_RGB_BIN,
            "rgb_offsets_path": FAST_ENCODED_RGB_OFFSETS,
            "depth_bin_path": FAST_ENCODED_DEPTH_BIN,
            "depth_offsets_path": FAST_ENCODED_DEPTH_OFFSETS,
            "normal_bin_path": FAST_ENCODED_NORMAL_BIN,
            "normal_offsets_path": FAST_ENCODED_NORMAL_OFFSETS,
            "normal_valids_path": FAST_ENCODED_NORMAL_VALIDS,
        }
        if all(name in present for name in enc_names.values()):
            encoded_cache_paths = {k: fast_dir / name for k, name in enc_names.items()}
            encoded_cache_paths["meta_path"] = pack_meta_path

        return SequenceRecord(
            dataset_name=self.dataset_name,
            split=self.split,
            sequence_name=scene_dir.name,
            sequence_root=scene_dir,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            normal_paths=normal_paths,
            mask_paths=None,
            anno_path=(scene_dir / "anno.npz") if (scene_dir / "anno.npz").exists() else None,
            info_path=(scene_dir / "info.npz") if (scene_dir / "info.npz").exists() else None,
            scene_info_path=(scene_dir / "scene_info.json") if (scene_dir / "scene_info.json").exists() else None,
            num_frames=num_frames,
            image_size=image_size,
            fast_dir=fast_dir,
            fast_anno_paths=fast_anno_paths if fast_anno_paths else None,
            encoded_cache_paths=encoded_cache_paths,
        )

    def _index_scene_original(self, scene_dir: Path) -> Optional[SequenceRecord]:
        """Index a classic PointOdyssey sequence (rgbs/ depths/ normals/ subdirs)."""
        from PIL import Image

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
            scene_dir=scene_dir, modal_name="depths",
            rgb_map=rgb_map, modal_map=depth_map, frame_ids=frame_ids,
        )
        normal_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir, modal_name="normals",
            rgb_map=rgb_map, modal_map=normal_map, frame_ids=frame_ids,
        )
        mask_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir, modal_name="masks",
            rgb_map=rgb_map, modal_map=mask_map, frame_ids=frame_ids,
        )

        with Image.open(rgb_paths[0]) as _img:
            image_size = (_img.height, _img.width)

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

    # ------------------------------------------------------------------ #
    #  Annotation loading                                                  #
    # ------------------------------------------------------------------ #

    def _load_anno(self, sequence_name: str) -> dict[str, Any]:
        if sequence_name in self._anno_cache:
            return self._anno_cache[sequence_name]

        r = self.get_record(sequence_name)

        # Fast path: individual .npy files with memory-mapped lazy loading
        if r.fast_anno_paths:
            anno: dict[str, Any] = {}
            for key in FAST_REQUIRED_ANNO_FILES:
                if key in r.fast_anno_paths:
                    anno[key] = np.load(r.fast_anno_paths[key], mmap_mode="r", allow_pickle=False)
            if "visibs" in r.fast_anno_paths:
                anno["visibs"] = np.load(r.fast_anno_paths["visibs"], mmap_mode="r", allow_pickle=False)
            elif "valids" in anno:
                anno["visibs"] = anno["valids"]
            self._anno_cache[sequence_name] = anno
            return anno

        if r.anno_path is None:
            raise FileNotFoundError(f"Missing anno.npz for sequence: {sequence_name}")

        # HDF5 chunked format (faster random-access than npz)
        h5_path = r.anno_path.with_suffix(".h5")
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, "r") as f:
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

    # ------------------------------------------------------------------ #
    #  Encoded frame cache                                                 #
    # ------------------------------------------------------------------ #

    def _get_encoded_cache(self, sequence_name: str, r: SequenceRecord) -> dict[str, Any]:
        """Open (or retrieve cached) memmap handles for a sequence's binary cache."""
        if sequence_name in self._encoded_cache_store:
            return self._encoded_cache_store[sequence_name]

        paths = r.encoded_cache_paths
        depth_suffix = (
            Path(r.depth_paths[0]).suffix.lower() if r.depth_paths else ".png"
        )
        normal_suffix = (
            Path(r.normal_paths[0]).suffix.lower() if r.normal_paths else ".jpg"
        )

        cache: dict[str, Any] = {
            "rgb_bin":      np.memmap(paths["rgb_bin_path"],    mode="r", dtype=np.uint8),
            "rgb_offsets":  np.load(paths["rgb_offsets_path"],  mmap_mode="r", allow_pickle=False),
            "depth_bin":    np.memmap(paths["depth_bin_path"],  mode="r", dtype=np.uint8),
            "depth_offsets": np.load(paths["depth_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_bin":   np.memmap(paths["normal_bin_path"], mode="r", dtype=np.uint8),
            "normal_offsets": np.load(paths["normal_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_valids": np.load(paths["normal_valids_path"], mmap_mode="r", allow_pickle=False),
            "depth_suffix":  depth_suffix,
            "normal_suffix": normal_suffix,
        }
        self._encoded_cache_store[sequence_name] = cache
        return cache

    @staticmethod
    def _slice_cache_bytes(bin_arr: np.ndarray, offsets: np.ndarray, frame_pos: int) -> np.ndarray:
        start = int(offsets[frame_pos])
        end = int(offsets[frame_pos + 1])
        return np.asarray(bin_arr[start:end], dtype=np.uint8)

    def _decode_rgb_from_cache(self, cache: dict, frame_pos: int) -> np.ndarray:
        encoded = self._slice_cache_bytes(cache["rgb_bin"], cache["rgb_offsets"], frame_pos)
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode RGB frame {frame_pos} from encoded cache")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _decode_depth_from_cache(self, cache: dict, frame_pos: int) -> np.ndarray:
        encoded = self._slice_cache_bytes(cache["depth_bin"], cache["depth_offsets"], frame_pos)
        suffix = cache["depth_suffix"]
        if suffix == ".npy":
            depth = np.load(io.BytesIO(encoded.tobytes()))
        else:
            depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to decode depth frame {frame_pos} from encoded cache")
        return depth.astype(np.float32)

    def _decode_normal_from_cache(self, cache: dict, frame_pos: int) -> Optional[np.ndarray]:
        is_valid = bool(np.asarray(cache["normal_valids"][frame_pos]).item())
        if not is_valid:
            return None
        encoded = self._slice_cache_bytes(cache["normal_bin"], cache["normal_offsets"], frame_pos)
        if len(encoded) == 0:
            return None
        suffix = cache["normal_suffix"]
        if suffix == ".npy":
            arr = np.load(io.BytesIO(encoded.tobytes())).astype(np.float32)
            return self._decode_normal_map(arr)
        normal = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if normal is None:
            return None
        if normal.ndim == 2:
            return normal.astype(np.float32)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return self._decode_normal_map(normal)

    # ------------------------------------------------------------------ #
    #  File-based frame reading (original format fallback)                 #
    # ------------------------------------------------------------------ #

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
            return self._decode_normal_map(arr.astype(np.float32))

        normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise IOError(f"Failed to read normal: {path}")

        if normal.ndim == 2:
            return normal.astype(np.float32)

        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return self._decode_normal_map(normal)

    def _decode_normal_map(self, normal: np.ndarray) -> np.ndarray:
        """
        Decode PointOdyssey normal maps to unit vectors in [-1, 1].

        Official PointOdyssey export stores Blender normal pass values in
        [-1, 1], converts them to uint16 PNG, and later re-saves those PNGs
        as JPG for the released dataset. The loader therefore needs to invert
        that visualization encoding instead of treating the file as plain RGB.
        """
        arr = normal.astype(np.float32, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return arr.astype(np.float32, copy=False)

        valid_threshold = 1e-6
        if np.issubdtype(normal.dtype, np.integer):
            max_val = 65535.0 if normal.dtype == np.uint16 else 255.0
            arr = arr / max_val
            arr = arr * 2.0 - 1.0
            # JPG re-encoding perturbs true zero-vectors to a tiny shell around 0.
            valid_threshold = 5e-2
        else:
            if arr.size == 0:
                return arr.astype(np.float32, copy=False)
            if arr.min() >= 0.0 and arr.max() <= 1.0:
                arr = arr * 2.0 - 1.0
            elif arr.min() < -1.0 or arr.max() > 1.0:
                arr = arr / 255.0
                arr = arr * 2.0 - 1.0

        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        finite = np.isfinite(arr).all(axis=-1, keepdims=True)
        valid = finite & (norm > valid_threshold)
        safe_norm = np.where(valid, norm, 1.0)
        decoded = np.where(valid, arr / safe_norm, 0.0)
        return decoded.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    #  Index helpers                                                       #
    # ------------------------------------------------------------------ #

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
