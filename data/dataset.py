import glob
import io
import json
import math
import os
import random
import re
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.patches import extract_local_patches


CANONICAL_QUERY_SPACE_CROP_NORMALIZED = 0
FAST_ANNOTATION_DIRNAME = "anno_fast"
FAST_FRAME_MANIFEST_FILENAME = "frame_manifest.json"
FAST_REQUIRED_ANNOTATION_KEYS = (
    "trajs_2d",
    "trajs_3d",
    "valids",
    "intrinsics",
    "extrinsics",
)
MOTION_BOUNDARY_CACHE_VERSION = 1
MOTION_BOUNDARY_PACKED_TEMPLATE = "motion_boundary_stride_{stride:02d}_packed.npy"
MOTION_BOUNDARY_META_TEMPLATE = "motion_boundary_stride_{stride:02d}_meta.json"
MOTION_BOUNDARY_BITORDER = "little"
FAST_FRAME_CACHE_META_FILENAME = "frame_cache_meta.json"
FAST_FRAME_CACHE_RGB_FILENAME = "rgb_frames.npy"
FAST_FRAME_CACHE_DEPTH_FILENAME = "depth_frames.npy"
FAST_FRAME_CACHE_NORMAL_FILENAME = "normal_frames.npy"
FAST_FRAME_CACHE_NORMAL_VALIDS_FILENAME = "normal_valids.npy"
FAST_ENCODED_FRAME_CACHE_META_FILENAME = "frame_pack_meta.json"
FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME = "rgb_frames.bin"
FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME = "rgb_frames_offsets.npy"
FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME = "depth_frames.bin"
FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME = "depth_frames_offsets.npy"
FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME = "normal_frames.bin"
FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME = "normal_frames_offsets.npy"
FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME = "normal_frames_valids.npy"


def crop_points_2d(points_2d, x0, y0):
    shifted = points_2d.copy().astype(np.float32)
    shifted[..., 0] -= float(x0)
    shifted[..., 1] -= float(y0)
    return shifted


def normalize_points_2d(points_2d, height, width):
    normalized = points_2d.copy().astype(np.float32)
    normalized[..., 0] /= max(float(width) - 1.0, 1.0)
    normalized[..., 1] /= max(float(height) - 1.0, 1.0)
    return normalized


def crop_intrinsics(intrinsics, x0, y0):
    cropped = intrinsics.copy().astype(np.float32)
    cropped[..., 0, 2] -= float(x0)
    cropped[..., 1, 2] -= float(y0)
    return cropped


def resize_intrinsics(intrinsics, src_h, src_w, dst_h, dst_w):
    resized = intrinsics.copy().astype(np.float32)
    scale_x = float(dst_w) / max(float(src_w), 1.0)
    scale_y = float(dst_h) / max(float(src_h), 1.0)
    resized[..., 0, 0] *= scale_x
    resized[..., 1, 1] *= scale_y
    resized[..., 0, 2] *= scale_x
    resized[..., 1, 2] *= scale_y
    return resized


def transform_intrinsics_for_crop_resize(intrinsics, x0, y0, crop_h, crop_w, dst_h, dst_w):
    intrinsics_crop = crop_intrinsics(intrinsics, x0=x0, y0=y0)
    intrinsics_resized = resize_intrinsics(
        intrinsics_crop,
        src_h=crop_h,
        src_w=crop_w,
        dst_h=dst_h,
        dst_w=dst_w,
    )
    return intrinsics_crop, intrinsics_resized


def decode_pointodyssey_depth(depth):
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / 65535.0 * 1000.0
    return depth.astype(np.float32)


def compute_motion_boundary_mask_for_frame(
    trajs_2d,
    valids,
    frame_idx,
    height,
    width,
    temporal_step=1,
):
    motion_map = np.zeros((height, width), dtype=np.float32)
    support_map = np.zeros((height, width), dtype=np.float32)
    points = trajs_2d[frame_idx]
    current_valid = (valids[frame_idx] > 0.5) & np.isfinite(points).all(axis=1)

    motion_mag = np.zeros((points.shape[0],), dtype=np.float32)
    motion_support = np.zeros((points.shape[0],), dtype=np.float32)

    next_idx = frame_idx + temporal_step
    if next_idx < trajs_2d.shape[0]:
        next_points = trajs_2d[next_idx]
        next_valid = (valids[next_idx] > 0.5) & np.isfinite(next_points).all(axis=1)
        pair_valid = current_valid & next_valid
        if np.any(pair_valid):
            motion_mag[pair_valid] += np.linalg.norm(next_points[pair_valid] - points[pair_valid], axis=1)
            motion_support[pair_valid] += 1.0

    prev_idx = frame_idx - temporal_step
    if prev_idx >= 0:
        prev_points = trajs_2d[prev_idx]
        prev_valid = (valids[prev_idx] > 0.5) & np.isfinite(prev_points).all(axis=1)
        pair_valid = current_valid & prev_valid
        if np.any(pair_valid):
            motion_mag[pair_valid] += np.linalg.norm(points[pair_valid] - prev_points[pair_valid], axis=1)
            motion_support[pair_valid] += 1.0

    motion_defined = motion_support > 0.0
    if not np.any(motion_defined):
        return np.zeros((height, width), dtype=bool)

    motion_mag[motion_defined] /= motion_support[motion_defined]
    src_xy = np.round(points[motion_defined]).astype(np.int32)
    if src_xy.size == 0:
        return np.zeros((height, width), dtype=bool)
    src_x = np.clip(src_xy[:, 0], 0, width - 1)
    src_y = np.clip(src_xy[:, 1], 0, height - 1)
    np.add.at(motion_map, (src_y, src_x), motion_mag[motion_defined])
    np.add.at(support_map, (src_y, src_x), 1.0)

    sampled = support_map > 0.0
    motion_map[sampled] /= support_map[sampled]
    motion_map = cv2.GaussianBlur(motion_map, ksize=(5, 5), sigmaX=0.0, sigmaY=0.0)
    support_mask = cv2.dilate(sampled.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    if not np.any(support_mask):
        return np.zeros((height, width), dtype=bool)

    sobel_x = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    support_values = magnitude[support_mask]
    if support_values.size == 0 or float(np.max(support_values)) <= 0.0:
        return np.zeros((height, width), dtype=bool)

    threshold = max(float(np.percentile(support_values, 85.0)), 1e-6)
    boundary = support_mask & (magnitude >= threshold)
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    return boundary


def collate_fn(batch):
    """Collate successful dataset samples."""
    samples = [sample for sample, success in batch if success and sample]
    if not samples:
        return {}
    from torch.utils.data import default_collate

    filtered_samples = []
    for sample in samples:
        sample = dict(sample)
        if sample.get("local_patches") is None:
            sample.pop("local_patches", None)
        filtered_samples.append(sample)

    if any("video_query" in sample for sample in filtered_samples):
        if not all("video_query" in sample for sample in filtered_samples):
            raise ValueError("video_query must be present for all samples in a batch when enabled")
        max_h = max(sample["video_query"].shape[-2] for sample in filtered_samples)
        max_w = max(sample["video_query"].shape[-1] for sample in filtered_samples)
        for sample in filtered_samples:
            video_query = sample["video_query"]
            pad_h = max_h - video_query.shape[-2]
            pad_w = max_w - video_query.shape[-1]
            if pad_h > 0 or pad_w > 0:
                sample["video_query"] = F.pad(video_query, (0, pad_w, 0, pad_h))

    if all("local_patches" not in sample for sample in filtered_samples) and all(
        "video_query" not in sample for sample in filtered_samples
    ):
        return default_collate(filtered_samples)

    for sample in filtered_samples:
        if sample.get("video_query") is None:
            sample.pop("video_query", None)
    return default_collate(filtered_samples)


class PointOdysseyDataset(Dataset):
    def __init__(
        self,
        dataset_location="/home/zbf/Desktop/d4rt/d4rt-pytorch/data/pointodyssey",
        dset="train",
        use_augs=False,
        S=48,
        N=32,
        strides=None,
        clip_step=2,
        quick=False,
        verbose=False,
        img_size=256,
        num_queries=2048,
        patch_size=9,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        cache_boundaries=True,
        sequence_name=None,
        query_mode="full",
        use_motion_boundaries=True,
        precompute_local_patches=True,
        return_query_video=False,
        local_patch_source="resized",
        return_aux_tensors=True,
        static_scene_frame_idx: Optional[int] = None,
    ):
        self.dataset_location = dataset_location
        self.dset = dset
        self.use_augs = use_augs
        self.S = S
        self.N = N
        self.strides = None if strides is None else list(strides)
        self.clip_step = clip_step
        self.quick = quick
        self.verbose = verbose
        self.img_size = img_size
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.cache_boundaries = cache_boundaries
        self.sequence_name = sequence_name
        self.query_mode = query_mode
        self.use_motion_boundaries = use_motion_boundaries
        self.precompute_local_patches = precompute_local_patches
        self.return_query_video = return_query_video
        self.local_patch_source = local_patch_source
        self.return_aux_tensors = return_aux_tensors
        self.static_scene_frame_idx = static_scene_frame_idx
        self._sequence_asset_cache = {}
        if self.query_mode not in {"full", "target_cam", "same_frame"}:
            raise ValueError(f"Unsupported query_mode: {self.query_mode}")
        if self.local_patch_source not in {"resized", "highres"}:
            raise ValueError(f"Unsupported local_patch_source: {self.local_patch_source}")

        self.root = os.path.join(dataset_location, dset)
        if not os.path.exists(self.root):
            print(f"Warning: Dataset root {self.root} does not exist!")
            self.dirs = []
        else:
            self.dirs = sorted(
                d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            )
            if sequence_name is not None:
                self.dirs = [d for d in self.dirs if d == sequence_name]
            if quick:
                self.dirs = self.dirs[:1]

        if verbose:
            fast_ready = sum(
                1
                for seq_name in self.dirs
                if self._resolve_fast_annotation_paths(os.path.join(self.root, seq_name)) is not None
            )
            motion_ready = sum(
                1
                for seq_name in self.dirs
                if self._resolve_motion_boundary_cache_paths(os.path.join(self.root, seq_name))
            )
            frame_ready = sum(
                1
                for seq_name in self.dirs
                if self._resolve_fast_frame_cache_paths(os.path.join(self.root, seq_name)) is not None
            )
            packed_frame_ready = sum(
                1
                for seq_name in self.dirs
                if self._resolve_encoded_frame_cache_paths(os.path.join(self.root, seq_name)) is not None
            )
            print(f"Found {len(self.dirs)} sequences in {self.root}")
            print(f"Fast annotation cache ready for {fast_ready}/{len(self.dirs)} sequences")
            print(f"Motion-boundary cache ready for {motion_ready}/{len(self.dirs)} sequences")
            print(f"Frame cache ready for {frame_ready}/{len(self.dirs)} sequences")
            print(f"Packed frame cache ready for {packed_frame_ready}/{len(self.dirs)} sequences")

    def _build_frame_file_map(self, files):
        frame_map = {}
        for path in files:
            match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
            if match is None:
                continue
            frame_map[int(match.group(1))] = path
        return frame_map

    def _build_frame_index_map(self, files):
        index_map = {}
        for idx, path in enumerate(files):
            match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
            if match is None:
                continue
            index_map[int(match.group(1))] = idx
        return index_map

    def _resolve_frame_path(self, frame_idx, files, frame_map):
        if frame_idx in frame_map:
            return frame_map[frame_idx]
        if 0 <= frame_idx < len(files):
            return files[frame_idx]
        raise FileNotFoundError(f"Missing file for frame index {frame_idx}")

    def _resolve_frame_position(self, frame_idx, files, index_map):
        if frame_idx in index_map:
            return index_map[frame_idx]
        if 0 <= frame_idx < len(files):
            return frame_idx
        raise FileNotFoundError(f"Missing cached frame index {frame_idx}")

    def __len__(self):
        return len(self.dirs)

    def _resolve_annotation_path(self, seq_path):
        anno_path = os.path.join(seq_path, "anno.npz")
        if os.path.exists(anno_path):
            return anno_path

        npzs = glob.glob(os.path.join(seq_path, "*.npz"))
        if not npzs:
            raise FileNotFoundError(f"No annotation found in {seq_path}")
        return npzs[0]

    def _resolve_fast_annotation_paths(self, seq_path):
        fast_dir = os.path.join(seq_path, FAST_ANNOTATION_DIRNAME)
        if not os.path.isdir(fast_dir):
            return None

        annotation_paths = {}
        for key in FAST_REQUIRED_ANNOTATION_KEYS:
            path = os.path.join(fast_dir, f"{key}.npy")
            if not os.path.exists(path):
                return None
            annotation_paths[key] = path

        visibs_path = os.path.join(fast_dir, "visibs.npy")
        if os.path.exists(visibs_path):
            annotation_paths["visibs"] = visibs_path
        return annotation_paths

    def _resolve_motion_boundary_cache_paths(self, seq_path):
        fast_dir = os.path.join(seq_path, FAST_ANNOTATION_DIRNAME)
        if not os.path.isdir(fast_dir):
            return {}

        cache_paths = {}
        for meta_path in sorted(glob.glob(os.path.join(fast_dir, "motion_boundary_stride_*_meta.json"))):
            match = re.search(r"motion_boundary_stride_(\d+)_meta\.json$", meta_path)
            if match is None:
                continue
            stride = int(match.group(1))
            packed_path = os.path.join(
                fast_dir,
                MOTION_BOUNDARY_PACKED_TEMPLATE.format(stride=stride),
            )
            if not os.path.exists(packed_path):
                continue
            cache_paths[stride] = {
                "meta_path": meta_path,
                "packed_path": packed_path,
            }
        return cache_paths

    def _resolve_fast_frame_cache_paths(self, seq_path):
        fast_dir = os.path.join(seq_path, FAST_ANNOTATION_DIRNAME)
        if not os.path.isdir(fast_dir):
            return None

        meta_path = os.path.join(fast_dir, FAST_FRAME_CACHE_META_FILENAME)
        rgb_path = os.path.join(fast_dir, FAST_FRAME_CACHE_RGB_FILENAME)
        depth_path = os.path.join(fast_dir, FAST_FRAME_CACHE_DEPTH_FILENAME)
        normal_path = os.path.join(fast_dir, FAST_FRAME_CACHE_NORMAL_FILENAME)
        normal_valids_path = os.path.join(fast_dir, FAST_FRAME_CACHE_NORMAL_VALIDS_FILENAME)
        required = (meta_path, rgb_path, depth_path, normal_path, normal_valids_path)
        if not all(os.path.exists(path) for path in required):
            return None
        return {
            "meta_path": meta_path,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "normal_path": normal_path,
            "normal_valids_path": normal_valids_path,
        }

    def _resolve_encoded_frame_cache_paths(self, seq_path):
        fast_dir = os.path.join(seq_path, FAST_ANNOTATION_DIRNAME)
        if not os.path.isdir(fast_dir):
            return None

        meta_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_META_FILENAME)
        rgb_bin_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_RGB_BIN_FILENAME)
        rgb_offsets_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_RGB_OFFSETS_FILENAME)
        depth_bin_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_DEPTH_BIN_FILENAME)
        depth_offsets_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_DEPTH_OFFSETS_FILENAME)
        normal_bin_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_NORMAL_BIN_FILENAME)
        normal_offsets_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_NORMAL_OFFSETS_FILENAME)
        normal_valids_path = os.path.join(fast_dir, FAST_ENCODED_FRAME_CACHE_NORMAL_VALIDS_FILENAME)
        required = (
            meta_path,
            rgb_bin_path,
            rgb_offsets_path,
            depth_bin_path,
            depth_offsets_path,
            normal_bin_path,
            normal_offsets_path,
            normal_valids_path,
        )
        if not all(os.path.exists(path) for path in required):
            return None
        return {
            "meta_path": meta_path,
            "rgb_bin_path": rgb_bin_path,
            "rgb_offsets_path": rgb_offsets_path,
            "depth_bin_path": depth_bin_path,
            "depth_offsets_path": depth_offsets_path,
            "normal_bin_path": normal_bin_path,
            "normal_offsets_path": normal_offsets_path,
            "normal_valids_path": normal_valids_path,
        }

    def _load_fast_frame_manifest(self, seq_path, annotation_paths):
        if annotation_paths is None:
            return None

        fast_dir = os.path.dirname(next(iter(annotation_paths.values())))
        manifest_path = os.path.join(fast_dir, FAST_FRAME_MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        frame_lists = {}
        for key in ("rgb_files", "depth_files", "normal_files"):
            rel_paths = manifest.get(key)
            if not rel_paths:
                return None
            frame_lists[key] = [os.path.join(seq_path, rel_path) for rel_path in rel_paths]
        return frame_lists

    def _find_frame_files(self, directory, patterns):
        for pattern in patterns:
            files = sorted(glob.glob(os.path.join(directory, pattern)))
            if files:
                return files
        return []

    def _load_annotations(self, assets, need_intrinsics):
        if assets["annotation_backend"] == "fast":
            annotation_paths = assets["annotation_paths"]
            trajs_2d = np.load(annotation_paths["trajs_2d"], mmap_mode="r", allow_pickle=False)
            trajs_3d = np.load(annotation_paths["trajs_3d"], mmap_mode="r", allow_pickle=False)
            valids = np.load(annotation_paths["valids"], mmap_mode="r", allow_pickle=False)
            visibilities = (
                np.load(annotation_paths["visibs"], mmap_mode="r", allow_pickle=False)
                if "visibs" in annotation_paths
                else valids
            )
            extrinsics = np.load(annotation_paths["extrinsics"], mmap_mode="r", allow_pickle=False)
            intrinsics = (
                np.load(annotation_paths["intrinsics"], mmap_mode="r", allow_pickle=False)
                if need_intrinsics
                else None
            )
            return trajs_2d, trajs_3d, intrinsics, valids, visibilities, extrinsics

        with np.load(assets["anno_path"], allow_pickle=True) as anno:
            trajs_2d = anno["trajs_2d"]
            trajs_3d = anno["trajs_3d"]
            intrinsics = anno["intrinsics"] if need_intrinsics else None
            valids = anno["valids"]
            visibilities = anno["visibs"] if "visibs" in anno else valids
            extrinsics = anno["extrinsics"]
        return trajs_2d, trajs_3d, intrinsics, valids, visibilities, extrinsics

    def _get_sequence_assets(self, seq_name, seq_path):
        cached = self._sequence_asset_cache.get(seq_name)
        if cached is not None:
            return cached

        rgb_dir = os.path.join(seq_path, "rgbs")
        depth_dir = os.path.join(seq_path, "depths")
        normal_dir = os.path.join(seq_path, "normals")

        annotation_paths = self._resolve_fast_annotation_paths(seq_path)
        fast_manifest = self._load_fast_frame_manifest(seq_path, annotation_paths)
        motion_boundary_cache_paths = self._resolve_motion_boundary_cache_paths(seq_path)
        encoded_frame_cache_paths = self._resolve_encoded_frame_cache_paths(seq_path)
        frame_cache_paths = self._resolve_fast_frame_cache_paths(seq_path)

        if fast_manifest is not None:
            rgb_files = fast_manifest["rgb_files"]
            depth_files = fast_manifest["depth_files"]
            normal_files = fast_manifest["normal_files"]
        else:
            rgb_files = self._find_frame_files(rgb_dir, ("*.jpg", "*.png"))
            depth_files = self._find_frame_files(depth_dir, ("*.png", "*.npy"))
            normal_files = self._find_frame_files(normal_dir, ("*.jpg", "*.png", "*.npy"))

        assets = {
            "annotation_backend": "fast" if annotation_paths is not None else "npz",
            "annotation_paths": annotation_paths,
            "anno_path": None if annotation_paths is not None else self._resolve_annotation_path(seq_path),
            "rgb_files": rgb_files,
            "depth_files": depth_files,
            "normal_files": normal_files,
            "rgb_file_map": self._build_frame_file_map(rgb_files),
            "depth_file_map": self._build_frame_file_map(depth_files),
            "normal_file_map": self._build_frame_file_map(normal_files),
            "rgb_index_map": self._build_frame_index_map(rgb_files),
            "depth_index_map": self._build_frame_index_map(depth_files),
            "normal_index_map": self._build_frame_index_map(normal_files),
            "motion_boundary_cache_paths": motion_boundary_cache_paths,
            "motion_boundary_cache_entries": {},
            "encoded_frame_cache_paths": encoded_frame_cache_paths,
            "encoded_frame_cache_entry": None,
            "frame_cache_paths": frame_cache_paths,
            "frame_cache_entry": None,
        }
        self._sequence_asset_cache[seq_name] = assets
        return assets

    def _load_cached_motion_boundary_masks(self, assets, frame_indices, stride, x0, y0, crop_h, crop_w):
        if not self.cache_boundaries:
            return None

        cache_paths = assets.get("motion_boundary_cache_paths") or {}
        entry = cache_paths.get(int(stride))
        if entry is None:
            return None

        cache_entries = assets.setdefault("motion_boundary_cache_entries", {})
        cached_entry = cache_entries.get(int(stride))
        if cached_entry is None:
            with open(entry["meta_path"], "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            packed = np.load(entry["packed_path"], mmap_mode="r", allow_pickle=False)
            cached_entry = {"meta": meta, "packed": packed}
            cache_entries[int(stride)] = cached_entry

        meta = cached_entry["meta"]
        packed = cached_entry["packed"]
        height = int(meta["height"])
        width = int(meta["width"])
        bitorder = meta.get("bitorder", MOTION_BOUNDARY_BITORDER)

        if y0 < 0 or x0 < 0 or y0 + crop_h > height or x0 + crop_w > width:
            return None

        packed_frames = np.asarray(packed[frame_indices])
        unpacked = np.unpackbits(packed_frames, axis=-1, bitorder=bitorder)[..., :width]
        cropped = unpacked[:, y0:y0 + crop_h, x0:x0 + crop_w].astype(bool, copy=False)
        return [cropped[idx] for idx in range(cropped.shape[0])]

    def _load_frame_cache_entry(self, assets):
        frame_cache_paths = assets.get("frame_cache_paths")
        if frame_cache_paths is None:
            return None

        cached_entry = assets.get("frame_cache_entry")
        if cached_entry is not None:
            return cached_entry

        with open(frame_cache_paths["meta_path"], "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cached_entry = {
            "meta": meta,
            "rgb": np.load(frame_cache_paths["rgb_path"], mmap_mode="r", allow_pickle=False),
            "depth": np.load(frame_cache_paths["depth_path"], mmap_mode="r", allow_pickle=False),
            "normal": np.load(frame_cache_paths["normal_path"], mmap_mode="r", allow_pickle=False),
            "normal_valids": np.load(frame_cache_paths["normal_valids_path"], mmap_mode="r", allow_pickle=False),
        }
        assets["frame_cache_entry"] = cached_entry
        return cached_entry

    def _load_encoded_frame_cache_entry(self, assets):
        encoded_frame_cache_paths = assets.get("encoded_frame_cache_paths")
        if encoded_frame_cache_paths is None:
            return None

        cached_entry = assets.get("encoded_frame_cache_entry")
        if cached_entry is not None:
            return cached_entry

        with open(encoded_frame_cache_paths["meta_path"], "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cached_entry = {
            "meta": meta,
            "rgb_bin": np.memmap(encoded_frame_cache_paths["rgb_bin_path"], mode="r", dtype=np.uint8),
            "rgb_offsets": np.load(encoded_frame_cache_paths["rgb_offsets_path"], mmap_mode="r", allow_pickle=False),
            "depth_bin": np.memmap(encoded_frame_cache_paths["depth_bin_path"], mode="r", dtype=np.uint8),
            "depth_offsets": np.load(encoded_frame_cache_paths["depth_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_bin": np.memmap(encoded_frame_cache_paths["normal_bin_path"], mode="r", dtype=np.uint8),
            "normal_offsets": np.load(encoded_frame_cache_paths["normal_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_valids": np.load(encoded_frame_cache_paths["normal_valids_path"], mmap_mode="r", allow_pickle=False),
        }
        assets["encoded_frame_cache_entry"] = cached_entry
        return cached_entry

    def _get_rngs(self, index):
        # [CRITICAL FIX] If training, we ALWAYS want random sampling, even if use_augs=False.
        # Otherwise the model overfits to a fixed set of query points (index=0 -> fixed seed).
        if self.use_augs or self.dset == 'train':
            return random, np.random
        return random.Random(index), np.random.default_rng(index)

    def _load_rgb(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_depth(self, path):
        if path.endswith(".npy"):
            depth = np.load(path)
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(path)
        return decode_pointodyssey_depth(depth)

    def _load_normal(self, path):
        if path.endswith(".npy"):
            normal = np.load(path).astype(np.float32)
        else:
            normal = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if normal is None:
                raise FileNotFoundError(path)
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normal = normal * 2.0 - 1.0
        return normal

    def _load_normal_or_mask(self, path, fallback_shape):
        try:
            return self._load_normal(path), True
        except (FileNotFoundError, OSError, ValueError):
            height, width = fallback_shape
            return np.zeros((height, width, 3), dtype=np.float32), False

    def _load_rgb_from_cache(self, cache_entry, frame_pos):
        image = np.asarray(cache_entry["rgb"][frame_pos])
        return image.astype(np.float32) / 255.0

    def _load_depth_from_cache(self, cache_entry, frame_pos):
        depth = np.asarray(cache_entry["depth"][frame_pos])
        return decode_pointodyssey_depth(depth)

    def _load_normal_from_cache(self, cache_entry, frame_pos):
        meta = cache_entry["meta"]
        normal = np.asarray(cache_entry["normal"][frame_pos])
        if meta.get("normal_storage") == "rgb_uint8":
            normal = normal.astype(np.float32) / 255.0
            normal = normal * 2.0 - 1.0
            return normal
        return normal.astype(np.float32)

    def _load_normal_or_mask_from_cache(self, cache_entry, frame_pos, fallback_shape):
        is_valid = bool(np.asarray(cache_entry["normal_valids"][frame_pos]).item())
        if not is_valid:
            height, width = fallback_shape
            return np.zeros((height, width, 3), dtype=np.float32), False
        return self._load_normal_from_cache(cache_entry, frame_pos), True

    def _slice_encoded_cache_bytes(self, encoded_bin, offsets, frame_pos):
        start = int(offsets[frame_pos])
        end = int(offsets[frame_pos + 1])
        return np.asarray(encoded_bin[start:end], dtype=np.uint8)

    def _load_rgb_from_encoded_cache(self, cache_entry, frame_pos):
        encoded = self._slice_encoded_cache_bytes(cache_entry["rgb_bin"], cache_entry["rgb_offsets"], frame_pos)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode RGB frame {frame_pos} from encoded cache")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _load_depth_from_encoded_cache(self, cache_entry, frame_pos, suffix):
        encoded = self._slice_encoded_cache_bytes(cache_entry["depth_bin"], cache_entry["depth_offsets"], frame_pos)
        if suffix == ".npy":
            depth = np.load(io.BytesIO(encoded.tobytes()))
        else:
            depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to decode depth frame {frame_pos} from encoded cache")
        return decode_pointodyssey_depth(depth)

    def _load_normal_from_encoded_cache(self, cache_entry, frame_pos, suffix):
        encoded = self._slice_encoded_cache_bytes(cache_entry["normal_bin"], cache_entry["normal_offsets"], frame_pos)
        if suffix == ".npy":
            normal = np.load(io.BytesIO(encoded.tobytes())).astype(np.float32)
            return normal
        if len(encoded) == 0:
            raise ValueError(f"Empty encoded buffer for frame {frame_pos}")
        normal = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise ValueError(f"Failed to decode normal frame {frame_pos} from encoded cache")
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normal = normal * 2.0 - 1.0
        return normal

    def _load_normal_or_mask_from_encoded_cache(self, cache_entry, frame_pos, suffix, fallback_shape):
        is_valid = bool(np.asarray(cache_entry["normal_valids"][frame_pos]).item())
        if not is_valid:
            height, width = fallback_shape
            return np.zeros((height, width, 3), dtype=np.float32), False
        try:
            return self._load_normal_from_encoded_cache(cache_entry, frame_pos, suffix), True
        except (ValueError, cv2.error):
            height, width = fallback_shape
            return np.zeros((height, width, 3), dtype=np.float32), False

    def _sample_stride(self, total_frames, py_rng):
        if self.strides is None:
            max_stride = max(1, total_frames // max(1, self.S))
            # 论文设定：随机 stride
            # 即使 use_augs=False (如验证或过拟合实验)，也应该随机采样 stride
            # 限制 stride 在 1-4 之间 (经验值，防止运动过大)
            limit = min(4, max_stride)
            return py_rng.randint(1, limit) # randint(a, b) 包含 b

        valid_strides = [stride for stride in self.strides if total_frames >= self.S * stride]
        if not valid_strides:
            return 1
        # 即使指定了 stride 列表，也应该随机选择（如果列表有多个）
        return py_rng.choice(valid_strides)

    def _sample_crop(self, height, width, py_rng):
        if not self.use_augs:
            # 验证/过拟合模式：固定中心裁剪
            crop_size = min(height, width)
            x0 = (width - crop_size) // 2
            y0 = (height - crop_size) // 2
            return x0, y0, crop_size, crop_size

        area = height * width
        min_ratio = 3.0 / 4.0
        max_ratio = 4.0 / 3.0

        for _ in range(10):
            target_area = py_rng.uniform(0.3, 1.0) * area
            aspect = math.exp(py_rng.uniform(math.log(min_ratio), math.log(max_ratio)))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            if 0 < crop_w <= width and 0 < crop_h <= height:
                x0 = py_rng.randint(0, width - crop_w)
                y0 = py_rng.randint(0, height - crop_h)
                break
        else:
            crop_w = width
            crop_h = height
            x0 = 0
            y0 = 0

        if py_rng.random() < 0.05:
            zoom = py_rng.uniform(0.7, 0.95)
            zoom_w = max(1, int(round(crop_w * zoom)))
            zoom_h = max(1, int(round(crop_h * zoom)))
            x0 = x0 + max(0, (crop_w - zoom_w) // 2)
            y0 = y0 + max(0, (crop_h - zoom_h) // 2)
            crop_w = zoom_w
            crop_h = zoom_h

        return x0, y0, crop_w, crop_h

    def _apply_color_aug(self, rgb_frames, py_rng):
        if not self.use_augs:
            return rgb_frames

        brightness = py_rng.uniform(0.6, 1.4)
        contrast = py_rng.uniform(0.6, 1.4)
        saturation = py_rng.uniform(0.6, 1.4)
        hue_delta = py_rng.uniform(-0.1, 0.1) * 180.0
        apply_gray = py_rng.random() < 0.2
        apply_blur = py_rng.random() < 0.4
        blur_sigma = py_rng.uniform(0.1, 2.0)

        augmented = []
        for frame in rgb_frames:
            image = np.clip(frame * brightness, 0.0, 1.0)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast + mean, 0.0, 1.0)

            hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 255.0)
            hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            if apply_gray:
                gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
                image = np.repeat(gray[..., None], 3, axis=-1)

            if apply_blur:
                image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=blur_sigma, sigmaY=blur_sigma)

            augmented.append(np.clip(image, 0.0, 1.0))

        return augmented

    def _compute_boundary_mask(self, depth):
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sobel_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            return np.zeros_like(depth, dtype=bool)

        values = magnitude[valid]
        if values.size == 0 or float(np.max(values)) <= 0.0:
            return np.zeros_like(depth, dtype=bool)

        threshold = max(float(np.percentile(values, 85.0)), 1e-6)
        return valid & (magnitude >= threshold)

    def _compute_motion_boundary_mask(self, clip_trajs_2d, clip_valid, frame_idx, crop_h, crop_w):
        return compute_motion_boundary_mask_for_frame(
            trajs_2d=clip_trajs_2d,
            valids=clip_valid,
            frame_idx=frame_idx,
            height=crop_h,
            width=crop_w,
            temporal_step=1,
        )


    def _build_transform_metadata(self, original_h, original_w, x0, y0, crop_h, crop_w):
        return {
            "canonical_space": torch.tensor(CANONICAL_QUERY_SPACE_CROP_NORMALIZED, dtype=torch.long),
            "original_hw": torch.tensor([float(original_h), float(original_w)], dtype=torch.float32),
            "crop_offset_xy": torch.tensor([float(x0), float(y0)], dtype=torch.float32),
            "crop_size_hw": torch.tensor([float(crop_h), float(crop_w)], dtype=torch.float32),
            "resized_hw": torch.tensor([float(self.img_size), float(self.img_size)], dtype=torch.float32),
        }

    def extract_patches(self, video, coords, patch_size=None):
        """
        Args:
            video: (S, H, W, 3) float32 in [0, 1]
            coords: (N, 3) with normalized (u, v, t_idx) in canonical crop-normalized space
        """
        patch_size = self.patch_size if patch_size is None else patch_size
        if torch.is_tensor(video):
            video_tensor = video.to(dtype=torch.float32)
        else:
            video_tensor = torch.from_numpy(video).to(dtype=torch.float32)
        if torch.is_tensor(coords):
            coords_tensor = coords.to(dtype=torch.float32)
        else:
            coords_tensor = torch.from_numpy(coords).to(dtype=torch.float32)

        frames_btchw = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        coords_uv = coords_tensor[:, :2].unsqueeze(0)
        t_src = coords_tensor[:, 2].round().long().unsqueeze(0)
        patches = extract_local_patches(frames_btchw, coords_uv, t_src, patch_size=patch_size)
        return patches.squeeze(0)

    def _sample_query_data(
        self,
        clip_trajs_2d,
        clip_world_3d,
        clip_valid,
        clip_vis,
        clip_extrinsics,
        crop_h,
        crop_w,
        normal_frames,
        normal_valid_frames,
        depth_frames,
        py_rng,
        np_rng,
        motion_boundary_masks=None,
    ):
        num_frames, num_points, _ = clip_trajs_2d.shape

        depth_boundary_masks = [self._compute_boundary_mask(depth) for depth in depth_frames]
        if self.use_motion_boundaries:
            if motion_boundary_masks is None:
                motion_boundary_masks = [
                    self._compute_motion_boundary_mask(clip_trajs_2d, clip_valid, frame_idx, crop_h, crop_w)
                    for frame_idx in range(num_frames)
                ]
        else:
            motion_boundary_masks = [
                np.zeros_like(depth_boundary_masks[frame_idx], dtype=bool)
                for frame_idx in range(num_frames)
            ]
        boundary_masks = [
            np.logical_or(depth_boundary_masks[frame_idx], motion_boundary_masks[frame_idx])
            for frame_idx in range(num_frames)
        ]
        valid_sources_by_frame = []
        boundary_sources_by_frame = []

        for frame_idx in range(num_frames):
            src_points = clip_trajs_2d[frame_idx]
            src_defined = clip_valid[frame_idx] > 0.5
            src_visible = clip_vis[frame_idx] > 0.5
            src_in_bounds = (
                (src_points[:, 0] >= 0.0)
                & (src_points[:, 0] < crop_w)
                & (src_points[:, 1] >= 0.0)
                & (src_points[:, 1] < crop_h)
            )
            src_world_finite = np.isfinite(clip_world_3d[frame_idx]).all(axis=-1)
            valid_src = np.where(src_defined & src_visible & src_in_bounds & src_world_finite)[0]
            valid_sources_by_frame.append(valid_src)

            if len(valid_src) == 0 or self.boundary_ratio <= 0:
                boundary_sources_by_frame.append(np.empty((0,), dtype=np.int64))
                continue

            src_xy = np.round(src_points[valid_src]).astype(np.int32)
            src_xy[:, 0] = np.clip(src_xy[:, 0], 0, crop_w - 1)
            src_xy[:, 1] = np.clip(src_xy[:, 1], 0, crop_h - 1)
            src_boundary = boundary_masks[frame_idx][src_xy[:, 1], src_xy[:, 0]]
            boundary_sources_by_frame.append(valid_src[src_boundary])

        valid_source_frames = [frame_idx for frame_idx, indices in enumerate(valid_sources_by_frame) if len(indices) > 0]
        if not valid_source_frames:
            raise RuntimeError("No valid source points available after augmentation")

        # D4RT Paper: Specialized Querying - Grid Sampling for Camera Pose/Intrinsics
        # We reserve a small portion of queries (e.g., 64) for grid sampling if needed for pose supervision
        # But per user request and paper "Method" section, this is usually for specific tasks.
        # However, the standard training mix includes everything.
        # To strictly follow the "Mix" description: "In a single batch... Group A... Group B".
        # The grid sampling for pose is described as "To derive relative camera pose... we create queries...".
        # It's often implemented as a separate task or mixed in.
        # Given the current structure, we implement the Temporal Sampling Mix (Group A/B) per query.

        N = self.num_queries
        valid_src_frames_arr = np.array(valid_source_frames, dtype=np.int64)

        # ── 1. Generate all source / target / camera frames at once ──────────────
        src_frames = np_rng.choice(valid_src_frames_arr, size=N)  # (N,)

        if self.query_mode == "same_frame":
            tgt_frames = src_frames.copy()
            cam_frames = src_frames.copy()
        elif self.query_mode == "target_cam":
            tgt_frames = np_rng.choice(num_frames, size=N)
            cam_frames = tgt_frames.copy()
        else:
            tgt_frames = np_rng.choice(num_frames, size=N)
            if self.t_tgt_eq_t_cam_ratio >= 1.0:
                cam_frames = tgt_frames.copy()
            else:
                use_same_cam = np_rng.random(size=N) < self.t_tgt_eq_t_cam_ratio
                cam_frames_alt = np_rng.choice(num_frames, size=N)
                cam_frames = np.where(use_same_cam, tgt_frames, cam_frames_alt)

        # ── 2. Point selection – group by source_frame (~S iterations) ���──────────
        use_bnd_mask = np_rng.random(size=N) < self.boundary_ratio  # (N,)
        pt_indices_np = np.empty(N, dtype=np.int64)

        for sf in np.unique(src_frames):
            mask = src_frames == sf
            q_idxs = np.where(mask)[0]
            valid_src = valid_sources_by_frame[sf]
            bnd_cands = boundary_sources_by_frame[sf]
            wants_bnd = use_bnd_mask[q_idxs] & (len(bnd_cands) > 0)

            bnd_q = q_idxs[wants_bnd]
            if len(bnd_q) > 0:
                pt_indices_np[bnd_q] = np_rng.choice(bnd_cands, size=len(bnd_q))

            nbnd_q = q_idxs[~wants_bnd]
            if len(nbnd_q) > 0:
                pt_indices_np[nbnd_q] = np_rng.choice(valid_src, size=len(nbnd_q))

        # ── 3. Batch coordinate lookups ───────────────────────────────────────────
        src_xy = clip_trajs_2d[src_frames, pt_indices_np]   # (N, 2)
        tgt_xy = clip_trajs_2d[tgt_frames, pt_indices_np]   # (N, 2)

        src_px_arr = np.clip(np.round(src_xy[:, 0]).astype(np.int32), 0, crop_w - 1)
        src_py_arr = np.clip(np.round(src_xy[:, 1]).astype(np.int32), 0, crop_h - 1)
        tgt_px_arr = np.clip(np.round(tgt_xy[:, 0]).astype(np.int32), 0, crop_w - 1)
        tgt_py_arr = np.clip(np.round(tgt_xy[:, 1]).astype(np.int32), 0, crop_h - 1)

        # ── 4. Boundary lookups (stack once, fancy-index) ─────────────────────────
        bnd_stack  = np.stack(boundary_masks,       axis=0)  # (S, H, W) bool
        dbnd_stack = np.stack(depth_boundary_masks, axis=0)  # (S, H, W) bool
        mbnd_stack = np.stack(motion_boundary_masks, axis=0) # (S, H, W) bool

        src_is_bnd  = bnd_stack [src_frames, src_py_arr, src_px_arr]  # (N,)
        src_is_dbnd = dbnd_stack[src_frames, src_py_arr, src_px_arr]  # (N,)
        src_is_mbnd = mbnd_stack[src_frames, src_py_arr, src_px_arr]  # (N,)

        # ── 5. Validity flags ─────────────────────────────────────────────────────
        target_defined_arr = clip_valid[tgt_frames, pt_indices_np] > 0.5   # (N,)
        tgt_vis_arr        = clip_vis  [tgt_frames, pt_indices_np] > 0.5   # (N,)
        tgt_in_bounds_arr  = (
            (tgt_xy[:, 0] >= 0.0) & (tgt_xy[:, 0] < crop_w) &
            (tgt_xy[:, 1] >= 0.0) & (tgt_xy[:, 1] < crop_h)
        )
        vis_flags = target_defined_arr & tgt_vis_arr & tgt_in_bounds_arr   # (N,)

        # ── 6. Batch 3D transforms (einsum replaces N individual 4×4 matmuls) ─────
        src_world = clip_world_3d[src_frames, pt_indices_np]  # (N, 3)
        tgt_world = clip_world_3d[tgt_frames, pt_indices_np]  # (N, 3)
        ones_col  = np.ones((N, 1), dtype=np.float32)
        src_world_h = np.concatenate([src_world, ones_col], axis=-1)  # (N, 4)
        tgt_world_h = np.concatenate([tgt_world, ones_col], axis=-1)  # (N, 4)
        cam_ext = clip_extrinsics[cam_frames]                          # (N, 4, 4)
        src_cam = np.einsum('nij,nj->ni', cam_ext, src_world_h)[:, :3].astype(np.float32)  # (N, 3)
        tgt_cam = np.einsum('nij,nj->ni', cam_ext, tgt_world_h)[:, :3].astype(np.float32)  # (N, 3)

        # ── 7. Normal lookups – grouped by target_frame (≤S iterations) ──────────
        normal_valid_arr = np.array(normal_valid_frames, dtype=bool)    # (S,)
        needs_normal     = vis_flags & normal_valid_arr[tgt_frames]      # (N,)
        target_normal_np = np.zeros((N, 3), dtype=np.float32)
        if needs_normal.any():
            for tf in np.unique(tgt_frames[needs_normal]):
                tm = needs_normal & (tgt_frames == tf)
                qi = np.where(tm)[0]
                target_normal_np[qi] = normal_frames[tf][tgt_py_arr[qi], tgt_px_arr[qi]]

        # ── 8. Mask computations ──────────────────────────────────────────────────
        src_finite = np.isfinite(src_cam).all(axis=-1)   # (N,)
        tgt_finite = np.isfinite(tgt_cam).all(axis=-1)   # (N,)
        has_valid_3d = target_defined_arr & src_finite & tgt_finite

        target_normal_t  = torch.from_numpy(target_normal_np)
        has_valid_normal = needs_normal & torch.isfinite(target_normal_t).all(dim=-1).numpy()

        # ── 9. Assemble output tensors ────────────────────────────────────────────
        cw1 = max(crop_w - 1, 1)
        ch1 = max(crop_h - 1, 1)
        targets = {
            "pos_2d":                   torch.from_numpy(np.stack([tgt_xy[:, 0] / cw1, tgt_xy[:, 1] / ch1], axis=-1).astype(np.float32)),
            "pos_3d":                   torch.from_numpy(tgt_cam),
            "visibility":               torch.from_numpy(vis_flags.astype(np.float32)),
            "displacement":             torch.from_numpy((tgt_cam - src_cam)),
            "normal":                   target_normal_t,
            "mask_3d":                  torch.from_numpy(has_valid_3d),
            "mask_2d":                  torch.from_numpy(vis_flags),
            "mask_vis":                 torch.from_numpy(target_defined_arr),
            "mask_disp":                torch.from_numpy(has_valid_3d),
            "mask_normal":              torch.from_numpy(has_valid_normal),
            "source_is_boundary":       torch.from_numpy(src_is_bnd),
            "source_is_depth_boundary": torch.from_numpy(src_is_dbnd),
            "source_is_motion_boundary":torch.from_numpy(src_is_mbnd),
            "point_indices":            torch.from_numpy(pt_indices_np),
        }
        coords_uv   = torch.from_numpy(np.stack([src_xy[:, 0] / cw1, src_xy[:, 1] / ch1], axis=-1).astype(np.float32))
        t_src       = torch.from_numpy(src_frames.astype(np.int64))
        t_tgt       = torch.from_numpy(tgt_frames.astype(np.int64))
        t_cam       = torch.from_numpy(cam_frames.astype(np.int64))

        return coords_uv, t_src, t_tgt, t_cam, targets

    def _resolve_static_scene_frame_idx(self, num_frames):
        if self.static_scene_frame_idx is None:
            return None

        frame_idx = int(self.static_scene_frame_idx)
        if frame_idx < 0:
            frame_idx += num_frames
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(
                f"static_scene_frame_idx={self.static_scene_frame_idx} is out of range for clip length {num_frames}"
            )
        return frame_idx

    def _apply_static_scene_degradation(
        self,
        frame_indices,
        rgb_frames,
        depth_frames,
        normal_frames,
        normal_valid_frames,
        clip_trajs_2d,
        clip_world_3d,
        clip_intrinsics,
        clip_valid,
        clip_vis,
        clip_extrinsics,
    ):
        anchor_idx = self._resolve_static_scene_frame_idx(len(rgb_frames))
        if anchor_idx is None:
            return (
                frame_indices,
                rgb_frames,
                depth_frames,
                normal_frames,
                normal_valid_frames,
                clip_trajs_2d,
                clip_world_3d,
                clip_intrinsics,
                clip_valid,
                clip_vis,
                clip_extrinsics,
            )

        num_frames = len(rgb_frames)
        anchor_frame_index = int(frame_indices[anchor_idx])
        anchor_rgb = rgb_frames[anchor_idx]
        anchor_depth = depth_frames[anchor_idx]
        anchor_normal = normal_frames[anchor_idx]
        anchor_normal_valid = bool(normal_valid_frames[anchor_idx])

        return (
            [anchor_frame_index] * num_frames,
            [anchor_rgb.copy() for _ in range(num_frames)],
            [anchor_depth.copy() for _ in range(num_frames)],
            [anchor_normal.copy() for _ in range(num_frames)],
            [anchor_normal_valid] * num_frames,
            np.repeat(clip_trajs_2d[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_world_3d[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            None if clip_intrinsics is None else np.repeat(clip_intrinsics[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_valid[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_vis[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
            np.repeat(clip_extrinsics[anchor_idx:anchor_idx + 1].copy(), num_frames, axis=0),
        )

    def __getitem__(self, index):
        seq_name = self.dirs[index]
        seq_path = os.path.join(self.root, seq_name)
        py_rng, np_rng = self._get_rngs(index)

        try:
            assets = self._get_sequence_assets(seq_name, seq_path)
            need_intrinsics = self.return_aux_tensors or self.static_scene_frame_idx is not None

            trajs_2d, trajs_3d, intrinsics, valids, visibilities, extrinsics = self._load_annotations(
                assets,
                need_intrinsics=need_intrinsics,
            )
            total_frames = trajs_2d.shape[0]

            if (
                trajs_2d.ndim != 3
                or trajs_3d.ndim != 3
                or (intrinsics is not None and intrinsics.ndim != 3)
                or valids.ndim != 2
                or visibilities.ndim != 2
            ):
                return {}, False

            stride = self._sample_stride(total_frames, py_rng)
            max_safe_start = max(0, total_frames - 1 - (self.S - 1) * stride)
            
            # 即使在 disable-train-augs 下，也恢复随机起始帧采样
            t_start = py_rng.randint(0, max_safe_start)
                
            frame_indices = [t_start + i * stride for i in range(self.S)]

            rgb_files = assets["rgb_files"]
            depth_files = assets["depth_files"]
            normal_files = assets["normal_files"]
            encoded_frame_cache_entry = self._load_encoded_frame_cache_entry(assets)
            frame_cache_entry = None if encoded_frame_cache_entry is not None else self._load_frame_cache_entry(assets)
            if encoded_frame_cache_entry is not None:
                rgb_positions = [
                    self._resolve_frame_position(t, rgb_files, assets["rgb_index_map"])
                    for t in frame_indices
                ]
                depth_positions = [
                    self._resolve_frame_position(t, depth_files, assets["depth_index_map"])
                    for t in frame_indices
                ]
                normal_positions = [
                    self._resolve_frame_position(t, normal_files, assets["normal_index_map"])
                    for t in frame_indices
                ]
                rgb_frames = [
                    self._load_rgb_from_encoded_cache(encoded_frame_cache_entry, frame_pos)
                    for frame_pos in rgb_positions
                ]
                depth_frames = [
                    self._load_depth_from_encoded_cache(
                        encoded_frame_cache_entry,
                        frame_pos,
                        os.path.splitext(depth_files[frame_pos])[1].lower(),
                    )
                    for frame_pos in depth_positions
                ]
                normal_frames = []
                normal_valid_frames = []
                for frame_idx, frame_pos in enumerate(normal_positions):
                    normal_frame, is_valid = self._load_normal_or_mask_from_encoded_cache(
                        encoded_frame_cache_entry,
                        frame_pos,
                        os.path.splitext(normal_files[frame_pos])[1].lower(),
                        rgb_frames[frame_idx].shape[:2],
                    )
                    normal_frames.append(normal_frame)
                    normal_valid_frames.append(is_valid)
            elif frame_cache_entry is not None:
                rgb_positions = [
                    self._resolve_frame_position(t, rgb_files, assets["rgb_index_map"])
                    for t in frame_indices
                ]
                depth_positions = [
                    self._resolve_frame_position(t, depth_files, assets["depth_index_map"])
                    for t in frame_indices
                ]
                normal_positions = [
                    self._resolve_frame_position(t, normal_files, assets["normal_index_map"])
                    for t in frame_indices
                ]
                rgb_frames = [
                    self._load_rgb_from_cache(frame_cache_entry, frame_pos)
                    for frame_pos in rgb_positions
                ]
                depth_frames = [
                    self._load_depth_from_cache(frame_cache_entry, frame_pos)
                    for frame_pos in depth_positions
                ]
                normal_frames = []
                normal_valid_frames = []
                for frame_idx, frame_pos in enumerate(normal_positions):
                    normal_frame, is_valid = self._load_normal_or_mask_from_cache(
                        frame_cache_entry,
                        frame_pos,
                        rgb_frames[frame_idx].shape[:2],
                    )
                    normal_frames.append(normal_frame)
                    normal_valid_frames.append(is_valid)
            else:
                rgb_paths = [self._resolve_frame_path(t, rgb_files, assets["rgb_file_map"]) for t in frame_indices]
                depth_paths = [self._resolve_frame_path(t, depth_files, assets["depth_file_map"]) for t in frame_indices]
                normal_paths = [self._resolve_frame_path(t, normal_files, assets["normal_file_map"]) for t in frame_indices]

                rgb_frames = [self._load_rgb(path) for path in rgb_paths]
                depth_frames = [self._load_depth(path) for path in depth_paths]
                normal_frames = []
                normal_valid_frames = []
                for frame_idx, path in enumerate(normal_paths):
                    normal_frame, is_valid = self._load_normal_or_mask(
                        path,
                        rgb_frames[frame_idx].shape[:2],
                    )
                    normal_frames.append(normal_frame)
                    normal_valid_frames.append(is_valid)

            clip_trajs_2d = trajs_2d[frame_indices].astype(np.float32)
            clip_world_3d = trajs_3d[frame_indices].astype(np.float32)
            clip_intrinsics = intrinsics[frame_indices].astype(np.float32) if intrinsics is not None else None
            clip_valid = valids[frame_indices].astype(np.float32)
            clip_vis = visibilities[frame_indices].astype(np.float32)
            clip_extrinsics = extrinsics[frame_indices].astype(np.float32)

            (
                frame_indices,
                rgb_frames,
                depth_frames,
                normal_frames,
                normal_valid_frames,
                clip_trajs_2d,
                clip_world_3d,
                clip_intrinsics,
                clip_valid,
                clip_vis,
                clip_extrinsics,
            ) = self._apply_static_scene_degradation(
                frame_indices=frame_indices,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                normal_frames=normal_frames,
                normal_valid_frames=normal_valid_frames,
                clip_trajs_2d=clip_trajs_2d,
                clip_world_3d=clip_world_3d,
                clip_intrinsics=clip_intrinsics,
                clip_valid=clip_valid,
                clip_vis=clip_vis,
                clip_extrinsics=clip_extrinsics,
            )

            original_h, original_w = rgb_frames[0].shape[:2]
            x0, y0, crop_w, crop_h = self._sample_crop(original_h, original_w, py_rng)
            cached_motion_boundary_masks = None
            if self.use_motion_boundaries and self.static_scene_frame_idx is None:
                cached_motion_boundary_masks = self._load_cached_motion_boundary_masks(
                    assets,
                    frame_indices=frame_indices,
                    stride=stride,
                    x0=x0,
                    y0=y0,
                    crop_h=crop_h,
                    crop_w=crop_w,
                )

            rgb_frames = [frame[y0:y0 + crop_h, x0:x0 + crop_w] for frame in rgb_frames]
            depth_frames = [depth[y0:y0 + crop_h, x0:x0 + crop_w] for depth in depth_frames]
            normal_frames = [normal[y0:y0 + crop_h, x0:x0 + crop_w] for normal in normal_frames]
            rgb_frames = self._apply_color_aug(rgb_frames, py_rng)

            clip_trajs_2d = crop_points_2d(clip_trajs_2d, x0=x0, y0=y0)
            in_bounds = (
                (clip_trajs_2d[..., 0] >= 0.0)
                & (clip_trajs_2d[..., 0] < crop_w)
                & (clip_trajs_2d[..., 1] >= 0.0)
                & (clip_trajs_2d[..., 1] < crop_h)
            )
            clip_vis = clip_vis * in_bounds.astype(np.float32)
            clip_intrinsics_crop = None
            clip_intrinsics_resized = None
            if clip_intrinsics is not None:
                clip_intrinsics_crop, clip_intrinsics_resized = transform_intrinsics_for_crop_resize(
                    clip_intrinsics,
                    x0=x0,
                    y0=y0,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    dst_h=self.img_size,
                    dst_w=self.img_size,
                )

            query_video = None
            if self.return_query_video:
                query_video = torch.stack(
                    [torch.from_numpy(frame).permute(2, 0, 1) for frame in rgb_frames],
                    dim=0,
                )

            resized_video = torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32)
            resized_depths = torch.zeros((self.S, 1, self.img_size, self.img_size), dtype=torch.float32)
            resized_normals = torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32)

            # 批量resize：torch比cv2循环快4倍
            rgb_stack = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2)
            depth_stack = torch.from_numpy(np.stack(depth_frames)).unsqueeze(1)
            normal_stack = torch.from_numpy(np.stack(normal_frames)).permute(0, 3, 1, 2)

            resized_video = F.interpolate(rgb_stack, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            resized_depths = F.interpolate(depth_stack, size=(self.img_size, self.img_size), mode='nearest')
            resized_normals = F.interpolate(normal_stack, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

            coords_uv, t_src, t_tgt, t_cam, targets = self._sample_query_data(
                clip_trajs_2d=clip_trajs_2d,
                clip_world_3d=clip_world_3d,
                clip_valid=clip_valid,
                clip_vis=clip_vis,
                clip_extrinsics=clip_extrinsics,
                crop_h=crop_h,
                crop_w=crop_w,
                normal_frames=normal_frames,
                normal_valid_frames=normal_valid_frames,
                depth_frames=depth_frames,
                motion_boundary_masks=cached_motion_boundary_masks,
                py_rng=py_rng,
                np_rng=np_rng,
            )

            if bool(targets["mask_normal"].any()):
                normal_indices = torch.where(targets["mask_normal"])[0]
                target_uv = targets["pos_2d"][normal_indices]
                target_frames = t_tgt[normal_indices]
                x = torch.clamp(torch.round(target_uv[:, 0] * (self.img_size - 1)).long(), 0, self.img_size - 1)
                y = torch.clamp(torch.round(target_uv[:, 1] * (self.img_size - 1)).long(), 0, self.img_size - 1)
                targets["normal"][normal_indices] = resized_normals[target_frames, :, y, x]

            local_patches = None
            if self.precompute_local_patches:
                patch_queries = torch.cat(
                    [coords_uv, t_src.unsqueeze(-1).to(torch.float32)],
                    dim=-1,
                )
                patch_video = resized_video.permute(0, 2, 3, 1)
                if self.local_patch_source == "highres":
                    patch_video = torch.from_numpy(np.stack(rgb_frames, axis=0))
                local_patches = self.extract_patches(
                    patch_video,
                    patch_queries,
                )

            transform_metadata = self._build_transform_metadata(
                original_h=original_h,
                original_w=original_w,
                x0=x0,
                y0=y0,
                crop_h=crop_h,
                crop_w=crop_w,
            )

            sample = {
                "video": resized_video,
                "coords": coords_uv,
                "t_src": t_src,
                "t_tgt": t_tgt,
                "t_cam": t_cam,
                "transform_metadata": transform_metadata,
                "aspect_ratio": torch.tensor(
                    [float(crop_w) / max(float(crop_h), 1.0)],
                    dtype=torch.float32,
                ),
                "targets": targets,
            }
            if self.return_aux_tensors:
                sample.update(
                    {
                        "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
                        "extrinsics": torch.from_numpy(clip_extrinsics),
                        "depths": resized_depths,
                        "normals": resized_normals,
                    }
                )
                if clip_intrinsics is not None:
                    sample["intrinsics_original"] = torch.from_numpy(clip_intrinsics)
                    sample["intrinsics_crop"] = torch.from_numpy(clip_intrinsics_crop)
                    sample["intrinsics"] = torch.from_numpy(clip_intrinsics_resized)
            if local_patches is not None:
                sample["local_patches"] = local_patches
            if query_video is not None:
                sample["video_query"] = query_video
            return sample, True

        except Exception as exc:
            print(f"Error loading sequence {seq_name}: {exc}")
            import traceback
            traceback.print_exc()
            return {}, False
