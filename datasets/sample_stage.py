"""
Sample-local staging for COS-backed datasets.

This module implements an optional builder-side staging step backed by a
node-local shared raw-file cache:

1. Build the exact file manifest for one sample
2. Materialize those files into a persistent local cache directory
3. Hard-link the cached files into a temporary sample directory
4. Let the existing adapter.load_clip() read from local disk
5. Delete the temporary sample directory after the sample is built

This keeps QuerySample spool semantics unchanged while finally exploiting
the sampler's short-term frame / sequence locality across nearby samples.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Optional

from qcloud_cos import CosConfig, CosS3Client


@dataclasses.dataclass(frozen=True)
class SampleStageConfig:
    backend: str
    stage_root: str
    sdk_workers: int = 8
    cos_timeout_s: int = 20
    download_retries: int = 2
    cache_max_bytes: int = 100 * 1024**3
    cache_low_watermark_ratio: float = 0.9
    cache_touch_interval_s: float = 30.0
    cache_scan_interval_s: float = 30.0
    window_radius: int = 48
    mount_root: str = "/data_cos"
    bucket: str = "hd-ai-data-1251882982"
    region: str = "ap-beijing"
    passwd_file: str = "/etc/passwd-s3fs-data_cos"
    enabled_datasets: tuple[str, ...] = ()
    scene_prefetch_datasets: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SampleStageConfig":
        datasets = raw.get("enabled_datasets", ())
        if isinstance(datasets, str):
            datasets = tuple(
                item.strip()
                for item in datasets.split(",")
                if item.strip()
            )
        else:
            datasets = tuple(str(item).strip() for item in datasets if str(item).strip())
        scene_prefetch_datasets = raw.get("scene_prefetch_datasets", ())
        if isinstance(scene_prefetch_datasets, str):
            scene_prefetch_datasets = tuple(
                item.strip()
                for item in scene_prefetch_datasets.split(",")
                if item.strip()
            )
        else:
            scene_prefetch_datasets = tuple(
                str(item).strip()
                for item in scene_prefetch_datasets
                if str(item).strip()
            )
        return cls(
            backend=str(raw.get("backend", "")).strip().lower(),
            stage_root=str(raw.get("stage_root", "")).strip(),
            sdk_workers=max(1, int(raw.get("sdk_workers", 8))),
            cos_timeout_s=max(1, int(raw.get("cos_timeout_s", 20))),
            download_retries=max(0, int(raw.get("download_retries", 2))),
            cache_max_bytes=max(0, int(raw.get("cache_max_bytes", 100 * 1024**3))),
            cache_low_watermark_ratio=min(
                0.99,
                max(0.50, float(raw.get("cache_low_watermark_ratio", 0.9))),
            ),
            cache_touch_interval_s=max(
                0.0, float(raw.get("cache_touch_interval_s", 30.0))
            ),
            cache_scan_interval_s=max(
                1.0, float(raw.get("cache_scan_interval_s", 30.0))
            ),
            window_radius=max(0, int(raw.get("window_radius", 48))),
            mount_root=str(raw.get("mount_root", "/data_cos")).strip(),
            bucket=str(raw.get("bucket", "hd-ai-data-1251882982")).strip(),
            region=str(raw.get("region", "ap-beijing")).strip(),
            passwd_file=str(raw.get("passwd_file", "/etc/passwd-s3fs-data_cos")).strip(),
            enabled_datasets=datasets,
            scene_prefetch_datasets=scene_prefetch_datasets,
        )


class SampleLocalStager:
    """Builder-local staging helper."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_tls', None)  # threading.local() is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tls = threading.local()

    def __init__(self, config: SampleStageConfig):
        if config.backend != "cos_sdk":
            raise ValueError(f"Unsupported sample stage backend: {config.backend!r}")
        self.config = config
        self.mount_root = Path(config.mount_root)
        self.stage_root = Path(config.stage_root)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = self.stage_root / "shared_raw_cache"
        self.cache_data_root = self.cache_root / "data"
        self.lock_root = self.cache_root / "locks"
        self.work_root = self.stage_root / "work"
        self.cache_data_root.mkdir(parents=True, exist_ok=True)
        self.lock_root.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        self._last_cache_scan_s = 0.0
        self._last_stage_stats: dict[str, Any] = {}

    def _get_client(self) -> CosS3Client:
        client = getattr(self._tls, "cos_client", None)
        if client is None:
            parts = Path(self.config.passwd_file).read_text().strip().split(":")
            if len(parts) == 2:
                secret_id, secret_key = parts
            elif len(parts) == 3:
                _bucket, secret_id, secret_key = parts
            else:
                raise ValueError(
                    f"Unsupported COS passwd file format: {self.config.passwd_file}"
                )
            cos_config = CosConfig(
                Region=self.config.region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Scheme="https",
                Timeout=self.config.cos_timeout_s,
            )
            client = CosS3Client(cos_config)
            self._tls.cos_client = client
        return client

    def supports(self, adapter: Any) -> bool:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        if dataset_name not in self.config.enabled_datasets:
            return False
        root = getattr(adapter, "root", None)
        if root is None:
            return False
        mount = str(self.mount_root).rstrip("/") + "/"
        return str(root).startswith(mount) or str(root) == str(self.mount_root)

    @staticmethod
    def _stage_blendedmvs_precomputed_enabled() -> bool:
        return os.getenv(
            "D4RT_BLENDEDMVS_STAGE_PRECOMPUTED", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

    @contextlib.contextmanager
    def stage_sample(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
        sample_tag: str,
    ) -> Iterator[Any]:
        if not self.supports(adapter):
            yield adapter
            return

        manifest = self._build_manifest(adapter, sequence_name, frame_indices)
        if not manifest:
            yield adapter
            return
        self._maybe_prefetch_scene(adapter, sequence_name)

        temp_dir = Path(
            tempfile.mkdtemp(prefix=f"sample_stage_{sample_tag}_", dir=self.work_root)
        )
        try:
            t_materialize0 = time.perf_counter()
            materialize_stats = self._materialize_manifest(manifest, temp_dir)
            materialize_s = time.perf_counter() - t_materialize0
            self._last_stage_stats = {
                "dataset": str(getattr(adapter, "dataset_name", "")),
                "sequence": sequence_name,
                "requested_frames": len(frame_indices),
                "manifest_files": len(manifest),
                "materialize_s": materialize_s,
                **materialize_stats,
            }
            staged_dataset_root = temp_dir / self._to_cos_key(Path(adapter.root))
            # scannetpp: warm _scene_cache from staged files so _get_scene_data
            # reads local cameras.txt/images.txt instead of COS mount
            if str(getattr(adapter, "dataset_name", "")) == "scannetpp":
                t_depth0 = time.perf_counter()
                if sequence_name not in adapter._scene_cache:
                    old_data_root = adapter.data_root
                    adapter.data_root = staged_dataset_root
                    try:
                        adapter._get_scene_data(sequence_name)
                    finally:
                        adapter.data_root = old_data_root
                        # restore scene_dir to COS path; drop if path restoration fails
                        sd = adapter._scene_cache.get(sequence_name)
                        if sd is not None:
                            try:
                                cos_scene_dir = old_data_root / Path(sd["scene_dir"]).relative_to(staged_dataset_root)
                                adapter._scene_cache[sequence_name] = dict(sd, scene_dir=cos_scene_dir)
                            except ValueError:
                                adapter._scene_cache.pop(sequence_name, None)
                depth_frame_indices = self._expand_frame_indices(
                    adapter,
                    sequence_name,
                    frame_indices,
                )
                self._prepare_scannetpp_depth_stage(
                    adapter,
                    sequence_name,
                    depth_frame_indices,
                    staged_dataset_root,
                )
                self._last_stage_stats["scannetpp_depth_stage_s"] = time.perf_counter() - t_depth0
            with self._rebase_adapter(adapter, sequence_name, staged_dataset_root):
                yield adapter
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_manifest(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        dataset_name = str(adapter.dataset_name)
        manifest_frame_indices = self._expand_frame_indices(
            adapter, sequence_name, frame_indices
        )
        if dataset_name == "kubric":
            return self._manifest_kubric(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "pointodyssey":
            return self._manifest_pointodyssey(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "dynamic_replica":
            return self._manifest_dynamic_replica(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "co3dv2":
            return self._manifest_co3dv2(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "blendedmvs":
            return self._manifest_blendedmvs(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "scannetpp":
            return self._manifest_scannetpp(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "mvssynth":
            return self._manifest_mvssynth(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "scannet":
            return self._manifest_scannet(adapter, sequence_name, manifest_frame_indices)
        return []

    def _expand_frame_indices(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[int]:
        radius = self.config.window_radius
        if not frame_indices:
            return frame_indices
        unique_indices = sorted(set(int(i) for i in frame_indices))
        if radius <= 0:
            return unique_indices
        try:
            num_frames = int(adapter.get_num_frames(sequence_name))
        except Exception:
            return unique_indices
        if num_frames <= 0:
            return unique_indices

        positive_gaps = [
            b - a for a, b in zip(unique_indices, unique_indices[1:]) if b > a
        ]
        if positive_gaps and min(positive_gaps) > 1 and len(set(positive_gaps)) == 1:
            # Strided clips should not pull every intermediate frame into the
            # raw cache.  Extend only along the observed stride so staging stays
            # proportional to the frames the adapter will actually load.
            stride = positive_gaps[0]
            extra_steps = radius // stride
            if extra_steps <= 0:
                return unique_indices
            expanded = set(unique_indices)
            first = unique_indices[0]
            last = unique_indices[-1]
            for step in range(1, extra_steps + 1):
                before = first - step * stride
                after = last + step * stride
                if before >= 0:
                    expanded.add(before)
                if after < num_frames:
                    expanded.add(after)
            return sorted(expanded)

        low = max(0, unique_indices[0] - radius)
        high = min(num_frames - 1, unique_indices[-1] + radius)
        return list(range(low, high + 1))

    def _manifest_kubric(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = [record.rank_path]
        if record.h5_path is not None:
            paths.append(record.h5_path)
            if record.depth_names is not None or getattr(record, "has_depth_dir", False):
                paths.extend(
                    record.depth_dir / adapter._depth_name_for_index(record, i)
                    for i in frame_indices
                )
        else:
            paths.append(record.ann_path)
        paths.extend(
            record.frame_dir / adapter._frame_name_for_index(record, i)
            for i in frame_indices
        )
        return self._dedupe_keep_order(paths)

    def _manifest_pointodyssey(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter.get_record(sequence_name)
        paths: list[Path] = []
        if record.scene_info_path is not None:
            paths.append(Path(record.scene_info_path))
        if self._pointodyssey_stage_anno_h5(record):
            anno_path = Path(record.anno_path) if record.anno_path is not None else None
            if anno_path is not None:
                h5_path = anno_path.with_suffix(".h5")
                paths.append(h5_path if h5_path.exists() else anno_path)
        for idx in frame_indices:
            paths.append(Path(record.rgb_paths[idx]))
            if record.depth_paths is not None:
                paths.append(Path(record.depth_paths[idx]))
            if record.normal_paths is not None:
                paths.append(Path(record.normal_paths[idx]))
        return self._dedupe_keep_order(paths)

    def _manifest_dynamic_replica(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            paths.append(adapter.split_root / record.image_rel_paths[idx])
            paths.append(adapter.split_root / record.depth_rel_paths[idx])
            traj_rel = record.traj_rel_paths[idx]
            if traj_rel is not None:
                paths.append(adapter.split_root / traj_rel)
        return self._dedupe_keep_order(paths)

    def _manifest_scannet(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        rec = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            paths.append(rec.image_dir / f"{idx:05d}.jpg")
            paths.append(rec.depth_dir / f"{idx:05d}.png")
            paths.append(rec.pose_dir  / f"{idx:05d}.txt")
        # precomputed.h5 stays on COS mount — h5py slices frames without full download
        return self._dedupe_keep_order(paths)

    def _manifest_scannetpp(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        # Use data_root directly to avoid _scene_cache staged-path pollution
        scene_dir = adapter.data_root / sequence_name
        colmap_dir = scene_dir / "iphone" / "colmap"
        frames_dir = scene_dir / "iphone" / "frames"
        paths: list[Path] = [
            colmap_dir / "cameras.txt",
            colmap_dir / "images.txt",
            scene_dir / "iphone" / "pose_intrinsic_imu.json",
        ]
        paths += [frames_dir / f"{i:06d}.jpg" for i in frame_indices]
        # depth.bin fetched via Range requests — not in manifest
        # precomputed.h5 itself is not staged.  When tracks are enabled, stage
        # the tiny chunk index so the adapter can fetch only needed h5 chunks
        # via COS Range requests.
        if getattr(adapter, "use_precomputed_tracks", False):
            index_path = scene_dir / "precomputed.h5_chunk_index.pkl"
            if index_path.exists():
                paths.append(index_path)
        return self._dedupe_keep_order(paths)

    def _prepare_scannetpp_depth_stage(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
        staged_dataset_root: Path,
    ) -> None:
        """Fetch depth frames via Range requests and prepare staging metadata."""
        # Use full_indices from cache if available, else read from COS path directly
        # Never use sd["scene_dir"] here — it may point to a staged path
        sd = adapter._scene_cache.get(sequence_name)
        if sd is not None and "full_indices" in sd:
            full_indices_arr = sd["full_indices"]
        else:
            from datasets.adapters.scannetpp import _join_frames
            full_indices_arr = _join_frames(adapter.data_root / sequence_name)["full_indices"]
        needed_full_indices = sorted(set(int(full_indices_arr[i]) for i in frame_indices))
        cos_scene_dir = adapter.data_root / sequence_name
        self._stage_scannetpp_depth(cos_scene_dir, staged_dataset_root / sequence_name, needed_full_indices)
        adapter._staged_depth_map_tmp = {fi: pos for pos, fi in enumerate(needed_full_indices)}

    def _stage_scannetpp_depth(
        self,
        original_scene_dir: Path,
        staged_scene_dir: Path,
        frame_indices: list[int],
    ) -> None:
        """Fetch only needed depth frames via COS Range requests and write a minimal depth.bin."""
        import pickle, lz4.block, struct
        from concurrent.futures import ThreadPoolExecutor

        rel_key = self._to_cos_key(original_scene_dir)
        depth_key = f"{rel_key}/iphone/depth.bin"
        index_key = f"{rel_key}/iphone/depth_chunk_index.pkl"

        # cache chunk index (~100KB)
        index_cache_path = self._ensure_cached(original_scene_dir / "iphone" / "depth_chunk_index.pkl")
        chunk_offsets: list[tuple[int, int]] = pickle.loads(index_cache_path.read_bytes())

        needed = sorted(set(frame_indices))

        def fetch_chunk(fi: int) -> tuple[int, bytes]:
            chunk_cache_path = self.cache_data_root / Path(f"{depth_key}.chunks") / f"{fi:08d}.bin"
            if chunk_cache_path.is_file():
                self._touch_cache_entry(chunk_cache_path)
                return fi, chunk_cache_path.read_bytes()

            offset, chunk_size = chunk_offsets[fi]
            lock_key = Path(f"{depth_key}.chunks") / f"{fi:08d}.lock"
            with self._path_lock(lock_key):
                if chunk_cache_path.is_file():
                    self._touch_cache_entry(chunk_cache_path)
                    return fi, chunk_cache_path.read_bytes()
                range_header = f"bytes={offset}-{offset + chunk_size - 1}"
                r = self._get_client().get_object(
                    Bucket=self.config.bucket,
                    Key=depth_key,
                    Range=range_header,
                )
                raw = r["Body"].get_raw_stream().read()
                chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = chunk_cache_path.with_name(
                    f".{chunk_cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
                )
                try:
                    tmp_path.write_bytes(raw)
                    os.replace(tmp_path, chunk_cache_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                self._touch_cache_entry(chunk_cache_path, force=True)
                self._maybe_evict_cache()
                return fi, raw

        with ThreadPoolExecutor(max_workers=min(16, len(needed))) as ex:
            chunks = dict(ex.map(lambda fi: fetch_chunk(fi), needed))

        # write a minimal depth.bin with only the needed frames (re-indexed 0..N-1)
        dst_depth = staged_scene_dir / "iphone" / "depth.bin"
        dst_depth.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_depth, "wb") as f:
            for fi in needed:
                chunk = chunks[fi]
                f.write(struct.pack("<I", len(chunk)))
                f.write(chunk)

    def _manifest_blendedmvs(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            frame_id = record.frame_ids[idx]
            paths.append(record.rgb_path(frame_id, adapter.use_masked))
            paths.append(record.depth_path(frame_id))
            paths.append(record.cam_path(frame_id))
        precompute_root = getattr(adapter, "precompute_root", None)
        stage_precomputed = self._stage_blendedmvs_precomputed_enabled()
        if stage_precomputed and precompute_root is not None:
            precomputed_npz = Path(precompute_root) / sequence_name / "precomputed.npz"
            precomputed_h5 = precomputed_npz.with_suffix(".h5")
            max_bytes = int(
                float(os.getenv("D4RT_BLENDEDMVS_STAGE_PRECOMPUTED_MAX_GB", "2.0"))
                * 1024**3
            )
            if precomputed_h5.exists():
                try:
                    if max_bytes <= 0 or precomputed_h5.stat().st_size <= max_bytes:
                        paths.append(precomputed_h5)
                except OSError:
                    pass
            elif precomputed_npz.exists():
                try:
                    if max_bytes <= 0 or precomputed_npz.stat().st_size <= max_bytes:
                        paths.append(precomputed_npz)
                except OSError:
                    pass
        return self._dedupe_keep_order(paths)

    def _manifest_co3dv2(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            frame_idx = max(0, min(int(idx), record.num_frames - 1))
            frame_number = record.frame_numbers[frame_idx]
            anno = adapter._get_frame_anno(
                record.category,
                record.sequence_name,
                frame_number,
            )
            paths.append(adapter.root / anno["image"]["path"])
            paths.append(adapter.root / anno["depth"]["path"])
            depth_mask_path = anno.get("depth", {}).get("mask_path")
            if depth_mask_path:
                paths.append(adapter.root / depth_mask_path)
        return self._dedupe_keep_order(paths)

    def _manifest_mvssynth(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            paths.append(record.image_path(idx))
            paths.append(record.depth_path(idx))
            paths.append(record.pose_path(idx))
        if frame_indices and frame_indices[0] != 0:
            # When precomputed tracks exist, MVSSynthAdapter reads pose 0 to
            # match the full-sequence origin shift used at precompute time.
            paths.append(record.pose_path(0))
        # precomputed.h5/npz stays on the original COS mount so h5py can slice
        # requested frames without downloading the whole sequence cache.
        return self._dedupe_keep_order(paths)

    @staticmethod
    def _dedupe_keep_order(paths: list[Path]) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path not in seen:
                out.append(path)
                seen.add(path)
        return out

    def _to_cos_key(self, path: Path) -> str:
        path = Path(path)
        try:
            rel_path = path.relative_to(self.mount_root)
        except ValueError:
            rel_path = path.resolve().relative_to(self.mount_root)
        return str(rel_path).replace(os.sep, "/")

    def _materialize_manifest(self, manifest: list[Path], temp_dir: Path) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "linked": 0,
            "copied": 0,
            "bytes": 0,
        }
        stats_lock = threading.Lock()

        def stage_one(src_path: Path) -> None:
            rel_key = Path(self._to_cos_key(src_path))
            cache_path = self.cache_data_root / rel_key
            was_hit = cache_path.is_file()
            dst = temp_dir / rel_key
            dst.parent.mkdir(parents=True, exist_ok=True)
            mode, size = self._link_cached_file(src_path, dst)
            with stats_lock:
                stats["cache_hits" if was_hit else "cache_misses"] += 1
                stats[mode] = int(stats.get(mode, 0)) + 1
                stats["bytes"] = int(stats.get("bytes", 0)) + int(size)

        with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
            futures = [executor.submit(stage_one, path) for path in manifest]
            for future in as_completed(futures):
                future.result()
        return stats

    def _maybe_prefetch_scene(self, adapter: Any, sequence_name: str) -> None:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        if dataset_name not in self.config.scene_prefetch_datasets:
            return
        # Non-blocking: check marker first, skip if already prefetched
        marker_path = self._get_scene_marker_path(adapter, sequence_name)
        if marker_path is not None and marker_path.is_file():
            return
        # Fire-and-forget background prefetch (don't block the builder)
        threading.Thread(
            target=self._do_scene_prefetch,
            args=(adapter, dataset_name, sequence_name),
            daemon=True,
        ).start()

    def _get_scene_marker_path(self, adapter: Any, sequence_name: str) -> Optional[Path]:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        try:
            if dataset_name == "blendedmvs":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(record.scene_dir)) / ".d4rt_scene_complete"
            elif dataset_name == "co3dv2":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(adapter.root / record.category / record.sequence_name)) / ".d4rt_scene_complete"
            elif dataset_name == "kubric":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(record.scene_dir)) / ".d4rt_scene_complete"
            elif dataset_name == "dynamic_replica":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(adapter.split_root / record.image_rel_paths[0])).parent.parent / ".d4rt_scene_complete"
            elif dataset_name == "scannetpp":
                return self.cache_data_root / Path(self._to_cos_key(adapter.data_root / sequence_name)) / ".d4rt_scene_complete"
        except Exception:
            return None
        return None

    def _do_scene_prefetch(self, adapter: Any, dataset_name: str, sequence_name: str) -> None:
        try:
            if dataset_name == "blendedmvs":
                self._prefetch_blendedmvs_scene(adapter, sequence_name)
            elif dataset_name == "co3dv2":
                self._prefetch_co3dv2_scene(adapter, sequence_name)
            elif dataset_name == "kubric":
                self._prefetch_kubric_scene(adapter, sequence_name)
            elif dataset_name == "dynamic_replica":
                self._prefetch_dynamic_replica_scene(adapter, sequence_name)
            elif dataset_name == "scannetpp":
                self._prefetch_scannetpp_scene(adapter, sequence_name)
        except Exception as e:
            print(f"[ScenePrefetch] {dataset_name}/{sequence_name} failed: {e}", flush=True)

    def _prefetch_blendedmvs_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(self._to_cos_key(record.scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._blendedmvs_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=blendedmvs\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _blendedmvs_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for frame_id in record.frame_ids:
            paths.append(record.rgb_path(frame_id, adapter.use_masked))
            paths.append(record.depth_path(frame_id))
            paths.append(record.cam_path(frame_id))
        return self._dedupe_keep_order(paths)

    def _prefetch_co3dv2_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(
            self._to_cos_key(adapter.root / record.category / record.sequence_name)
        )
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._co3dv2_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=co3dv2\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _co3dv2_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for fn in record.frame_numbers:
            anno = adapter._get_frame_anno(record.category, record.sequence_name, fn)
            paths.append(adapter.root / anno["image"]["path"])
            paths.append(adapter.root / anno["depth"]["path"])
            depth_mask_path = anno.get("depth", {}).get("mask_path")
            if depth_mask_path:
                paths.append(adapter.root / depth_mask_path)
        return self._dedupe_keep_order(paths)

    def _prefetch_kubric_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(self._to_cos_key(record.scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._kubric_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=kubric\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _kubric_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = [record.rank_path]
        if record.h5_path is not None:
            paths.append(record.h5_path)
            if record.depth_names is not None or getattr(record, "has_depth_dir", False):
                paths.extend(
                    record.depth_dir / adapter._depth_name_for_index(record, i)
                    for i in range(record.num_frames)
                )
        else:
            paths.append(record.ann_path)
        paths.extend(
            record.frame_dir / adapter._frame_name_for_index(record, i)
            for i in range(record.num_frames)
        )
        return self._dedupe_keep_order(paths)

    def _prefetch_dynamic_replica_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(
            self._to_cos_key(adapter.split_root / record.image_rel_paths[0])
        ).parent.parent
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._dynamic_replica_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=dynamic_replica\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _dynamic_replica_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in range(record.num_frames):
            paths.append(adapter.split_root / record.image_rel_paths[idx])
            paths.append(adapter.split_root / record.depth_rel_paths[idx])
            traj_rel = record.traj_rel_paths[idx]
            if traj_rel is not None:
                paths.append(adapter.split_root / traj_rel)
        return self._dedupe_keep_order(paths)

    def _prefetch_scannetpp_scene(self, adapter: Any, sequence_name: str) -> None:
        scene_dir = adapter.data_root / sequence_name
        scene_rel_key = Path(self._to_cos_key(scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._scannetpp_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=scannetpp\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _scannetpp_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        scene_dir = adapter.data_root / sequence_name
        colmap_dir = scene_dir / "iphone" / "colmap"
        frames_dir = scene_dir / "iphone" / "frames"
        paths: list[Path] = [
            colmap_dir / "cameras.txt",
            colmap_dir / "images.txt",
            scene_dir / "iphone" / "pose_intrinsic_imu.json",
        ]
        num_frames = adapter.get_num_frames(sequence_name)
        paths += [frames_dir / f"{i:06d}.jpg" for i in range(num_frames)]
        if getattr(adapter, "use_precomputed_tracks", False):
            index_path = scene_dir / "precomputed.h5_chunk_index.pkl"
            if index_path.exists():
                paths.append(index_path)
        return self._dedupe_keep_order(paths)

    def _link_cached_file(self, src_path: Path, dst: Path) -> tuple[str, int]:
        for attempt in range(2):
            cache_path = self._ensure_cached(src_path)
            try:
                os.link(cache_path, dst)
                return "linked", cache_path.stat().st_size
            except FileNotFoundError:
                if attempt == 0:
                    continue
                raise
            except OSError as exc:
                if exc.errno in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                    try:
                        shutil.copy2(cache_path, dst)
                        return "copied", cache_path.stat().st_size
                    except FileNotFoundError:
                        if attempt == 0:
                            continue
                        raise
                raise
        raise FileNotFoundError(src_path)

    def _ensure_cached(self, src_path: Path) -> Path:
        src_path = Path(src_path)
        rel_key = Path(self._to_cos_key(src_path))
        cache_path = self.cache_data_root / rel_key
        if cache_path.is_file():
            self._touch_cache_entry(cache_path)
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._path_lock(rel_key):
            if cache_path.is_file():
                self._touch_cache_entry(cache_path)
                return cache_path
            self._download_to_cache(src_path, cache_path, rel_key)
        self._touch_cache_entry(cache_path, force=True)
        self._maybe_evict_cache()
        return cache_path

    def _download_to_cache(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
        key = rel_key.as_posix()
        tmp_path = cache_path.with_name(f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}")
        last_exc: Optional[BaseException] = None
        for attempt in range(self.config.download_retries + 1):
            try:
                response = self._get_client().get_object(
                    Bucket=self.config.bucket,
                    Key=key,
                )
                stream = response["Body"].get_raw_stream()
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(tmp_path, cache_path)
                return
            except BaseException as exc:
                last_exc = exc
                tmp_path.unlink(missing_ok=True)
                if attempt < self.config.download_retries:
                    time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                    continue
                raise
            finally:
                tmp_path.unlink(missing_ok=True)
        if last_exc is not None:
            raise last_exc

    def _touch_cache_entry(self, cache_path: Path, force: bool = False) -> None:
        interval = self.config.cache_touch_interval_s
        if interval <= 0:
            return
        now = time.time()
        if not force:
            try:
                if now - cache_path.stat().st_mtime < interval:
                    return
            except OSError:
                return
        try:
            os.utime(cache_path, (now, now), follow_symlinks=False)
        except OSError:
            pass

    @contextlib.contextmanager
    def _path_lock(self, rel_key: Path) -> Iterator[None]:
        digest = hashlib.sha1(rel_key.as_posix().encode("utf-8")).hexdigest()
        lock_path = self.lock_root / f"{digest}.lock"
        with open(lock_path, "a+b") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _maybe_evict_cache(self) -> None:
        max_bytes = self.config.cache_max_bytes
        if max_bytes <= 0:
            return

        now = time.time()
        if now - self._last_cache_scan_s < self.config.cache_scan_interval_s:
            return

        eviction_lock = self.lock_root / "eviction.lock"
        with open(eviction_lock, "a+b") as lock_f:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                self._last_cache_scan_s = now
                self._evict_cache_locked()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _evict_cache_locked(self) -> None:
        entries: list[tuple[float, int, Path]] = []
        total_bytes = 0
        for path in self.cache_data_root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith(".") or ".part." in path.name:
                continue
            try:
                rel_parts = path.relative_to(self.cache_data_root).parts
            except ValueError:
                continue
            if any(part.startswith(".d4rt_") for part in rel_parts):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total_bytes += stat.st_size
            entries.append((stat.st_mtime, stat.st_size, path))

        if total_bytes <= self.config.cache_max_bytes:
            return

        target_bytes = int(self.config.cache_max_bytes * self.config.cache_low_watermark_ratio)
        entries.sort(key=lambda item: item[0])
        for _, size, path in entries:
            if total_bytes <= target_bytes:
                break
            self._invalidate_scene_marker_for_path(path)
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
            total_bytes -= size

    def _invalidate_scene_marker_for_path(self, cache_path: Path) -> None:
        try:
            rel_parts = cache_path.relative_to(self.cache_data_root).parts
        except ValueError:
            return
        if len(rel_parts) < 4:
            return
        if cache_path.name == ".d4rt_scene_complete":
            return

        # BlendedMVS: hdu_datasets/BlendedMVS/<scene>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "BlendedMVS" and len(rel_parts) >= 3:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], ".d4rt_scene_complete"
            )
            marker_path.unlink(missing_ok=True)
            return

        # Co3Dv2: hdu_datasets/Co3Dv2/<category>/<sequence>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Co3Dv2" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # Kubric: hdu_datasets/Kubric/<scene_id>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Kubric" and len(rel_parts) >= 3:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # DynamicReplica: hdu_datasets/Dynamic_Replica/<split>/<seq>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Dynamic_Replica" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # ScanNet++: hdu_datasets/scannetpp/data/<scene>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "scannetpp" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _rebase_adapter(
        self,
        adapter: Any,
        sequence_name: str,
        staged_dataset_root: Path,
    ) -> Iterator[None]:
        dataset_name = str(adapter.dataset_name)
        if dataset_name == "kubric":
            old_root = adapter.root
            old_record = adapter._name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(old_root),
                ann_path=staged_dataset_root / old_record.ann_path.relative_to(old_root),
                rank_path=staged_dataset_root / old_record.rank_path.relative_to(old_root),
                h5_path=(
                    staged_dataset_root / old_record.h5_path.relative_to(old_root)
                    if old_record.h5_path is not None
                    else None
                ),
                trajs_2d_path=(
                    staged_dataset_root / old_record.trajs_2d_path.relative_to(old_root)
                    if old_record.trajs_2d_path is not None
                    else None
                ),
            )
            adapter.root = staged_dataset_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "pointodyssey":
            old_root = adapter.root
            old_record = adapter.name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_root=self._rebase_pathlike(
                    old_record.sequence_root, old_root, staged_dataset_root
                ),
                rgb_paths=self._rebase_pathlike_list(
                    old_record.rgb_paths, old_root, staged_dataset_root
                ),
                depth_paths=self._rebase_pathlike_list(
                    old_record.depth_paths, old_root, staged_dataset_root
                ),
                normal_paths=self._rebase_pathlike_list(
                    old_record.normal_paths, old_root, staged_dataset_root
                ),
                mask_paths=self._rebase_pathlike_list(
                    old_record.mask_paths, old_root, staged_dataset_root
                ),
                # Keep heavy annotation files on /data_cos so h5py can slice
                # the needed frames directly instead of downloading the whole
                # per-scene anno.h5 into the sample stage.
                anno_path=(
                    self._rebase_pathlike(
                        old_record.anno_path, old_root, staged_dataset_root
                    )
                    if self._pointodyssey_stage_anno_h5(old_record)
                    else old_record.anno_path
                ),
                info_path=old_record.info_path,
                scene_info_path=self._rebase_pathlike(
                    old_record.scene_info_path, old_root, staged_dataset_root
                ),
                fast_dir=old_record.fast_dir,
                fast_anno_paths=old_record.fast_anno_paths,
                encoded_cache_paths=old_record.encoded_cache_paths,
            )
            adapter.root = staged_dataset_root
            adapter.name_to_record[sequence_name] = new_record
            adapter._scene_info_cache.clear()
            adapter._encoded_cache_store.clear()
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.name_to_record[sequence_name] = old_record
                adapter._scene_info_cache.clear()
                adapter._encoded_cache_store.clear()
            return

        if dataset_name == "dynamic_replica":
            old_root = adapter.root
            old_split_root = adapter.split_root
            old_record = adapter._name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_dir=staged_dataset_root / old_record.sequence_dir.relative_to(old_root),
            )
            adapter.root = staged_dataset_root
            adapter.split_root = staged_dataset_root / adapter.split
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.split_root = old_split_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "blendedmvs":
            old_root = adapter.root
            old_precompute_root = getattr(adapter, "precompute_root", None)
            old_record = adapter._name_to_record[sequence_name]
            new_record = old_record.__class__(
                scene_id=old_record.scene_id,
                scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(old_root),
                frame_ids=old_record.frame_ids,
            )
            new_precompute_root = old_precompute_root
            if (
                old_precompute_root is not None
                and self._stage_blendedmvs_precomputed_enabled()
            ):
                try:
                    new_precompute_root = (
                        staged_dataset_root / Path(old_precompute_root).relative_to(old_root)
                    )
                except ValueError:
                    new_precompute_root = old_precompute_root
            adapter.root = staged_dataset_root
            adapter.precompute_root = new_precompute_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "scannetpp":
            old_data_root = adapter.data_root
            old_scene_cache = adapter._scene_cache.get(sequence_name)
            if old_scene_cache is not None:
                new_scene_data = dict(old_scene_cache)
                new_scene_data["scene_dir"] = staged_dataset_root / old_scene_cache["scene_dir"].relative_to(old_data_root)
                precomputed_dir = old_scene_cache.get("_precomputed_dir", old_scene_cache["scene_dir"])
                try:
                    cached_scene_dir = self.cache_data_root / self._to_cos_key(old_scene_cache["scene_dir"])
                    if (
                        (cached_scene_dir / "precomputed.h5").is_file()
                        or (cached_scene_dir / "precomputed.npz").is_file()
                    ):
                        precomputed_dir = cached_scene_dir
                except Exception:
                    pass
                new_scene_data["_precomputed_dir"] = precomputed_dir
                index_dir = precomputed_dir
                try:
                    cached_scene_dir = self.cache_data_root / self._to_cos_key(old_scene_cache["scene_dir"])
                    if (cached_scene_dir / "precomputed.h5_chunk_index.pkl").is_file():
                        index_dir = cached_scene_dir
                except Exception:
                    pass
                new_scene_data["_precomputed_index_dir"] = index_dir
                # _staged_depth_map set by _prepare_scannetpp_depth_stage in stage_sample
                staged_depth_map = getattr(adapter, "_staged_depth_map_tmp", None)
                if staged_depth_map is not None:
                    new_scene_data["_staged_depth_map"] = staged_depth_map
                adapter._scene_cache[sequence_name] = new_scene_data
                adapter._depth_chunk_cache.pop(sequence_name, None)
            adapter.data_root = staged_dataset_root
            try:
                yield
            finally:
                adapter.data_root = old_data_root
                if old_scene_cache is not None:
                    adapter._scene_cache[sequence_name] = old_scene_cache
                else:
                    adapter._scene_cache.pop(sequence_name, None)
                adapter._depth_chunk_cache.pop(sequence_name, None)
                adapter.__dict__.pop("_staged_depth_map_tmp", None)
            return

        if dataset_name == "scannet":
            old_root = adapter.root  # /data_cos/hdu_datasets/scannet
            old_scans_dir = adapter._scans_dir
            old_precompute_root = adapter.precompute_root
            old_record = adapter._name_to_record[sequence_name]
            def _rb(p: Path) -> Path:
                try:
                    return staged_dataset_root / p.relative_to(old_root)
                except ValueError:
                    return p
            from datasets.adapters.scannet import _SequenceRecord as _ScanNetRecord
            new_record = _ScanNetRecord(
                scene_id=old_record.scene_id,
                scene_dir=_rb(old_record.scene_dir),
                processed_dir=_rb(old_record.processed_dir),
                image_dir=_rb(old_record.image_dir),
                depth_dir=_rb(old_record.depth_dir),
                pose_dir=_rb(old_record.pose_dir),
                precomputed_path=old_record.precomputed_path,  # keep on COS for h5py slicing
                mesh_path=old_record.mesh_path,
                num_frames=old_record.num_frames,
                color_hw=(old_record.color_height, old_record.color_width),
                depth_hw=(old_record.depth_height, old_record.depth_width),
                has_axis_alignment=old_record.has_axis_alignment,
            )
            adapter.root = staged_dataset_root
            adapter._scans_dir = staged_dataset_root / old_scans_dir.relative_to(old_root)
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter._scans_dir = old_scans_dir
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "co3dv2":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            old_track_precompute_root = adapter.track_precompute_root
            old_record = adapter._uid_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_dir=(
                    staged_dataset_root
                    / old_record.category
                    / old_record.sequence_name
                ),
                # Co3Dv2 COS planned mode stages only raw RGB/depth/mask files.
                # Keep precomputed tracks disabled in staged samples so h5/npz
                # checks cannot fall back to the slow /data_cos mount.
                has_precomputed=False,
            )
            adapter.root = staged_dataset_root
            adapter.precompute_root = None
            adapter.track_precompute_root = None
            adapter._uid_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter.track_precompute_root = old_track_precompute_root
                adapter._uid_to_record[sequence_name] = old_record
            return

        if dataset_name == "mvssynth":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            old_record = adapter._name_to_record[sequence_name]
            new_record = old_record.__class__(
                sequence_id=old_record.sequence_id,
                sequence_dir=staged_dataset_root / old_record.sequence_dir.relative_to(old_root),
                num_frames=old_record.num_frames,
            )
            adapter.root = staged_dataset_root
            # Keep precomputed_root on the original mount, matching ScanNet++:
            # raw per-frame files are staged, while h5/npz caches are sliced in
            # place instead of copied into each sample directory.
            adapter.precompute_root = old_precompute_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        yield

    @staticmethod
    def _pointodyssey_stage_anno_h5(record: Any) -> bool:
        enabled = os.getenv("D4RT_POINTODYSSEY_STAGE_ANNO_H5", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return enabled and getattr(record, "anno_path", None) is not None

    @staticmethod
    def _rebase_pathlike(
        value: Any,
        old_root: Path,
        staged_dataset_root: Path,
    ) -> Any:
        if value is None:
            return None
        path_value = Path(value)
        try:
            return staged_dataset_root / path_value.relative_to(old_root)
        except Exception:
            return value

    def _rebase_pathlike_list(
        self,
        values: Any,
        old_root: Path,
        staged_dataset_root: Path,
    ) -> Any:
        if values is None:
            return None
        return [
            self._rebase_pathlike(value, old_root, staged_dataset_root)
            for value in values
        ]


def build_sample_stager(raw_config: Optional[dict[str, Any]]) -> Optional[SampleLocalStager]:
    if not raw_config:
        return None
    config = SampleStageConfig.from_dict(raw_config)
    if not config.backend:
        return None
    return SampleLocalStager(config)
