"""Unit tests for datasets/sample_stage.py."""

import shutil
import sys
import tempfile
from pathlib import Path
from types import MethodType

sys.path.insert(0, ".")

from datasets.adapters.blendedmvs import _SequenceRecord as BlendedMVSRecord
from datasets.adapters.mvssynth import _SequenceRecord as MVSSynthRecord
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class FakeBlendedMVSAdapter:
    dataset_name = "blendedmvs"

    def __init__(self, root: Path, scene_id: str, frame_ids: list[str], use_masked: bool = False):
        self.root = root
        self.use_masked = use_masked
        record = BlendedMVSRecord(scene_id=scene_id, scene_dir=root / scene_id, frame_ids=frame_ids)
        self._name_to_record = {scene_id: record}

    def _get_record(self, sequence_name: str):
        return self._name_to_record[sequence_name]


class FakeMVSSynthAdapter:
    dataset_name = "mvssynth"

    def __init__(self, root: Path, sequence_id: str, num_frames: int = 4):
        self.root = root
        self.precompute_root = root
        record = MVSSynthRecord(
            sequence_id=sequence_id,
            sequence_dir=root / sequence_id,
            num_frames=num_frames,
        )
        self._name_to_record = {sequence_id: record}

    def _get_record(self, sequence_name: str):
        return self._name_to_record[sequence_name]

    def get_num_frames(self, sequence_name: str) -> int:
        return self._name_to_record[sequence_name].num_frames


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_scene(root: Path, scene_id: str, num_frames: int = 4) -> list[str]:
    frame_ids = [f"{idx:08d}" for idx in range(num_frames)]
    for frame_id in frame_ids:
        _write_text(root / scene_id / "blended_images" / f"{frame_id}.jpg", f"rgb-{frame_id}")
        _write_text(root / scene_id / "rendered_depth_maps" / f"{frame_id}.pfm", f"depth-{frame_id}")
        _write_text(root / scene_id / "cams" / f"{frame_id}_cam.txt", f"cam-{frame_id}")
    return frame_ids


def _make_fake_mvssynth_sequence(root: Path, sequence_id: str, num_frames: int = 4) -> None:
    for idx in range(num_frames):
        _write_text(root / sequence_id / "images" / f"{idx:04d}.png", f"rgb-{idx}")
        _write_text(root / sequence_id / "depths" / f"{idx:04d}.exr", f"depth-{idx}")
        _write_text(root / sequence_id / "poses" / f"{idx:04d}.json", f"pose-{idx}")


def test_blendedmvs_scene_prefetch_caches_entire_scene_once():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        dataset_root = mount_root / "hdu_datasets" / "BlendedMVS"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        scene_id = "scene_001"
        frame_ids = _make_fake_scene(dataset_root, scene_id, num_frames=4)
        adapter = FakeBlendedMVSAdapter(dataset_root, scene_id, frame_ids)

        stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(stage_root),
                sdk_workers=4,
                mount_root=str(mount_root),
                passwd_file=str(passwd_file),
                enabled_datasets=("blendedmvs",),
                scene_prefetch_datasets=("blendedmvs",),
            )
        )

        download_calls: list[str] = []

        def fake_download(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
            download_calls.append(rel_key.as_posix())
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, cache_path)

        stager._download_to_cache = MethodType(fake_download, stager)

        with stager.stage_sample(
            adapter=adapter,
            sequence_name=scene_id,
            frame_indices=[0, 1],
            sample_tag="t0",
        ):
            pass

        expected_files = 3 * len(frame_ids)
        assert len(download_calls) == expected_files

        cache_scene_dir = (
            stage_root
            / "shared_raw_cache"
            / "data"
            / "hdu_datasets"
            / "BlendedMVS"
            / scene_id
        )
        assert (cache_scene_dir / ".d4rt_scene_complete").is_file()
        assert len([p for p in cache_scene_dir.rglob("*") if p.is_file() and not p.name.startswith(".d4rt_scene_complete")]) == expected_files

        with stager.stage_sample(
            adapter=adapter,
            sequence_name=scene_id,
            frame_indices=[2, 3],
            sample_tag="t1",
        ):
            pass

        assert len(download_calls) == expected_files


def test_invalidate_blendedmvs_scene_marker():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(stage_root),
                mount_root=str(mount_root),
                passwd_file=str(passwd_file),
                enabled_datasets=("blendedmvs",),
                scene_prefetch_datasets=("blendedmvs",),
            )
        )

        scene_dir = (
            stager.cache_data_root
            / "hdu_datasets"
            / "BlendedMVS"
            / "scene_001"
        )
        marker_path = scene_dir / ".d4rt_scene_complete"
        file_path = scene_dir / "blended_images" / "00000000.jpg"
        _write_text(marker_path, "ok")
        _write_text(file_path, "rgb")

        stager._invalidate_scene_marker_for_path(file_path)
        assert not marker_path.exists()


def test_mvssynth_manifest_stages_raw_frames_and_pose0_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        dataset_root = mount_root / "hdu_datasets" / "GTAV_1080"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        sequence_id = "0001"
        _make_fake_mvssynth_sequence(dataset_root, sequence_id, num_frames=8)
        adapter = FakeMVSSynthAdapter(dataset_root, sequence_id, num_frames=8)
        stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(stage_root),
                mount_root=str(mount_root),
                passwd_file=str(passwd_file),
                enabled_datasets=("mvssynth",),
                window_radius=0,
            )
        )

        manifest = stager._build_manifest(adapter, sequence_id, [2, 3])
        rel_manifest = [path.relative_to(dataset_root / sequence_id).as_posix() for path in manifest]

        assert rel_manifest == [
            "images/0002.png",
            "depths/0002.exr",
            "poses/0002.json",
            "images/0003.png",
            "depths/0003.exr",
            "poses/0003.json",
            "poses/0000.json",
        ]
        assert all("precomputed" not in path.name for path in manifest)


def test_mvssynth_stage_sample_rebases_record_but_keeps_precompute_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        dataset_root = mount_root / "hdu_datasets" / "GTAV_1080"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        sequence_id = "0001"
        _make_fake_mvssynth_sequence(dataset_root, sequence_id, num_frames=4)
        adapter = FakeMVSSynthAdapter(dataset_root, sequence_id, num_frames=4)
        stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(stage_root),
                sdk_workers=2,
                mount_root=str(mount_root),
                passwd_file=str(passwd_file),
                enabled_datasets=("mvssynth",),
            )
        )

        def fake_download(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, cache_path)

        stager._download_to_cache = MethodType(fake_download, stager)

        old_record = adapter._get_record(sequence_id)
        with stager.stage_sample(
            adapter=adapter,
            sequence_name=sequence_id,
            frame_indices=[1, 2],
            sample_tag="mvssynth",
        ):
            staged_record = adapter._get_record(sequence_id)
            assert adapter.root != dataset_root
            assert staged_record is not old_record
            assert staged_record.sequence_dir == adapter.root / sequence_id
            assert staged_record.image_path(1).is_file()
            assert staged_record.depth_path(1).is_file()
            assert staged_record.pose_path(0).is_file()
            assert adapter.precompute_root == dataset_root

        assert adapter.root == dataset_root
        assert adapter._get_record(sequence_id) is old_record
        assert adapter.precompute_root == dataset_root
