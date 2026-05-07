"""End-to-end test for PlannedMixtureDataset with DataLoader and set_epoch()."""

import sys
import tempfile
import time

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataclasses import dataclass
from typing import Any


# Reuse mock classes from test_builder_integration.py (must be top-level for pickle)
@dataclass
class MockClip:
    video: torch.Tensor
    cameras: Any = None
    dataset_name: str = "mock"
    sequence_name: str = "scene_001"


class MockAdapter:
    def __init__(self, dataset_name="mock_dataset"):
        self.dataset_name = dataset_name
        self.valid_sequences = [f"scene_{i:03d}" for i in range(20)]

    def list_sequences(self):
        return self.valid_sequences

    def get_num_frames(self, sequence_name):
        return 100

    def load_clip(self, sequence_name, frame_indices):
        n_frames = len(frame_indices)
        return MockClip(
            video=torch.randn(n_frames, 3, 64, 64),
            sequence_name=sequence_name,
        )


class MockTransform:
    def __call__(self, clip, rng=None):
        return clip


@dataclass
class MockQuerySample:
    video: torch.Tensor
    dataset_name: str = "mock"
    sequence_name: str = "scene_001"


class MockQueryBuilder:
    def __call__(self, clip, py_rng=None):
        return MockQuerySample(
            video=clip.video,
            dataset_name=clip.dataset_name,
            sequence_name=clip.sequence_name,
        )


def custom_collate(batch):
    """Custom collate function that handles MockQuerySample dataclasses."""
    if len(batch) == 0:
        return None

    # Stack videos and collect metadata
    videos = torch.stack([sample.video for sample in batch])
    dataset_names = [sample.dataset_name for sample in batch]
    sequence_names = [sample.sequence_name for sample in batch]

    # Return a single MockQuerySample with batched data
    return MockQuerySample(
        video=videos,
        dataset_name=dataset_names[0],  # Just use first for simplicity
        sequence_name=sequence_names[0],
    )


def test_planned_dataset_basic_consumption():
    """PlannedMixtureDataset can be consumed via DataLoader with SequentialSampler."""
    from datasets.planned_dataset import PlannedMixtureDataset
    from datasets.sampling import DatasetSampler
    from datasets.mixture import MixtureSampler

    tmpdir = tempfile.mkdtemp()
    try:
        # Create mock adapter and sampler
        adapter = MockAdapter("ds0")
        ds_sampler = DatasetSampler(
            adapter=adapter,
            clip_len=8,
            sequence_locality_size=2,
        )
        mixture_sampler = MixtureSampler(
            samplers=[ds_sampler],
            dataset_weights=[1.0],
            dataset_locality_size=2,
        )

        # Create PlannedMixtureDataset
        dataset = PlannedMixtureDataset(
            adapters=[adapter],
            dataset_weights=[1.0],
            transform=MockTransform(),
            query_builder=MockQueryBuilder(),
            mixture_sampler=mixture_sampler,
            clip_len=8,
            epoch_size=10,
            seed=42,
            rank=0,
            world_size=1,
            spool_dir=tmpdir,
            prefetch_depth=3,
            builder_workers=1,
        )

        # Consume via DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=SequentialSampler(dataset),
            num_workers=0,
            collate_fn=custom_collate,
        )

        samples = []
        for batch in loader:
            assert batch.video.shape[0] == 2  # batch_size
            assert batch.video.shape[1] == 8  # clip_len
            assert batch.video.shape[2] == 3  # channels
            samples.append(batch)

        assert len(samples) == 5  # 10 samples / batch_size=2

        # Cleanup
        dataset.cleanup()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("✓ test_planned_dataset_basic_consumption passed")


def test_planned_dataset_set_epoch():
    """set_epoch() rebuilds pipeline and advances generation."""
    from datasets.planned_dataset import PlannedMixtureDataset
    from datasets.sampling import DatasetSampler
    from datasets.mixture import MixtureSampler

    tmpdir = tempfile.mkdtemp()
    try:
        adapter = MockAdapter("ds0")
        ds_sampler = DatasetSampler(
            adapter=adapter,
            clip_len=8,
            sequence_locality_size=2,
        )
        mixture_sampler = MixtureSampler(
            samplers=[ds_sampler],
            dataset_weights=[1.0],
            dataset_locality_size=2,
        )

        dataset = PlannedMixtureDataset(
            adapters=[adapter],
            dataset_weights=[1.0],
            transform=MockTransform(),
            query_builder=MockQueryBuilder(),
            mixture_sampler=mixture_sampler,
            clip_len=8,
            epoch_size=6,
            seed=42,
            rank=0,
            world_size=1,
            spool_dir=tmpdir,
            prefetch_depth=2,
            builder_workers=1,
        )

        # Epoch 0
        loader0 = DataLoader(dataset, batch_size=2, sampler=SequentialSampler(dataset), num_workers=0, collate_fn=custom_collate)
        samples0 = list(loader0)
        assert len(samples0) == 3  # 6 samples / 2

        # Advance to epoch 1
        dataset.set_epoch(1)
        loader1 = DataLoader(dataset, batch_size=2, sampler=SequentialSampler(dataset), num_workers=0, collate_fn=custom_collate)
        samples1 = list(loader1)
        assert len(samples1) == 3

        # Samples should be different (different epoch → different RNG seeds)
        # We can't directly compare tensors (random), but we can verify no crash

        dataset.cleanup()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("✓ test_planned_dataset_set_epoch passed")


def test_planned_dataset_multi_epoch_no_contamination():
    """Multiple epochs don't read stale generation bundles."""
    from datasets.planned_dataset import PlannedMixtureDataset
    from datasets.sampling import DatasetSampler
    from datasets.mixture import MixtureSampler

    tmpdir = tempfile.mkdtemp()
    try:
        adapter = MockAdapter("ds0")
        ds_sampler = DatasetSampler(
            adapter=adapter,
            clip_len=8,
            sequence_locality_size=2,
        )
        mixture_sampler = MixtureSampler(
            samplers=[ds_sampler],
            dataset_weights=[1.0],
            dataset_locality_size=2,
        )

        dataset = PlannedMixtureDataset(
            adapters=[adapter],
            dataset_weights=[1.0],
            transform=MockTransform(),
            query_builder=MockQueryBuilder(),
            mixture_sampler=mixture_sampler,
            clip_len=8,
            epoch_size=4,
            seed=42,
            rank=0,
            world_size=1,
            spool_dir=tmpdir,
            prefetch_depth=2,
            builder_workers=1,
        )

        # Run 3 epochs
        for epoch in range(3):
            dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=2, sampler=SequentialSampler(dataset), num_workers=0, collate_fn=custom_collate)
            samples = list(loader)
            assert len(samples) == 2  # 4 samples / 2
            # If generation isolation fails, we'd get TimeoutError or stale data

        dataset.cleanup()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("✓ test_planned_dataset_multi_epoch_no_contamination passed")


if __name__ == "__main__":
    test_planned_dataset_basic_consumption()
    test_planned_dataset_set_epoch()
    test_planned_dataset_multi_epoch_no_contamination()
    print("\nAll PlannedMixtureDataset end-to-end tests passed!")
