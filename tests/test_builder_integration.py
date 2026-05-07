"""End-to-end integration tests for the sample builder pipeline.

These tests exercise the full multiprocessing path:
plan -> enqueue -> forkserver builder -> spool write -> spool read.
Uses mock adapters to avoid real dataset dependencies.
"""

import sys
import tempfile
import time
import multiprocessing as mp
import random as _random_module
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import torch
from typing import Any


@dataclass
class MockClip:
    """Minimal clip structure returned by mock adapter."""
    video: torch.Tensor
    cameras: Any = None
    dataset_name: str = "mock"
    sequence_name: str = "scene_001"


class MockAdapter:
    """Pickle-safe mock adapter for builder integration tests."""

    def __init__(self, dataset_name="mock_dataset"):
        self.dataset_name = dataset_name

    def load_clip(self, sequence_name, frame_indices):
        n_frames = len(frame_indices)
        return MockClip(
            video=torch.randn(n_frames, 3, 64, 64),
            sequence_name=sequence_name,
        )


class MockTransform:
    """Pickle-safe mock transform — acts as identity, returns a dict-like
    TransformResult that the mock query builder can consume."""

    def __call__(self, clip, rng=None):
        return clip


@dataclass
class MockQuerySample:
    """Minimal QuerySample-like output with all required attributes.

    The builder checks sample.video.shape[0] (line 132 of sample_builder.py),
    so .video must be a tensor with shape [n_frames, ...].
    """
    video: torch.Tensor                         # [S,3,H,W]
    highres_video: Any = None
    depths: Any = None
    normals: Any = None
    coords: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))
    t_src: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    t_tgt: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    t_cam: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    intrinsics: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 3, 3))
    extrinsics: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 4, 4))
    targets: dict = field(default_factory=dict)
    local_patches: Any = None
    transform_metadata: dict = field(default_factory=dict)
    aspect_ratio: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0]))
    dataset_name: str = "mock"
    sequence_name: str = "scene_001"
    metadata: dict = field(default_factory=dict)



class MockQueryBuilder:
    """Pickle-safe mock query builder."""

    def __call__(self, clip, py_rng=None):
        return MockQuerySample(
            video=clip.video,
            dataset_name=clip.dataset_name,
            sequence_name=clip.sequence_name,
        )


def test_builder_basic_pipeline():
    """Full pipeline: plan specs -> enqueue -> builder process -> spool -> read back."""
    from datasets.planning import SampleSpec
    from datasets.sample_spool import SampleSpool
    from datasets.sample_builder import start_builder_process, stop_builder_processes

    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        ctx = mp.get_context("forkserver")
        shared_gen = ctx.Value("i", 0)
        input_queue = ctx.Queue(maxsize=32)
        output_queue = ctx.Queue()

        # Start one builder
        proc = start_builder_process(
            builder_id=0,
            adapters=[MockAdapter("ds0"), MockAdapter("ds1")],
            transform=MockTransform(),
            query_builder=MockQueryBuilder(),
            spool=spool,
            input_queue=input_queue,
            output_queue=output_queue,
            clip_len=8,
            current_generation=shared_gen,
        )

        # Create and enqueue specs
        n_samples = 5
        for i in range(n_samples):
            rng = _random_module.Random(42 + i)
            spec = SampleSpec(
                local_index=i,
                global_index=i,
                dataset_idx=0,
                sequence_name=f"scene_{i:03d}",
                frame_indices=list(range(8)),
                rng_state=rng.getstate(),
                generation=0,
            )
            input_queue.put(spec)

        # Wait for all bundles
        for i in range(n_samples):
            sample = spool.wait_for_bundle(i, generation=0, timeout=30.0)
            assert sample.video.shape == (8, 3, 64, 64), f"Wrong shape: {sample.video.shape}"

        # Shutdown
        stop_builder_processes([proc], input_queue, timeout=5.0)
        assert not proc.is_alive()

    print("test_builder_basic_pipeline passed")


def test_builder_generation_gate():
    """Builder discards specs from old generations."""
    from datasets.planning import SampleSpec
    from datasets.sample_spool import SampleSpool
    from datasets.sample_builder import start_builder_process, stop_builder_processes

    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(1)  # current gen = 1

        ctx = mp.get_context("forkserver")
        shared_gen = ctx.Value("i", 1)  # builder sees gen 1
        input_queue = ctx.Queue(maxsize=32)
        output_queue = ctx.Queue()

        proc = start_builder_process(
            builder_id=0,
            adapters=[MockAdapter()],
            transform=MockTransform(),
            query_builder=MockQueryBuilder(),
            spool=spool,
            input_queue=input_queue,
            output_queue=output_queue,
            clip_len=8,
            current_generation=shared_gen,
        )

        # Enqueue a stale spec (generation=0, but current is 1)
        rng = _random_module.Random(42)
        stale_spec = SampleSpec(
            local_index=0, global_index=0, dataset_idx=0,
            sequence_name="scene_stale", frame_indices=list(range(8)),
            rng_state=rng.getstate(), generation=0,
        )
        input_queue.put(stale_spec)

        # Enqueue a current spec (generation=1)
        fresh_spec = SampleSpec(
            local_index=0, global_index=0, dataset_idx=0,
            sequence_name="scene_fresh", frame_indices=list(range(8)),
            rng_state=rng.getstate(), generation=1,
        )
        input_queue.put(fresh_spec)

        # Only the fresh spec should produce a gen=1 bundle
        sample = spool.wait_for_bundle(0, generation=1, timeout=30.0)
        assert sample is not None

        # Stale spec should NOT have produced a gen=0 bundle
        assert not spool.is_ready(0, generation=0), "Stale spec should have been discarded"

        stop_builder_processes([proc], input_queue, timeout=5.0)

    print("test_builder_generation_gate passed")


def test_builder_multiple_workers():
    """Multiple builders process specs concurrently."""
    from datasets.planning import SampleSpec
    from datasets.sample_spool import SampleSpool
    from datasets.sample_builder import start_builder_process, stop_builder_processes

    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        ctx = mp.get_context("forkserver")
        shared_gen = ctx.Value("i", 0)
        input_queue = ctx.Queue(maxsize=64)
        output_queue = ctx.Queue()

        # Start 2 builders
        procs = []
        for bid in range(2):
            p = start_builder_process(
                builder_id=bid,
                adapters=[MockAdapter()],
                transform=MockTransform(),
                query_builder=MockQueryBuilder(),
                spool=spool,
                input_queue=input_queue,
                output_queue=output_queue,
                clip_len=8,
                current_generation=shared_gen,
            )
            procs.append(p)

        # Enqueue 10 specs
        n_samples = 10
        for i in range(n_samples):
            rng = _random_module.Random(42 + i)
            spec = SampleSpec(
                local_index=i, global_index=i, dataset_idx=0,
                sequence_name=f"scene_{i:03d}", frame_indices=list(range(8)),
                rng_state=rng.getstate(), generation=0,
            )
            input_queue.put(spec)

        # All should complete
        for i in range(n_samples):
            sample = spool.wait_for_bundle(i, generation=0, timeout=30.0)
            assert sample.video.shape[0] == 8

        stop_builder_processes(procs, input_queue, timeout=5.0)

    print("test_builder_multiple_workers passed")


if __name__ == "__main__":
    test_builder_basic_pipeline()
    test_builder_generation_gate()
    test_builder_multiple_workers()
    print("\nAll builder integration tests passed!")
