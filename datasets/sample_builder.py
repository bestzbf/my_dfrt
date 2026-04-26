"""
Sample builder for background QuerySample construction.

This module implements background worker processes that build QuerySample
bundles from SampleSpec objects, enabling prefetch from COS.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random as _random_module
import traceback
from typing import Any, Optional

from .adapters.base import BaseAdapter
from .planning import SampleSpec
from .query_builder import D4RTQueryBuilder, QuerySample
from .sample_spool import SampleSpool
from .transforms import GeometryTransformPipeline


class SampleBuilder:
    """Background process that builds QuerySample bundles from SampleSpecs.

    Each builder runs in a separate process and:
    1. Reads SampleSpec from input queue
    2. Calls adapter.load_clip() -> transform() -> query_builder()
    3. Writes QuerySample to spool directory
    4. Signals completion via output queue

    Error handling:
    - Transient I/O errors: retry same spec (up to 3 attempts)
    - Structural errors: write error marker, fail fast
    """

    def __init__(
        self,
        builder_id: int,
        adapters: list[BaseAdapter],
        transform: GeometryTransformPipeline,
        query_builder: D4RTQueryBuilder,
        spool: SampleSpool,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        clip_len: int,
    ):
        """
        Args:
            builder_id: Unique ID for this builder process
            adapters: List of dataset adapters (one per dataset in mixture)
            transform: Transform pipeline
            query_builder: Query builder
            spool: Sample spool for writing bundles
            input_queue: Queue for receiving SampleSpec objects
            output_queue: Queue for signaling completion
            clip_len: Expected clip length
        """
        self.builder_id = builder_id
        self.adapters = adapters
        self.transform = transform
        self.query_builder = query_builder
        self.spool = spool
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clip_len = clip_len

    def run(self) -> None:
        """Main loop for builder process."""
        while True:
            try:
                # Get next spec from queue (blocking)
                spec = self.input_queue.get(timeout=1.0)

                if spec is None:
                    # Shutdown signal
                    break

                # Build sample with retry logic
                success = self._build_sample_with_retry(spec)

                # Signal completion
                self.output_queue.put((spec.sample_index, success))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Builder {self.builder_id}] Unexpected error: {e}")
                traceback.print_exc()

    def _build_sample_with_retry(self, spec: SampleSpec, max_attempts: int = 3) -> bool:
        """Build a sample with retry logic for transient errors.

        Args:
            spec: SampleSpec to build
            max_attempts: Maximum number of retry attempts

        Returns:
            True if successful, False if failed
        """
        for attempt in range(max_attempts):
            try:
                sample = self._build_sample(spec)

                # Validate sample
                if sample.video.shape[0] != self.clip_len:
                    raise RuntimeError(
                        f"Sample has {sample.video.shape[0]} frames, expected {self.clip_len}"
                    )

                # Write to spool
                self.spool.write_bundle(spec.sample_index, sample)
                return True

            except Exception as e:
                if attempt < max_attempts - 1:
                    # Transient error, retry
                    print(
                        f"[Builder {self.builder_id}] Retry {attempt+1}/{max_attempts} "
                        f"for sample {spec.sample_index}: {e}"
                    )
                else:
                    # Structural error, write error marker
                    print(
                        f"[Builder {self.builder_id}] Failed sample {spec.sample_index} "
                        f"after {max_attempts} attempts: {e}"
                    )
                    self.spool.write_error(spec.sample_index, e)
                    return False

        return False

    def _build_sample(self, spec: SampleSpec) -> QuerySample:
        """Build a single QuerySample from a SampleSpec.

        Args:
            spec: SampleSpec containing dataset_idx, sequence_name, frame_indices, rng_state

        Returns:
            QuerySample ready for training
        """
        # Get adapter for this dataset
        adapter = self.adapters[spec.dataset_idx]

        # Load clip
        clip = adapter.load_clip(spec.sequence_name, spec.frame_indices)

        # Restore RNG state (same RNG used for sampling, transform, query_builder)
        rng = _random_module.Random()
        rng.setstate(spec.rng_state)

        # Apply transform
        result = self.transform(clip, rng=rng)

        # Build query sample
        sample = self.query_builder(result, py_rng=rng)

        return sample


def start_builder_process(
    builder_id: int,
    adapters: list[BaseAdapter],
    transform: GeometryTransformPipeline,
    query_builder: D4RTQueryBuilder,
    spool: SampleSpool,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    clip_len: int,
) -> mp.Process:
    """Start a builder process.

    Args:
        builder_id: Unique ID for this builder
        adapters: List of dataset adapters
        transform: Transform pipeline
        query_builder: Query builder
        spool: Sample spool
        input_queue: Input queue for SampleSpecs
        output_queue: Output queue for completion signals
        clip_len: Expected clip length

    Returns:
        Started multiprocessing.Process
    """
    builder = SampleBuilder(
        builder_id=builder_id,
        adapters=adapters,
        transform=transform,
        query_builder=query_builder,
        spool=spool,
        input_queue=input_queue,
        output_queue=output_queue,
        clip_len=clip_len,
    )

    process = mp.Process(target=builder.run, name=f"SampleBuilder-{builder_id}")
    process.start()
    return process


def stop_builder_processes(
    processes: list[mp.Process],
    input_queue: mp.Queue,
    timeout: float = 5.0,
) -> None:
    """Stop all builder processes gracefully.

    Args:
        processes: List of builder processes
        input_queue: Input queue (for sending shutdown signals)
        timeout: Timeout for joining processes
    """
    # Send shutdown signals
    for _ in processes:
        input_queue.put(None)

    # Wait for processes to finish
    for p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            print(f"Warning: Builder process {p.name} did not terminate, killing...")
            p.terminate()
            p.join(timeout=1.0)
