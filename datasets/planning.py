"""
Sample planning for deterministic prefetch in planned mode.

This module generates deterministic sample plans for each rank, enabling
predictable I/O patterns and efficient prefetching from COS.
"""

from __future__ import annotations

import pickle
import random as _random_module
from dataclasses import dataclass
from typing import Any, Optional

from .mixture import MixtureSampler


@dataclass
class SampleSpec:
    """Specification for a single sample to be built by a background worker.

    Attributes:
        dataset_idx: Index into the mixture's dataset list
        sequence_name: Name of the sequence to load
        frame_indices: List of frame indices to load
        rng_state: Serialized RNG state after sampling (for transform + query_builder)
        sample_index: Global sample index (for debugging)
    """
    dataset_idx: int
    sequence_name: str
    frame_indices: list[int]
    rng_state: tuple[Any, ...]  # Result of random.getstate()
    sample_index: int


class SamplePlanner:
    """Generates deterministic sample plans for a rank.

    This planner simulates the MixtureSampler's locality behavior to produce
    a deterministic sequence of SampleSpecs. Unlike the online mode where each
    DataLoader worker has its own sampler state, planned mode has one explicit
    plan per rank.

    Key differences from online mode:
    - Locality is rank-global, not per-worker
    - No implicit worker scheduling effects
    - Fully deterministic given seed and epoch
    - RNG state is captured for reproducible transforms
    """

    def __init__(
        self,
        mixture_sampler: MixtureSampler,
        seed: int,
        epoch: int = 0,
    ):
        """
        Args:
            mixture_sampler: The MixtureSampler to simulate
            seed: Base random seed
            epoch: Current epoch number
        """
        self.mixture_sampler = mixture_sampler
        self.seed = seed
        self.epoch = epoch

    def generate_plan(self, start_index: int, count: int) -> list[SampleSpec]:
        """Generate a deterministic plan of sample specs.

        Args:
            start_index: Starting sample index (usually rank * samples_per_rank)
            count: Number of samples to generate

        Returns:
            List of SampleSpec objects
        """
        plan: list[SampleSpec] = []

        for i in range(count):
            sample_index = start_index + i

            # Create RNG with deterministic seed (same as MixtureDataset.__getitem__)
            rng_seed = self.seed + sample_index
            rng = _random_module.Random(rng_seed)

            # Sample using the mixture sampler
            dataset_idx, sequence_name, frame_indices = self.mixture_sampler.sample(rng)

            # Capture RNG state after sampling (will be used for transform + query_builder)
            rng_state = rng.getstate()

            spec = SampleSpec(
                dataset_idx=dataset_idx,
                sequence_name=sequence_name,
                frame_indices=frame_indices,
                rng_state=rng_state,
                sample_index=sample_index,
            )
            plan.append(spec)

        return plan

    def reset_for_epoch(self, epoch: int) -> None:
        """Reset planner for a new epoch.

        Args:
            epoch: New epoch number
        """
        self.epoch = epoch
        # Reset locality state in the mixture sampler
        self.mixture_sampler.reset_locality_state()


def serialize_sample_spec(spec: SampleSpec) -> bytes:
    """Serialize a SampleSpec to bytes for IPC."""
    return pickle.dumps(spec)


def deserialize_sample_spec(data: bytes) -> SampleSpec:
    """Deserialize a SampleSpec from bytes."""
    return pickle.loads(data)
