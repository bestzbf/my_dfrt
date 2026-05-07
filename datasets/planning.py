"""
Sample planning for deterministic prefetch in planned mode.

This module generates deterministic sample plans for each rank, enabling
predictable I/O patterns and efficient prefetching from COS.

Key design decisions:
- **Single-RNG-per-sample:** each sample creates one ``Random(seed + sample_index)``
  matching online mode's per-sample RNG structure (``MixtureDataset.__getitem__``).
  The same RNG drives ``mixture_sampler.sample()`` and is then captured as
  ``rng_state`` for downstream transform / query_builder.
- Each rank only *keeps* the items where ``global_index % world_size == rank``.
- ``local_index`` is the position within a rank's sub-plan (0..count_per_rank-1),
  used as the spool file name.
- ``global_index`` is the absolute sample ID, used for RNG seed computation.
- ``generation`` tracks epoch transitions so builders can discard stale tasks.
"""

from __future__ import annotations

import pickle
import random as _random_module
from dataclasses import dataclass
from typing import Any

from .mixture import MixtureSampler


@dataclass
class SampleSpec:
    """Specification for a single sample to be built by a background worker.

    Attributes:
        local_index: Position in this rank's plan (0..count-1), used for spool filename.
        global_index: Absolute sample ID across all ranks, used for RNG seed.
        dataset_idx: Index into the mixture's dataset list.
        sequence_name: Name of the sequence to load.
        frame_indices: List of frame indices to load.
        rng_state: Serialized RNG state after sampling (for transform + query_builder).
        generation: Epoch generation; builders discard tasks from older generations.
    """
    local_index: int
    global_index: int
    dataset_idx: int
    sequence_name: str
    frame_indices: list[int]
    rng_state: tuple[Any, ...]
    generation: int


class SamplePlanner:
    """Generates deterministic sample plans for a single rank.

    Uses a **single-RNG-per-sample** design that matches online mode's
    per-sample RNG structure (``MixtureDataset.__getitem__``): each sample
    creates one ``Random(seed + sample_index)`` which first drives
    ``mixture_sampler.sample(rng)`` and then continues into transform /
    query_builder via the captured ``rng_state``.

    **Known difference from online mode:** The inter-sample locality patterns
    differ because online mode's locality evolves through DataLoader worker
    scheduling (non-deterministic), while planned mode uses independent
    per-sample RNGs (deterministic but no implicit locality).

    The planner iterates ``total_samples = count_per_rank * world_size``
    and keeps only items where ``global_index % world_size == rank``.
    """

    def __init__(
        self,
        mixture_sampler: MixtureSampler,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        reshuffle_each_epoch: bool = True,
    ):
        self.mixture_sampler = mixture_sampler
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.reshuffle_each_epoch = reshuffle_each_epoch

    def generate_plan(
        self, epoch: int, count_per_rank: int, epoch_size: int, generation: int | None = None,
    ) -> list[SampleSpec]:
        """Generate a deterministic plan of SampleSpecs for this rank.

        Each sample uses a single RNG seeded from (seed, epoch, sample_index),
        matching the online mode's per-sample RNG structure (mixture.py).  The
        same RNG drives sampling, then its post-sampling state is captured as
        ``rng_state`` for downstream transform / query_builder.

        **Padding behavior (PyTorch DistributedSampler semantics):** When
        ``total_samples = count_per_rank * world_size`` exceeds ``epoch_size``,
        padding samples reuse early indices via ``sample_index = global_index % epoch_size``.
        This ensures all ranks process the same number of samples while maintaining
        deterministic RNG seeding that wraps around the epoch boundary.

        **Known difference from online mode:** in online mode, MixtureSampler's
        locality state (dataset blocks, sequence blocks) evolves implicitly
        through DataLoader worker scheduling, which is non-deterministic.  In
        planned mode, locality is not preserved because each sample creates an
        independent RNG — the sampling decisions are identical to what online
        mode would produce for the same (seed, epoch, index) tuple, but the
        inter-sample locality patterns differ.  This is an intentional trade-off:
        deterministic per-sample reproducibility over implicit locality.

        Args:
            epoch: Current epoch number.
            count_per_rank: Number of samples each rank should produce.
            epoch_size: Total number of unique samples in the epoch (before padding).
            generation: Generation tag stamped into every SampleSpec.
                If *None*, defaults to *epoch* for backward compat.

        Returns:
            List of ``SampleSpec`` with ``len == count_per_rank``.
        """
        if generation is None:
            generation = epoch

        total_samples = count_per_rank * self.world_size

        # Reset locality so every rank starts from the same clean state.
        self.reset_for_epoch()

        plan: list[SampleSpec] = []
        local_index = 0

        for global_index in range(total_samples):
            # Wrap sample_index to epoch_size (PyTorch DistributedSampler padding).
            # When total_samples > epoch_size, padding samples reuse early indices.
            sample_index = global_index % epoch_size

            # Single RNG per sample, matching online mode's seeding.
            if self.reshuffle_each_epoch:
                sample_seed = self.seed + epoch * epoch_size + sample_index
            else:
                sample_seed = self.seed + sample_index
            rng = _random_module.Random(sample_seed)

            # Sampling consumes the RNG (same structure as online mode).
            dataset_idx, sequence_name, frame_indices = (
                self.mixture_sampler.sample(rng)
            )

            # Capture post-sampling RNG state for transform / query_builder.
            rng_state = rng.getstate()

            if global_index % self.world_size == self.rank:
                plan.append(SampleSpec(
                    local_index=local_index,
                    global_index=global_index,
                    dataset_idx=dataset_idx,
                    sequence_name=sequence_name,
                    frame_indices=frame_indices,
                    rng_state=rng_state,
                    generation=generation,
                ))
                local_index += 1

        return plan

    def reset_for_epoch(self) -> None:
        """Reset the mixture sampler's locality state for a fresh epoch."""
        self.mixture_sampler.reset_locality_state()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_sample_spec(spec: SampleSpec) -> bytes:
    """Serialize a SampleSpec to bytes for IPC."""
    return pickle.dumps(spec)


def deserialize_sample_spec(data: bytes) -> SampleSpec:
    """Deserialize a SampleSpec from bytes."""
    return pickle.loads(data)
