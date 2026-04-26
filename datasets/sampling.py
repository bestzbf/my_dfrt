"""
Single-dataset sampler for D4RT training.

Handles:
- Sequence sampling (uniform or weighted)
- Frame window sampling (random / sequential / dataset-adaptive stride)
- Clip length validation
- RNG state management

Usage:
    sampler = DatasetSampler(
        adapter=pointodyssey_adapter,
        clip_len=48,
        sampling_mode='stride',
    )
    sequence_name, frame_indices = sampler.sample(rng)
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass
from typing import Literal, Optional

from datasets.adapters.base import BaseAdapter


@dataclass
class SamplerConfig:
    """Configuration for dataset sampling."""
    clip_len: int = 8
    sampling_mode: Literal['random', 'sequential', 'stride'] = 'random'
    min_frames: int = 8
    sequence_weights: Optional[dict[str, float]] = None
    sequence_locality_size: int = 1
    frame_locality_radius: Optional[int] = None


# Dataset-adaptive stride policy.
#
# Design principles:
# - Dynamic video datasets favour small strides to preserve motion continuity.
# - RGB-D scan sequences are usually very dense, so stride 1-2 is sufficient.
# - Static multi-view datasets benefit from moderately larger view spacing.
#
# Weights are intentionally biased toward smaller / safer strides unless the
# dataset is inherently multi-view-oriented.
_DATASET_STRIDE_POLICY: dict[str, tuple[list[int], list[float]]] = {
    # Dynamic / tracking-heavy video datasets
    'dynamic_replica': ([1, 2, 3], [0.50, 0.30, 0.20]),
    'kubric':          ([1, 2, 4], [0.45, 0.35, 0.20]),
    'tartanair':       ([1, 2, 3], [0.50, 0.30, 0.20]),
    'pointodyssey':    ([1, 2, 4], [0.45, 0.35, 0.20]),
    'vkitti2':         ([1, 2, 3], [0.50, 0.30, 0.20]),
    'waymo':           ([1, 2, 3], [0.55, 0.30, 0.15]),

    # Dense RGB-D / scan sequences
    'scannet':         ([1, 2],    [0.65, 0.35]),
    'scannetpp':       ([1, 2],    [0.65, 0.35]),

    # Static multi-view datasets (blendedmvs uses stride=1 only due to low overlap)
    'co3dv2':          ([1, 2, 4], [0.50, 0.35, 0.15]),
    'blendedmvs':      ([1],       [1.0]),
    'mvssynth':        ([1, 2, 4], [0.55, 0.30, 0.15]),
}

# Conservative fallback for unknown datasets.
_DEFAULT_STRIDE_CANDIDATES = [1, 2, 3, 4]
_DEFAULT_STRIDE_WEIGHTS = [0.50, 0.25, 0.15, 0.10]


@dataclass
class _SequenceBlockState:
    """Worker-local reusable sequence block for short-term sampling locality."""
    sequence_name: str
    num_frames: int
    remaining_uses: int
    stride: Optional[int] = None
    last_start_idx: Optional[int] = None


class DatasetSampler:
    """
    Single-dataset sampler.

    Args:
        adapter: Dataset adapter (PointOdyssey, ScanNet, etc.)
        clip_len: Number of frames per clip
        sampling_mode: 'random', 'sequential', or 'stride'
        min_frames: Minimum frames required in a sequence
        sequence_weights: Optional per-sequence sampling weights
        allowed_sequences: If set, only these sequence names will be sampled.
                           Useful for single-scene training mode.

    Notes:
        sampling_mode='stride' uses a dataset-adaptive temporal/view stride
        policy instead of a single global max_stride. This is more reliable for
        mixed training across dynamic video, RGB-D scan, and static multi-view
        datasets.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        clip_len: int = 8,
        sampling_mode: Literal['random', 'sequential', 'stride'] = 'random',
        min_frames: int = 8,
        sequence_weights: Optional[dict[str, float]] = None,
        allowed_sequences: Optional[list[str]] = None,
        custom_stride_range: Optional[tuple[int, int]] = None,
        sequence_locality_size: int = 1,
        frame_locality_radius: Optional[int] = None,
    ):
        self.adapter = adapter
        self.clip_len = clip_len
        self.sampling_mode = sampling_mode
        self.min_frames = min_frames
        self.sequence_weights = sequence_weights
        self.allowed_sequences_set = set(allowed_sequences) if allowed_sequences is not None else None
        self.custom_stride_range = custom_stride_range
        self.sequence_locality_size = max(1, int(sequence_locality_size))
        self.frame_locality_radius = (
            None if frame_locality_radius is None else max(0, int(frame_locality_radius))
        )
        self._active_sequence: Optional[_SequenceBlockState] = None

        # Filter sequences by minimum frame count (and whitelist if given)
        self.valid_sequences = self._filter_sequences()

        if len(self.valid_sequences) == 0:
            if self.allowed_sequences_set is not None:
                raise ValueError(
                    f"None of the requested sequences {sorted(self.allowed_sequences_set)} "
                    f"were found (or had enough frames) in {adapter.dataset_name}. "
                    f"Available sequences: {adapter.list_sequences()[:10]}"
                )
            raise ValueError(
                f"No valid sequences found with >= {min_frames} frames in {adapter.dataset_name}"
            )

        # Build sampling weights
        self.sampling_probs = self._build_sampling_probs()

    def _filter_sequences(self) -> list[str]:
        """Filter sequences that have enough frames, and optionally by whitelist."""
        valid = []
        for seq_name in self.adapter.list_sequences():
            if self.allowed_sequences_set is not None and seq_name not in self.allowed_sequences_set:
                continue
            try:
                num_frames = self.adapter.get_num_frames(seq_name)
                if num_frames >= max(self.min_frames, self.clip_len):
                    valid.append(seq_name)
            except Exception:
                pass  # Skip sequences with missing or corrupt files
        return valid

    def _build_sampling_probs(self) -> list[float]:
        """Build sampling probabilities for each valid sequence."""
        if self.sequence_weights is None:
            return [1.0 / len(self.valid_sequences)] * len(self.valid_sequences)

        weights = [self.sequence_weights.get(seq_name, 1.0) for seq_name in self.valid_sequences]
        total = sum(weights)
        return [w / total for w in weights]

    def _get_stride_policy(self) -> tuple[list[int], list[float]]:
        """Return dataset-adaptive stride candidates and weights."""
        dataset_name = str(getattr(self.adapter, 'dataset_name', '')).lower()
        return _DATASET_STRIDE_POLICY.get(
            dataset_name,
            (_DEFAULT_STRIDE_CANDIDATES, _DEFAULT_STRIDE_WEIGHTS),
        )

    def _sample_stride(self, num_frames: int, rng: _random_module.Random) -> int:
        """Sample a valid stride using the dataset-adaptive policy or custom range."""
        if self.custom_stride_range is not None:
            min_stride, max_stride = self.custom_stride_range
            valid_strides = []
            for stride in range(min_stride, max_stride + 1):
                if num_frames >= 1 + (self.clip_len - 1) * stride:
                    valid_strides.append(stride)
            if not valid_strides:
                return 1
            return rng.choice(valid_strides)

        candidates, weights = self._get_stride_policy()

        valid_candidates = []
        valid_weights = []
        for stride, weight in zip(candidates, weights):
            # Need num_frames >= start + (clip_len - 1) * stride + 1.
            if num_frames >= 1 + (self.clip_len - 1) * stride:
                valid_candidates.append(stride)
                valid_weights.append(weight)

        if not valid_candidates:
            return 1

        total = sum(valid_weights)
        valid_probs = [w / total for w in valid_weights]
        return rng.choices(valid_candidates, weights=valid_probs, k=1)[0]

    def reset_locality_state(self) -> None:
        """Drop any reusable sequence block for this worker-local sampler copy."""
        self._active_sequence = None

    def _sample_sequence_name(self, rng: _random_module.Random) -> str:
        return rng.choices(self.valid_sequences, weights=self.sampling_probs, k=1)[0]

    def _start_sequence_block(self, rng: _random_module.Random) -> _SequenceBlockState:
        sequence_name = self._sample_sequence_name(rng)
        num_frames = self.adapter.get_num_frames(sequence_name)
        state = _SequenceBlockState(
            sequence_name=sequence_name,
            num_frames=num_frames,
            remaining_uses=self.sequence_locality_size,
        )
        if self.sampling_mode == 'stride':
            state.stride = self._sample_stride(num_frames, rng)
        return state

    def _sample_start_idx(
        self,
        max_start: int,
        rng: _random_module.Random,
        previous_start: Optional[int] = None,
    ) -> int:
        if max_start <= 0:
            return 0
        if previous_start is None or self.frame_locality_radius is None or self.frame_locality_radius <= 0:
            return rng.randint(0, max_start)

        low = max(0, previous_start - self.frame_locality_radius)
        high = min(max_start, previous_start + self.frame_locality_radius)
        if low > high:
            low, high = 0, max_start
        return rng.randint(low, high)

    def sample(self, rng: Optional[_random_module.Random] = None) -> tuple[str, list[int]]:
        """
        Sample a sequence and frame indices.

        Returns:
            (sequence_name, frame_indices)
        """
        if rng is None:
            rng = _random_module.Random()

        block_state = self._active_sequence
        if block_state is None or block_state.remaining_uses <= 0:
            block_state = self._start_sequence_block(rng)

        frame_indices = self._sample_frame_indices(block_state.num_frames, rng, block_state=block_state)
        block_state.remaining_uses -= 1
        self._active_sequence = block_state if block_state.remaining_uses > 0 else None
        return block_state.sequence_name, frame_indices

    def _sample_frame_indices(
        self,
        num_frames: int,
        rng: _random_module.Random,
        block_state: Optional[_SequenceBlockState] = None,
    ) -> list[int]:
        """Sample frame indices based on sampling mode."""
        if self.sampling_mode == 'random':
            # Random sampling with replacement.
            # Useful for stress-testing, but usually not ideal for coherent video clips.
            return sorted(rng.choices(range(num_frames), k=self.clip_len))

        if self.sampling_mode == 'sequential':
            # Sample one contiguous temporal window.
            if num_frames == self.clip_len:
                if block_state is not None:
                    block_state.last_start_idx = 0
                return list(range(num_frames))
            start_idx = self._sample_start_idx(
                num_frames - self.clip_len,
                rng,
                previous_start=None if block_state is None else block_state.last_start_idx,
            )
            if block_state is not None:
                block_state.last_start_idx = start_idx
            return list(range(start_idx, start_idx + self.clip_len))

        if self.sampling_mode == 'stride':
            # Dataset-adaptive random stride sampling.
            # This keeps dynamic video clips temporally coherent while allowing
            # larger view spacing on static multi-view datasets.
            stride = block_state.stride if block_state is not None and block_state.stride is not None else self._sample_stride(num_frames, rng)
            if block_state is not None:
                block_state.stride = stride
            max_start = num_frames - (self.clip_len - 1) * stride

            if max_start <= 0:
                # Graceful fallback for very short sequences.
                if block_state is not None:
                    block_state.last_start_idx = 0
                return list(range(min(self.clip_len, num_frames)))

            start_idx = self._sample_start_idx(
                max_start - 1,
                rng,
                previous_start=None if block_state is None else block_state.last_start_idx,
            )
            if block_state is not None:
                block_state.last_start_idx = start_idx
            return [start_idx + i * stride for i in range(self.clip_len)]

        raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def __len__(self) -> int:
        """Number of valid sequences."""
        return len(self.valid_sequences)

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return self.adapter.dataset_name
