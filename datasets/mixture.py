"""
Multi-dataset mixture sampler and dataset for D4RT training.

Combines multiple datasets with configurable sampling weights.

Usage:
    mixture = MixtureDataset(
        adapters=[pointodyssey_adapter, scannet_adapter],
        dataset_weights=[0.7, 0.3],
        clip_len=8,
        img_size=256,
        use_augs=True,
    )
    sample = mixture[0]  # Returns QuerySample

Single-scene usage (only one sequence):
    mixture = MixtureDataset(
        adapters=[pointodyssey_adapter],
        allowed_sequences_per_adapter=[['ani']],
        clip_len=8,
        img_size=256,
    )
"""

from __future__ import annotations

import multiprocessing as _mp
import random as _random_module
from typing import Optional

from datasets.adapters.base import BaseAdapter
from datasets.query_builder import D4RTQueryBuilder, QuerySample
from datasets.sampling import DatasetSampler
from datasets.transforms import GeometryTransformPipeline


class MixtureSampler:
    """
    Multi-dataset sampler with configurable weights.

    Args:
        samplers: List of DatasetSampler instances
        dataset_weights: Sampling weight for each dataset (will be normalized)
    """

    def __init__(
        self,
        samplers: list[DatasetSampler],
        dataset_weights: Optional[list[float]] = None,
    ):
        if len(samplers) == 0:
            raise ValueError("samplers cannot be empty")

        self.samplers = samplers

        # Normalize weights
        if dataset_weights is None:
            dataset_weights = [1.0] * len(samplers)

        if len(dataset_weights) != len(samplers):
            raise ValueError(
                f"dataset_weights length {len(dataset_weights)} != samplers length {len(samplers)}"
            )

        total = sum(dataset_weights)
        self.dataset_probs = [w / total for w in dataset_weights]

    def sample(self, rng: Optional[_random_module.Random] = None) -> tuple[int, str, list[int]]:
        """
        Sample a dataset, sequence, and frame indices.

        Returns:
            (dataset_idx, sequence_name, frame_indices)
        """
        if rng is None:
            rng = _random_module.Random()

        # Sample dataset
        dataset_idx = rng.choices(range(len(self.samplers)), weights=self.dataset_probs, k=1)[0]

        # Sample from that dataset
        sequence_name, frame_indices = self.samplers[dataset_idx].sample(rng)

        return dataset_idx, sequence_name, frame_indices

    def get_dataset_names(self) -> list[str]:
        """Get list of dataset names."""
        return [s.get_dataset_name() for s in self.samplers]


class MixtureDataset:
    """
    Multi-dataset PyTorch-style dataset.

    Combines multiple adapters with transforms and query builder.

    Args:
        adapters: List of dataset adapters
        dataset_weights: Sampling weight for each dataset
        clip_len: Number of frames per clip
        img_size: Target image size (square)
        use_augs: Whether to use data augmentation
        num_queries: Number of queries per sample
        boundary_ratio: Ratio of boundary samples
        t_tgt_eq_t_cam_ratio: Ratio of t_tgt == t_cam samples
        seed: Random seed for reproducibility
        allowed_sequences_per_adapter: Optional list (one entry per adapter) of
            sequence name whitelists. Pass None for an adapter to allow all its
            sequences. Example: [['ani', 'ani11_new_'], None] — restricts the
            first adapter to two sequences while the second is unrestricted.
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        dataset_weights: Optional[list[float]] = None,
        clip_len: int = 8,
        img_size: int = 256,
        use_augs: bool = True,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        seed: int = 42,
        allowed_sequences_per_adapter: Optional[list[Optional[list[str]]]] = None,
        sampling_mode: str = 'stride',
        epoch_size: int = 10000,
        custom_stride_range: Optional[tuple[int, int]] = None,
        precompute_patches: bool = True,
        precompute_from_highres: bool = False,
        allow_track_fallback: bool = False,
        reshuffle_each_epoch: bool = False,
    ):
        self.adapters = adapters
        self.clip_len = clip_len
        self.seed = seed
        self.epoch_size = epoch_size
        self.reshuffle_each_epoch = reshuffle_each_epoch
        # Keep epoch visible across worker-local dataset copies when DataLoader
        # uses persistent workers.
        self._shared_epoch = _mp.Value("q", 0, lock=False)

        # Build samplers for each dataset
        samplers = []
        for i, adapter in enumerate(adapters):
            allowed = None
            if allowed_sequences_per_adapter is not None:
                allowed = allowed_sequences_per_adapter[i]
            samplers.append(
                DatasetSampler(
                    adapter=adapter,
                    clip_len=clip_len,
                    allowed_sequences=allowed,
                    sampling_mode=sampling_mode,
                    custom_stride_range=custom_stride_range,
                )
            )

        # Build mixture sampler
        self.mixture_sampler = MixtureSampler(
            samplers=samplers,
            dataset_weights=dataset_weights,
        )

        # Build transform pipeline
        self.transform = GeometryTransformPipeline(
            img_size=img_size,
            use_augs=use_augs,
        )

        # Build query builder
        self.query_builder = D4RTQueryBuilder(
            num_queries=num_queries,
            boundary_ratio=boundary_ratio,
            t_tgt_eq_t_cam_ratio=t_tgt_eq_t_cam_ratio,
            precompute_patches=precompute_patches,
            precompute_from_highres=precompute_from_highres,
            allow_track_fallback=allow_track_fallback,
        )

    @property
    def current_epoch(self) -> int:
        return int(self._shared_epoch.value)

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.value = int(epoch)

    def __getitem__(self, index: int) -> QuerySample:
        """
        Get a training sample.

        Args:
            index: Sample index (used as RNG seed)

        Returns:
            QuerySample with video, coords, targets, masks, etc.
        """
        # Create RNG from index + seed
        sample_index = index
        if self.reshuffle_each_epoch:
            sample_index = self.current_epoch * self.epoch_size + index
        rng = _random_module.Random(self.seed + sample_index)

        # Sample dataset, sequence, and frames; retry on corrupt files or wrong shape
        for attempt in range(10):
            dataset_idx, sequence_name, frame_indices = self.mixture_sampler.sample(rng)
            adapter = self.adapters[dataset_idx]
            try:
                clip = adapter.load_clip(sequence_name, frame_indices)
                result = self.transform(clip, rng=rng)
                sample = self.query_builder(result, py_rng=rng)
                if sample.video.shape[0] != self.clip_len:
                    raise RuntimeError(f"Sample has {sample.video.shape[0]} frames, expected {self.clip_len}")
                break
            except Exception as e:
                if attempt >= 2:
                    print(f"Warning: skipping bad sample (attempt {attempt+1}): {e}")
                rng = _random_module.Random(self.seed + index + attempt + 1)
        else:
            raise RuntimeError(f"Failed to load a valid sample after 10 attempts at index {index}")

        return sample

    def __len__(self) -> int:
        """
        Dataset length (arbitrary, since we sample randomly).
        Set to a reasonable epoch size.
        """
        return self.epoch_size

    def get_dataset_names(self) -> list[str]:
        """Get list of dataset names in the mixture."""
        return self.mixture_sampler.get_dataset_names()
