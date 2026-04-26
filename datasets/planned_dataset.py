"""
Planned dataset for deterministic sample-bundle prefetch.

This module provides PlannedMixtureDataset, a drop-in replacement for
MixtureDataset that uses background processes to pre-build QuerySample
bundles from a deterministic plan.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch.utils.data

from .adapters.base import BaseAdapter
from .mixture import MixtureSampler
from .planning import SamplePlanner
from .query_builder import D4RTQueryBuilder, QuerySample
from .sample_builder import start_builder_process, stop_builder_processes
from .sample_spool import SampleSpool
from .transforms import GeometryTransformPipeline


class PlannedMixtureDataset(torch.utils.data.Dataset):
    """Drop-in replacement for MixtureDataset with planned sample-bundle prefetch.

    Key differences from MixtureDataset:
    - Generates deterministic sample plan at init
    - Spawns background builder processes
    - __getitem__ waits for pre-built bundles instead of building on-demand
    - Locality is rank-global, not per-worker

    This enables efficient prefetch from COS by predicting I/O patterns.
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        dataset_weights: list[float],
        transform: GeometryTransformPipeline,
        query_builder: D4RTQueryBuilder,
        mixture_sampler: MixtureSampler,
        clip_len: int,
        seed: int,
        epoch_size: int,
        reshuffle_each_epoch: bool = True,
        builder_workers: int = 2,
        prefetch_depth: int = 32,
        spool_dir: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            adapters: List of dataset adapters
            dataset_weights: Sampling weights for each dataset
            transform: Transform pipeline
            query_builder: Query builder
            mixture_sampler: Mixture sampler (for generating plan)
            clip_len: Number of frames per clip
            seed: Random seed
            epoch_size: Number of samples per epoch
            reshuffle_each_epoch: Whether to reshuffle each epoch
            builder_workers: Number of background builder processes
            prefetch_depth: Number of samples to prefetch ahead
            spool_dir: Directory for spool files (default: /tmp/d4rt_spool_rank{rank}/)
            rank: DDP rank
            world_size: DDP world size
        """
        self.adapters = adapters
        self.dataset_weights = dataset_weights
        self.transform = transform
        self.query_builder = query_builder
        self.mixture_sampler = mixture_sampler
        self.clip_len = clip_len
        self.seed = seed
        self.epoch_size = epoch_size
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.builder_workers = builder_workers
        self.prefetch_depth = prefetch_depth
        self.rank = rank
        self.world_size = world_size

        self.current_epoch = 0

        # Create spool directory
        if spool_dir is None:
            spool_dir = Path(tempfile.gettempdir()) / f"d4rt_spool_rank{rank}"
        self.spool = SampleSpool(spool_dir, rank=rank, cleanup_on_init=True)

        # Create planner
        self.planner = SamplePlanner(
            mixture_sampler=mixture_sampler,
            seed=seed,
            epoch=0,
        )

        # Generate initial plan
        self.current_plan = self.planner.generate_plan(start_index=0, count=epoch_size)

        # Create multiprocessing queues
        self.input_queue = mp.Queue(maxsize=prefetch_depth * 2)
        self.output_queue = mp.Queue()

        # Start builder processes
        self.builder_processes = []
        for i in range(builder_workers):
            process = start_builder_process(
                builder_id=i,
                adapters=adapters,
                transform=transform,
                query_builder=query_builder,
                spool=self.spool,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                clip_len=clip_len,
            )
            self.builder_processes.append(process)

        # Enqueue initial prefetch batch
        for i in range(min(prefetch_depth, len(self.current_plan))):
            self.input_queue.put(self.current_plan[i])

        self.next_enqueue_index = min(prefetch_depth, len(self.current_plan))

    def __getitem__(self, index: int) -> QuerySample:
        """Get a pre-built QuerySample from the spool.

        Args:
            index: Sample index (within current epoch)

        Returns:
            QuerySample loaded from spool
        """
        # Wait for bundle to be ready
        sample = self.spool.wait_for_bundle(index, timeout=300.0)

        # Enqueue next spec (sliding window prefetch)
        if self.next_enqueue_index < len(self.current_plan):
            self.input_queue.put(self.current_plan[self.next_enqueue_index])
            self.next_enqueue_index += 1

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch and regenerate plan.

        Args:
            epoch: New epoch number
        """
        self.current_epoch = epoch

        # Reset planner for new epoch
        self.planner.reset_for_epoch(epoch)

        # Generate new plan
        sample_index_offset = epoch * self.epoch_size if self.reshuffle_each_epoch else 0
        self.current_plan = self.planner.generate_plan(
            start_index=sample_index_offset,
            count=self.epoch_size,
        )

        # Clear spool
        self.spool.cleanup()

        # Enqueue initial prefetch batch for new epoch
        self.next_enqueue_index = 0
        for i in range(min(self.prefetch_depth, len(self.current_plan))):
            self.input_queue.put(self.current_plan[i])
            self.next_enqueue_index += 1

    def get_dataset_names(self) -> list[str]:
        """Get list of dataset names in the mixture."""
        return [adapter.dataset_name for adapter in self.adapters]

    def cleanup(self) -> None:
        """Clean up resources (stop builder processes, clean spool)."""
        # Stop builder processes
        stop_builder_processes(self.builder_processes, self.input_queue, timeout=5.0)

        # Clean up spool
        self.spool.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


def set_dataset_epoch(dataset: Any, epoch: int) -> None:
    """Set epoch for a dataset (works for both MixtureDataset and PlannedMixtureDataset).

    Args:
        dataset: Dataset instance
        epoch: Epoch number
    """
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(dataset, "current_epoch"):
        dataset.current_epoch = epoch
