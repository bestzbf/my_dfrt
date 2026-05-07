"""
Planned dataset for deterministic sample-bundle prefetch.

Design invariants
-----------------
- __getitem__(i) corresponds to plan[i]; external sampler MUST be
  SequentialSampler (enforced by train_mixture.py).
- Epoch transitions fully tear down and rebuild the prefetch pipeline
  (queues + builder processes + spool generation) so there is zero
  chance of cross-epoch pollution.
- All child processes are created via the "forkserver" mp context to
  avoid CUDA-fork issues.  Adapters / transforms / query_builder are
  pure-CPU objects and safe to pickle across the forkserver boundary.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
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

logger = logging.getLogger(__name__)

_mp_ctx = mp.get_context("forkserver")


class PlannedMixtureDataset(torch.utils.data.Dataset):
    """Drop-in Dataset with planned prefetch.

    Epoch transitions destroy old queues / processes and create fresh
    ones rather than hot-switching, guaranteeing zero cross-epoch
    pollution.

    **epoch_size padding:** When ``world_size`` does not evenly divide
    ``epoch_size``, each rank processes ``math.ceil(epoch_size / world_size)``
    samples, so the effective global sample count is
    ``ceil(epoch_size / world_size) * world_size`` — potentially slightly more
    than ``epoch_size``.  This matches PyTorch ``DistributedSampler``'s padding
    behavior and ensures no rank runs out of work early.

    **Cleanup:** Call ``cleanup()`` or let the destructor run to stop builder
    processes.  If called mid-epoch (before all samples are consumed), builders
    may not exit gracefully within the 5-second timeout and will be terminated.
    This is safe but may log warnings.  Normal training (full epoch consumption
    before ``set_epoch()``) avoids this.
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
        max_spool_bytes: int = 2 * 1024**3,
        sample_stage_config: Optional[dict[str, Any]] = None,
    ):
        # Store config for pipeline rebuilds.
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
        self.max_spool_bytes = max_spool_bytes
        self.sample_stage_config = sample_stage_config
        self._wait_log = os.getenv("D4RT_PLANNED_WAIT_LOG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

        self.current_epoch = 0
        self._generation: int = 0

        # Spool persists across epochs; generation tag isolates files.
        if spool_dir is None:
            spool_dir = str(
                Path(tempfile.gettempdir()) / f"d4rt_spool_rank{rank}"
            )
        self.spool = SampleSpool(
            spool_dir, rank=rank, cleanup_on_init=True,
            max_spool_bytes=max_spool_bytes,
        )
        self.spool.set_generation(self._generation)

        self.planner = SamplePlanner(
            mixture_sampler=mixture_sampler,
            seed=seed, rank=rank, world_size=world_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
        )

        # Mutable pipeline state -- rebuilt every epoch.
        self.input_queue: Optional[mp.Queue] = None
        self.output_queue: Optional[mp.Queue] = None
        self.builder_processes: list[mp.Process] = []
        self._shared_generation: Optional[Any] = None
        self.current_plan: list = []
        self.next_enqueue_index: int = 0

        # Initial plan + pipeline for epoch 0.
        self.current_plan = self.planner.generate_plan(
            epoch=0, count_per_rank=math.ceil(self.epoch_size / self.world_size),
            epoch_size=self.epoch_size, generation=self._generation,
        )
        self._start_pipeline()

    # ------------------------------------------------------------------
    # Pipeline lifecycle (rebuilt each epoch)
    # ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        """Create fresh queues, builder processes, and seed prefetch window."""
        self._shared_generation = _mp_ctx.Value("i", self._generation)
        self.input_queue = _mp_ctx.Queue(maxsize=self.prefetch_depth * 2)
        self.output_queue = _mp_ctx.Queue()

        self.builder_processes = []
        for i in range(self.builder_workers):
            proc = start_builder_process(
                builder_id=i,
                adapters=self.adapters,
                transform=self.transform,
                query_builder=self.query_builder,
                spool=self.spool,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                clip_len=self.clip_len,
                current_generation=self._shared_generation,
                sample_stage_config=self.sample_stage_config,
            )
            self.builder_processes.append(proc)

        # Seed the prefetch window.
        limit = min(self.prefetch_depth, len(self.current_plan))
        for i in range(limit):
            self.input_queue.put(self.current_plan[i])
        self.next_enqueue_index = limit

    def _stop_pipeline(self) -> None:
        """Tear down builder processes and queues.  Safe to call repeatedly."""
        if self.builder_processes:
            stop_builder_processes(
                self.builder_processes, self.input_queue, timeout=5.0,
            )
            self.builder_processes = []
        # Let old queues be GC'd; new ones are created in _start_pipeline.
        self.input_queue = None
        self.output_queue = None
        self._shared_generation = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Padded to ensure equal work across ranks (see class docstring).
        return math.ceil(self.epoch_size / self.world_size)

    def __getitem__(self, index: int) -> QuerySample:
        """Return the pre-built bundle for *index*.

        The index MUST arrive sequentially (0, 1, 2, ...) because we use
        SequentialSampler.  The spool blocks until the bundle with the
        matching (index, generation) pair is ready.
        """
        sample = self._wait_with_worker_check(index)

        # Slide the prefetch window forward.
        if self.next_enqueue_index < len(self.current_plan):
            self.input_queue.put(self.current_plan[self.next_enqueue_index])
            self.next_enqueue_index += 1

        return sample

    def _wait_with_worker_check(self, index: int, total_timeout: float = 600.0, check_interval: float = 5.0, max_requeue: int = 3) -> QuerySample:
        """Wait for spool bundle, failing fast if all workers have died.

        After ``max_requeue`` consecutive failures for the same index, the
        failing spec is substituted with a randomly chosen spec from a
        different position in the plan so training can continue.
        """
        import dataclasses
        import random
        import time
        deadline = time.time() + total_timeout
        last_alive_log = 0.0
        requeue_count = 0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                alive = sum(1 for p in self.builder_processes if p.is_alive())
                spec_info = self._format_plan_spec(index)
                spool_info = self._format_spool_summary()
                raise TimeoutError(
                    f"Sample g{self._generation:04d}_{index} did not become ready "
                    f"within {total_timeout}s. alive_workers={alive}/{len(self.builder_processes)} "
                    f"{spec_info} {spool_info} Spool dir: {self.spool.spool_dir}"
                )
            try:
                return self.spool.wait_for_bundle(
                    index, self._generation, timeout=min(check_interval, remaining)
                )
            except TimeoutError:
                pass
            except RuntimeError as e:
                # Builder wrote an error marker (e.g. timeout, COS failure).
                # Delete the marker and re-enqueue the spec so a worker retries it.
                self.spool._get_error_path(index, self._generation).unlink(missing_ok=True)
                requeue_count += 1
                if requeue_count >= max_requeue and len(self.current_plan) > 1:
                    # Persistent failure: substitute with a random spec from a
                    # different plan position so this slot doesn't block forever.
                    alt_indices = [i for i in range(len(self.current_plan)) if i != index]
                    alt_idx = random.choice(alt_indices)
                    alt_spec = self.current_plan[alt_idx]
                    sub_spec = dataclasses.replace(alt_spec, local_index=index, generation=self._generation)
                    self.input_queue.put(sub_spec)
                    requeue_count = 0
                    print(
                        f"[PlannedDataset] sample g{self._generation:04d}_{index} failed "
                        f"{max_requeue}x, substituting with plan[{alt_idx}] "
                        f"(dataset_idx={sub_spec.dataset_idx})",
                        flush=True,
                    )
                elif index < len(self.current_plan):
                    self.input_queue.put(self.current_plan[index])
                    print(f"[PlannedDataset] sample g{self._generation:04d}_{index} failed ({e}), re-enqueued", flush=True)

            alive_workers = [p for p in self.builder_processes if p.is_alive()]
            dead_workers = [p for p in self.builder_processes if not p.is_alive()]

            # Periodically log worker status
            now = time.time()
            if self._wait_log and now - last_alive_log > 30.0:
                print(
                    f"[PlannedDataset] waiting g{self._generation:04d}_{index} "
                    f"alive={len(alive_workers)}/{len(self.builder_processes)} "
                    f"elapsed={total_timeout - remaining:.0f}s "
                    f"{self._format_plan_spec(index)} {self._format_spool_summary()}",
                    flush=True,
                )
                last_alive_log = now

            # Hard fail if ALL workers are dead — no one will ever write the bundle.
            if dead_workers and not alive_workers and not self.spool.is_ready(index, self._generation):
                exitcodes = [p.exitcode for p in dead_workers]
                raise RuntimeError(
                    f"All {len(dead_workers)} builder workers died (exitcodes={exitcodes}). "
                    f"Sample g{self._generation:04d}_{index} cannot be produced. "
                    f"Check stderr for builder crash traceback / faulthandler output."
                )

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def _format_plan_spec(self, index: int) -> str:
        """Compact description of the sample currently blocking consumption."""
        try:
            spec = self.current_plan[index]
            dataset = self.adapters[spec.dataset_idx].dataset_name
            frame_min = min(spec.frame_indices) if spec.frame_indices else None
            frame_max = max(spec.frame_indices) if spec.frame_indices else None
            return (
                f"spec=dataset:{dataset},seq:{spec.sequence_name},"
                f"frames:{len(spec.frame_indices)}[{frame_min}..{frame_max}],"
                f"global:{spec.global_index}"
            )
        except Exception as exc:
            return f"spec=<unavailable:{type(exc).__name__}>"

    def _format_spool_summary(self) -> str:
        """Summarize current-generation ready indices for head-of-line diagnosis."""
        try:
            prefix = f"g{self._generation:04d}_"
            ready_indices = []
            for path in self.spool.spool_dir.iterdir():
                name = path.name
                if not (name.startswith(prefix) and name.endswith(".ready")):
                    continue
                try:
                    ready_indices.append(int(name.split("_", 1)[1].split(".", 1)[0]))
                except Exception:
                    continue
            if not ready_indices:
                return "spool_ready=0"
            ready_indices.sort()
            return (
                f"spool_ready={len(ready_indices)},"
                f"min={ready_indices[0]},max={ready_indices[-1]}"
            )
        except Exception as exc:
            return f"spool_ready=<unavailable:{type(exc).__name__}>"

    def set_epoch(self, epoch: int) -> None:
        """Advance to *epoch*: tear down old pipeline, rebuild from scratch.

        This is intentionally heavier than a hot-switch -- it guarantees
        zero cross-epoch pollution by destroying old queues and processes
        before creating new ones.
        """
        if (
            epoch == self.current_epoch
            and self.builder_processes
            and self.current_plan
        ):
            return

        self.current_epoch = epoch

        # 1. Kill old builders + queues.
        self._stop_pipeline()

        # 2. Bump generation.
        self._generation += 1

        # 3. Purge stale spool files.
        self.spool.set_generation(self._generation)

        # 4. Re-plan.
        self.current_plan = self.planner.generate_plan(
            epoch=epoch, count_per_rank=math.ceil(self.epoch_size / self.world_size),
            epoch_size=self.epoch_size, generation=self._generation,
        )

        # 5. Spin up fresh pipeline.
        self._start_pipeline()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_dataset_names(self) -> list[str]:
        return [adapter.dataset_name for adapter in self.adapters]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop builder processes and remove spool files."""
        self._stop_pipeline()
        self.spool.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# ======================================================================
# Utility
# ======================================================================

def set_dataset_epoch(dataset: Any, epoch: int) -> None:
    """Set epoch on a dataset (works for both MixtureDataset and PlannedMixtureDataset)."""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(dataset, "current_epoch"):
        dataset.current_epoch = epoch
