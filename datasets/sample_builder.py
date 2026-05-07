"""
Sample builder for background QuerySample construction.

This module implements background worker processes that build QuerySample
bundles from SampleSpec objects, enabling prefetch from COS.

Key design decisions:
- Uses forkserver context to avoid inheriting CUDA/NCCL state from parent.
- Checks spec.generation against a shared counter so stale specs from
  previous epochs are silently discarded instead of built and spooled.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random as _random_module
import threading
import time
import traceback
from typing import Any, Optional

from .adapters.base import BaseAdapter
from .planning import SampleSpec
from .query_builder import D4RTQueryBuilder, QuerySample
from .sample_stage import build_sample_stager
from .sample_spool import SampleSpool
from .transforms import GeometryTransformPipeline


class SampleBuilder:
    """Background process that builds QuerySample bundles from SampleSpecs.

    Each builder runs in a separate process and:
    1. Reads SampleSpec from input queue
    2. Checks generation -- discards stale specs
    3. Calls adapter.load_clip() -> transform() -> query_builder()
    4. Writes QuerySample to spool directory
    5. Signals completion via output queue

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
        current_generation: "mp.Value",
        sample_stage_config: Optional[dict[str, Any]] = None,
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
            current_generation: Shared integer updated by the main process on
                epoch transitions.  Specs whose generation is older than this
                value are silently discarded.
        """
        self.builder_id = builder_id
        self.adapters = adapters
        self.transform = transform
        self.query_builder = query_builder
        self.spool = spool
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clip_len = clip_len
        self.current_generation = current_generation
        self.sample_stage_config = sample_stage_config
        self._sample_stager = None
        self._last_profile: dict[str, Any] = {}
        self._profile_builder = os.getenv("D4RT_PROFILE_BUILDER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._profile_builder_all = os.getenv("D4RT_PROFILE_BUILDER_ALL", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._profile_builder_threshold_s = float(
            os.getenv("D4RT_PROFILE_BUILDER_THRESHOLD_S", "5.0")
        )
        self._verbose_builder = os.getenv("D4RT_VERBOSE_BUILDER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._enable_faulthandler = os.getenv("D4RT_BUILDER_FAULTHANDLER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop for builder process."""
        enable_faulthandler = getattr(self, "_enable_faulthandler", False)
        verbose_builder = getattr(self, "_verbose_builder", False)
        if enable_faulthandler:
            import faulthandler, signal, sys
            faulthandler.enable(file=sys.stderr, all_threads=True)
            # Do not register SIGTERM: multiprocessing.terminate() uses it for
            # normal cleanup, and dumping a stack on expected shutdown is noisy.
            for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
                try:
                    faulthandler.register(sig, file=sys.stderr, chain=True)
                except Exception:
                    pass
        if verbose_builder:
            print(f"[Builder {self.builder_id}] STARTED pid={os.getpid()}", flush=True)
        try:
            while True:
                try:
                    # Get next spec from queue (blocking with short timeout so we
                    # can re-check for shutdown even if the queue is empty).
                    spec: SampleSpec = self.input_queue.get(timeout=1.0)

                    if spec is None:
                        # Shutdown signal
                        if verbose_builder:
                            print(f"[Builder {self.builder_id}] received None, EXITING normally", flush=True)
                        break

                    # --- generation gate ---
                    if spec.generation < self.current_generation.value:
                        # Stale spec from a previous epoch; drop it.
                        continue

                    # Build sample with retry logic
                    success = self._build_sample_with_retry(spec)

                    # Signal completion (use local_index so the coordinator knows
                    # which spool slot was written).
                    self.output_queue.put((spec.local_index, success))

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Builder {self.builder_id}] Unexpected error: {e}", flush=True)
                    traceback.print_exc()
        except BaseException as e:
            print(f"[Builder {self.builder_id}] FATAL EXIT: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if verbose_builder:
                print(f"[Builder {self.builder_id}] EXITED pid={os.getpid()}", flush=True)

    # ------------------------------------------------------------------
    # Build with retry
    # ------------------------------------------------------------------

    def _build_sample_with_retry(self, spec: SampleSpec, max_attempts: int = 3) -> bool:
        """Build a sample with retry logic for transient errors.

        Args:
            spec: SampleSpec to build
            max_attempts: Maximum number of retry attempts

        Returns:
            True if successful, False if failed
        """
        build_timeout = float(os.environ.get("D4RT_BUILD_TIMEOUT", "120"))
        for attempt in range(max_attempts):
            attempt_start = time.perf_counter()
            try:
                if build_timeout > 0:
                    result_box: list = []
                    exc_box: list = []
                    def _run():
                        try:
                            result_box.append(self._build_sample(spec))
                        except Exception as e:
                            exc_box.append(e)
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    t.join(timeout=build_timeout)
                    if t.is_alive():
                        t.join(timeout=70)
                        adapter = self.adapters[spec.dataset_idx]
                        dataset_name = getattr(adapter, "dataset_name", type(adapter).__name__)
                        if hasattr(adapter, '_scene_cache'):
                            adapter._scene_cache.pop(spec.sequence_name, None)
                        if hasattr(adapter, '_depth_chunk_cache'):
                            adapter._depth_chunk_cache.pop(spec.sequence_name, None)

                        # Get timing info if available (from partial execution)
                        timing_info = ""
                        if hasattr(self, '_last_load_timing') and self._last_load_timing:
                            timing = self._last_load_timing
                            timing_info = (
                                f" | partial_timing: "
                                f"scene_data={timing.get('scene_data_s', 0)*1000:.0f}ms "
                                f"rgb_load={timing.get('rgb_load_s', 0)*1000:.0f}ms "
                                f"depth_load={timing.get('depth_load_s', 0)*1000:.0f}ms "
                                f"precomputed={timing.get('precomputed_s', 0)*1000:.0f}ms"
                            )

                        err = TimeoutError(
                            f"_build_sample timed out after {build_timeout}s "
                            f"(spec={spec.local_index}, dataset_idx={spec.dataset_idx}, "
                            f"dataset={dataset_name}, seq={spec.sequence_name}, "
                            f"frames={len(spec.frame_indices)}){timing_info}"
                        )
                        total_s = time.perf_counter() - attempt_start

                        # Print detailed timeout info
                        print(
                            f"[Timeout] dataset={dataset_name} seq={spec.sequence_name} "
                            f"frames={len(spec.frame_indices)} timeout={build_timeout}s "
                            f"total_elapsed={total_s:.1f}s{timing_info}",
                            flush=True,
                        )

                        self._maybe_print_profile(spec=spec, attempt=attempt, success=False,
                                                   total_s=total_s, error=err)
                        # Don't retry on timeout — stale threads exhaust COS connections
                        self.spool.write_error(spec.local_index, err, spec.generation)
                        return False
                    if exc_box:
                        raise exc_box[0]
                    sample = result_box[0]
                else:
                    sample = self._build_sample(spec)

                # Validate frame count
                if sample.video.shape[0] != self.clip_len:
                    raise RuntimeError(
                        f"Sample has {sample.video.shape[0]} frames, "
                        f"expected {self.clip_len}"
                    )

                # Wait for disk headroom, then write
                t_wait0 = time.perf_counter()
                self.spool.wait_for_space()
                t_wait_space = time.perf_counter() - t_wait0
                t_write0 = time.perf_counter()
                self.spool.write_bundle(
                    spec.local_index, sample, spec.generation
                )
                t_write = time.perf_counter() - t_write0
                total_s = time.perf_counter() - attempt_start

                # Print detailed timing for slow samples (>5s) or timeout cases
                slow_threshold = float(os.environ.get("D4RT_SLOW_SAMPLE_THRESHOLD_S", "5.0"))
                if total_s > slow_threshold and hasattr(self, '_last_load_timing') and self._last_load_timing:
                    timing = self._last_load_timing
                    adapter = self.adapters[spec.dataset_idx]
                    dataset_name = getattr(adapter, "dataset_name", type(adapter).__name__)
                    print(
                        f"[SlowSample] dataset={dataset_name} seq={spec.sequence_name} "
                        f"frames={len(spec.frame_indices)} total_load={timing.get('total_s', 0)*1000:.0f}ms | "
                        f"scene_data={timing.get('scene_data_s', 0)*1000:.0f}ms "
                        f"rgb_load={timing.get('rgb_load_s', 0)*1000:.0f}ms "
                        f"rgb_resize={timing.get('rgb_resize_s', 0)*1000:.0f}ms "
                        f"depth_load={timing.get('depth_load_s', 0)*1000:.0f}ms "
                        f"depth_resize={timing.get('depth_resize_s', 0)*1000:.0f}ms "
                        f"precomputed={timing.get('precomputed_s', 0)*1000:.0f}ms "
                        f"process={timing.get('process_s', 0)*1000:.0f}ms | "
                        f"stage={self._last_profile.get('stage_s', 0)*1000:.0f}ms "
                        f"transform={self._last_profile.get('transform_s', 0)*1000:.0f}ms "
                        f"query={self._last_profile.get('query_s', 0)*1000:.0f}ms | "
                        f"total_build={total_s*1000:.0f}ms",
                        flush=True,
                    )

                self._maybe_print_profile(
                    spec=spec,
                    attempt=attempt,
                    success=True,
                    total_s=total_s,
                    wait_space_s=t_wait_space,
                    write_s=t_write,
                )
                return True

            except Exception as e:
                total_s = time.perf_counter() - attempt_start
                self._maybe_print_profile(
                    spec=spec,
                    attempt=attempt,
                    success=False,
                    total_s=total_s,
                    error=e,
                )
                if attempt < max_attempts - 1:
                    print(
                        f"[Builder {self.builder_id}] Retry {attempt + 1}/{max_attempts} "
                        f"for sample {spec.local_index}: {e}"
                    )
                else:
                    print(
                        f"[Builder {self.builder_id}] Failed sample {spec.local_index} "
                        f"after {max_attempts} attempts: {e}"
                    )
                    self.spool.write_error(
                        spec.local_index, e, spec.generation
                    )
                    return False

        return False  # unreachable, but keeps mypy happy

    # ------------------------------------------------------------------
    # Core build logic
    # ------------------------------------------------------------------

    def _build_sample(self, spec: SampleSpec) -> QuerySample:
        """Build a single QuerySample from a SampleSpec.

        Args:
            spec: SampleSpec with dataset_idx, sequence_name, frame_indices,
                  rng_state, and global_index (used as RNG seed).

        Returns:
            QuerySample ready for training
        """
        adapter = self.adapters[spec.dataset_idx]
        self._last_profile = {
            "dataset": getattr(adapter, "dataset_name", type(adapter).__name__),
            "sequence": spec.sequence_name,
            "frame_min": min(spec.frame_indices) if spec.frame_indices else None,
            "frame_max": max(spec.frame_indices) if spec.frame_indices else None,
            "frame_count": len(spec.frame_indices),
            "stage_s": 0.0,
            "load_s": 0.0,
            "transform_s": 0.0,
            "query_s": 0.0,
        }

        sample_stager = self._get_sample_stager()
        sample_tag = f"b{self.builder_id}_g{spec.generation}_i{spec.local_index}"
        stage_ctx = (
            sample_stager.stage_sample(
                adapter,
                spec.sequence_name,
                spec.frame_indices,
                sample_tag=sample_tag,
            )
            if sample_stager is not None
            else _nullcontext(adapter)
        )
        t_stage0 = time.perf_counter()
        with stage_ctx as staged_adapter:
            t_stage = time.perf_counter() - t_stage0
            self._last_profile["stage_s"] = t_stage
            t_load0 = time.perf_counter()
            clip = staged_adapter.load_clip(spec.sequence_name, spec.frame_indices)
            t_load = time.perf_counter() - t_load0
            self._last_profile["load_s"] = t_load

        # Extract detailed timing from clip metadata if available
        self._last_load_timing = clip.metadata.pop("_load_timing", None)

        # Restore RNG state (deterministic transforms + query building)
        rng = _random_module.Random()
        rng.setstate(spec.rng_state)

        # Apply geometry transforms
        t_transform0 = time.perf_counter()
        result = self.transform(clip, rng=rng)
        t_transform = time.perf_counter() - t_transform0
        self._last_profile["transform_s"] = t_transform

        # Build query sample
        t_query0 = time.perf_counter()
        sample = self.query_builder(result, py_rng=rng)
        t_query = time.perf_counter() - t_query0
        self._last_profile["query_s"] = t_query
        sample.metadata.update(
            {
                "planned_generation": spec.generation,
                "planned_local_index": spec.local_index,
                "planned_global_index": spec.global_index,
                "planned_dataset_idx": spec.dataset_idx,
                "planned_frame_indices": list(spec.frame_indices),
                "planned_frame_count": len(spec.frame_indices),
                "planned_frame_min": min(spec.frame_indices) if spec.frame_indices else None,
                "planned_frame_max": max(spec.frame_indices) if spec.frame_indices else None,
                "builder_id": self.builder_id,
            }
        )

        self._last_profile.update(
            stage_s=t_stage,
            load_s=t_load,
            transform_s=t_transform,
            query_s=t_query,
        )

        return sample

    def _maybe_print_profile(
        self,
        spec: SampleSpec,
        attempt: int,
        success: bool,
        total_s: float,
        wait_space_s: float = 0.0,
        write_s: float = 0.0,
        error: Optional[Exception] = None,
    ) -> None:
        if not self._profile_builder:
            return
        if not self._profile_builder_all and total_s < self._profile_builder_threshold_s:
            return

        p = self._last_profile
        status = "ok" if success else "fail"
        msg = (
            f"[BuilderProfile b{self.builder_id}] "
            f"status={status} g{spec.generation:04d}_{spec.local_index:08d} "
            f"attempt={attempt + 1} total={total_s:.3f}s "
            f"dataset={p.get('dataset', '<unknown>')} "
            f"seq={p.get('sequence', spec.sequence_name)} "
            f"frames={p.get('frame_count', len(spec.frame_indices))}"
            f"[{p.get('frame_min', None)}..{p.get('frame_max', None)}] "
            f"stage={p.get('stage_s', 0.0):.3f}s "
            f"load={p.get('load_s', 0.0):.3f}s "
            f"transform={p.get('transform_s', 0.0):.3f}s "
            f"query={p.get('query_s', 0.0):.3f}s "
            f"wait_space={wait_space_s:.3f}s "
            f"write={write_s:.3f}s"
        )
        if error is not None:
            msg += f" error={type(error).__name__}: {error}"
        print(msg, flush=True)

    def _get_sample_stager(self):
        if self._sample_stager is None:
            self._sample_stager = build_sample_stager(self.sample_stage_config)
        return self._sample_stager


class _nullcontext:
    def __init__(self, value: Any):
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ======================================================================
# Process lifecycle helpers
# ======================================================================


def start_builder_process(
    builder_id: int,
    adapters: list[BaseAdapter],
    transform: GeometryTransformPipeline,
    query_builder: D4RTQueryBuilder,
    spool: SampleSpool,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    clip_len: int,
    current_generation: "mp.Value",
    sample_stage_config: Optional[dict[str, Any]] = None,
) -> mp.Process:
    """Start a builder process using the *forkserver* start method.

    Using forkserver (instead of the default fork) avoids inheriting
    CUDA/NCCL context from the parent process, which would otherwise
    cause hangs or crashes in the child.

    Args:
        builder_id: Unique ID for this builder
        adapters: List of dataset adapters
        transform: Transform pipeline
        query_builder: Query builder
        spool: Sample spool
        input_queue: Input queue for SampleSpecs
        output_queue: Output queue for completion signals
        clip_len: Expected clip length
        current_generation: Shared mp.Value(ctypes.c_int64) that the main
            process bumps on each epoch transition

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
        current_generation=current_generation,
        sample_stage_config=sample_stage_config,
    )

    ctx = mp.get_context("forkserver")
    process = ctx.Process(target=builder.run, name=f"SampleBuilder-{builder_id}")
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
