"""
Sample spool for managing local QuerySample bundle storage.

This module handles the lifecycle of pre-built QuerySample bundles:
- Writing bundles atomically (.building -> .ready)
- Waiting for bundles to become ready
- Cleaning up consumed bundles
- Error marker handling
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Optional

from .query_builder import QuerySample


class SampleSpool:
    """Manages local storage of pre-built QuerySample bundles.

    Each bundle is stored as a pickle file with atomic rename:
    - {index:08d}.building (temp file during construction)
    - {index:08d}.ready (final file, ready for consumption)
    - {index:08d}.error (error marker with exception info)

    Bundles are deleted immediately after loading to save disk space.
    """

    def __init__(
        self,
        spool_dir: str | Path,
        rank: int = 0,
        cleanup_on_init: bool = True,
    ):
        """
        Args:
            spool_dir: Directory for spool files (e.g., /tmp/d4rt_spool_rank0/)
            rank: Rank ID (for multi-GPU training)
            cleanup_on_init: Whether to clean up existing spool files on init
        """
        self.spool_dir = Path(spool_dir)
        self.rank = rank
        self.spool_dir.mkdir(parents=True, exist_ok=True)

        if cleanup_on_init:
            self._cleanup_all()

        self._ready_set: set[int] = set()
        self._error_set: set[int] = set()

    def _cleanup_all(self) -> None:
        """Remove all existing spool files."""
        for pattern in ("*.building", "*.ready", "*.error"):
            for p in self.spool_dir.glob(pattern):
                try:
                    p.unlink()
                except Exception:
                    pass

    def _get_building_path(self, index: int) -> Path:
        return self.spool_dir / f"{index:08d}.building"

    def _get_ready_path(self, index: int) -> Path:
        return self.spool_dir / f"{index:08d}.ready"

    def _get_error_path(self, index: int) -> Path:
        return self.spool_dir / f"{index:08d}.error"

    def write_bundle(self, index: int, sample: QuerySample) -> None:
        """Write a QuerySample bundle atomically.

        Args:
            index: Sample index
            sample: QuerySample to write
        """
        building_path = self._get_building_path(index)
        ready_path = self._get_ready_path(index)

        # Write to temp file
        with open(building_path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename
        os.replace(building_path, ready_path)
        self._ready_set.add(index)

    def write_error(self, index: int, exception: Exception) -> None:
        """Write an error marker for a failed sample.

        Args:
            index: Sample index
            exception: Exception that occurred
        """
        error_path = self._get_error_path(index)
        error_info = {
            "index": index,
            "exception_type": type(exception).__name__,
            "exception_str": str(exception),
        }
        with open(error_path, "wb") as f:
            pickle.dump(error_info, f)
        self._error_set.add(index)

    def wait_for_bundle(
        self,
        index: int,
        timeout: float = 300.0,
        poll_interval: float = 0.1,
    ) -> QuerySample:
        """Wait for a bundle to become ready and load it.

        Args:
            index: Sample index to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            QuerySample loaded from disk

        Raises:
            TimeoutError: If bundle doesn't become ready within timeout
            RuntimeError: If bundle has an error marker
        """
        ready_path = self._get_ready_path(index)
        error_path = self._get_error_path(index)

        start_time = time.time()
        while True:
            # Check for error marker
            if error_path.exists():
                with open(error_path, "rb") as f:
                    error_info = pickle.load(f)
                raise RuntimeError(
                    f"Sample {index} failed during building: "
                    f"{error_info['exception_type']}: {error_info['exception_str']}"
                )

            # Check if ready
            if ready_path.exists():
                with open(ready_path, "rb") as f:
                    sample = pickle.load(f)

                # Clean up immediately after loading
                try:
                    ready_path.unlink()
                    self._ready_set.discard(index)
                except Exception:
                    pass

                return sample

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Sample {index} did not become ready within {timeout}s. "
                    f"Spool dir: {self.spool_dir}"
                )

            time.sleep(poll_interval)

    def is_ready(self, index: int) -> bool:
        """Check if a bundle is ready (non-blocking).

        Args:
            index: Sample index

        Returns:
            True if bundle is ready
        """
        return self._get_ready_path(index).exists()

    def has_error(self, index: int) -> bool:
        """Check if a bundle has an error marker.

        Args:
            index: Sample index

        Returns:
            True if bundle has error
        """
        return self._get_error_path(index).exists()

    def get_spool_size(self) -> int:
        """Get total size of spool directory in bytes."""
        total = 0
        for p in self.spool_dir.glob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def get_ready_count(self) -> int:
        """Get number of ready bundles."""
        return len(list(self.spool_dir.glob("*.ready")))

    def cleanup(self) -> None:
        """Clean up all spool files."""
        self._cleanup_all()
        self._ready_set.clear()
        self._error_set.clear()
