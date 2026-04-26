"""Pickle-based disk cache for adapter index building."""

from __future__ import annotations

import contextlib
import os
import pickle
from pathlib import Path
from typing import Callable, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

T = TypeVar('T')


@contextlib.contextmanager
def _cache_build_lock(lock_path: Path):
    """Serialize expensive cold builds across DDP ranks/processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_or_build(build_fn: Callable[[], T], cache_path: Path) -> T:
    """Load pickled index from cache_path, or call build_fn and save result.

    Safe for concurrent DDP use: cold builds are serialized by a lock file, and
    writes use a temp file + os.replace (atomic). To force rebuild, delete the
    cache file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # Corrupt cache — fall through and rebuild

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _cache_build_lock(lock_path):
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # Corrupt cache — rebuild under lock.

        result = build_fn()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a per-process temp file to avoid clobbering partially-written
        # output if the builder crashes.
        tmp = cache_path.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            os.replace(tmp, cache_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return result
