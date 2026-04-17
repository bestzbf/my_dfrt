"""Pickle-based disk cache for adapter index building."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar('T')


def load_or_build(build_fn: Callable[[], T], cache_path: Path) -> T:
    """Load pickled index from cache_path, or call build_fn and save result.

    Safe for concurrent DDP use: writes via a temp file + os.replace (atomic).
    To force rebuild, delete the cache file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # Corrupt cache — fall through and rebuild

    result = build_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a per-process temp file to avoid DDP ranks clobbering each other.
    # The first rank to finish wins; others silently skip the rename.
    tmp = cache_path.with_suffix(f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        # os.replace is atomic on POSIX; if another rank already wrote the
        # cache this will still succeed (last writer wins, both are valid).
        os.replace(tmp, cache_path)
    except OSError:
        # Another rank already wrote it — clean up our temp file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return result
