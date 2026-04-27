"""Unit tests for datasets/sample_spool.py"""

import sys
import tempfile
import time
import threading
from pathlib import Path

sys.path.insert(0, ".")

import torch
from datasets.sample_spool import SampleSpool


def _make_fake_sample():
    """Create a minimal fake QuerySample-like object for testing."""
    # Using a simple dict instead of QuerySample to avoid complex dependencies
    return {
        "video": torch.randn(48, 3, 256, 256),
        "coords": torch.randn(2048, 2),
        "dataset_name": "test",
        "sequence_name": "scene_001",
    }


def test_write_and_read_bundle():
    """Write a bundle, then read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        sample = _make_fake_sample()
        spool.write_bundle(0, sample)

        assert spool.is_ready(0), "Bundle should be ready after write"

        loaded = spool.wait_for_bundle(0, timeout=1.0)
        assert torch.allclose(loaded["video"], sample["video"])
        assert loaded["dataset_name"] == "test"

        # Should be cleaned up after loading
        assert not spool.is_ready(0), "Bundle should be cleaned up after loading"

    print("✓ test_write_and_read_bundle passed")


def test_error_marker():
    """Write an error marker and verify it raises on wait."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        spool.write_error(5, RuntimeError("test error"))

        assert spool.has_error(5)

        try:
            spool.wait_for_bundle(5, timeout=1.0)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "test error" in str(e)

    print("✓ test_error_marker passed")


def test_timeout():
    """Verify timeout works when bundle never appears."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        start = time.time()
        try:
            spool.wait_for_bundle(99, timeout=0.5, poll_interval=0.1)
            assert False, "Should have raised TimeoutError"
        except TimeoutError:
            elapsed = time.time() - start
            assert elapsed >= 0.4, "Should have waited at least ~0.5s"

    print("✓ test_timeout passed")


def test_concurrent_write_read():
    """Writer in another thread, reader waits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        sample = _make_fake_sample()

        def writer():
            time.sleep(0.3)
            spool.write_bundle(10, sample)

        t = threading.Thread(target=writer)
        t.start()

        loaded = spool.wait_for_bundle(10, timeout=2.0, poll_interval=0.05)
        t.join()

        assert torch.allclose(loaded["video"], sample["video"])

    print("✓ test_concurrent_write_read passed")


def test_multiple_bundles():
    """Write and read multiple bundles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        samples = {}
        for i in range(5):
            s = _make_fake_sample()
            s["sequence_name"] = f"scene_{i:03d}"
            spool.write_bundle(i, s)
            samples[i] = s

        assert spool.get_ready_count() == 5

        for i in range(5):
            loaded = spool.wait_for_bundle(i, timeout=1.0)
            assert loaded["sequence_name"] == f"scene_{i:03d}"

        assert spool.get_ready_count() == 0

    print("✓ test_multiple_bundles passed")


def test_cleanup():
    """Verify cleanup removes all files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        for i in range(3):
            spool.write_bundle(i, _make_fake_sample())
        spool.write_error(10, RuntimeError("oops"))

        assert spool.get_ready_count() == 3

        spool.cleanup()
        assert spool.get_ready_count() == 0
        assert not spool.has_error(10)

    print("✓ test_cleanup passed")


def test_spool_size():
    """Verify spool size tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        spool.write_bundle(0, _make_fake_sample())
        size = spool.get_spool_size()
        assert size > 0, "Spool should have non-zero size after writing"

    print("✓ test_spool_size passed")


if __name__ == "__main__":
    test_write_and_read_bundle()
    test_error_marker()
    test_timeout()
    test_concurrent_write_read()
    test_multiple_bundles()
    test_cleanup()
    test_spool_size()
    print("\nAll spool tests passed!")
