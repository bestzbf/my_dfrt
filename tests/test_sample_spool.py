"""Updated unit tests for datasets/sample_spool.py with generation isolation"""

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
    return {
        "video": torch.randn(8, 3, 256, 256),
        "coords": torch.randn(512, 2),
        "dataset_name": "test",
        "sequence_name": "scene_001",
    }


def test_write_and_read_bundle_with_generation():
    """Write a bundle with generation, then read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        sample = _make_fake_sample()
        spool.write_bundle(0, sample, generation=0)

        assert spool.is_ready(0, generation=0), "Bundle should be ready after write"

        loaded = spool.wait_for_bundle(0, generation=0, timeout=1.0)
        assert torch.allclose(loaded["video"], sample["video"])
        assert loaded["dataset_name"] == "test"

        # Should be cleaned up after loading
        assert not spool.is_ready(0, generation=0), "Bundle should be cleaned up after loading"

    print("✓ test_write_and_read_bundle_with_generation passed")


def test_generation_isolation():
    """Old generation files are purged when set_generation is called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)

        # Write bundles for generation 0
        spool.set_generation(0)
        spool.write_bundle(0, _make_fake_sample(), generation=0)
        spool.write_bundle(1, _make_fake_sample(), generation=0)
        assert spool.is_ready(0, generation=0)
        assert spool.is_ready(1, generation=0)

        # Switch to generation 1 — old files should be purged
        spool.set_generation(1)
        assert not spool.is_ready(0, generation=0), "Gen 0 files should be purged"
        assert not spool.is_ready(1, generation=0), "Gen 0 files should be purged"

        # Write new bundles for generation 1
        spool.write_bundle(0, _make_fake_sample(), generation=1)
        assert spool.is_ready(0, generation=1)
        assert not spool.is_ready(0, generation=0)

    print("✓ test_generation_isolation passed")


def test_disk_watermark():
    """wait_for_space blocks when spool size exceeds max_spool_bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set a very small limit (1 KB)
        spool = SampleSpool(tmpdir, rank=0, max_spool_bytes=1024)
        spool.set_generation(0)

        # Write a large sample (should exceed 1 KB)
        large_sample = {
            "video": torch.randn(8, 3, 256, 256),  # ~2 MB
            "coords": torch.randn(512, 2),
        }
        spool.write_bundle(0, large_sample, generation=0)

        # Now spool is over limit, wait_for_space should block
        start = time.time()
        try:
            spool.wait_for_space(timeout=0.5)
            assert False, "Should have raised TimeoutError"
        except TimeoutError:
            elapsed = time.time() - start
            assert elapsed >= 0.4, "Should have waited at least ~0.5s"

        # Consume the bundle to free space
        spool.wait_for_bundle(0, generation=0, timeout=1.0)

        # Now wait_for_space should succeed
        spool.wait_for_space(timeout=1.0)

    print("✓ test_disk_watermark passed")


def test_error_marker_with_generation():
    """Write an error marker with generation and verify it raises on wait."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        spool.write_error(5, RuntimeError("test error"), generation=0)

        assert spool.has_error(5, generation=0)

        try:
            spool.wait_for_bundle(5, generation=0, timeout=1.0)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "test error" in str(e)

    print("✓ test_error_marker_with_generation passed")


def test_timeout_with_generation():
    """Verify timeout works when bundle never appears."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        start = time.time()
        try:
            spool.wait_for_bundle(99, generation=0, timeout=0.5, poll_interval=0.1)
            assert False, "Should have raised TimeoutError"
        except TimeoutError:
            elapsed = time.time() - start
            assert elapsed >= 0.4, "Should have waited at least ~0.5s"

    print("✓ test_timeout_with_generation passed")


def test_concurrent_write_read_with_generation():
    """Writer in another thread, reader waits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)
        sample = _make_fake_sample()

        def writer():
            time.sleep(0.3)
            spool.write_bundle(10, sample, generation=0)

        t = threading.Thread(target=writer)
        t.start()

        loaded = spool.wait_for_bundle(10, generation=0, timeout=2.0, poll_interval=0.05)
        t.join()

        assert torch.allclose(loaded["video"], sample["video"])

    print("✓ test_concurrent_write_read_with_generation passed")


def test_multiple_bundles_with_generation():
    """Write and read multiple bundles with generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        samples = {}
        for i in range(5):
            s = _make_fake_sample()
            s["sequence_name"] = f"scene_{i:03d}"
            spool.write_bundle(i, s, generation=0)
            samples[i] = s

        assert spool.get_ready_count() == 5

        for i in range(5):
            loaded = spool.wait_for_bundle(i, generation=0, timeout=1.0)
            assert loaded["sequence_name"] == f"scene_{i:03d}"

        assert spool.get_ready_count() == 0

    print("✓ test_multiple_bundles_with_generation passed")


def test_cleanup():
    """Verify cleanup removes all files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        for i in range(3):
            spool.write_bundle(i, _make_fake_sample(), generation=0)
        spool.write_error(10, RuntimeError("oops"), generation=0)

        assert spool.get_ready_count() == 3

        spool.cleanup()
        assert spool.get_ready_count() == 0
        assert not spool.has_error(10, generation=0)

    print("✓ test_cleanup passed")


def test_spool_size():
    """Verify spool size tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        spool.write_bundle(0, _make_fake_sample(), generation=0)
        size = spool.get_spool_size()
        assert size > 0, "Spool should have non-zero size after writing"

    print("✓ test_spool_size passed")


def test_generation_file_naming():
    """Verify generation prefix in filenames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spool = SampleSpool(tmpdir, rank=0)
        spool.set_generation(0)

        spool.write_bundle(5, _make_fake_sample(), generation=0)

        # Check that file has generation prefix
        files = list(Path(tmpdir).glob("*.ready"))
        assert len(files) == 1
        assert files[0].name.startswith("g0000_"), f"File should start with g0000_, got {files[0].name}"

        # Switch generation and write again
        spool.set_generation(1)
        spool.write_bundle(5, _make_fake_sample(), generation=1)

        files = list(Path(tmpdir).glob("*.ready"))
        assert len(files) == 1
        assert files[0].name.startswith("g0001_"), f"File should start with g0001_, got {files[0].name}"

    print("✓ test_generation_file_naming passed")


if __name__ == "__main__":
    test_write_and_read_bundle_with_generation()
    test_generation_isolation()
    test_disk_watermark()
    test_error_marker_with_generation()
    test_timeout_with_generation()
    test_concurrent_write_read_with_generation()
    test_multiple_bundles_with_generation()
    test_cleanup()
    test_spool_size()
    test_generation_file_naming()
    print("\nAll spool tests passed!")
