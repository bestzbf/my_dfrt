"""Benchmark ScanNet++ I/O: staged local depth.bin + h5 chunk cache vs COS mount."""
import os
import sys
import time
import pickle
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DEPTH_H, DEPTH_W = 192, 256
CLIP_LEN = 48
COS_SCENE = "/data_cos/hdu_datasets/scannetpp/data/00777c41d4"


def find_staged_scene():
    """Find a staged scene with depth.bin."""
    base = "/tmp/d4rt_sample_stage/work"
    for d in glob.glob(f"{base}/sample_stage_*/hdu_datasets/scannetpp/data/*/"):
        depth = os.path.join(d, "iphone", "depth.bin")
        if os.path.exists(depth) and os.path.getsize(depth) > 1_000_000:
            return d.rstrip("/")
    return None


def count_depth_frames(depth_path):
    """Count frames in a depth.bin by reading chunk headers."""
    import lz4.block
    count = 0
    with open(depth_path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            chunk_size = int.from_bytes(hdr, byteorder="little")
            f.seek(chunk_size, 1)
            count += 1
    return count


def bench_depth_staged(depth_path, n_frames, label):
    """Read depth.bin without chunk index (sequential scan)."""
    import lz4.block
    import zlib

    # First count frames and build index on the fly
    t0 = time.perf_counter()
    offsets = []
    with open(depth_path, "rb") as f:
        while True:
            pos = f.tell()
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            chunk_size = int.from_bytes(hdr, byteorder="little")
            offsets.append((pos + 4, chunk_size))
            f.seek(chunk_size, 1)
    index_time = time.perf_counter() - t0

    total = len(offsets)
    if total == 0:
        print(f"  [{label}] no frames found")
        return 0

    n = min(n_frames, total)
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(total, size=n, replace=False).tolist())

    t0 = time.perf_counter()
    frames = []
    with open(depth_path, "rb") as f:
        for fi in indices:
            offset, chunk_size = offsets[fi]
            f.seek(offset)
            chunk = f.read(chunk_size)
            try:
                dec = lz4.block.decompress(chunk, uncompressed_size=DEPTH_H * DEPTH_W * 2)
                depth = np.frombuffer(dec, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).astype(np.float32) / 1000.0
            except Exception:
                dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(dec, dtype=np.float32).reshape(DEPTH_H, DEPTH_W)
            frames.append(depth.copy())
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {n}/{total} frames, index={index_time*1000:.1f}ms, "
          f"read={elapsed:.3f}s, {elapsed/n*1000:.1f}ms/frame")
    return elapsed


def bench_depth_cos_with_index(n_frames, label):
    """Read depth.bin from COS mount using pre-built chunk index."""
    import lz4.block
    import zlib

    depth_path = os.path.join(COS_SCENE, "iphone", "depth.bin")
    index_path = os.path.join(COS_SCENE, "iphone", "depth_chunk_index.pkl")

    t0 = time.perf_counter()
    with open(index_path, "rb") as f:
        offsets = pickle.load(f)
    index_time = time.perf_counter() - t0

    total = len(offsets)
    n = min(n_frames, total)
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(total, size=n, replace=False).tolist())

    t0 = time.perf_counter()
    frames = []
    with open(depth_path, "rb") as f:
        for fi in indices:
            offset, chunk_size = offsets[fi]
            f.seek(offset)
            chunk = f.read(chunk_size)
            try:
                dec = lz4.block.decompress(chunk, uncompressed_size=DEPTH_H * DEPTH_W * 2)
                depth = np.frombuffer(dec, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).astype(np.float32) / 1000.0
            except Exception:
                dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(dec, dtype=np.float32).reshape(DEPTH_H, DEPTH_W)
            frames.append(depth.copy())
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {n}/{total} frames, index={index_time*1000:.1f}ms, "
          f"read={elapsed:.3f}s, {elapsed/n*1000:.1f}ms/frame")
    return elapsed


def bench_h5_chunk_cache(n_frames, label):
    """Read precomputed data from h5 chunk cache (local SSD)."""
    import h5py

    cache_root = "/tmp/d4rt_sample_stage/shared_raw_cache/data/.d4rt_h5_range_chunks"
    if not os.path.exists(cache_root):
        print(f"  [{label}] SKIP - no chunk cache")
        return 0

    # Find a cached h5 chunk dir
    dirs = os.listdir(cache_root)
    if not dirs:
        print(f"  [{label}] SKIP - empty cache")
        return 0

    # Read one chunk to measure speed
    sample_dir = os.path.join(cache_root, dirs[0])
    files = os.listdir(sample_dir)
    if not files:
        print(f"  [{label}] SKIP - no files in {dirs[0]}")
        return 0

    # Read a few cached chunks
    t0 = time.perf_counter()
    total_bytes = 0
    count = 0
    for fname in files[:n_frames]:
        fpath = os.path.join(sample_dir, fname)
        data = open(fpath, "rb").read()
        total_bytes += len(data)
        count += 1
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {count} chunks, {elapsed:.3f}s, {total_bytes/1e6:.1f}MB, "
          f"{elapsed/count*1000:.1f}ms/chunk")
    return elapsed


def bench_h5_cos_mount(n_frames, label):
    """Read precomputed.h5 from COS mount."""
    import h5py

    h5_path = os.path.join(COS_SCENE, "precomputed.h5")

    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        total = f["trajs_2d"].shape[0]
        n = min(n_frames, total)
        rng = np.random.RandomState(42)
        indices = sorted(rng.choice(total, size=n, replace=False).tolist())

        keys = ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]
        result = {}
        for key in keys:
            if key in f:
                result[key] = f[key][indices]
    elapsed = time.perf_counter() - t0
    total_bytes = sum(v.nbytes for v in result.values())
    print(f"  [{label}] {n} frames, {elapsed:.3f}s, {elapsed/n*1000:.1f}ms/frame, "
          f"{total_bytes/1e6:.1f}MB")
    return elapsed


def main():
    print("=" * 70)
    print("ScanNet++ I/O Benchmark: Staged Local vs COS Mount")
    print("=" * 70)

    staged = find_staged_scene()
    print(f"COS scene:   {COS_SCENE}")
    print(f"Staged scene: {staged}")

    if staged:
        staged_depth = os.path.join(staged, "iphone", "depth.bin")
        print(f"  staged depth.bin: {os.path.getsize(staged_depth)/1e6:.1f}MB")
    cos_depth = os.path.join(COS_SCENE, "iphone", "depth.bin")
    cos_h5 = os.path.join(COS_SCENE, "precomputed.h5")
    print(f"  COS depth.bin: {os.path.getsize(cos_depth)/1e6:.1f}MB")
    print(f"  COS precomputed.h5: {os.path.getsize(cos_h5)/1e6:.1f}MB")

    # --- depth.bin ---
    print(f"\n{'='*50}")
    print("DEPTH.BIN")
    print(f"{'='*50}")

    if staged:
        print(f"\nLocal staged ({os.path.basename(staged)}):")
        bench_depth_staged(staged_depth, CLIP_LEN, "staged cold")
        bench_depth_staged(staged_depth, CLIP_LEN, "staged warm")

    print(f"\nCOS mount:")
    bench_depth_cos_with_index(CLIP_LEN, "cos cold")
    bench_depth_cos_with_index(CLIP_LEN, "cos warm")

    # --- precomputed.h5 ---
    print(f"\n{'='*50}")
    print("PRECOMPUTED.H5")
    print(f"{'='*50}")

    print(f"\nCOS mount (h5py):")
    bench_h5_cos_mount(CLIP_LEN, "cos h5 cold")
    bench_h5_cos_mount(CLIP_LEN, "cos h5 warm")

    print(f"\nH5 chunk cache (local SSD):")
    bench_h5_chunk_cache(CLIP_LEN, "chunk cache")

    # --- summary ---
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"The staged pipeline downloads depth.bin + frames to local SSD,")
    print(f"then uses COS range requests for precomputed.h5 chunks (cached to SSD).")
    print(f"The COS FUSE mount has ~600ms/frame random seek latency.")
    print(f"Staged local SSD should be ~10-50x faster for depth.bin.")


if __name__ == "__main__":
    main()
