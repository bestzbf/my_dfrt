"""Benchmark ScanNet++ data loading: depth.bin and precomputed.h5/npz.
Focus on indexed reads (what training actually uses)."""
import os
import sys
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DATA_ROOT = "/data_cos/hdu_datasets/scannetpp/data"
SCENE = "00777c41d4"
SCENE_DIR = os.path.join(DATA_ROOT, SCENE)
DEPTH_PATH = os.path.join(SCENE_DIR, "iphone", "depth.bin")
DEPTH_INDEX_PATH = os.path.join(SCENE_DIR, "iphone", "depth_chunk_index.pkl")
H5_PATH = os.path.join(SCENE_DIR, "precomputed.h5")
NPZ_PATH = os.path.join(SCENE_DIR, "precomputed.npz")
H5_INDEX_PATH = os.path.join(SCENE_DIR, "precomputed.h5_chunk_index.pkl")

DEPTH_H, DEPTH_W = 192, 256
CLIP_LEN = 48


def load_depth_index():
    """Load the pre-built depth chunk offset index."""
    with open(DEPTH_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def bench_depth_indexed(chunk_offsets, n_frames, total_frames, label=""):
    """Read N random frames using chunk index."""
    import lz4.block
    import zlib

    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(total_frames, size=n_frames, replace=False).tolist())

    t0 = time.perf_counter()
    frames = []
    with open(DEPTH_PATH, "rb") as f:
        for fi in indices:
            offset, chunk_size = chunk_offsets[fi]
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
    total_mb = sum(f.nbytes for f in frames) / 1e6
    print(f"  [depth indexed {label}] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame, {total_mb:.1f}MB data")
    return elapsed


def bench_depth_indexed_sequential(chunk_offsets, n_frames, total_frames):
    """Read N sequential frames (contiguous in file)."""
    import lz4.block
    import zlib

    start = max(0, total_frames // 2 - n_frames // 2)
    indices = list(range(start, start + n_frames))

    t0 = time.perf_counter()
    frames = []
    with open(DEPTH_PATH, "rb") as f:
        for fi in indices:
            offset, chunk_size = chunk_offsets[fi]
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
    print(f"  [depth indexed sequential] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame")
    return elapsed


def bench_h5_open():
    """Time to open h5 file."""
    import h5py
    t0 = time.perf_counter()
    f = h5py.File(H5_PATH, "r")
    elapsed = time.perf_counter() - t0
    keys = list(f.keys())
    shapes = {k: f[k].shape for k in keys}
    f.close()
    print(f"  [h5 open] {elapsed:.3f}s")
    for k, s in shapes.items():
        print(f"    {k}: shape={s}")
    return elapsed


def bench_h5_read(n_frames, total_frames, label=""):
    """Read N random frames from h5."""
    import h5py

    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(total_frames, size=n_frames, replace=False).tolist())

    t0 = time.perf_counter()
    with h5py.File(H5_PATH, "r") as f:
        keys = ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]
        result = {}
        for key in keys:
            if key in f:
                result[key] = f[key][indices]
    elapsed = time.perf_counter() - t0
    total_bytes = sum(v.nbytes for v in result.values())
    print(f"  [h5 read {label}] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame, {total_bytes/1e6:.1f}MB")
    return elapsed


def bench_h5_sequential(n_frames, total_frames):
    """Read N sequential frames from h5."""
    import h5py

    start = max(0, total_frames // 2 - n_frames // 2)
    indices = list(range(start, start + n_frames))

    t0 = time.perf_counter()
    with h5py.File(H5_PATH, "r") as f:
        keys = ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]
        result = {}
        for key in keys:
            if key in f:
                result[key] = f[key][indices]
    elapsed = time.perf_counter() - t0
    print(f"  [h5 sequential] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame")
    return elapsed


def bench_npz_read(n_frames, total_frames, label=""):
    """Read N random frames from npz."""
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(total_frames, size=n_frames, replace=False).tolist())

    t0 = time.perf_counter()
    data = np.load(NPZ_PATH, allow_pickle=False)
    result = {}
    for key in ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]:
        if key in data:
            result[key] = data[key][indices]
    data.close()
    elapsed = time.perf_counter() - t0
    total_bytes = sum(v.nbytes for v in result.values())
    print(f"  [npz read {label}] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame, {total_bytes/1e6:.1f}MB")
    return elapsed


def main():
    print("=" * 70)
    print(f"ScanNet++ I/O Benchmark — COS mount")
    print(f"Scene: {SCENE}")
    print(f"Clip length: {CLIP_LEN} frames")
    print("=" * 70)

    # --- file sizes ---
    print(f"\nFile sizes:")
    for p in [DEPTH_PATH, H5_PATH, NPZ_PATH, DEPTH_INDEX_PATH, H5_INDEX_PATH]:
        if os.path.exists(p):
            print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f}MB")

    # --- depth.bin ---
    print(f"\n{'='*50}")
    print("DEPTH.BIN (indexed read, what training uses)")
    print(f"{'='*50}")

    t0 = time.perf_counter()
    chunk_offsets = load_depth_index()
    idx_time = time.perf_counter() - t0
    total_depth_frames = len(chunk_offsets)
    print(f"  chunk index: {total_depth_frames} chunks, loaded in {idx_time*1000:.1f}ms")

    # cold read
    bench_depth_indexed(chunk_offsets, CLIP_LEN, total_depth_frames, "cold")
    # warm read (file cache)
    bench_depth_indexed(chunk_offsets, CLIP_LEN, total_depth_frames, "warm")
    # sequential
    bench_depth_indexed_sequential(chunk_offsets, CLIP_LEN, total_depth_frames)
    # different sizes
    for n in [8, 16, 32, 64]:
        bench_depth_indexed(chunk_offsets, n, total_depth_frames, f"n={n}")

    # --- precomputed.h5 ---
    print(f"\n{'='*50}")
    print("PRECOMPUTED.H5")
    print(f"{'='*50}")

    bench_h5_open()
    import h5py
    with h5py.File(H5_PATH, "r") as f:
        total_h5 = f["trajs_2d"].shape[0]

    bench_h5_read(CLIP_LEN, total_h5, "cold")
    bench_h5_read(CLIP_LEN, total_h5, "warm")
    bench_h5_sequential(CLIP_LEN, total_h5)
    for n in [8, 16, 32, 64]:
        bench_h5_read(n, total_h5, f"n={n}")

    # --- precomputed.npz ---
    print(f"\n{'='*50}")
    print("PRECOMPUTED.NPZ")
    print(f"{'='*50}")

    npz_data = np.load(NPZ_PATH, allow_pickle=False)
    total_npz = npz_data["trajs_2d"].shape[0]
    npz_data.close()

    bench_npz_read(CLIP_LEN, total_npz, "cold")
    bench_npz_read(CLIP_LEN, total_npz, "warm")

    # --- full clip simulation ---
    print(f"\n{'='*50}")
    print(f"FULL CLIP SIMULATION ({CLIP_LEN} frames)")
    print(f"{'='*50}")

    import lz4.block
    import zlib
    rng = np.random.RandomState(42)
    depth_indices = sorted(rng.choice(total_depth_frames, size=CLIP_LEN, replace=False).tolist())
    h5_indices = sorted(rng.choice(total_h5, size=CLIP_LEN, replace=False).tolist())

    # depth
    t0 = time.perf_counter()
    with open(DEPTH_PATH, "rb") as f:
        for fi in depth_indices:
            offset, chunk_size = chunk_offsets[fi]
            f.seek(offset)
            chunk = f.read(chunk_size)
            try:
                dec = lz4.block.decompress(chunk, uncompressed_size=DEPTH_H * DEPTH_W * 2)
                depth = np.frombuffer(dec, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).astype(np.float32) / 1000.0
            except Exception:
                dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(dec, dtype=np.float32).reshape(DEPTH_H, DEPTH_W)
    depth_time = time.perf_counter() - t0

    # h5
    t0 = time.perf_counter()
    with h5py.File(H5_PATH, "r") as f:
        for key in ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]:
            if key in f:
                _ = f[key][h5_indices]
    h5_time = time.perf_counter() - t0

    print(f"  depth.bin (indexed): {depth_time:.3f}s")
    print(f"  precomputed.h5:      {h5_time:.3f}s")
    print(f"  total I/O:           {depth_time + h5_time:.3f}s")
    print(f"  bottleneck:          {'depth' if depth_time > h5_time else 'h5'}")


if __name__ == "__main__":
    main()
