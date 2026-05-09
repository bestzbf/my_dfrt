"""Benchmark ScanNet++ on COS mount vs local SSD (sample_stage cache)."""
import os
import sys
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DEPTH_H, DEPTH_W = 192, 256
CLIP_LEN = 48

# COS mount path
COS_SCENE = "/data_cos/hdu_datasets/scannetpp/data/00777c41d4"
# Local staged path
LOCAL_SCENE = "/tmp/d4rt_sample_stage/work/sample_stage_b9_g0_i900_jccghsf2/hdu_datasets/scannetpp/data/9ed27ac522"


def find_local_scene():
    """Find a local staged scene with all needed files."""
    import glob
    base = "/tmp/d4rt_sample_stage/work"
    for d in glob.glob(f"{base}/sample_stage_*/hdu_datasets/scannetpp/data/*/"):
        depth = os.path.join(d, "iphone", "depth.bin")
        idx = os.path.join(d, "iphone", "depth_chunk_index.pkl")
        h5 = os.path.join(d, "precomputed.h5")
        if all(os.path.exists(f) for f in [depth, idx, h5]):
            return d.rstrip("/")
    return None


def bench_depth_indexed(scene_dir, n_frames, label, chunk_offsets=None):
    """Read N random frames using chunk index."""
    import lz4.block
    import zlib

    depth_path = os.path.join(scene_dir, "iphone", "depth.bin")
    index_path = os.path.join(scene_dir, "iphone", "depth_chunk_index.pkl")

    if chunk_offsets is None:
        with open(index_path, "rb") as f:
            chunk_offsets = pickle.load(f)
    total = len(chunk_offsets)

    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(min(n_frames, total), size=min(n_frames, total), replace=False).tolist())

    t0 = time.perf_counter()
    frames = []
    with open(depth_path, "rb") as f:
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
    print(f"  [{label}] {len(frames)} frames, {elapsed:.3f}s, "
          f"{elapsed/len(frames)*1000:.1f}ms/frame")
    return elapsed


def bench_h5_read(scene_dir, n_frames, label):
    """Read N random frames from h5."""
    import h5py

    h5_path = os.path.join(scene_dir, "precomputed.h5")
    if not os.path.exists(h5_path):
        print(f"  [{label}] SKIP - no h5 file")
        return 0

    with h5py.File(h5_path, "r") as f:
        total = f["trajs_2d"].shape[0]

    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(min(n_frames, total), size=min(n_frames, total), replace=False).tolist())

    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        keys = ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]
        result = {}
        for key in keys:
            if key in f:
                result[key] = f[key][indices]
    elapsed = time.perf_counter() - t0
    total_bytes = sum(v.nbytes for v in result.values())
    print(f"  [{label}] {n_frames} frames, {elapsed:.3f}s, "
          f"{elapsed/n_frames*1000:.1f}ms/frame, {total_bytes/1e6:.1f}MB")
    return elapsed


def main():
    print("=" * 70)
    print("ScanNet++ I/O Benchmark: COS mount vs Local SSD")
    print("=" * 70)

    # Find local scene
    local_scene = find_local_scene()
    if local_scene is None:
        print("ERROR: No local staged scene found!")
        sys.exit(1)
    print(f"COS scene:   {COS_SCENE}")
    print(f"Local scene: {local_scene}")

    # Check file sizes
    for label, scene in [("COS", COS_SCENE), ("Local", local_scene)]:
        depth = os.path.join(scene, "iphone", "depth.bin")
        h5 = os.path.join(scene, "precomputed.h5")
        if os.path.exists(depth):
            print(f"  [{label}] depth.bin: {os.path.getsize(depth)/1e6:.1f}MB")
        if os.path.exists(h5):
            print(f"  [{label}] precomputed.h5: {os.path.getsize(h5)/1e6:.1f}MB")

    # --- depth.bin benchmark ---
    print(f"\n{'='*50}")
    print("DEPTH.BIN — indexed random read")
    print(f"{'='*50}")

    for label, scene in [("COS mount", COS_SCENE), ("Local SSD", local_scene)]:
        print(f"\n{label}:")
        # load index
        idx_path = os.path.join(scene, "iphone", "depth_chunk_index.pkl")
        with open(idx_path, "rb") as f:
            offsets = pickle.load(f)
        print(f"  chunks: {len(offsets)}")

        # cold
        bench_depth_indexed(scene, CLIP_LEN, f"{label} cold", offsets)
        # warm
        bench_depth_indexed(scene, CLIP_LEN, f"{label} warm", offsets)

    # --- precomputed.h5 benchmark ---
    print(f"\n{'='*50}")
    print("PRECOMPUTED.H5 — frame slice read")
    print(f"{'='*50}")

    for label, scene in [("COS mount", COS_SCENE), ("Local SSD", local_scene)]:
        print(f"\n{label}:")
        bench_h5_read(scene, CLIP_LEN, f"{label} cold")
        bench_h5_read(scene, CLIP_LEN, f"{label} warm")

    # --- full clip simulation ---
    print(f"\n{'='*50}")
    print(f"FULL CLIP SIMULATION ({CLIP_LEN} frames)")
    print(f"{'='*50}")

    import h5py

    for label, scene in [("COS mount", COS_SCENE), ("Local SSD", local_scene)]:
        depth_path = os.path.join(scene, "iphone", "depth.bin")
        h5_path = os.path.join(scene, "precomputed.h5")
        idx_path = os.path.join(scene, "iphone", "depth_chunk_index.pkl")

        if not all(os.path.exists(f) for f in [depth_path, h5_path, idx_path]):
            print(f"  [{label}] SKIP - missing files")
            continue

        with open(idx_path, "rb") as f:
            offsets = pickle.load(f)
        total_depth = len(offsets)
        with h5py.File(h5_path, "r") as f:
            total_h5 = f["trajs_2d"].shape[0]

        import lz4.block, zlib
        rng = np.random.RandomState(42)
        d_idx = sorted(rng.choice(total_depth, size=CLIP_LEN, replace=False).tolist())
        h_idx = sorted(rng.choice(total_h5, size=CLIP_LEN, replace=False).tolist())

        t0 = time.perf_counter()
        with open(depth_path, "rb") as f:
            for fi in d_idx:
                off, sz = offsets[fi]
                f.seek(off)
                chunk = f.read(sz)
                try:
                    dec = lz4.block.decompress(chunk, uncompressed_size=DEPTH_H * DEPTH_W * 2)
                except Exception:
                    dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
        depth_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        with h5py.File(h5_path, "r") as f:
            for key in ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]:
                if key in f:
                    _ = f[key][h_idx]
        h5_t = time.perf_counter() - t0

        print(f"\n  {label}:")
        print(f"    depth.bin: {depth_t:.3f}s")
        print(f"    h5:        {h5_t:.3f}s")
        print(f"    total:     {depth_t + h5_t:.3f}s")


if __name__ == "__main__":
    main()
