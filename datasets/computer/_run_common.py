"""
Shared precompute logic used by all run_*.py scripts.

For each sequence in an adapter, loads all frames, computes normals
and tracks, and writes a compressed .npz cache file.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from computer.depth_to_normals import compute_normals_sequence
from computer.depth_to_tracks import compute_tracks
from adapters.base import BaseAdapter


# --------------------------------------------------------------------------- #
# Per-sequence worker (runs in subprocess)
# --------------------------------------------------------------------------- #

def _process_sequence(args: tuple) -> tuple[str, str | None]:
    """
    Process one sequence.  Returns (seq_name, error_message_or_None).
    Must be a top-level function for multiprocessing pickle.
    """
    seq_name, adapter_state, output_root, num_points, overwrite = args

    out_path = Path(output_root) / seq_name / "precomputed.npz"
    if out_path.exists() and not overwrite:
        return seq_name, None   # already done

    try:
        # Reconstruct adapter in subprocess
        adapter_cls, adapter_kwargs = adapter_state
        adapter = adapter_cls(**adapter_kwargs)

        info = adapter.get_sequence_info(seq_name)
        num_frames = info["num_frames"]

        # Load all frames for this sequence
        all_indices = list(range(num_frames))
        clip = adapter.load_clip(seq_name, all_indices)

        depths     = clip.depths       # list[np.ndarray] [H,W]
        intrinsics = clip.intrinsics   # [T,3,3]
        extrinsics = clip.extrinsics   # [T,4,4] w2c

        if depths is None:
            return seq_name, "no depth"

        H, W = depths[0].shape

        # ---- normals ----
        normals = compute_normals_sequence(depths, intrinsics)   # [T,H,W,3]

        # ---- tracks ----
        tracks = compute_tracks(
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_points=num_points,
        )

        # ---- save ----
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            normals         = normals.astype(np.float16),   # [T,H,W,3] half to save space
            trajs_2d        = tracks["trajs_2d"].astype(np.float32),
            trajs_3d_world  = tracks["trajs_3d_world"].astype(np.float32),
            valids          = tracks["valids"],
            visibs          = tracks["visibs"],
            ref_frame       = np.array(tracks["ref_frame"],  dtype=np.int32),
            num_frames      = np.array(num_frames,           dtype=np.int32),
            num_points      = np.array(tracks["num_points"], dtype=np.int32),
        )
        return seq_name, None

    except Exception:
        return seq_name, traceback.format_exc()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_precompute(
    adapter: BaseAdapter,
    output_root: Path,
    num_points: int = 8000,
    workers: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Run precomputation for all sequences in an adapter.

    Args:
        adapter:     Any BaseAdapter instance.
        output_root: Directory where precomputed.npz files will be written.
                     Structure: output_root / <sequence_name> / precomputed.npz
        num_points:  Number of track points per sequence.
        workers:     Number of parallel worker processes.
        overwrite:   Re-compute even if output already exists.
    """
    sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[precompute] {adapter.dataset_name}: {total} sequences → {output_root}")

    # Build a picklable description of the adapter so workers can reconstruct it
    adapter_state = _make_adapter_state(adapter)

    job_args = [
        (seq, adapter_state, str(output_root), num_points, overwrite)
        for seq in sequences
    ]

    done = 0
    failed = []

    if workers <= 1:
        # Single-process mode (easier to debug)
        for args in tqdm(job_args, desc=adapter.dataset_name):
            seq_name, err = _process_sequence(args)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_sequence, a): a[0] for a in job_args}
            with tqdm(total=total, desc=adapter.dataset_name) as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")

    print(f"[precompute] done {done - len(failed)}/{total}, failed {len(failed)}")
    if failed:
        print("[precompute] Failed sequences:")
        for seq_name, err in failed:
            print(f"  {seq_name}:\n    {err.splitlines()[-1]}")


# --------------------------------------------------------------------------- #
# Adapter state serialisation (for subprocess pickling)
# --------------------------------------------------------------------------- #

def _make_adapter_state(adapter: BaseAdapter) -> tuple:
    """
    Return (adapter_class, kwargs_dict) so a subprocess can re-instantiate.
    Covers the four supported adapters.
    """
    cls = type(adapter)
    name = cls.__name__

    if name == "ScanNetAdapter":
        return cls, {
            "root": str(adapter.root),
            "depth_scale": adapter.depth_scale,
            "default_pose_convention": adapter.default_pose_convention,
        }
    elif name == "Co3Dv2Adapter":
        return cls, {
            "root": str(adapter.root),
            "categories": adapter.categories if hasattr(adapter, "categories") else None,
            "subset_name": adapter.subset_name if hasattr(adapter, "subset_name") else "fewview_train",
            "split": adapter.split if hasattr(adapter, "split") else "train",
        }
    elif name == "BlendedMVSAdapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "use_masked": adapter.use_masked if hasattr(adapter, "use_masked") else False,
            "verbose": False,
        }
    elif name == "MVSSynthAdapter":
        return cls, {
            "root": str(adapter.root),
            "verbose": False,
        }
    elif name == "TartanAirAdapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "camera": adapter.camera if hasattr(adapter, "camera") else "left",
            "verbose": False,
        }
    elif name == "VKITTI2Adapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "camera": adapter.camera if hasattr(adapter, "camera") else "Camera_0",
            "verbose": False,
        }
    elif name == "WaymoAdapter":
        # Waymo is TFRecord-based; disable on-the-fly RAFT flow for precompute
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "training",
            "extract_flow": False,
            "verbose": False,
        }
    else:
        raise ValueError(f"Unsupported adapter type for multiprocessing: {name}")
