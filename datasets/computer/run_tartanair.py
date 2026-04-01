"""
Precompute normals + tracks for TartanAir sequences.

TartanAir has RGB + depth + optical-flow but NO tracks/normals/visibility.
This script derives normals (from depth+intrinsics) and pseudo-static tracks
(from depth+pose, Strategy A) and caches them as precomputed.npz.

The output directory structure mirrors the adapter's sequence names so that
TartanAirAdapter can load the cache with:
    <output_root> / <seq_name> / precomputed.npz

Usage:
    python run_tartanair.py \\
        --root /data2/d4rt/datasets/TartanAir \\
        [--output-root /data2/d4rt/datasets/TartanAir] \\
        [--camera left] \\
        [--num-points 8000] \\
        [--workers 4] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # /data2/d4rt/code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # /data2/d4rt/code/datasets

from computer._run_common import run_precompute
from adapters.TartanAir import TartanAirAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute TartanAir normals + pseudo-static tracks from depth+pose"
    )
    parser.add_argument("--root", required=True,
                        help="TartanAir root directory (contains P001/, P003/, …)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--camera", default="left", choices=["left", "right"],
                        help="Which camera to use (default: left)")
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Number of track points to sample per sequence (default: 8000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if precomputed.npz already exists")
    args = parser.parse_args()

    adapter = TartanAirAdapter(root=args.root, camera=args.camera, verbose=False)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    run_precompute(
        adapter=adapter,
        output_root=output_root,
        num_points=args.num_points,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
