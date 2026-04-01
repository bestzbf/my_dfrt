"""
Precompute normals + tracks for Virtual KITTI 2 sequences.

VirtualKitti2 has RGB + depth + optical-flow but NO tracks/normals/visibility.
This script derives normals (from depth+intrinsics) and pseudo-static tracks
(from depth+pose, Strategy A) and caches them as precomputed.npz.

The output directory structure mirrors the adapter's sequence names so that
VKITTI2Adapter can load the cache with:
    <output_root> / <seq_name> / precomputed.npz
  e.g.  <output_root> / Scene01_clone / precomputed.npz

Usage:
    python run_vkitti2.py \\
        --root /data2/d4rt/datasets/VirtualKitti \\
        [--output-root /data2/d4rt/datasets/VirtualKitti] \\
        [--camera Camera_0] \\
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
from adapters.VirtualKitti import VKITTI2Adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute VirtualKitti2 normals + pseudo-static tracks from depth+pose"
    )
    parser.add_argument("--root", required=True,
                        help="VirtualKitti2 root directory (contains Scene01/, Scene02/, …)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--camera", default="Camera_0", choices=["Camera_0", "Camera_1"],
                        help="Which camera to use (default: Camera_0)")
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Number of track points to sample per sequence (default: 8000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if precomputed.npz already exists")
    args = parser.parse_args()

    adapter = VKITTI2Adapter(root=args.root, camera=args.camera, verbose=False)
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
