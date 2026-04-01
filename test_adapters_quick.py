#!/usr/bin/env python3
"""Quick test: skip Co3Dv2 (31834 seqs), test all others + mixture."""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader

DATASETS = [
    ("pointodyssey",    "/data2/d4rt/datasets/PointOdyssey"),
    ("kubric",          "/data2/d4rt/datasets/kubric"),
    ("dynamic_replica", "/data1/d4rt/datasets/Dynamic_Replica"),
    ("scannet",         "/data2/d4rt/datasets/scannet/scannet"),
    ("blendedmvs",      "/data2/d4rt/datasets/BlendedMVS"),
    ("mvssynth",        "/data2/d4rt/datasets/MVS-Synth/GTAV_1080"),
    ("tartanair",       "/data2/d4rt/datasets/TartanAir"),
    ("vkitti2",         "/data2/d4rt/datasets/VirtualKitti"),
]
WEIGHTS = [0.30, 0.18, 0.14, 0.12, 0.10, 0.08, 0.05, 0.03]

def test_adapter(name, root):
    t0 = time.time()
    try:
        adapter = create_adapter(name, root=root, split='train')
        seqs = adapter.list_sequences()
        t1 = time.time()
        seq = seqs[0]
        info = adapter.get_sequence_info(seq)
        n = info['num_frames']
        frames = list(range(min(4, n)))
        clip = adapter.load_clip(seq, frames)
        t2 = time.time()
        print(f"  ✓ {name:20s}  seqs={len(seqs):5d}  init={t1-t0:.1f}s  clip={t2-t1:.2f}s  "
              f"img={clip.images[0].shape}  "
              f"tracks={'Y' if clip.trajs_2d is not None else 'N'}  "
              f"depth={'Y' if clip.depths is not None else 'N'}")
        return adapter
    except Exception as e:
        print(f"  ✗ {name:20s}  ERROR: {e}")
        return None

print("=" * 70)
print("Step 1: Adapter init + clip load")
print("=" * 70)
adapters, weights = [], []
for (name, root), w in zip(DATASETS, WEIGHTS):
    a = test_adapter(name, root)
    if a:
        adapters.append(a)
        weights.append(w)

print(f"\n{len(adapters)}/{len(DATASETS)} adapters OK\n")

print("=" * 70)
print("Step 2: MixtureDataset - 5 samples")
print("=" * 70)
try:
    ds = MixtureDataset(adapters=adapters, dataset_weights=weights,
                        clip_len=4, img_size=256, num_queries=256, seed=42)
    print(f"✓ Created  names={ds.get_dataset_names()}")
    for i in range(5):
        t0 = time.time()
        s = ds[i]
        dt = time.time() - t0
        m3 = s.targets['mask_3d'].sum().item()
        m2 = s.targets['mask_2d'].sum().item()
        print(f"  [{i}] {s.dataset_name:20s}  video={tuple(s.video.shape)}  "
              f"mask_3d={m3}/256  mask_2d={m2}/256  t={dt:.2f}s")
except Exception as e:
    print(f"✗ {e}"); traceback.print_exc(); sys.exit(1)

print()
print("=" * 70)
print("Step 3: DataLoader batch")
print("=" * 70)
try:
    loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=d4rt_collate_fn)
    t0 = time.time()
    batch = next(iter(loader))
    dt = time.time() - t0
    print(f"✓ batch  t={dt:.2f}s")
    print(f"  video:        {tuple(batch['video'].shape)}")
    print(f"  coords:       {tuple(batch['coords'].shape)}")
    print(f"  intrinsics:   {tuple(batch['intrinsics'].shape)}")
    print(f"  extrinsics:   {tuple(batch['extrinsics'].shape)}")
    print(f"  dataset_names:{batch['dataset_names']}")
    for k, v in batch['targets'].items():
        print(f"  targets[{k:25s}]: {tuple(v.shape)}")
except Exception as e:
    print(f"✗ {e}"); traceback.print_exc()

print("\nDone.")
