#!/usr/bin/env python3
"""Test each adapter individually, then test mixture (no Co3Dv2 for speed)."""

import sys
import time
import yaml
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader

DATASETS = [
    ("pointodyssey",  "/data2/d4rt/datasets/PointOdyssey"),
    ("kubric",        "/data2/d4rt/datasets/kubric"),
    ("dynamic_replica", "/data1/d4rt/datasets/Dynamic_Replica"),
    ("scannet",       "/data2/d4rt/datasets/scannet/scannet"),
    ("co3dv2",        "/data2/d4rt/datasets/Co3Dv2"),
    ("blendedmvs",    "/data2/d4rt/datasets/BlendedMVS"),
    ("mvssynth",      "/data2/d4rt/datasets/MVS-Synth/GTAV_1080"),
    ("tartanair",     "/data2/d4rt/datasets/TartanAir"),
    ("vkitti2",       "/data2/d4rt/datasets/VirtualKitti"),
]

WEIGHTS = [0.27, 0.16, 0.12, 0.11, 0.11, 0.08, 0.07, 0.06, 0.05]

def test_adapter(name, root):
    """Test a single adapter: init + load one clip."""
    t0 = time.time()
    try:
        adapter = create_adapter(name, root=root, split='train')
        seqs = adapter.list_sequences()
        t1 = time.time()
        # Load one clip from first sequence
        seq = seqs[0]
        info = adapter.get_sequence_info(seq)
        n = info['num_frames']
        frames = list(range(min(4, n)))  # 4 frames is enough for a quick check
        clip = adapter.load_clip(seq, frames)
        t2 = time.time()
        print(f"  ✓ {name:20s}  seqs={len(seqs):5d}  "
              f"init={t1-t0:.1f}s  clip={t2-t1:.1f}s  "
              f"img={clip.images[0].shape}  "
              f"has_tracks={'Yes' if clip.trajs_2d is not None else 'No':3s}  "
              f"has_depth={'Yes' if clip.depths is not None else 'No':3s}")
        return adapter
    except Exception as e:
        print(f"  ✗ {name:20s}  ERROR: {e}")
        traceback.print_exc()
        return None

print("=" * 70)
print("Step 1: Test each adapter individually")
print("=" * 70)
adapters = []
for name, root in DATASETS:
    adapter = test_adapter(name, root)
    if adapter is not None:
        adapters.append(adapter)

print()
print(f"Adapters loaded: {len(adapters)} / {len(DATASETS)}")
print()

if len(adapters) == 0:
    print("No adapters loaded, cannot test mixture.")
    sys.exit(1)

print("=" * 70)
print("Step 2: Test MixtureDataset + load 3 samples")
print("=" * 70)
weights = WEIGHTS[:len(adapters)]
try:
    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=4,      # short clip for speed
        img_size=256,
        num_queries=256, # fewer queries for speed
        use_augs=True,
        seed=42,
    )
    print(f"✓ MixtureDataset created  names={dataset.get_dataset_names()}")
    for i in range(3):
        t0 = time.time()
        sample = dataset[i]
        dt = time.time() - t0
        print(f"  sample[{i}]: dataset={sample.dataset_name:20s}  seq={sample.sequence_name[:20]:20s}  "
              f"video={tuple(sample.video.shape)}  coords={tuple(sample.coords.shape)}  "
              f"mask_3d={sample.targets['mask_3d'].sum().item():4d}/{sample.targets['mask_3d'].shape[0]}  "
              f"mask_2d={sample.targets['mask_2d'].sum().item():4d}/{sample.targets['mask_2d'].shape[0]}  "
              f"t={dt:.2f}s")
except Exception as e:
    print(f"✗ MixtureDataset failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("Step 3: Test DataLoader (batch_size=2, num_workers=2)")
print("=" * 70)
try:
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=d4rt_collate_fn)
    t0 = time.time()
    batch = next(iter(loader))
    dt = time.time() - t0
    print(f"✓ DataLoader batch  t={dt:.2f}s")
    print(f"  video:       {tuple(batch['video'].shape)}")
    print(f"  coords:      {tuple(batch['coords'].shape)}")
    print(f"  intrinsics:  {tuple(batch['intrinsics'].shape)}")
    print(f"  extrinsics:  {tuple(batch['extrinsics'].shape)}")
    print(f"  dataset_names: {batch['dataset_names']}")
    t3 = batch['targets']
    for k, v in t3.items():
        print(f"  targets[{k}]: {tuple(v.shape)}")
except Exception as e:
    print(f"✗ DataLoader failed: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("All tests done.")
