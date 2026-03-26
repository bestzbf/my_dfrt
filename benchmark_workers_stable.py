#!/usr/bin/env python3
"""稳定的worker sweep测试：固定数据集，多次测量取平均"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import PointOdysseyDataset, collate_fn

# 创建dataset
dataset = PointOdysseyDataset(
    dataset_location="/data2/d4rt/datasets/PointOdyssey_fast",
    dset="train",
    S=48,
    img_size=256,
    num_queries=2048,
    use_augs=True,
    precompute_local_patches=True,
    local_patch_source="highres",
)

# 固定使用前40个场景（避免随机性）
subset = Subset(dataset, list(range(min(40, len(dataset)))))

batch_size = 8
prefetch_factor = 4
worker_configs = [8, 12, 16, 20, 24]
num_batches = 3  # 每个配置测3个batch

print("稳定Worker Sweep测试")
print("=" * 60)
print(f"固定数据集: 前{len(subset)}个场景")
print(f"batch_size={batch_size}, prefetch_factor={prefetch_factor}")
print(f"每个配置测{num_batches}个batch\n")

results = {}

for num_workers in worker_configs:
    print(f"测试 NUM_WORKERS={num_workers}...")

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    batch_times = []
    iterator = iter(loader)

    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        print(f"  batch {i+1}: {elapsed:.3f}s")

    avg_time = sum(batch_times[1:]) / len(batch_times[1:])  # 跳过第一个batch（冷启动）
    results[num_workers] = {
        'all': batch_times,
        'avg_excluding_first': avg_time,
        'first': batch_times[0],
    }
    print(f"  → 平均(跳过首batch): {avg_time:.3f}s\n")

    del loader

print("\n" + "=" * 60)
print("汇总结果（跳过首batch冷启动）:")
print("-" * 60)
for nw in worker_configs:
    avg = results[nw]['avg_excluding_first']
    first = results[nw]['first']
    print(f"NUM_WORKERS={nw:2d}: 首batch={first:6.3f}s, 稳定平均={avg:6.3f}s")

best_nw = min(results.keys(), key=lambda k: results[k]['avg_excluding_first'])
print(f"\n最优配置: NUM_WORKERS={best_nw}")
