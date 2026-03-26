#!/usr/bin/env python3
"""精确测量 dataset __getitem__ 各部分耗时"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import PointOdysseyDataset

# 创建 dataset
dataset = PointOdysseyDataset(
    dataset_location="/data2/d4rt/datasets/PointOdyssey_fast",
    dset="train",
    S=48,
    img_size=256,
    num_queries=2048,
    patch_size=9,
    use_augs=True,
    precompute_local_patches=True,
    local_patch_source="highres",
    use_motion_boundaries=True,
    return_aux_tensors=False,
)

# 插桩所有可能的耗时函数
methods_to_track = [
    '_load_rgb_from_encoded_cache',
    '_load_depth_from_encoded_cache',
    '_load_normal_or_mask_from_encoded_cache',
    '_apply_color_aug',
    '_sample_query_data',
    '_compute_boundary_mask',
    'extract_patches',
]

timings = {m: [] for m in methods_to_track}

for method_name in methods_to_track:
    original = getattr(dataset, method_name)
    def make_wrapper(orig, name):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = orig(*args, **kwargs)
            timings[name].append(time.perf_counter() - t0)
            return result
        return wrapper
    setattr(dataset, method_name, make_wrapper(original, method_name))

# 测试 3 个样本
print("测试 3 个样本的详细耗时分解...\n")
total_times = []

for idx in range(3):
    t_start = time.perf_counter()
    sample, ok = dataset[idx]
    t_total = time.perf_counter() - t_start
    total_times.append(t_total)

    print(f"样本 {idx}: 总耗时 {t_total:.3f}s")

print(f"\n平均总耗时: {np.mean(total_times):.3f}s")
print("\n各函数平均耗时:")
print("-" * 50)

accounted = 0
for method_name in methods_to_track:
    calls = timings[method_name]
    if calls:
        total = sum(calls)
        avg = total / len(calls)
        count = len(calls)
        print(f"{method_name:40s}: {total:.3f}s ({count:3d} 调用, 平均 {avg*1000:.1f}ms)")
        accounted += total

print("-" * 50)
print(f"已统计部分总计: {accounted:.3f}s")
print(f"未统计部分: {np.mean(total_times) - accounted:.3f}s ({(np.mean(total_times) - accounted) / np.mean(total_times) * 100:.1f}%)")
