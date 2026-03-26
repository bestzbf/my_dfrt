#!/usr/bin/env python3
"""精确测量 __getitem__ 各阶段耗时（避免重复计数）"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import PointOdysseyDataset

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

# 手动插桩 __getitem__ 关键阶段
original_getitem = dataset.__getitem__

def instrumented_getitem(index):
    timings = {}

    seq_name = dataset.dirs[index]
    seq_path = dataset.root + "/" + seq_name
    py_rng, np_rng = dataset._get_rngs(index)

    t0 = time.perf_counter()
    assets = dataset._get_sequence_assets(seq_name, seq_path)
    timings['get_assets'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trajs_2d, trajs_3d, intrinsics, valids, visibilities, extrinsics = dataset._load_annotations(assets, need_intrinsics=False)
    timings['load_annotations'] = time.perf_counter() - t0

    total_frames = trajs_2d.shape[0]
    stride = dataset._sample_stride(total_frames, py_rng)
    t_start = py_rng.randint(0, max(0, total_frames - 1 - (dataset.S - 1) * stride))
    frame_indices = [t_start + i * stride for i in range(dataset.S)]

    # 加载帧（JPEG解码）
    t0 = time.perf_counter()
    encoded_cache = dataset._load_encoded_frame_cache_entry(assets)
    rgb_files = assets["rgb_files"]
    depth_files = assets["depth_files"]
    normal_files = assets["normal_files"]

    rgb_positions = [dataset._resolve_frame_position(t, rgb_files, assets["rgb_index_map"]) for t in frame_indices]
    depth_positions = [dataset._resolve_frame_position(t, depth_files, assets["depth_index_map"]) for t in frame_indices]
    normal_positions = [dataset._resolve_frame_position(t, normal_files, assets["normal_index_map"]) for t in frame_indices]

    rgb_frames = [dataset._load_rgb_from_encoded_cache(encoded_cache, pos) for pos in rgb_positions]
    depth_frames = [dataset._load_depth_from_encoded_cache(encoded_cache, pos, ".png") for pos in depth_positions]
    normal_frames = []
    for idx, pos in enumerate(normal_positions):
        nf, _ = dataset._load_normal_or_mask_from_encoded_cache(encoded_cache, pos, ".png", rgb_frames[idx].shape[:2])
        normal_frames.append(nf)
    timings['decode_frames'] = time.perf_counter() - t0

    # Crop
    original_h, original_w = rgb_frames[0].shape[:2]
    x0, y0, crop_w, crop_h = dataset._sample_crop(original_h, original_w, py_rng)
    rgb_frames = [f[y0:y0+crop_h, x0:x0+crop_w] for f in rgb_frames]
    depth_frames = [d[y0:y0+crop_h, x0:x0+crop_w] for d in depth_frames]
    normal_frames = [n[y0:y0+crop_h, x0:x0+crop_w] for n in normal_frames]

    # 颜色增强
    t0 = time.perf_counter()
    rgb_frames = dataset._apply_color_aug(rgb_frames, py_rng)
    timings['color_aug'] = time.perf_counter() - t0

    # Resize (cv2.resize循环)
    import cv2
    import torch
    t0 = time.perf_counter()
    resized_video = torch.zeros((dataset.S, 3, dataset.img_size, dataset.img_size), dtype=torch.float32)
    for frame_idx in range(dataset.S):
        rgb_resized = cv2.resize(rgb_frames[frame_idx], (dataset.img_size, dataset.img_size))
        resized_video[frame_idx] = torch.from_numpy(rgb_resized).permute(2, 0, 1)
    timings['resize_frames'] = time.perf_counter() - t0

    return timings

# 测试3个样本
print("详细耗时分解（避免重复计数）:\n")
all_timings = []
for idx in range(3):
    t_total_start = time.perf_counter()
    timings = instrumented_getitem(idx)
    t_total = time.perf_counter() - t_total_start
    all_timings.append(timings)
    print(f"样本 {idx}: 总耗时 {t_total:.3f}s")
    for k, v in timings.items():
        print(f"  {k:20s}: {v:.3f}s ({v/t_total*100:.1f}%)")
    print()

# 平均值
print("=" * 50)
print("3个样本平均:")
for key in all_timings[0].keys():
    avg = sum(t[key] for t in all_timings) / 3
    print(f"  {key:20s}: {avg:.3f}s")
