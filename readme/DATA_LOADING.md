# D4RT 数据加载模块说明

## 概述

`data/dataset.py` 实现了 PointOdyssey 数据集的加载器，用于 D4RT (Dense 4D Reconstruction Transformer) 模型训练。该加载器支持多种优化策略，包括快速缓存、运动边界检测和灵活的查询采样。

## 核心类：PointOdysseyDataset

### 初始化参数

```python
PointOdysseyDataset(
    dataset_location="/path/to/pointodyssey",  # 数据集根目录
    dset="train",                               # 数据集分割: train/val/test
    use_augs=False,                             # 是否使用数据增强
    S=48,                                       # 时间序列长度（帧数）
    N=32,                                       # 每个序列的点数
    strides=None,                               # 帧采样步长列表
    clip_step=2,                                # 剪辑步长
    quick=False,                                # 快速模式（仅加载1个序列）
    verbose=False,                              # 详细输出
    img_size=256,                               # 输出图像尺寸
    num_queries=2048,                           # 查询点数量
    patch_size=9,                               # ��部patch大小
    boundary_ratio=0.3,                         # 边界采样比例
    t_tgt_eq_t_cam_ratio=0.4,                  # t_tgt==t_cam 的查询比例
    cache_boundaries=True,                      # 使用运动边界缓存
    sequence_name=None,                         # 指定序列名称
    query_mode="full",                          # 查询模式: full/target_cam/same_frame
    use_motion_boundaries=True,                 # 使用运动边界
    precompute_local_patches=True,              # 预计算局部patches
    return_query_video=False,                   # 返回查询视频
    local_patch_source="resized",               # patch来源: resized/highres
    return_aux_tensors=True,                    # 返回辅助张量
    static_scene_frame_idx=None,                # 静态场景帧索引
)
```

## 数据加载流程

### 1. 序列资产管理

数据集支持三种加载后端：

#### 1.1 快速注释缓存 (Fast Annotation Cache)
- 位置: `{sequence}/anno_fast/`
- 文件:
  - `trajs_2d.npy`: 2D轨迹 (T, N, 2)
  - `trajs_3d.npy`: 3D轨迹 (T, N, 3)
  - `valids.npy`: 有效性掩码 (T, N)
  - `visibs.npy`: 可见性掩码 (T, N)
  - `intrinsics.npy`: 相机内参 (T, 3, 3)
  - `extrinsics.npy`: 相机外参 (T, 4, 4)
  - `frame_manifest.json`: 帧文件清单

#### 1.2 编码帧缓存 (Encoded Frame Cache)
- 位置: `{sequence}/anno_fast/`
- 文件:
  - `rgb_frames.bin` + `rgb_frames_offsets.npy`: 压缩RGB帧
  - `depth_frames.bin` + `depth_frames_offsets.npy`: 压缩深度帧
  - `normal_frames.bin` + `normal_frames_offsets.npy`: 压缩法线帧
  - `normal_frames_valids.npy`: 法线有效性
  - `frame_pack_meta.json`: 元数据

优势：使用 JPEG/PNG 压缩，大幅减少磁盘占用和I/O时间

#### 1.3 原始帧缓存 (Raw Frame Cache)
- 位置: `{sequence}/anno_fast/`
- 文件:
  - `rgb_frames.npy`: 未压缩RGB (T, H, W, 3)
  - `depth_frames.npy`: 深度 (T, H, W)
  - `normal_frames.npy`: 法线 (T, H, W, 3)
  - `normal_valids.npy`: 法线有效性 (T,)
  - `frame_cache_meta.json`: 元数据

#### 1.4 原始文件加载
- 从 `rgbs/`, `depths/`, `normals/` 目录逐帧加载
- 支持 `.jpg`, `.png`, `.npy` 格式

### 2. 时间采样策略

```python
# 步长采样
stride = random.randint(1, min(4, total_frames // S))

# 起始帧采样
t_start = random.randint(0, total_frames - 1 - (S - 1) * stride)

# 帧索引
frame_indices = [t_start + i * stride for i in range(S)]
```

### 3. 空间裁剪与增强

#### 3.1 裁剪策略

- `use_augs=False`（验证/过拟合模式）：固定中心裁剪，`crop_size = min(H, W)`
- `use_augs=True`（训练模式）：随机裁剪
  - 目标面积：原图的 30%~100%
  - 宽高比：3/4 ~ 4/3（对数均匀分布）
  - 5% 概率额外缩放（zoom 0.7~0.95）

#### 3.2 颜色增强（仅 use_augs=True）

| 增强项 | 范围 |
|--------|------|
| 亮度 | 0.6 ~ 1.4 |
| 对比度 | 0.6 ~ 1.4 |
| 饱和度 | 0.6 ~ 1.4 |
| 色调偏移 | ±18° |
| 灰度化 | 20% 概率 |
| 高斯模糊 | 40% 概率，σ=0.1~2.0 |

### 4. 查询点采样

每个样本采样 `num_queries`（默认2048）个查询点，遵循 D4RT 论文的采样策略：

#### 4.1 空间采样（边界过采样）
- **70%** 查询：从所有有效点中随机采样
- **30%** 查询（`boundary_ratio=0.3`）：从边界点中采样
  - 深度边界：对深度图做 Sobel 梯度，取 top-15% 梯度点
  - 运动边界：对2D轨迹运动幅度做 Sobel 梯度，取 top-15% 梯度点
  - 最终边界 = 深度边界 ∪ 运动边界

#### 4.2 时间采样（query_mode="full"）
- **40%** 查询（`t_tgt_eq_t_cam_ratio=0.4`）：`t_tgt = t_cam`（目标帧=相机帧）
- **60%** 查询：`t_tgt` 和 `t_cam` 独立随机采样

`query_mode` 选项：
- `full`：完整时间采样（默认）
- `target_cam`：强制 `t_tgt = t_cam`
- `same_frame`：强制 `t_src = t_tgt = t_cam`

#### 4.3 有效点过滤条件
- `valids[t, i] > 0.5`：标注有效
- `visibs[t, i] > 0.5`：点可见
- 点在裁剪区域内
- 3D 世界坐标有限（非 NaN/Inf）

### 5. 运动边界缓存

预计算的运动边界以 bit-packed 格式存储：

```
anno_fast/
  motion_boundary_stride_01_packed.npy   # bit-packed 边界掩码
  motion_boundary_stride_01_meta.json    # 元数据（height, width, bitorder）
  motion_boundary_stride_02_packed.npy
  motion_boundary_stride_02_meta.json
  ...
```

加载时按 stride 选择对应缓存，并裁剪到当前 crop 区域。

---

## 数据集目录结构

```
pointodyssey/
├── train/
│   ├── seq_000001/
│   │   ├── rgbs/           # RGB帧: 0001.jpg, 0002.jpg, ...
│   │   ├── depths/         # 深度帧: 0001.png (uint16, /65535*1000=meters)
│   │   ├── normals/        # 法线帧: 0001.jpg (RGB编码, *2-1=[-1,1])
│   │   ├── anno.npz        # 原始标注（或 anno_fast/ 目录）
│   │   └── anno_fast/
│   │       ├── trajs_2d.npy
│   │       ├── trajs_3d.npy
│   │       ├── valids.npy
│   │       ├── visibs.npy
│   │       ├── intrinsics.npy
│   │       ├── extrinsics.npy
│   │       ├── frame_manifest.json
│   │       ├── frame_pack_meta.json        # 编码帧缓存元数据
│   │       ├── rgb_frames.bin              # 编码RGB
│   │       ├── rgb_frames_offsets.npy
│   │       ├── depth_frames.bin
│   │       ├── depth_frames_offsets.npy
│   │       ├── normal_frames.bin
│   │       ├── normal_frames_offsets.npy
│   │       ├── normal_frames_valids.npy
│   │       ├── motion_boundary_stride_01_packed.npy
│   │       └── motion_boundary_stride_01_meta.json
│   └── seq_000002/
│       └── ...
└── val/
    └── ...
```

---

## __getitem__ 返回值

每个样本返回 `(sample_dict, success_bool)`，`sample_dict` 包含：

### 核心字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `video` | (S, 3, H, W) | 归一化RGB视频，float32 [0,1] |
| `coords` | (N_q, 2) | 查询点归一化坐标 (u,v) ∈ [0,1] |
| `t_src` | (N_q,) | 查询点所在源帧索引 |
| `t_tgt` | (N_q,) | 目标帧索引 |
| `t_cam` | (N_q,) | 相机参考帧索引 |
| `aspect_ratio` | (1,) | 裁剪区域宽高比 |
| `targets` | dict | 监督信号（见下） |
| `transform_metadata` | dict | 裁剪/缩放变换元数据 |

### targets 字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `pos_2d` | (N_q, 2) | 目标帧2D位置，归一化 [0,1] |
| `pos_3d` | (N_q, 3) | 目标点在相机坐标系下的3D位置 |
| `displacement` | (N_q, 3) | 3D位移向量 (tgt_cam - src_cam) |
| `visibility` | (N_q,) | 目标点可见性 (0/1) |
| `normal` | (N_q, 3) | 目标点法线向量 |
| `mask_2d` | (N_q,) | 2D监督有效掩码 |
| `mask_3d` | (N_q,) | 3D监督有效掩码 |
| `mask_disp` | (N_q,) | 位移监督有效掩码 |
| `mask_vis` | (N_q,) | 可见性监督有效掩码 |
| `mask_normal` | (N_q,) | 法线监督有效掩码 |
| `source_is_boundary` | (N_q,) | 查询点是否在边界上 |
| `source_is_depth_boundary` | (N_q,) | 是否在深度边界 |
| `source_is_motion_boundary` | (N_q,) | 是否在运动边界 |
| `point_indices` | (N_q,) | 原始点索引 |

### 辅助字段（return_aux_tensors=True）

| 字段 | 形状 | 说明 |
|------|------|------|
| `frame_indices` | (S,) | 实际帧索引 |
| `extrinsics` | (S, 4, 4) | 相机外参矩阵 |
| `depths` | (S, 1, H, W) | 深度图 |
| `normals` | (S, 3, H, W) | 法线图 |
| `intrinsics` | (S, 3, 3) | 缩放后相机内参 |
| `intrinsics_crop` | (S, 3, 3) | 裁剪后相机内参 |
| `intrinsics_original` | (S, 3, 3) | 原始相机内参 |

### 可选字段

| 字段 | 条件 | 说明 |
|------|------|------|
| `local_patches` | precompute_local_patches=True | (N_q, 3, P, P) 查询点局部patch |
| `video_query` | return_query_video=True | (S, 3, H_orig, W_orig) 原始分辨率视频 |

---

## 坐标系约定

- **2D坐标**：归一化到 [0, 1]，`u = x / (W-1)`，`v = y / (H-1)`
- **3D坐标**：相机坐标系，单位为数据集原始单位（米）
- **深度编码**：uint16 PNG → `depth_m = value / 65535.0 * 1000.0`
- **法线编码**：uint8 RGB → `normal = pixel/255.0 * 2.0 - 1.0`
- **外参矩阵**：世界坐标 → 相机坐标，`p_cam = E @ [p_world; 1]`

---

## 缓存构建脚本

```bash
# 1. 构建快速标注缓存（必须先做）
python build_pointodyssey_fast_cache.py

# 2. 构建帧缓存（原始 npy 格式）
python build_pointodyssey_frame_cache.py

# 3. 构建编码帧缓存（推荐，节省磁盘）
python build_pointodyssey_packed_frame_cache.py

# 4. 构建运动边界缓存（加速训练）
python build_pointodyssey_motion_boundary_cache.py
```

---

## 数据集重复因子（dataset_repeat_factor）

训练脚本 `train.py` 支持通过 `dataset_repeat_factor` 对数据集进行整倍数扩展，底层使用 `torch.utils.data.ConcatDataset` 实现：

```python
dataset_repeat_factor = getattr(args, 'dataset_repeat_factor', 32)  # 默认 32
if dataset_repeat_factor > 1:
    train_dataset = ConcatDataset([train_dataset] * dataset_repeat_factor)
```

### 作用

PointOdyssey 数据集中每个序列的**可采样片段数**通常远大于序列本身的数量（每次 `__getitem__` 都会随机采样不同的时间起点和裁剪区域），因此单个 epoch 扫描一遍索引列表会严重低估数据集的真实容量。通过将数据集重复 N 倍，可以：

- **增大每个 epoch 的迭代步数**，让每条序列在一个 epoch 内被多次以不同随机参数���样；
- **匹配更大的 batch size** 或更长的 warmup/训练计划，而无需调整 `--epochs`；
- **统一 `steps_per_epoch`**，使 LR 调度曲线（warmup + cosine decay）的步数单位与实际训练量对应。

### 注意事项

- 重复因子仅影响 DataLoader 的索引范围，底层的 `PointOdysseyDataset` 对象只有一份，不会额外占用内存；
- 由于每次 `__getitem__` 都重新随机采样，重复索引并**不会**返回相同的样本；
- 如需自定义，可通过 YAML 配置文件添加 `dataset_repeat_factor: <N>`，或在 `train.py` 中直接修改默认值（当前默认 `32`）。

---

## 使用示例

```python
from data import PointOdysseyDataset, collate_fn
from torch.utils.data import DataLoader

dataset = PointOdysseyDataset(
    dataset_location="/path/to/pointodyssey",
    dset="train",
    use_augs=True,
    S=48,
    num_queries=2048,
    img_size=256,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_fn,
    num_workers=8,
)

for batch in loader:
    video = batch["video"]          # (B, S, 3, 256, 256)
    coords = batch["coords"]        # (B, N_q, 2)
    targets = batch["targets"]
    pos_3d = targets["pos_3d"]      # (B, N_q, 3)
    mask_3d = targets["mask_3d"]    # (B, N_q)
```

---

## 性能说明

加载速度优先级（从快到慢）：

1. **编码帧缓存**（`frame_pack_meta.json` 存在）：memmap + JPEG解码，最快
2. **原始帧缓存**（`frame_cache_meta.json` 存在）：memmap 直接读取
3. **原始文件**（`rgbs/`, `depths/`, `normals/`）：逐帧磁盘读取，最慢

建议训练时使用编码帧缓存 + 运动边界缓存以获得最佳 I/O 性能。

