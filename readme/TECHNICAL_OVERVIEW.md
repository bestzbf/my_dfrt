# D4RT 技术实现全览

> Dynamic 4D Reconstruction and Tracking
> 本文档覆盖整个代码库的所有技术细节，包括模型架构、数据流、损失函数、训练流程、数据管线和推理算法。

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [核心概念：统一查询接口](#3-核心概念统一查询接口)
4. [模型架构](#4-模型架构)
   - 4.1 [顶层模型 D4RT](#41-顶层模型-d4rt)
   - 4.2 [编码器 D4RTEncoder](#42-编码器-d4rtencoder)
   - 4.3 [解码器 D4RTDecoder](#43-解码器-d4rtdecoder)
   - 4.4 [嵌入模块](#44-嵌入模块)
5. [数据管线](#5-数据管线)
   - 5.1 [PointOdyssey 数据集结构](#51-pointodyssey-数据集结构)
   - 5.2 [四种帧缓存后端](#52-四种帧缓存后端)
   - 5.3 [运动边界缓存](#53-运动边界缓存)
   - 5.4 [Clip 采样流程](#54-clip-采样流程)
   - 5.5 [数据增强](#55-数据增强)
   - 5.6 [Query 采样策略](#56-query-采样策略)
   - 5.7 [Dataset 输出格式](#57-dataset-输出格式)
   - 5.8 [collate_fn](#58-collate_fn)
6. [损失函数](#6-损失函数)
   - 6.1 [Scale-Invariant 3D Loss（主损失）](#61-scale-invariant-3d-loss主损失)
   - 6.2 [2D Reprojection Loss](#62-2d-reprojection-loss)
   - 6.3 [Visibility Loss](#63-visibility-loss)
   - 6.4 [Displacement Loss](#64-displacement-loss)
   - 6.5 [Normal Loss](#65-normal-loss)
   - 6.6 [Confidence Loss（Kendall 公式）](#66-confidence-losskendall-公式)
   - 6.7 [总损失与权重](#67-总损失与权重)
   - 6.8 [Confidence 调度](#68-confidence-调度)
7. [训练流程](#7-训练流程)
   - 7.1 [分布式设置](#71-分布式设置)
   - 7.2 [优化器与 LR 调度](#72-优化器与-lr-调度)
   - 7.3 [梯度累积与 Query Chunking](#73-梯度累积与-query-chunking)
   - 7.4 [大 Batch 支持（ConcatDataset）](#74-大-batch-支持concatdataset)
   - 7.5 [AMP 混合精度](#75-amp-混合精度)
   - 7.6 [Checkpoint 管理](#76-checkpoint-管理)
   - 7.7 [训练命令示例](#77-训练命令示例)
8. [稠密跟踪算法（Algorithm 1）](#8-稠密跟踪算法algorithm-1)
9. [工具模块](#9-工具模块)
10. [配置文件说明](#10-配置文件说明)
11. [关键超参数速查表](#11-关键超参数速查表)
12. [数据预处理流程（prepare 脚本）](#12-数据预处理流程prepare-脚本)

---

## 1. 项目概述

D4RT（Dynamic 4D Reconstruction and Tracking）是一个前馈式 Transformer 模型，将视频编码为**全局场景表示（Global Scene Representation，GSR）**，然后通过轻量解码器将任意时空查询 `(u, v, t_src, t_tgt, t_cam)` 解码为：

| 输出 | 维度 | 含义 |
|------|------|------|
| `pos_3d` | `(B, N_q, 3)` | 相机坐标系下的 3D 位置（米） |
| `pos_2d` | `(B, N_q, 2)` | 归一化 2D 投影坐标 `[0,1]` |
| `visibility` | `(B, N_q, 1)` | 可见性 logit（sigmoid→概率） |
| `displacement` | `(B, N_q, 3)` | 3D 运动位移向量 |
| `normal` | `(B, N_q, 3)` | L2 归一化表面法向量 |
| `confidence` | `(B, N_q, 1)` | Kendall 不确定性参数 `s`（`s = -log c`） |

统一查询接口支持多任务推理，仅通过改变时间戳组合即可切换任务：

| 任务 | `t_src` | `t_tgt` | `t_cam` |
|------|---------|---------|---------|
| 点轨迹跟踪 | 固定 | 变化 | = `t_tgt` |
| 单帧深度图 | = `t` | = `t` | = `t` |
| 点云（参考系） | 变化 | 变化 | 固定（参考帧） |
| 相机外参 | 固定 | 固定 | 变化 |

---

## 2. 目录结构

```
/data1/zbf/my_dfrt/
├── configs/
│   ├── d4rt_base.yaml                          # 基础训练配置
│   └── d4rt_pointodyssey_curriculum_base.yaml  # PointOdyssey 课程训练配置
├── data/
│   ├── __init__.py
│   └── dataset.py                              # PointOdysseyDataset + collate_fn
├── losses/
│   ├── __init__.py
│   └── losses.py                               # D4RTLoss 及所有子损失
├── models/
│   ├── __init__.py
│   ├── d4rt.py                                 # 顶层 D4RT 模型
│   ├── encoder.py                              # D4RTEncoder（LocalBlock + GlobalBlock）
│   ├── decoder.py                              # D4RTDecoder（CrossAttention）
│   ├── embeddings.py                           # Fourier/Timestep/Patch 嵌入
│   └── dense_tracking.py                       # Algorithm 1 稠密跟踪
├── utils/
│   ├── camera.py                               # 相机投影工具
│   ├── geometry.py                             # 几何变换
│   ├── losses.py                               # 共享损失工具
│   ├── metrics.py                              # 深度评估指标
│   ├── misc.py                                 # 杂项工具
│   ├── patches.py                              # 5D grid_sample patch 提取
│   └── visualization.py                        # 可视化
├── scripts/
│   ├── prepare_pointodyssey_local_dataset.py   # 一键数据预处理
│   ├── build_pointodyssey_fast_cache.py        # 构建 anno_fast 注释缓存
│   ├── build_pointodyssey_packed_frame_cache.py# 构建 packed 帧缓存（.bin）
│   ├── build_pointodyssey_motion_boundary_cache.py # 构建运动边界缓存
│   ├── check_pointodyssey_sanity.py            # 数据集几何完整性检查
│   └── profile_pointodyssey_pipeline.py        # 数据加载性能测试
├── readme/
│   ├── DATA_LOADING.md
│   ├── MODEL_ARCHITECTURE.md
│   └── TECHNICAL_OVERVIEW.md                   # ← 本文档
├── train.py                                    # 主训练脚本（1652 行）
├── visualize_tracks.py                         # 轨迹可视化
└── environment.yml                             # Conda 环境
```

---

## 3. 核心概念：统一查询接口

D4RT 的核心设计是**两阶段前馈**：

```
video (B, T, H, W, C)
    │
    ▼  encode()  ─────────────────── 一次编码，O(1) 开销
encoder_features (B, N_enc, D)      ← Global Scene Representation F
    │
    ▼  decode()  ─────────────────── 可对任意数量/类型的查询解码
predictions (B, N_q, *)
```

**查询五元组** `(u, v, t_src, t_tgt, t_cam)`：

| 变量 | 类型 | 含义 |
|------|------|------|
| `u, v` | `float [0,1]` | 查询点在源帧中的归一化像素坐标 |
| `t_src` | `int [0, T)` | 查询点所在的源帧索引 |
| `t_tgt` | `int [0, T)` | 目标帧索引（预测该点在此帧的位置） |
| `t_cam` | `int [0, T)` | 相机参考帧索引（3D 坐标表达于此帧相机系下） |

---

## 4. 模型架构

### 4.1 顶层模型 D4RT

**文件**：`models/d4rt.py`

```python
class D4RT(nn.Module):
    def __init__(
        self,
        encoder_variant: str = 'base',   # 'base'|'large'|'huge'|'giant'
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: tuple = (2, 16, 16), # 3D patch kernel (T, H, W)
        decoder_depth: int = 8,
        decoder_num_heads: int = 12,
        max_timesteps: int = 128,        # TimestepEmbedding 查询表大小
        query_patch_size: int = 9,       # 查询局部 RGB patch 尺寸
        videomae_model: str = None,      # HF 模型 ID 或本地路径
        patch_provider: str = 'auto',    # patch 提供方式
        ...
    )
```

**各 variant 解码器默认配置**（`create_d4rt()`）：

| Variant | `decoder_depth` | `decoder_num_heads` |
|---------|----------------|---------------------|
| base    | 6              | 12                  |
| large   | 6              | 16                  |
| huge    | 8              | 16                  |
| giant   | 8              | 16                  |

**主要方法**：

| 方法 | 说明 |
|------|------|
| `encode(video, aspect_ratio)` | 视频 → GSR，返回 `(B, N, D)` |
| `decode(encoder_features, frames, coords, t_src, t_tgt, t_cam, ...)` | GSR + 查询 → 预测字典 |
| `forward(video, coords, ...)` | 完整前向，`encode` + `decode` |
| `predict_depth(video, ...)` | 对所有帧生成深度图 |
| `predict_point_tracks(video, query_points, query_frames, ...)` | 轨迹预测 |
| `predict_point_cloud(video, reference_frame, ...)` | 统一参考系点云 |

---

### 4.2 编码器 D4RTEncoder

**文件**：`models/encoder.py`

#### 各 variant 配置

| Variant | `embed_dim` | `depth`（stages） | `num_heads` | 参数量（估计） |
|---------|-------------|-------------------|-------------|----------------|
| base    | 768         | 12（6 stages）    | 12          | ~86M           |
| large   | 1024        | 24（12 stages）   | 16          | ~307M          |
| huge    | 1280        | 32（16 stages）   | 16          | ~633M          |
| giant   | 1408        | 40（20 stages）   | 16          | ~1.1B          |

#### 核心数据结构：`[B, T, P, C]` 统一状态

编码器在内部将 token 保持为 4D 张量 `(B, T, P, C)`，其中：
- `B`：batch size
- `T` = `temporal_size // patch_size[0]`（时间 patch 数，默认 `48//2=24`）
- `P` = `(img_size // patch_size[1]) * (img_size // patch_size[2])`（空间 patch 数，默认 `(256//16)^2=256`）
- `C` = `embed_dim`

#### 子模块

**PatchEmbed3D**（3D 卷积 Patch 嵌入）

```python
self.proj = nn.Conv3d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=(2, 16, 16),   # 默认：时间步长=2，空间步长=16
    stride=(2, 16, 16),
)
# 输入: (B, C, T, H, W) → 输出: (B, N, embed_dim)，N = T'*H'*W'
```

**FactorizedPositionEncoding3D**（分解式三维位置编码）

将时间、行、列编码分解为三个独立的可学习参数，相加得到 3D 位置编码：
```python
self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, 1, embed_dim))
self.row_embed  = nn.Parameter(torch.zeros(1, 1, num_rows, 1, embed_dim))
self.col_embed  = nn.Parameter(torch.zeros(1, 1, 1, num_cols, embed_dim))
pos = time_embed + row_embed + col_embed  # 广播相加
```
参数量 = `(T' + H' + W') * D`，远小于 3D 全连接位置编码的 `T'*H'*W'*D`。

**EncoderStage**（Local → Global 交替注意力）

每个 stage 由一个 `LocalBlock` + 一个 `GlobalBlock` 组成：

```
LocalBlock:
  x: (B, T, P, C) → reshape → (B*T, P, C)
  → EfficientAttention（每帧内部自注意力）
  → reshape → (B, T, P, C)

GlobalBlock:
  x: (B, T, P, C) → reshape → (B, T*P, C)
  → EfficientAttention（全序列跨帧注意力）
  → reshape → (B, T, P, C)
```

`depth` 必须为偶数（每 stage = 2 层），`num_stages = depth // 2`。

**EfficientAttention**

使用 `torch.nn.functional.scaled_dot_product_attention`，自动选择 FlashAttention / memory-efficient / math 后端。

**Blackwell GPU（sm_103+）兼容**：在 `_is_blackwell()` 为 True 时，强制使用 MATH 后端，绕过 cuDNN frontend bug：
```python
if _is_blackwell():
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(q, k, v, ...)
```

**Aspect Ratio Token**

将原始视频宽高比编码为一个额外 token，拼接到每帧 token 末尾：
```python
self.aspect_ratio_embed = nn.Linear(1, embed_dim)
# aspect_ratio: (B, 1) → (B, embed_dim) → 拼接到 patch_tokens [B, T, P+1, C]
```
输出时默认剥除该 token（`keep_special_tokens_in_output=False`）。

**VideoMAE 权重初始化**

`_load_videomae_weights(model_name)`：从 HuggingFace VideoMAE 加载权重，映射到自定义编码器的 stage 子块（分别对应 local_block 和 global_block）。支持 Q/K/V 权重从三个独立矩阵合并到一个 QKV 矩阵的格式转换。

**梯度检查点**

```python
model.encoder.gradient_checkpointing_enable()
# 训练前向时对每个 stage 使用 torch.utils.checkpoint.checkpoint()
```

#### 编码器前向流程

```
video → canonicalize_video() → (B, C, T, H, W)
  → PatchEmbed3D → (B, N, D) → reshape → (B, T, P, D)
  → FactorizedPositionEncoding3D（加位置编码）
  → build aspect_ratio_token → cat → (B, T, P+1, D)
  → for stage in stages:
      LocalBlock → GlobalBlock
  → LayerNorm
  → 剥除 special token，flatten → (B, T*P, D)  ← GSR
```

---

### 4.3 解码器 D4RTDecoder

**文件**：`models/decoder.py`

#### 架构概述

```
query (u,v,t_src,t_tgt,t_cam)
  │
  ├─ FourierEmbedding(u,v)     → (B, N, D)
  ├─ TimestepEmbedding(t_src)  → (B, N, D)
  ├─ TimestepEmbedding(t_tgt)  → (B, N, D)
  ├─ TimestepEmbedding(t_cam)  → (B, N, D)
  └─ PatchEmbeddingFast(u,v,t_src, frames) → (B, N, D)
         ↓ 所有分量相加
  query_base = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb
         ↓ + learnable query_token
         ↓ query_mlp（2层 MLP 混合）
  query: (B, N_q, D)
         ↓
  for block in blocks:          # depth 个 DecoderBlock
      CrossAttention(query → encoder_features)
      MLP
         ↓
  LayerNorm
         ↓ 6个输出头
  ┌─ head_3d:    Linear(D, 3)  → pos_3d
  ├─ head_2d:    Linear(D, 2)  → delta_2d → pos_2d = coords + delta_2d
  ├─ head_vis:   Linear(D, 1)  → visibility logit
  ├─ head_disp:  Linear(D, 3)  → displacement
  ├─ head_normal:Linear(D, 3)  → normal → F.normalize
  └─ head_conf:  Linear(D, 1)  → s，clamp(-5.0, 10.0)
```

**关键设计**：查询之间**不做自注意力**，每个 query 独立经过 cross-attention 查询 GSR，完全并行。

#### DecoderBlock

```python
class DecoderBlock(nn.Module):
    # Pre-LN 结构
    def forward(self, query, encoder_features):
        memory = self.norm_kv(encoder_features)
        query = query + self.cross_attn(
            query=self.norm1(query),
            key=memory,
            value=memory,
        )
        query = query + self.mlp(self.norm2(query))
        return query
```

`disable_cross_attention=True` 时跳过 cross-attention，仅保留 MLP，用于消融实验。

#### 2D Head：残差设计

```python
delta_2d = self.head_2d(query)          # 预测残差偏移
pos_2d = coords + delta_2d              # 最终 2D 位置 = 输入坐标 + 残差
```
配合 `--zero-init-2d-residual-head`，初始时 `pos_2d = coords`（恒等映射），从零开始学习偏移。

#### Confidence Head：Kendall log-parameterization

```python
s = self.head_conf(query)
s = torch.clamp(s, min=-5.0, max=10.0)
# 网络输出 s，实际置信度 c = exp(-s)
# s→-∞ 时 c→+∞（极高置信），s→+∞ 时 c→0（极低置信）
# clamp min=-5.0：防止 exp(-s) 超过 exp(5)≈148，避免梯度爆炸
```

#### 3D Head 模式

| `debug_3d_head_mode` | 结构 |
|----------------------|------|
| `linear`（默认） | `Linear(D, 3)` |
| `mlp256` | `Linear(D,256) → ReLU → Linear(256,256) → ReLU → Linear(256,3)` |

#### Patch Provider

| Provider | 说明 |
|----------|------|
| `auto` | 有 `local_patches` 则用 precomputed_resized，否则 sampled_resized |
| `sampled_resized` | 在线从 resized 帧（256×256）用 grid_sample 采样 |
| `precomputed_resized` | 使用 dataset 预计算的 patch（默认最快） |
| `sampled_highres` | 在线从原始高分辨率帧采样（需要 transform_metadata） |
| `precomputed_highres` | 使用从高分辨率帧预计算的 patch |

---

### 4.4 嵌入模块

**文件**：`models/embeddings.py`

#### FourierEmbedding（2D 坐标 → 嵌入）

```python
# 频率带：2^0, 2^1, ..., 2^(L-1)，L = num_frequencies = 64
freqs = 2.0 ** torch.linspace(0, L-1, L)

# 对每个坐标 (u, v) 计算多频率 sin/cos：
coords_freq = coords.unsqueeze(-1) * freqs * (2π)   # (B, N, 2, L)
fourier_features = cat([sin(coords_freq), cos(coords_freq)], dim=-1)  # (B, N, 2, 2L)
fourier_features = reshape(B, N, 4L)

# 线性投影到 embed_dim
output = Linear(4L → D)(fourier_features)           # (B, N, D)
```
注意：sin/cos 计算在 fp32 下进行以防止高频溢出，投影后转回 AMP dtype。

#### TimestepEmbedding（离散时间步 → 嵌入）

```python
# 三个独立可学习嵌入表，各 max_timesteps=128 个条目
self.src_embedding = nn.Embedding(128, D)
self.tgt_embedding = nn.Embedding(128, D)
self.cam_embedding = nn.Embedding(128, D)

src_emb = self.src_embedding(t_src)   # (B, N, D)
tgt_emb = self.tgt_embedding(t_tgt)   # (B, N, D)
cam_emb = self.cam_embedding(t_cam)   # (B, N, D)
```

#### PatchEmbeddingFast（局部 RGB patch → 嵌入）

使用 `torch.nn.functional.grid_sample`（5D 模式）从视频张量中向量化提取 patch：

```python
# input_5d: (B, C, T, H, W)
# grid_xyz:  (B, N, ps, ps, 3)  ← 在 (x, y, z=时间) 上采样
patches = F.grid_sample(input_5d, grid_xyz,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=True)
# output: (B, C, N, ps, ps) → permute → (B, N, C, ps, ps)

# MLP 嵌入：
# patch_dim = ps * ps * 3 = 9*9*3 = 243
patches_flat = reshape(B, N, 243)
embedding = MLP(243 → D → D)(patches_flat)   # GELU 激活
```

---

## 5. 数据管线

**文件**：`data/dataset.py`

### 5.1 PointOdyssey 数据集结构

```
/data2/d4rt/datasets/PointOdyssey_fast/
├── train/          # 131 个序列
│   ├── <seq_name>/
│   │   ├── rgbs/               # RGB 图像（jpg/png）
│   │   ├── depths/             # 深度图（png uint16 或 npy）
│   │   ├── normals/            # 法向量图
│   │   └── anno_fast/          # 预计算缓存（核心）
│   │       ├── trajs_2d.npy            # (T, N_pts, 2) 像素轨迹
│   │       ├── trajs_3d.npy            # (T, N_pts, 3) 世界坐标 3D 轨迹
│   │       ├── valids.npy              # (T, N_pts) 标注有效性
│   │       ├── visibs.npy              # (T, N_pts) 可见性
│   │       ├── intrinsics.npy          # (T, 3, 3) 相机内参
│   │       ├── extrinsics.npy          # (T, 4, 4) 相机外参（世界→相机）
│   │       ├── frame_manifest.json     # 帧文件路径清单
│   │       ├── rgb_frames.bin          # Packed JPEG 帧（字节流）
│   │       ├── rgb_frames_offsets.npy  # 帧偏移数组
│   │       ├── depth_frames.bin
│   │       ├── depth_frames_offsets.npy
│   │       ├── normal_frames.bin
│   │       ├── normal_frames_offsets.npy
│   │       ├── normal_frames_valids.npy # 法向量有效性标志
│   │       ├── frame_pack_meta.json    # Packed 缓存元数据
│   │       ├── motion_boundary_stride_01_packed.npy  # 运动边界（stride=1）
│   │       ├── motion_boundary_stride_01_meta.json
│   │       ├── motion_boundary_stride_02_packed.npy  # 运动边界（stride=2）
│   │       └── ...（stride 3, 4）
├── val/            # 15 个序列
└── test/           # 13 个序列
```

**深度图解码**：`decode_pointodyssey_depth()`
- 浮点 npy：直接使用
- uint16 png：`depth = uint16_value / 65535.0 * 1000.0`（单位：毫米→米）

### 5.2 四种帧缓存后端

Dataset `__getitem__` 按优先级选择后端：

```
优先级 1（最快）: Packed Encoded Cache（.bin + offsets）
优先级 2: Raw Frame Cache（mmap .npy）
优先级 3: 直接文件读取（.jpg/.png/.npy）
```

**Packed Encoded Cache**（主要使用）：

```python
# 所有帧的 JPEG/PNG 字节流拼接在一个 .bin 文件中
# offsets[i]..offsets[i+1] 定位第 i 帧字节
encoded = np.memmap('rgb_frames.bin', mode='r', dtype=np.uint8)
offsets = np.load('rgb_frames_offsets.npy')

start = offsets[frame_pos]
end   = offsets[frame_pos + 1]
frame_bytes = encoded[start:end]
image = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
```

### 5.3 运动边界缓存

**格式**：每帧的运动边界 mask 以 1-bit 打包存储：

```python
# 写入时：
packed = np.packbits(boundary_mask, axis=-1, bitorder='little')
# 读取时：
unpacked = np.unpackbits(packed_frames, axis=-1, bitorder='little')[..., :width]
cropped = unpacked[:, y0:y0+crop_h, x0:x0+crop_w].astype(bool)
```

**运动边界计算算法**（`compute_motion_boundary_mask_for_frame()`）：
1. 对每个有效点计算前后帧的平均运动幅度
2. 将运动幅度散点图用 Gaussian Blur（5×5）平滑
3. 用 Sobel 算子计算运动幅度梯度大小
4. 取 85 百分位作为阈值，超过阈值的区域为运动边界
5. 对结果做膨胀（5×5，1次迭代）扩大边界范围

### 5.4 Clip 采样流程

```python
# 1. 随机选择 stride（1~4，根据总帧数限制）
stride = randint(1, min(4, total_frames // S))

# 2. 随机选择起始帧
max_safe_start = total_frames - 1 - (S - 1) * stride
t_start = randint(0, max_safe_start)

# 3. 等间隔采样 S 帧
frame_indices = [t_start + i * stride for i in range(S)]
# 默认 S=48，产生 48 帧 clip
```

### 5.5 数据增强

**仅训练时（`use_augs=True`）启用**：

**随机裁剪**（`_sample_crop()`）：
```python
# 10 次尝试，随机面积比 [0.3, 1.0]，随机宽高比 [3/4, 4/3]
target_area = uniform(0.3, 1.0) * H * W
aspect = exp(uniform(log(3/4), log(4/3)))
crop_w = sqrt(target_area * aspect)
crop_h = sqrt(target_area / aspect)
# 5% 概率额外应用 zoom-in（缩放到 70%~95%）
```

**颜色增强**（`_apply_color_aug()`）：

| 增强项 | 范围 |
|--------|------|
| 亮度 | `[0.6, 1.4]` |
| 对比度 | `[0.6, 1.4]` |
| 饱和度 | `[0.6, 1.4]` |
| 色调偏移 | `[-18°, +18°]`（HSV 空间） |
| 灰度化 | 20% 概率 |
| Gaussian Blur | 40% 概率，sigma `[0.1, 2.0]` |

**验证模式**：固定中心裁剪，不做颜色增强，但仍随机采样时间起点和 stride（`_get_rngs` 在 train split 始终随机）。

### 5.6 Query 采样策略

**边界过采样**（30% / 70% 混合）：
```python
if len(boundary_candidates) > 0 and random() < boundary_ratio(0.3):
    point_idx = choice(boundary_candidates)   # 从边界点采样
else:
    point_idx = choice(valid_sources)         # 从所有有效点随机采样
```

边界 = 深度边界 `∪` 运动边界（Sobel 梯度 > 85 百分位）。

**时间采样模式**（`query_mode`）：

| 模式 | `t_src` | `t_tgt` | `t_cam` | 用途 |
|------|---------|---------|---------|------|
| `full` | 随机 | 随机 | 随机（40% = `t_tgt`） | 完整 D4RT 训练 |
| `target_cam` | 随机 | 随机 | = `t_tgt` | 限制 t_cam=t_tgt |
| `same_frame` | 随机 | = `t_src` | = `t_src` | 纯深度/法向 |

**`t_tgt_eq_t_cam_ratio=0.4`**：`full` 模式下 40% 的查询采样 `t_cam = t_tgt`，60% 完全独立随机。

**3D 坐标转换**（核心标注逻辑）：
```python
# 将世界坐标系 3D 点变换到 t_cam 帧的相机坐标系
src_world = trajs_3d[t_src, point_idx]            # 世界坐标
tgt_world = trajs_3d[t_tgt, point_idx]

# extrinsics: 世界→相机的变换矩阵 (4×4)
src_cam_h = extrinsics[t_cam] @ [src_world, 1.0]
tgt_cam_h = extrinsics[t_cam] @ [tgt_world, 1.0]

target_pos_3d = tgt_cam_h[:3]           # 目标 3D 标注
target_disp   = tgt_cam_h[:3] - src_cam_h[:3]  # 位移标注
```

### 5.7 Dataset 输出格式

每个样本返回 `(sample_dict, success_flag)`：

```python
sample = {
    # 核心输入
    "video":       Tensor (S, 3, img_size, img_size),   # [0,1] RGB
    "coords":      Tensor (N_q, 2),                      # 归一化查询坐标
    "t_src":       Tensor (N_q,) long,
    "t_tgt":       Tensor (N_q,) long,
    "t_cam":       Tensor (N_q,) long,
    "aspect_ratio":Tensor (1,),                          # crop_w / crop_h

    # 变换元数据（用于高分辨率 patch 采样）
    "transform_metadata": {
        "canonical_space": Tensor scalar,
        "original_hw":    Tensor (2,),
        "crop_offset_xy": Tensor (2,),
        "crop_size_hw":   Tensor (2,),
        "resized_hw":     Tensor (2,),
    },

    # 训练标注
    "targets": {
        "pos_2d":      Tensor (N_q, 2),    # 归一化 2D 目标位置
        "pos_3d":      Tensor (N_q, 3),    # 相机坐标系 3D 目标位置
        "visibility":  Tensor (N_q,),      # 0/1 可见性
        "displacement":Tensor (N_q, 3),    # 3D 位移向量
        "normal":      Tensor (N_q, 3),    # 表面法向量
        "mask_3d":     BoolTensor (N_q,),  # 3D loss 有效 mask
        "mask_2d":     BoolTensor (N_q,),  # 2D loss 有效 mask
        "mask_vis":    BoolTensor (N_q,),  # vis loss 有效 mask
        "mask_disp":   BoolTensor (N_q,),  # disp loss 有效 mask
        "mask_normal": BoolTensor (N_q,),  # normal loss 有效 mask
        "source_is_boundary":       BoolTensor (N_q,),
        "source_is_depth_boundary": BoolTensor (N_q,),
        "source_is_motion_boundary":BoolTensor (N_q,),
        "point_indices": LongTensor (N_q,),
    },

    # 可选：预计算 patch（precompute_local_patches=True 时存在）
    "local_patches": Tensor (N_q, C, ps, ps),

    # 可选：高分辨率查询视频（return_query_video=True 时存在）
    "video_query": Tensor (S, C, H_orig, W_orig),

    # 可选辅助数据（return_aux_tensors=True 时存在）
    "frame_indices": LongTensor (S,),
    "extrinsics":    Tensor (S, 4, 4),
    "depths":        Tensor (S, 1, img_size, img_size),
    "normals":       Tensor (S, 3, img_size, img_size),
    "intrinsics":    Tensor (S, 3, 3),  # resize 后
}
```

**`__len__`** 返回序列数量（131/15/13）。每次 `__getitem__` 随机采样一个 clip。

### 5.8 collate_fn

```python
def collate_fn(batch):
    # 过滤失败样本
    samples = [sample for sample, success in batch if success and sample]
    # 处理 video_query 的变长 padding
    # 处理 local_patches 的 None 过滤
    return default_collate(filtered_samples)
```

---

## 6. 损失函数

**文件**：`losses/losses.py`

### 6.1 Scale-Invariant 3D Loss（主损失）

**完整公式**：

```
Step 1: mean-depth 归一化（按 t_cam 分组）
  μ_z(group) = |mean(z_i for i in group, z_i valid)|   ← 先取均值再取绝对值
  p̂_i = p_i / μ_z(group)        （预测与目标分别归一化）

Step 2: log 变换（抑制远处点）
  L(x) = sign(x) · log(1 + |x|)

Step 3: 逐点 L1
  ℓ_i = mean_j |L(p̂_pred_ij) - L(p̂_tgt_ij)|   (j ∈ {x,y,z})

Step 4: 置信度加权
  loss_3d = sum_i [exp(-s_i) · ℓ_i · mask_i] / sum_i(mask_i)
```

**分组归一化**（`normalize_groups = t_cam`）：每个 `t_cam` 值对应一个独立的归一化组，使多视角查询在各自参考系下归一化，避免跨视角的尺度耦合。

```python
for group_id in unique(t_cam):
    group_mask = (t_cam == group_id) & valid
    group_mean_z = z[group_mask].mean()                    # 先取 z 的均值（可能为负）
    normalizer[group_mask] = clamp(abs(group_mean_z), min=1e-6)  # 再取绝对值
    # 注意：这是 |mean(z)|，不是 mean(|z|)；当所有点 z>0 时两者近似相等
```

**`debug_3d_loss_mode='raw_l1'`**（消融实验）：`loss_3d` 直接使用相机坐标系 L1，跳过归一化和 log 变换。

**注意**：`loss_raw_3d`（无归一化、无 log 变换的原始 L1）**始终被计算**并记录，不论 `debug_3d_loss_mode` 取何值，目的是监控绝对误差。当 `lambda_raw_3d=0.0` 时它不参与总损失，但键值仍存在于 losses 字典中。

### 6.2 2D Reprojection Loss

```python
loss_2d = mean(|pred_2d - target_2d| · mask_2d) / sum(mask_2d)
```
L1 损失，mask = 目标可见且在 crop 范围内。

### 6.3 Visibility Loss

```python
loss_vis = BCE_with_logits(pred_vis_logit, target_vis_float) · mask_vis
```
Binary cross-entropy，mask = `target_defined`（标注有效）。

### 6.4 Displacement Loss

```python
loss_disp = mean(|pred_disp - (tgt_cam - src_cam)| · mask_disp)
```
L1 损失，mask = 3D 标注有效（`has_valid_3d`）。

### 6.5 Normal Loss

```python
pred_n  = F.normalize(pred_normal, dim=-1)
tgt_n   = F.normalize(target_normal, dim=-1)
loss_normal = mean((1 - dot(pred_n, tgt_n)) · mask_normal)
```
余弦相似度损失，范围 `[0, 2]`，mask = 可见且法向量有效。

### 6.6 Confidence Loss（Kendall 公式）

使用 Kendall & Gal (2017) 的异方差不确定性建模，网络输出 `s = -log(c)`（而非直接输出置信度 `c`）：

```
完整 Kendall 目标函数（3D 损失部分）：
  L = exp(-s) · ℓ_3d + s

分拆：
  loss_3d = exp(-s) · ℓ_3d    → 置信度加权的 3D 误差
  loss_conf = s                → 正则项，防止 s → -∞
```

**参数范围约束**（`head_conf` 输出后）：
```python
s = torch.clamp(s, min=-5.0, max=10.0)
# min=-5.0: exp(-s) 最大 ≈ 148，置信度上限，防止梯度爆炸
# max=10.0: 防止 s 无限增大（相当于置信度→0）
```

### 6.7 总损失与权重

```python
loss = (
    λ_3d   · loss_3d    +   # 默认 1.0
    λ_raw3d· loss_raw_3d+   # 默认 0.0（仅 debug）
    λ_2d   · loss_2d    +   # 默认 0.1
    λ_vis  · loss_vis   +   # 默认 0.1
    λ_disp · loss_disp  +   # 默认 0.1
    λ_normal·loss_normal+   # 默认 0.5
    λ_conf · loss_conf      # 默认 0.2（配合置信度调度）
)
```

**记录的额外 metrics**（不参与梯度）：

| Key | 含义 |
|-----|------|
| `metric_raw_3d_l1` | 原始相机坐标 L1 误差（用于监控） |
| `metric_raw_3d_euclidean` | 欧式距离误差 |
| `metric_conf_mean` | 平均置信度参数 `s` |
| `metric_pred_abs_depth_mean` | 预测绝对深度均值 |
| `metric_target_abs_depth_mean` | GT 绝对深度均值 |
| `metric_pred_norm_depth_mean` | 归一化后预测深度均值 |
| `metric_target_norm_depth_mean` | 归一化后 GT 深度均值 |

### 6.8 Confidence 调度

置信度损失线性 ramp-up，防止训练初期置信度坍塌：

```python
# penalty_factor：λ_conf 从 0 线性增长到目标值（唯一实际生效的调度维度）
penalty_factor = linear_ramp(step, start=conf_ramp_start_step,
                              ramp=conf_ramp_steps)
current_lambda_conf = λ_conf · penalty_factor

# weighting_factor：被计算并传入 criterion.set_confidence_schedule()，
# 但当前代码在 _reduce_point_loss() 中并不读取该字段——
# set_confidence_schedule() 会将 use_confidence_weighting 强制置为 True，
# exp(-s) 权重始终全量应用。实际上对 exp(-s) 参与度的控制完全通过 lambda_conf 实现。
weighting_factor = linear_ramp(step, start=conf_weighting_start_step,
                                ramp=conf_weighting_ramp_steps)  # 存储但未在损失计算中使用
```

**实现细节**：`D4RTLoss.set_confidence_schedule()` 总是将 `self.use_confidence_weighting = True`，即 `exp(-s)` 加权从调度开启的第一步起就完全生效。`confidence_weighting_factor` 字段虽被存储，但 `_reduce_point_loss()` 并不读取该字段。若要在训练初期关闭 exp(-s) 加权，需将 `conf_ramp_start_step` 设为正数（通过 `lambda_conf=0` 间接实现）。

---

## 7. 训练流程

**文件**：`train.py`

### 7.1 分布式设置

```python
# 支持 torchrun / SLURM 启动
rank       = os.environ["RANK"]
world_size = os.environ["WORLD_SIZE"]
local_rank = os.environ["LOCAL_RANK"]

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")  # world_size > 1 时

model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

**Blackwell GPU 修复**（`main()` 入口处）：
```python
cap = torch.cuda.get_device_capability()
if cap[0] >= 10:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
```

### 7.2 优化器与 LR 调度

**AdamW with weight decay 分离**：

```python
# bias、norm、ln 参数不施加 weight decay
decay_params = [p for name, p in model.named_parameters()
                if "bias" not in name
                and "norm" not in name.lower()
                and "ln" not in name.lower()]
no_decay_params = ...  # 其余参数

optimizer = AdamW([
    {"params": decay_params,    "weight_decay": 0.03},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
```

**Cosine LR 调度**（带线性 warmup）：
```
step < warmup_steps:      lr = peak_lr * step / warmup_steps
step >= warmup_steps:
  progress = (step - warmup_steps) / (total_steps - warmup_steps)
  cosine   = 0.5 * (1 + cos(π * progress))
  lr = min_lr + (peak_lr - min_lr) * cosine
```

默认：`peak_lr=1e-4`，`min_lr=1e-6`，`warmup_steps=2500`。

### 7.3 梯度累积与 Query Chunking

**梯度累积**（`gradient_accumulation_steps`）：
```python
is_accumulating = (accum_count % gradient_accumulation_steps) != 0
with model.no_sync() if is_accumulating else nullcontext():
    loss.backward()

if accum_count % gradient_accumulation_steps == 0:
    clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    step += 1
```

**Query Chunking**（`query_chunk_size`，用于低显存解码）：
```python
# 编码一次，分块解码
encoder_features = model.encode(video)

for start, end in iter_query_slices(total_queries, chunk_size):
    predictions = model.decode(encoder_features, ..., coords[:, start:end], ...)
    loss = criterion(predictions, targets[:, start:end])
    # 每个 chunk 的 loss 按 chunk 比例加权，累加梯度
    (loss * chunk_weight / gradient_accumulation_steps).backward(retain_graph=...)
```

### 7.4 大 Batch 支持（ConcatDataset）

PointOdysseyDataset 的 `__len__` 返回序列数（131），对于 `bs=32` 时每 epoch 只有 4 个 batch（drop_last 后）。解决方案：

```python
dataset_repeat_factor = 32  # 默认值
train_dataset = ConcatDataset([train_dataset] * dataset_repeat_factor)
# 131 * 32 = 4192 个"虚拟样本"
# 每次 __getitem__ 仍随机采样不同 clip，实质上是从同一序列多次随机采样
```

对于 `bs=32`，每 epoch 变为 `4192 // 32 = 131` 个 batch，等效于对每个序列多次随机采样。

### 7.5 AMP 混合精度

```python
# 自动选择 dtype
if torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16   # A100/H100/B200 等
else:
    amp_dtype = torch.float16    # 旧 GPU

# GradScaler 仅在 float16 时启用（bf16 不需要）
scaler = torch.amp.GradScaler("cuda",
    enabled=(amp and amp_dtype == torch.float16))

with torch.autocast(device_type='cuda', dtype=amp_dtype):
    encoder_features = model.encode(...)
    predictions = model.decode(...)
    losses = criterion(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 10.0)
scaler.step(optimizer)
scaler.update()
```

### 7.6 Checkpoint 管理

- **保存频率**：每 `save_epochs` epoch 保存一次，外加最终 epoch
- **同时维护** `checkpoint_epoch_{N:04d}.pth` + `checkpoint_latest.pth`
- **最多保留 20 个**：超出时自动删除最旧的
- **内容**：`model_state_dict`、`optimizer_state_dict`、`scheduler_state_dict`、`scaler_state_dict`、`step`、`epoch`、`args`、`train_metrics`、`val_metrics`

**加载模式**：
- 全状态恢复（`--resume`）：严格匹配模型权重，同时恢复 optimizer/scheduler/scaler
- 仅模型权重（`--resume-model-only`）：允许 missing/unexpected keys，跳过优化器状态
- 预训练权重（`--pretrained-weights`）：从检查点初始化模型，不恢复训练状态
- 编码器权重（`--pretrained-encoder`）：只加载 `encoder.*` 前缀的参数

**Key 归一化**：加载时自动剥除 `module.` / `model.` 前缀（DDP 训练 vs 单卡推理兼容）。

### 7.7 训练命令示例

**单 GPU（调试/小实验）**：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/d4rt_pointodyssey_curriculum_base.yaml \
    --data-root /data2/d4rt/datasets/PointOdyssey_fast \
    --output-dir outputs/exp001 \
    --batch-size 1 \
    --num-queries 2048 \
    --epochs 100
```

**多 GPU（torchrun）**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    --config configs/d4rt_base.yaml \
    --data-root /data2/d4rt/datasets/PointOdyssey_fast \
    --output-dir outputs/exp002 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --num-queries 2048 \
    --amp \
    --gradient-checkpointing \
    --epochs 100
```

**大 Batch（bs=32，充分利用 275GB VRAM）**：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --data-root /data2/d4rt/datasets/PointOdyssey_fast \
    --output-dir outputs/bs32 \
    --batch-size 32 \
    --num-queries 2048 \
    --num-workers 32 \
    --prefetch-factor 2 \
    --amp \
    --gradient-checkpointing \
    --epochs 200 \
    --lr 1e-4 \
    --warmup-steps 2500
```

---

## 8. 稠密跟踪算法（Algorithm 1）

**文件**：`models/dense_tracking.py`

对视频中所有像素进行高效稠密跟踪，复杂度从 `O(T²HW)` 降至 `O(THW / visibility_ratio)`（约 5-15× 加速）。

**核心思想**：使用**占用网格**记录已处理像素，只对未处理的像素发起新的跟踪查询：

```
算法 1：稠密跟踪
─────────────────────────────────────
输入：视频 V (B=1, T, H, W, C)，模型 M
输出：稠密轨迹集合 T_all

1. F ← M.encode(V)                         # 计算 GSR（只做一次）
2. G ← {false}^(T × H/s × W/s)             # 初始化占用网格（s=spatial_stride）
3. T_all ← ∅

4. while any(G = false):
5.   B_unvisited ← sample(~G, batch_size)   # 从未访问像素中随机取 batch

6.   for each (t_src, y, x) in B_unvisited:
7.     for t_tgt in range(T):
8.       Q ← (u, v, t_src, t_tgt, t_cam=t_tgt)
9.       P ← M.decode(F, Q)                  # 预测轨迹点

10.    对所有 t_tgt：可见性 > threshold 的像素 → G[t_tgt, y', x'] = True
11.   G[t_src, y, x] = True                  # 标记源像素已访问

12.  T_all ← T_all ∪ {轨迹}

输出 T_all
─────────────────────────────────────
```

**配置参数**（`DenseTrackingConfig`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 256 | 每批处理的轨迹数 |
| `visibility_threshold` | 0.5 | 可见性判断阈值（sigmoid 后） |
| `min_track_length` | 2 | 最短有效轨迹帧数 |
| `spatial_stride` | 1 | 空间采样步长（1=全分辨率） |

**两个跟踪模式**：
- `track_all_pixels()`：在各帧的相机坐标系下表达 3D 位置（`t_cam = t_tgt`）
- `track_all_pixels_to_world()`：所有 3D 位置统一表达在 `reference_frame` 相机坐标系下（`t_cam = reference_frame`）

---

## 9. 工具模块

**`utils/patches.py`**

```python
extract_local_patches(frames_btchw, coords, t_src, patch_size)
# 使用 5D grid_sample 批量提取局部 patch
# frames_btchw: (B, T, C, H, W)
# 输出: (B, N, C, ps, ps)

extract_local_patches_with_valid_hw(frames_btchw, coords, t_src, patch_size, valid_hw)
# 支持 batch 内不同有效裁剪尺寸（用于高分辨率 patch provider）
# valid_hw: (B, 2) 每个样本的有效 crop 尺寸
```

两者均使用 `padding_mode='border'`（边界重复填充），`align_corners=True`，`mode='bilinear'`。

**`utils/metrics.py`**

```python
# 深度评估指标（scale-invariant / shift-invariant 对齐后）
compute_depth_metrics(pred, target, mask):
    → abs_rel, sq_rel, rmse, log_rmse, δ1, δ2, δ3
```

**`utils/camera.py`**

```python
# 相机投影、Umeyama 对齐（用于位姿估计评估）
```

---

## 10. 配置文件说明

### `configs/d4rt_base.yaml`（基础配置）

```yaml
encoder: base              # ViT-B，768d，12 层（6 stages）
decoder_depth: 6
img_size: 256              # 输入分辨率
num_frames: 48             # clip 长度
patch_size: 9              # 查询局部 patch 大小

batch_size: 1              # 论文设定：每卡 bs=1
num_queries: 256           # 每 batch 查询数
steps: 500000              # 训练总步数
lr: 1.0e-4
min_lr: 1.0e-6
warmup_steps: 2500
weight_decay: 0.03
grad_clip: 10.0
amp: true

lambda_3d: 1.0
lambda_2d: 0.1
lambda_vis: 0.1
lambda_disp: 0.1
lambda_normal: 0.5
lambda_conf: 0.2
```

### `configs/d4rt_pointodyssey_curriculum_base.yaml`（课程训练配置）

```yaml
# 开局仅训练基础 3D 重建（关闭所有辅助损失）
lambda_vis: 0.0
lambda_disp: 0.0
lambda_normal: 0.0
lambda_conf: 0.0

# 从 same_frame 模式开始（最简单任务：深度估计）
query_mode: same_frame

# VideoMAE 预训练权重初始化
videomae_model: MCG-NJU/videomae-base
patch_provider: auto

# 梯度检查点（节省显存）
gradient_checkpointing: true
query_chunk_size: 256      # 分块解码，降低 peak 显存
```

---

## 11. 关键超参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder` | `base` | 编码器规模：base/large/huge/giant |
| `--decoder-depth` | variant 默认 | 解码器层数 |
| `--img-size` | `256` | 输入分辨率 |
| `--num-frames` | `48` | Clip 帧数（T） |
| `--patch-size` | `9` | 查询 RGB patch 大小（9×9） |
| `--batch-size` | `1` | 每 GPU batch size |
| `--gradient-accumulation-steps` | `1` | 梯度累积步数 |
| `--num-queries` | `2048` | 每 batch 查询数（N_q） |
| `--query-chunk-size` | `0`（不分块） | 解码分块大小 |
| `--epochs` | `100` | 训练 epoch 数 |
| `--steps` | `0`（按 epoch） | 指定总优化步数（优先级高于 epochs） |
| `--lr` | `1e-4` | 峰值学习率 |
| `--min-lr` | `1e-6` | 最小学习率 |
| `--warmup-steps` | `2500` | warmup 步数 |
| `--weight-decay` | `0.03` | 权重衰减（bias/norm 不施加） |
| `--grad-clip` | `10.0` | 梯度裁剪（L2 范数） |
| `--amp` | `False` | 开启混合精度 |
| `--gradient-checkpointing` | `False` | 开启梯度检查点 |
| `--lambda-3d` | `1.0` | 3D loss 权重 |
| `--lambda-2d` | `0.1` | 2D loss 权重 |
| `--lambda-vis` | `0.1` | 可见性 loss 权重 |
| `--lambda-disp` | `0.1` | 位移 loss 权重 |
| `--lambda-normal` | `0.5` | 法向量 loss 权重 |
| `--lambda-conf` | `0.2` | 置信度 loss 权重 |
| `--query-mode` | `full` | 查询时间模式 |
| `--t-tgt-eq-t-cam-ratio` | `0.4` | t_cam=t_tgt 的采样概率 |
| `--boundary-ratio` | `0.3`（内置） | 边界 query 比例 |
| `--num-workers` | `4` | DataLoader worker 数 |
| `--prefetch-factor` | `2` | 每 worker 预取 batch 数 |
| `--save-epochs` | `5` | 每 N epoch 保存 checkpoint |
| `--val-every-epochs` | `1` | 每 N epoch 做验证 |

---

## 12. 数据预处理流程（prepare 脚本）

**一键处理命令**：

```bash
conda run -n d4rt python scripts/prepare_pointodyssey_local_dataset.py \
    --src-root /data2/d4rt/datasets/PointOdyssey \
    --dst-root /data2/d4rt/datasets/PointOdyssey_fast \
    --splits train val test \
    --workers 8 \
    --verify-sample
```

**处理步骤**（按 `prepare_pointodyssey_local_dataset.py` 执行顺序）：

1. **`build_pointodyssey_fast_cache.py`**：从 `anno.npz` 提取并分别保存为独立 `.npy` 文件（支持 mmap 按需读取），生成 `anno_fast/` 目录
2. **`build_pointodyssey_packed_frame_cache.py`**：将 RGB、depth、normal 图像编码为紧凑 `.bin` 文件 + 偏移数组，同时生成 `frame_pack_meta.json`
3. **`build_pointodyssey_motion_boundary_cache.py`**：对每个序列、每个 stride（1~4）预计算运动边界 mask 并以 bit-packed 格式存储
4. **`check_pointodyssey_sanity.py`**（`--verify-sample`）：从处理后的数据集加载一个 batch，验证几何一致性

**每个序列处理后的新增文件**（`anno_fast/` 目录）：

| 文件 | 大小估计 | 说明 |
|------|---------|------|
| `trajs_2d.npy` | ~(T,N,2) float32 | 2D 轨迹 |
| `trajs_3d.npy` | ~(T,N,3) float32 | 3D 轨迹（世界坐标） |
| `valids.npy` | ~(T,N) float32 | 有效性 |
| `visibs.npy` | ~(T,N) float32 | 可见性 |
| `intrinsics.npy` | (T,3,3) float32 | 内参 |
| `extrinsics.npy` | (T,4,4) float32 | 外参 |
| `rgb_frames.bin` | ~序列帧数×~50KB | JPEG 编码帧 |
| `rgb_frames_offsets.npy` | (T+1,) int64 | 帧偏移 |
| `depth_frames.bin` | 变化 | 深度帧 |
| `normal_frames.bin` | 变化 | 法向量帧 |
| `motion_boundary_stride_01_packed.npy` | ~(T, ceil(H/8)) | 1-bit 运动边界 |
| `motion_boundary_stride_0{2,3,4}_packed.npy` | 同上 | 多 stride |

**PointOdyssey Sanity Gate**（训练前自动执行）：

训练脚本 `main()` 中在数据加载之前自动执行 `check_pointodyssey_sanity.py`（可用 `--skip-pointodyssey-sanity` 跳过）：
- 从指定 split（优先 `sample` > `val` > `train`）随机加载几帧
- 验证 2D/3D 轨迹在相机投影下的几何一致性
- 若失败则中止训练并打印错误原因

---

*本文档基于代码库 `/data1/zbf/my_dfrt/` 的完整分析编写，覆盖所有 `.py` 源文件的实现细节。*
