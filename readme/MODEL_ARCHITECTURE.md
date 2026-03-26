# D4RT 模型架构说明

## 概述

D4RT (Dense 4D Reconstruction Transformer) 是一个前馈式 Transformer 模型，将视频编码为**全局场景表示 F**（Global Scene Representation），然后通过轻量解码器对任意时空查询点 $(u, v, t_{src}, t_{tgt}, t_{cam})$ 独立预测 3D/2D 位置、可见性、运动位移、法线和置信度。

**核心设计原则：**
- **一次编码，任意解码**：编码器输出 F 在解码阶段固定，解码器针对每个查询点独立运行，不同查询间无交互
- **统一查询接口**：同一模型通过不同的 $(t_{src}, t_{tgt}, t_{cam})$ 组合，支持点追踪、深度估计、点云重建等多种任务
- **分离时空注意力**：编码器通过 Local→Global 交替 Stage 分离帧内空间相关性与跨帧时序相关性

```
video  (B, T, C, H, W)  +  aspect_ratio (B, 1)
            │
     ┌──────▼──────┐
     │   Encoder   │  ← PatchEmbed3D + FactorizedPosEmbed
     │             │  ← EncoderStage × N_stages
     │             │      LocalBlock  (帧内自注意力)
     │             │      GlobalBlock (全局时空注意力)
     └──────┬──────┘
            │  encoder_features F: (B, T'×P, D)  [固定不变]
            │
     ┌──────▼──────┐  ← query = (u,v,t_src,t_tgt,t_cam) + local_patch
     │   Decoder   │  ← build_query: FourierEmb + TimestepEmb + PatchEmb
     │             │  ← DecoderBlock × depth (CrossAttn + MLP)
     └──────┬──────┘
            │
     ┌──────▼───────────────────────────────────┐
     │ pos_3d (B,N_q,3)  pos_2d (B,N_q,2)      │
     │ visibility (B,N_q,1)  normal (B,N_q,3)   │
     │ displacement (B,N_q,3)  confidence (B,N_q,1) │
     └──────────────────────────────────────────┘
```

---

## 顶层模型：D4RT

**文件**：`models/d4rt.py` · `class D4RT(nn.Module)`

### 初始化参数

```python
D4RT(
    encoder_variant='base',               # 编码器规格: base/large/huge/giant
    img_size=256,                         # 输入图像空间尺寸（正方形）
    temporal_size=48,                     # 输入视频帧数
    patch_size=(2, 16, 16),               # 时空 patch 大小 (T_p, H_p, W_p)
    decoder_depth=8,                      # 解码器 DecoderBlock 层数
    decoder_num_heads=12,                 # 解码器注意力头数
    max_timesteps=128,                    # TimestepEmbedding 最大索引
    query_patch_size=9,                   # 查询局部 patch 边长（像素）
    videomae_model=None,                  # VideoMAE HuggingFace 模型名或本地路径
    patch_provider='auto',                # patch 提供策略（见解码器章节）
    drop_rate=0.0,                        # MLP Dropout 比率
    attn_drop_rate=0.0,                   # 注意力 Dropout 比率
    disable_query_patch_embedding=False,  # 调试：关闭 patch 嵌入项
    disable_query_timestep_embedding=False, # 调试：关闭时间步嵌入项
    disable_decoder_cross_attention=False,  # 调试：关闭解码器交叉注意力
    debug_3d_head_mode='linear',          # 3D 预测头：'linear' 或 'mlp256'
)
```

### 预设规格（`create_d4rt`）

```python
decoder_configs = {
    'base':  dict(decoder_depth=6,  decoder_num_heads=12),
    'large': dict(decoder_depth=6,  decoder_num_heads=16),
    'huge':  dict(decoder_depth=8,  decoder_num_heads=16),
    'giant': dict(decoder_depth=8,  decoder_num_heads=16),
}
```

### 查询接口的任务统一性

D4RT 通过同一个五元组 $(u, v, t_{src}, t_{tgt}, t_{cam})$ 表示所有任务：

| 任务                  | $t_{src}$    | $t_{tgt}$    | $t_{cam}$    | $(u, v)$         |
|-----------------------|--------------|--------------|--------------|------------------|
| **点追踪**（2D/3D）   | 固定（查询帧）| 变化（每帧）  | = $t_{tgt}$  | 固定（查询点）    |
| **点云重建**          | 变化（各帧）  | 变化（各帧）  | 固定（参考帧）| 稠密网格         |
| **深度图**            | = $t_{tgt}$  | = $t_{cam}$  | 变化（每帧）  | 稠密网格         |
| **相机外参估计**      | 固定          | 变化          | = $t_{tgt}$  | 稠密网格         |
| **单帧深度**（静态）  | = $t_{tgt}$  | = $t_{cam}$  | = t          | 每像素           |

**含义解释**：
- `t_src`：查询点被定义在哪一帧（即 (u,v) 对应的参考帧）
- `t_tgt`：希望预测该点在哪一帧的位置
- `t_cam`：输出 3D 坐标所在的相机坐标系基准帧（即以哪一帧的相机作为世界坐标原点）

### 主要方法签名

```python
# 编码：视频 → 全局场景表示
encoder_features = model.encode(
    video,          # (B, T, C, H, W) 或 (B, C, T, H, W) 或 (B, T, H, W, C)
    aspect_ratio,   # (B, 1)  width/height 比值，或旧式 (B, 2)
)  # → (B, N_enc, D)

# 解码：查询 → 预测
predictions = model.decode(
    encoder_features,   # (B, N_enc, D)
    frames,             # (B, T, C, H, W)  用于 patch 提取
    coords,             # (B, N_q, 2)  归一化 [0,1]
    t_src,              # (B, N_q)  long
    t_tgt,              # (B, N_q)  long
    t_cam,              # (B, N_q)  long
    local_patches,      # (B, N_q, 3, P, P) 可选预计算 patch
    transform_metadata, # dict 可选几何元数据
)  # → dict[str, Tensor]

# 完整前向（编码+解码）
predictions = model.forward(video, coords, t_src, t_tgt, t_cam, aspect_ratio, ...)

# 高层推理接口（@no_grad）
depth   = model.predict_depth(video, aspect_ratio, output_resolution)
tracks  = model.predict_point_tracks(video, query_points, query_frames, aspect_ratio)
cloud   = model.predict_point_cloud(video, reference_frame, aspect_ratio, stride)
```

### 输出字典

```python
{
    'pos_3d':       (B, N_q, 3),   # 相机坐标系 XYZ（t_cam 帧）
    'pos_2d':       (B, N_q, 2),   # 归一化 [0,1] 的 2D 位置（t_tgt 帧）
    'visibility':   (B, N_q, 1),   # 可见性 logit（sigmoid → 概率）
    'displacement': (B, N_q, 3),   # 3D 运动位移 = pos_3d(tgt_cam) - pos_3d(src_cam)
    'normal':       (B, N_q, 3),   # L2 归一化表面法线
    'confidence':   (B, N_q, 1),   # Kendall s 参数，s = -log(conf)
}
```

---

## 编码器：D4RTEncoder

**文件**：`models/encoder.py` · `class D4RTEncoder(nn.Module)`

### 完整前向传播步骤

```
输入: video (B, C_in, T, H, W)  [canonicalize_video 统一格式]

Step 1: PatchEmbed3D
  Conv3D(3, D, kernel=(T_p,H_p,W_p), stride=(T_p,H_p,W_p))
  (B, C, T, H, W) → (B, D, T', H', W') → flatten → (B, T'×H'×W', D)
  reshape → (B, T', P, D)    其中 P = H'×W' = 空间 token 数/帧

Step 2: FactorizedPositionEncoding3D
  pos = time_embed[T'] + row_embed[H'] + col_embed[W']   # 广播相加
  patch_tokens += pos    (B, T', P, D)

Step 3: aspect_ratio_token（可选，use_aspect_ratio_token=True）
  ar_scalar = canonicalize_aspect_ratio(aspect_ratio)    # (B, 1)  width/height
  ar_token  = Linear(1→D)(ar_scalar)                    # (B, D)
  ar_token 扩展为 (B, T', 1, D)，cat 到 patch_tokens 末尾
  → x: (B, T', P+1, D)

Step 4: EncoderStage × num_stages（每 stage = LocalBlock + GlobalBlock）
  维护统一状态 x: (B, T', P_total, D)
  （支持 gradient_checkpointing）

Step 5: LayerNorm(x)

Step 6: _flatten_output_tokens
  去掉 aspect_ratio_token 列（若有）: x[:, :, :P, :]
  flatten: (B, T', P, D) → (B, T'×P, D)

输出: encoder_features (B, N_enc, D)
  N_enc = T' × P = (T/T_p) × (H/H_p) × (W/W_p)
```

### 各规格维度参数

对于默认配置 img_size=256, temporal_size=48, patch_size=(2,16,16)：

| 规格   | D（embed_dim）| depth | num_stages | num_heads | MLP 隐藏维度 | N_enc（token 数）| 编码器参数量（约）|
|--------|--------------|-------|------------|-----------|------------|-----------------|-----------------|
| base   | 768          | 12    | 6          | 12        | 3072        | 6144            | ~86M            |
| large  | 1024         | 24    | 12         | 16        | 4096        | 6144            | ~307M           |
| huge   | 1280         | 32    | 16         | 16        | 5120        | 6144            | ~632M           |
| giant  | 1408         | 40    | 20         | 16        | 5632        | 6144            | ~1.0B           |

N_enc = (48/2) × (256/16) × (256/16) = 24 × 16 × 16 = **6144**

### PatchEmbed3D

```python
class PatchEmbed3D(nn.Module):
    # 核心：单层 Conv3D，无激活，无 bias 外的非线性
    proj = nn.Conv3d(
        in_channels=3,
        out_channels=embed_dim,
        kernel_size=patch_size,   # (2, 16, 16)
        stride=patch_size,        # 与 kernel 相同，无重叠
    )
    # 参数量 = 3 × 2 × 16 × 16 × D + D = 1536D + D
    # base: 1536×768 + 768 ≈ 1.18M 参数
```

前向：
```
(B, 3, T, H, W)
  → Conv3D → (B, D, T/2, H/16, W/16)
  → flatten(2) → (B, D, T'×H'×W')
  → transpose(1,2) → (B, N, D)
  reshape → (B, T', P, D)   P = H'×W'
```

### FactorizedPositionEncoding3D

将时空位置分解为三个可学习的 1D 参数，**相加**（不拼接）：

```python
# 参数形状（广播兼容）
time_embed  = nn.Parameter(zeros(1, T', 1,  1,  D))   # 时间位置
row_embed   = nn.Parameter(zeros(1, 1,  H', 1,  D))   # 行位置
col_embed   = nn.Parameter(zeros(1, 1,  1,  W', D))   # 列位置

pos = time_embed + row_embed + col_embed   # 广播 → (1, T', H', W', D)
```

优势：参数量 `(T' + H' + W') × D` << 完整 3D 编码 `T'×H'×W'×D`。
base 配置：(24 + 16 + 16) × 768 = 43,008 参数（vs 完整 3D: 4,718,592 参数）。

初始化：`trunc_normal_(std=0.02)`，可选学习 `scale` 参数整体缩放。

### LocalBlock（帧内自注意力）

```python
class LocalBlock(nn.Module):
    norm1 = nn.LayerNorm(D)
    attn  = EfficientAttention(D, num_heads, ...)   # Self-Attention
    norm2 = nn.LayerNorm(D)
    mlp   = MLP(D, D*4)                             # FFN

    def forward(self, x):  # x: (B, T', P, D)
        B, T, P, C = x.shape
        tokens = x.reshape(B * T, P, C)    # ← 关键：每帧独立
        tokens = tokens + attn(norm1(tokens))
        tokens = tokens + mlp(norm2(tokens))
        return tokens.reshape(B, T, P, C)
```

- 每帧 P 个 token 做自注意力，**不跨帧**
- 时间复杂度：O(T' × P² × D)（每帧独立）
- base 配置 单次：O(24 × 256² × 768) ≈ 1.2B FLOPs

### GlobalBlock（全局时空注意力）

```python
class GlobalBlock(nn.Module):
    norm1 = nn.LayerNorm(D)
    attn  = EfficientAttention(D, num_heads, ...)
    norm2 = nn.LayerNorm(D)
    mlp   = MLP(D, D*4)

    def forward(self, x):  # x: (B, T', P, D)
        B, T, P, C = x.shape
        tokens = x.reshape(B, T * P, C)   # ← 关键：全部 token 展平
        tokens = tokens + attn(norm1(tokens))
        tokens = tokens + mlp(norm2(tokens))
        return tokens.reshape(B, T, P, C)
```

- 全部 T'×P 个 token 做自注意力，**跨帧感知全局时序**
- 时间复杂度：O((T'×P)² × D)
- base 配置 单次：O(6144² × 768) ≈ 28.9B FLOPs（计算量最大的部分）

### EncoderStage = LocalBlock → GlobalBlock

```python
class EncoderStage(nn.Module):
    local_block  = LocalBlock(...)
    global_block = GlobalBlock(...)

    def forward(self, x):  # (B, T', P, D)
        x = self.local_block(x)    # 帧内相关
        x = self.global_block(x)   # 全局时空相关
        return x
```

两个 block 均为 **Pre-LN** 残差结构：
```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

### EfficientAttention（通用自注意力）

```python
class EfficientAttention(nn.Module):
    # 参数
    qkv  = nn.Linear(D, D*3, bias=True)   # Q、K、V 一次投影
    proj = nn.Linear(D, D)                 # 输出投影

    def forward(self, x):  # (B, N, D)
        B, N, C = x.shape
        qkv = self.qkv(x)                                  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)

        # F.scaled_dot_product_attention 自动选择最优 SDPA 后端
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=...)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

`head_dim = D / num_heads`：
- base：768 / 12 = 64
- large：1024 / 16 = 64
- huge：1280 / 16 = 80
- giant：1408 / 16 = 88

### Aspect Ratio Token

```python
# 构建
ar_scalar = canonicalize_aspect_ratio(aspect_ratio)  # (B, 1)  = W/H
ar_token  = self.aspect_ratio_embed(ar_scalar)        # Linear(1→D) → (B, D)
ar_token  = ar_token.unsqueeze(1).unsqueeze(2)        # (B, 1, 1, D)
ar_token  = ar_token.expand(-1, T', -1, -1)           # (B, T', 1, D)

# 追加到 patch_tokens 末尾
x = torch.cat([patch_tokens, ar_token], dim=2)        # (B, T', P+1, D)
```

- 在 LocalBlock 中，aspect_ratio_token 与每帧 patch 一起做自注意力
- 在 GlobalBlock 中，参与全局时空注意力
- 输出时通过 `_flatten_output_tokens` 去掉，**不传入解码器**

`canonicalize_aspect_ratio` 支持两种输入格式：
- `(B, 1)` 直接作为 W/H 标量
- `(B, 2)` 两种语义：若 `col[0] × col[1] ≈ 1` 则直接取 col[0]（倒数格式），否则 col[0]/col[1]

### 权重初始化

#### VideoMAE 初始化（默认推荐）

```
VideoMAE: MCG-NJU/videomae-{base,large,huge}
  │
  ├─ patch_embed.proj.weight/bias  ←  backbone.embeddings.patch_embeddings.projection
  │
  └─ stages[i].local_block / global_block  ←  backbone.encoder.layer[j]
       attn.qkv.weight  = cat([query.weight, key.weight, value.weight])
       attn.qkv.bias    = cat([q_bias, zeros(D), v_bias])  # k_bias=0
       attn.proj        ← attention.output.dense
       mlp.fc1          ← intermediate.dense
       mlp.fc2          ← output.dense
       norm1            ← layernorm_before
       norm2            ← layernorm_after
  │
  └─ norm.weight/bias  ←  backbone.layernorm
```

#### 检查点兼容性（`_load_from_state_dict`）

编码器内置遗留键名映射，可加载旧版检查点中的 `blocks.N.*` 格式：

```
旧键: encoder.blocks.{N}.{attn/mlp/norm...}
  N 为偶数 → stages[N//2].local_block.*
  N 为奇数 → stages[N//2].global_block.*

特殊: attn.attention.* → attn.*（旧式双层命名）
      temporal_embed   → time_embed（旧版时间嵌入键名）
```

#### 梯度检查点

```python
model.encoder.gradient_checkpointing_enable()
# 效果：每个 EncoderStage 的前向用 torch.utils.checkpoint.checkpoint 包装
# 节省显存代价：重计算一次前向（增加约 30-40% 计算时间）
```

---

## 解码器：D4RTDecoder

**文件**：`models/decoder.py` · `class D4RTDecoder(nn.Module)`

### 完整模块结构

```python
D4RTDecoder(
    embed_dim=768,          # 与编码器 embed_dim 匹配
    depth=6,                # DecoderBlock 层数
    num_heads=12,           # 交叉注意力头数
    mlp_ratio=4.0,          # FFN 隐层扩张比
    max_timesteps=128,      # TimestepEmbedding 表大小
    patch_size=9,           # 局部 patch 边长
    patch_provider='auto',  # patch 来源策略
    num_fourier_freqs=64,   # FourierEmbedding 频率数
    drop_rate=0.0,
    attn_drop_rate=0.0,
    disable_query_patch_embedding=False,
    disable_query_timestep_embedding=False,
    disable_cross_attention=False,
    debug_3d_head_mode='linear',
)
```

### 完整前向传播步骤

```
输入: encoder_features (B, N_enc, D)
      frames (B, T, C, H, W)
      coords (B, N_q, 2)   [u,v ∈ [0,1]]
      t_src  (B, N_q)      long
      t_tgt  (B, N_q)      long
      t_cam  (B, N_q)      long
      local_patches (B, N_q, 3, 9, 9) 可选

─── build_query() ─────────────────────────────────────────

Step 1: FourierEmbedding(coords) → coord_emb   (B, N_q, D)
  coords (fp32) × freqs [2^0..2^63] × 2π → sin/cos
  → fourier_features (B, N_q, 256) → Linear(256, D)

Step 2: TimestepEmbedding(t_src, t_tgt, t_cam)
  src_emb = src_embedding(t_src)   (B, N_q, D)   nn.Embedding(128, D)
  tgt_emb = tgt_embedding(t_tgt)   (B, N_q, D)   nn.Embedding(128, D)
  cam_emb = cam_embedding(t_cam)   (B, N_q, D)   nn.Embedding(128, D)

Step 3: PatchEmbeddingFast(frames, coords, t_src, local_patches)
  → patch_emb (B, N_q, D)
  [见 patch provider 章节]

Step 4: 求和 + 可学习 base token
  raw = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb
  raw = raw + query_token.expand(B, N_q, D)    # nn.Parameter(1,1,D)

Step 5: query_mlp（非线性混合）
  query = Linear(D,D) → GELU → Linear(D,D)    (B, N_q, D)

─── DecoderBlock × depth ──────────────────────────────────

Step 6: 对每层 DecoderBlock:
  memory = LayerNorm(encoder_features)         # (B, N_enc, D)
  q_norm = LayerNorm(query)                    # (B, N_q, D)

  # 交叉注意力: 查询 attend 到编码器全部 token
  q_proj: Linear(D, D)  → Q: (B, N_q, H, D/H)
  k_proj: Linear(D, D)  → K: (B, N_enc, H, D/H)
  v_proj: Linear(D, D)  → V: (B, N_enc, H, D/H)
  attn_out = SDPA(Q, K, V)                    # (B, N_q, D)

  query = query + proj(attn_out)               # 残差
  query = query + MLP(LayerNorm(query))        # FFN 残差

Step 7: LayerNorm(query)    (B, N_q, D)

─── 输出头 ────────────────────────────────────────────────

Step 8: 六个并行线性层
  pos_3d      = head_3d(query)                # (B, N_q, 3)
  delta_2d    = head_2d(query)                # (B, N_q, 2)  残差
  pos_2d      = coords + delta_2d             # identity 初始化友好
  visibility  = head_vis(query)               # (B, N_q, 1)  logit
  displacement= head_disp(query)              # (B, N_q, 3)
  normal_raw  = head_normal(query)            # (B, N_q, 3)
  normal      = F.normalize(normal_raw, dim=-1)
  s           = head_conf(query)              # (B, N_q, 1)
  s           = clamp(s, min=-5.0, max=10.0)
```

### 查询 Token 各组成部分详解

#### 坐标嵌入：FourierEmbedding

```python
class FourierEmbedding(nn.Module):
    # freqs: [1, 2, 4, ..., 2^63]  (共 64 个，注册为 buffer)
    # 投影层
    proj = nn.Linear(256, embed_dim)   # 256 = 2坐标 × 2(sin+cos) × 64频率

    def forward(self, coords):  # coords: (B, N, 2) fp32 强制
        # coords_freq: (B, N, 2, 64) = coords × freqs × 2π
        fourier = cat([sin(coords_freq), cos(coords_freq)], dim=-1)  # (B, N, 2, 128)
        fourier = fourier.reshape(B, N, 256)
        return proj(fourier)   # (B, N, D)
```

低频率（freqs[0]=1）捕获全局位置，高频率（freqs[63]=2^63）捕获精细空间细节。

#### 时间步嵌入：TimestepEmbedding

```python
class TimestepEmbedding(nn.Module):
    src_embedding = nn.Embedding(128, D)   # 源帧索引嵌入
    tgt_embedding = nn.Embedding(128, D)   # 目标帧索引嵌入
    cam_embedding = nn.Embedding(128, D)   # 相机帧索引嵌入
    # 初始化: normal(std=0.02)
```

三组完全独立的查找表，保证 t_src/t_tgt/t_cam 语义解耦。
参数量：3 × 128 × D（base: 3 × 128 × 768 = 294,912 参数）。

#### 局部 Patch 嵌入：PatchEmbeddingFast

```python
class PatchEmbeddingFast(nn.Module):
    mlp = Sequential(
        Linear(9×9×3, D),   # 243 → D
        GELU(),
        Linear(D, D),
    )
    offsets = meshgrid(-4..4, -4..4)    # (9, 9, 2) 偏移量 buffer

    def forward(self, frames, coords, t_src, local_patches=None):
        if local_patches is not None:
            return embed_patches(local_patches)   # 直接嵌入
        # 否则实时提取（调用 extract_local_patches）
        patches = extract_local_patches(frames, coords, t_src, patch_size=9)
        return embed_patches(patches)    # (B, N_q, D)
```

`extract_local_patches`（`utils/patches.py`）使用 **5D grid_sample**：
```python
# input_5d: (B, C, T, H, W)
# grid_xyz: (B, N_q, 9, 9, 3)  = [x_offset, y_offset, t_normalized]
patches = F.grid_sample(input_5d, grid_xyz,
    mode='bilinear', padding_mode='border', align_corners=True)
# → (B, C, N_q, 9, 9) → permute → (B, N_q, C, 9, 9)
```

高分辨率 patch 提取（`sampled_highres`）使用 `extract_local_patches_with_valid_hw`，
额外接受 `valid_hw (B, 2)` 参数，将坐标限制在实际裁剪区域内再缩放到填充后的张量坐标系。

#### 可学习基础 Query Token

```python
query_token = nn.Parameter(torch.zeros(1, 1, D))   # 初始化 trunc_normal(std=0.02)
# 广播到 (B, N_q, D) 后与其他嵌入相加
```

#### Query MLP（混合层）

```python
query_mlp = Sequential(
    Linear(D, D),
    GELU(),
    Linear(D, D),
)
```

将五个嵌入项求和后做非线性变换，而不是直接输入 DecoderBlock，给模型更强的特征混合能力。

### DecoderBlock 结构

```python
class DecoderBlock(nn.Module):
    norm1    = nn.LayerNorm(D)    # 归一化 query
    norm_kv  = nn.LayerNorm(D)    # 归一化 encoder features
    cross_attn = CrossAttention(D, num_heads, ...)
    norm2    = nn.LayerNorm(D)
    mlp      = MLP(D, D*mlp_ratio)

    def forward(self, query, encoder_features):
        if not disable_cross_attention:
            memory = norm_kv(encoder_features)
            query = query + cross_attn(
                query=norm1(query),   # Q
                key=memory,           # K
                value=memory,         # V
            )
        query = query + mlp(norm2(query))
        return query
```

**关键设计：无查询自注意力**。N_q 个查询点在所有 DecoderBlock 中均独立处理，Q 只通过交叉注意力感知全局场景 F，而不感知其他查询点。这使得：
- 查询数量在推理时可灵活变化
- 可以任意分块（chunking）处理大批量查询而不影响结果

### CrossAttention 实现

```python
class CrossAttention(nn.Module):
    q_proj = nn.Linear(D, D, bias=True)    # 仅 Query 的 Q 投影
    k_proj = nn.Linear(D, D, bias=True)    # encoder_features 的 K 投影
    v_proj = nn.Linear(D, D, bias=True)    # encoder_features 的 V 投影
    proj   = nn.Linear(D, D)               # 输出投影

    def forward(self, query, key, value=None, mask=None):
        # query: (B, N_q, D)      N_q 通常 2048
        # key:   (B, N_enc, D)    N_enc = 6144 (base)
        Q = q_proj(query).reshape(B, N_q,  H, head_dim).transpose(1,2)
        K = k_proj(key).reshape(B, N_enc, H, head_dim).transpose(1,2)
        V = v_proj(key).reshape(B, N_enc, H, head_dim).transpose(1,2)
        attn = SDPA(Q, K, V)    # (B, H, N_q, head_dim)
        return proj(attn.transpose(1,2).reshape(B, N_q, D))
```

注意力矩阵形状：`(B, H, N_q, N_enc)` = `(B, 12, 2048, 6144)`（base）

### 输出头详解

| 头           | 网络结构                              | 输出形状    | 语义与后处理                                                  |
|-------------|---------------------------------------|------------|---------------------------------------------------------------|
| `head_3d`   | `Linear(D, 3)` 或 `Linear→ReLU→Linear→ReLU→Linear`（mlp256）| `(B,N_q,3)` | 相机坐标 (X,Y,Z)，单位米，t_cam 帧坐标系       |
| `head_2d`   | `Linear(D, 2)`                        | `(B,N_q,2)` | 预测**残差**：pos_2d = coords + delta_2d，使恒等映射易于学习  |
| `head_vis`  | `Linear(D, 1)`                        | `(B,N_q,1)` | logit，`sigmoid(vis) > 0.5` 为可见；BCEWithLogits loss         |
| `head_disp` | `Linear(D, 3)`                        | `(B,N_q,3)` | 3D 运动位移向量 = pos_3d@t_cam - pos_3d@src_cam               |
| `head_normal`| `Linear(D, 3)` + `F.normalize`       | `(B,N_q,3)` | L2 归一化单位法线向量                                         |
| `head_conf` | `Linear(D, 1)` + `clamp(-5, 10)`     | `(B,N_q,1)` | Kendall 置信度参数 $s = -\log(\sigma)$，裁剪避免梯度消失/爆炸 |

**置信度参数说明**：
采用 Kendall & Gal (2017) 的对数参数化，$s = -\log(\text{conf})$，则 $\text{conf} = e^{-s}$。损失中置信度作为 3D 误差的权重：$L_{3d} \propto e^{-s} \cdot \|y_{pred} - y_{gt}\| + s$。
裁剪范围 `[-5, 10]`：下限 -5 使最大置信权重 $e^5 \approx 148$（之前 -2 → 7.4，过于宽松），上限 10 防止 s 无限增大逃避 3D 学习。

### Patch Provider 策略详解

| `patch_provider`       | 数据来源        | 需要 `local_patches` | 需要 `transform_metadata` | 速度  |
|------------------------|----------------|---------------------|---------------------------|-------|
| `auto`                 | 自动判断        | 可选                 | 可选                       | -     |
| `sampled_resized`      | 实时从缩放帧采样 | 否                   | 否                         | 中    |
| `precomputed_resized`  | Dataset 预计算  | **必须**             | 否                         | 最快  |
| `sampled_highres`      | 实时从原始裁剪帧采样 | 否              | **必须**                   | 慢    |
| `precomputed_highres`  | Dataset 预计算（高分辨率）| **必须** | 否                         | 快    |

`transform_metadata` 包含字段：
```python
{
    'canonical_space':  (B, 1),    # 必须为 0（crop-normalized 坐标）
    'original_hw':      (B, 2),    # 原始图像分辨率
    'crop_offset_xy':   (B, 2),    # 裁剪偏移（像素）
    'crop_size_hw':     (B, 2),    # 裁剪区域实际大小
    'resized_hw':       (B, 2),    # 缩放后分辨率
}
```

---

## 嵌入模块总览

**文件**：`models/embeddings.py`

### FourierEmbedding（详细实现）

```python
class FourierEmbedding(nn.Module):
    # 频率：等比数列，公比 2
    freqs = 2.0 ** arange(0, 64)   # [1, 2, 4, ..., 9.22e18]，register_buffer
    proj  = nn.Linear(256, embed_dim)

    def forward(self, coords):  # (B, N, 2)
        # 强制 fp32 避免高频项溢出
        coords_fp32 = coords.to(torch.float32)
        # 展开频率维度: (B, N, 2, 64)
        coords_freq = coords_fp32.unsqueeze(-1) * freqs * 2π
        # 正弦余弦特征: (B, N, 2, 128)
        fourier = cat([sin(coords_freq), cos(coords_freq)], dim=-1)
        # 展平: (B, N, 256) → 投影回 fp16/bf16
        return proj(fourier.reshape(B, N, 256).to(proj.weight.dtype))
```

参数量：256 × D + D（base: 256 × 768 + 768 ≈ 197K）

### TimestepEmbedding（详细实现）

```python
class TimestepEmbedding(nn.Module):
    src_embedding = nn.Embedding(max_T=128, embed_dim=D)
    tgt_embedding = nn.Embedding(max_T=128, embed_dim=D)
    cam_embedding = nn.Embedding(max_T=128, embed_dim=D)
    # 初始化: normal(std=0.02)

    def forward(self, t_src, t_tgt, t_cam):
        # 各自独立查表
        return (
            src_embedding(t_src),   # (B, N_q, D)
            tgt_embedding(t_tgt),   # (B, N_q, D)
            cam_embedding(t_cam),   # (B, N_q, D)
        )
```

三组嵌入表参数量：3 × 128 × D（base: 3 × 128 × 768 ≈ 295K）

### PatchEmbeddingFast vs PatchEmbedding

| 特性 | `PatchEmbeddingFast`（**实际使用**）| `PatchEmbedding`（旧版，参考实现）|
|------|-----------------------------------|---------------------------------|
| Patch 提取方式 | 5D `F.grid_sample`（向量化）| 双重 Python for loop（慢）|
| 边界处理 | `padding_mode='border'` 自动 | 手动 `F.pad` 分支逻辑 |
| 支持预计算 patch | ✅ `embed_patches()` 方法 | ❌ |
| 批量处理 | 所有 (B×N_q) 并行 | 逐 B、逐 N 串行 |

---

## 注意力实现与 GPU 兼容性

编码器（`EfficientAttention`）和解码器（`CrossAttention`）均统一使用：

```python
x = F.scaled_dot_product_attention(q, k, v,
    dropout_p=self.attn_drop if self.training else 0.0
)
```

PyTorch 自动选择最优后端：
1. **FlashAttention**（首选）：IO-aware，显存 O(N) 而非 O(N²)
2. **Memory-efficient**：无 FlashAttention 时的备选
3. **Math**（标准实现）：精确但显存开销大

**Blackwell GPU 适配（B100/B200/B300，sm_103+）**：
```python
def _is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

if _is_blackwell():
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(...)
```
原因：PyTorch 2.11 中 Blackwell 的 cuDNN frontend 与 flash/mem-efficient backend 存在兼容问题（kernel 编译失败），强制回退到 math backend。训练脚本 `train.py` 在启动时也全局设置：
```python
if cap[0] >= 10:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
```

---

## 稠密追踪：DenseTracker（Algorithm 1）

**文件**：`models/dense_tracking.py`

### 算法思想

朴素方法追踪所有像素需要为每个 (t_src, t_tgt) 对发起 H×W 次查询，总计 T²×H×W 次。
DenseTracker 利用**轨迹的传递性**：若点 P 在 t=0 时可见且在 t=3 时也可见，则 t=3 时该像素已被"追踪到"，无需再作为独立查询起点。

通过维护**占用网格 G**（标记哪些 (t, y, x) 位置已被访问），避免冗余查询：

```
Algorithm 1: DenseTracker.track_all_pixels

G ← zeros(T, H_eff, W_eff, bool)   # 占用网格
tracks ← []

while not G.all():
    # 采样 batch_size 个未访问像素作为新轨迹起点
    unvisited = nonzero(~G)
    batch_idx = unvisited[randperm(len(unvisited))[:batch_size]]
    t_src, y_idx, x_idx = batch_idx.T

    # 构建归一化坐标
    u = (x_idx * stride + stride/2) / W
    v = (y_idx * stride + stride/2) / H
    coords = stack([u, v], dim=-1).unsqueeze(0)   # (1, n, 2)

    # 遍历所有目标帧，运行解码器
    for t_tgt in range(T):
        outputs = model.decode(F, frames, coords,
            t_src, t_tgt, t_cam=t_tgt)
        # 记录 pos_3d, pos_2d, sigmoid(visibility)

    # 更新占用网格：可见轨迹点标记为已访问
    for each track i, each frame t:
        if vis[i,t] > threshold:
            (x_grid, y_grid) = pos_2d[i,t] 映射到网格坐标
            G[t, y_grid, x_grid] = True

    # 源点本身也标记为已访问
    G[t_src, y_idx, x_idx] = True

    tracks.append((pos_3d, pos_2d, visibility, ...))
```

### 复杂度分析

设平均可见性比率为 r（每个轨迹平均覆盖 r×T 个帧）：
- 朴素法：O(T² × H × W) 次解码器调用
- DenseTracker：O(T × H × W / r) 次（每次解码 T 帧）
- 实际加速比：T × r ≈ 5–15×（r 约为 0.3–0.7）

### 两种追踪模式

```python
# 模式 1：每帧在各自相机坐标系下
tracks = tracker.track_all_pixels(video, reference_frame, ...)
# t_cam = t_tgt（每帧用自身相机坐标系），pos_3d 含义随帧变化

# 模式 2：统一世界坐标系
tracks = tracker.track_all_pixels_to_world(video, reference_frame=0, ...)
# t_cam = reference_frame（固定），所有 pos_3d 在同一坐标系下可直接比较
```

---

## 参数量估算（base 规格）

| 组件                         | 参数量（约）  |
|------------------------------|-------------|
| PatchEmbed3D (Conv3D)        | 1.2M        |
| FactorizedPositionEncoding3D | 43K         |
| aspect_ratio_embed           | 1K          |
| EncoderStage × 6             |             |
| - LocalBlock × 6             | 6 × 7.1M = 42.6M |
| - GlobalBlock × 6            | 6 × 7.1M = 42.6M |
| Encoder LayerNorm            | 1.5K        |
| **编码器合计**               | **~86M**    |
| FourierEmbedding             | 197K        |
| TimestepEmbedding × 3        | 295K        |
| PatchEmbeddingFast           | 383K        |
| query_token                  | 0.8K        |
| query_mlp                    | 1.2M        |
| DecoderBlock × 6             | 6 × 14.2M = 85M |
| Decoder LayerNorm            | 1.5K        |
| 6 输出头                     | 768×(3+2+1+3+3+1) = 10K |
| **解码器合计**               | **~87M**    |
| **D4RT base 总计**           | **~173M**   |

> 每个 EncoderBlock（LocalBlock 或 GlobalBlock）参数量：
> - Attn QKV: D × 3D = 3D² ≈ 1.77M (base)
> - Attn proj: D × D ≈ 0.59M
> - MLP fc1/fc2: D×4D + 4D×D = 8D² ≈ 4.72M
> - LayerNorm × 2: 2 × 2D ≈ 3K
> - **合计 ≈ 7.09M** (base)

---

## 前向传播完整数据流

以 base 规格，B=1，T=48，H=W=256，N_q=2048 为例：

```
输入 video:  (1, 48, 3, 256, 256)     → 75.5M fp32 元素 ≈ 290MB

─── 编码器 ───────────────────────────────────────────────────────────────

canonicalize_video → (1, 3, 48, 256, 256)  [BCTHW]

PatchEmbed3D:
  Conv3D → (1, 768, 24, 16, 16)
  flatten + transpose → (1, 6144, 768)
  reshape → (1, 24, 256, 768)              [B, T', P, D]

+ FactorizedPosEmbed: (1, 24, 256, 768)   [广播相加]

+ ar_token: (1, 24, 1, 768) → cat → (1, 24, 257, 768)

EncoderStage × 6:
  LocalBlock:  reshape→(24, 257, 768)  → Self-Attn → reshape back
  GlobalBlock: reshape→(1, 6168, 768)  → Self-Attn → reshape back

LayerNorm → (1, 24, 257, 768)
去掉 ar_token → (1, 24, 256, 768)
flatten → (1, 6144, 768)                   encoder_features

─── 解码器 ───────────────────────────────────────────────────────────────

build_query(frames, coords, t_src, t_tgt, t_cam):
  coords (1, 2048, 2)
  FourierEmb → coord_emb (1, 2048, 768)
  TimestepEmb → src/tgt/cam_emb (1, 2048, 768) × 3
  PatchEmb → patch_emb (1, 2048, 768)  [5D grid_sample on (1,3,48,256,256)]
  sum + query_token → (1, 2048, 768)
  query_mlp → query (1, 2048, 768)

DecoderBlock × 6:
  CrossAttn:
    Q: (1, 2048, 768) → (1, 12, 2048, 64)
    K,V: (1, 6144, 768) → (1, 12, 6144, 64)
    SDPA(Q, K, V) → (1, 12, 2048, 64) → (1, 2048, 768)
  MLP: (1, 2048, 768) → (1, 2048, 3072) → (1, 2048, 768)

LayerNorm → (1, 2048, 768)

6 个线性头 → 各自输出，pos_2d = coords + delta_2d

─── 输出 ────────────────────────────────────────────────────────────────

pos_3d:      (1, 2048, 3)
pos_2d:      (1, 2048, 2)
visibility:  (1, 2048, 1)
displacement:(1, 2048, 3)
normal:      (1, 2048, 3)  [L2 归一化]
confidence:  (1, 2048, 1)  [clamp(-5,10)]
```

---

## 使用示例

```python
from models import create_d4rt, DenseTracker, DenseTrackingConfig
import torch

# ── 创建模型 ────────────────────────────────────────────────
model = create_d4rt(
    variant='base',
    img_size=256,
    temporal_size=48,
    patch_size=(2, 16, 16),
    query_patch_size=9,
)
model = model.cuda().eval()

# ── 两阶段推理（多次解码共享编码结果）──────────────────────
B, T, H, W = 1, 48, 256, 256
video = torch.randn(B, T, 3, H, W).cuda()        # (B, T, C, H, W)
ar    = torch.tensor([[1.0]]).cuda()              # 正方形，aspect=1

with torch.no_grad():
    features = model.encode(video, ar)            # (1, 6144, 768)，只算一次

    # 查询 1：点追踪（t_src 固定，t_tgt 变化）
    N = 512
    coords  = torch.rand(B, N, 2).cuda()
    t_src   = torch.zeros(B, N, dtype=torch.long).cuda()
    for t in range(T):
        t_tgt = torch.full((B, N), t, dtype=torch.long).cuda()
        t_cam = t_tgt.clone()
        preds = model.decode(features, video, coords, t_src, t_tgt, t_cam)
        # preds['pos_3d']: (1, 512, 3) 该帧 3D 位置

    # 查询 2：深度图（t_src=t_tgt=t_cam，全像素网格）
    u = torch.linspace(0, 1, W).cuda()
    v = torch.linspace(0, 1, H).cuda()
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
    coords_depth = torch.stack([grid_u, grid_v], -1).reshape(1, -1, 2)
    t_all = torch.zeros(1, H*W, dtype=torch.long).cuda()   # 第 0 帧
    preds_depth = model.decode(features, video, coords_depth, t_all, t_all, t_all)
    depth = preds_depth['pos_3d'][..., 2].reshape(1, H, W)  # Z 分量即深度

# ── 稠密追踪 ─────────────────────────────────────────────────
tracker = DenseTracker(model, DenseTrackingConfig(
    batch_size=256,
    visibility_threshold=0.5,
    spatial_stride=2,    # 每隔一像素采样
))
video_hwc = video.permute(0, 1, 3, 4, 2)    # (B, T, H, W, C)
tracks = tracker.track_all_pixels(video_hwc, reference_frame=0)
print(tracks['tracks_3d'].shape)   # (N_tracks, 48, 3)
print(tracks['visibility'].shape)  # (N_tracks, 48)
```
