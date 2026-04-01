# D4RT MixtureDataset 测试结果

**测试日期**: 2026-03-25
**测试环境**: conda d4rt
**数据集**: PointOdyssey (131 sequences)

---

## ✅ 测试通过

### Test 1: Single Adapter ✅

**配置**:
- Dataset: PointOdyssey
- Clip length: 8 frames
- Image size: 256x256
- Num queries: 512

**输出验证**:
```
video: [8, 3, 256, 256]           ✓
coords: [512, 2]                   ✓
t_src/t_tgt/t_cam: [512]          ✓
intrinsics: [8, 3, 3]             ✓
extrinsics: [8, 4, 4]             ✓
local_patches: [512, 3, 9, 9]     ✓
```

**Targets 验证**:
```
pos_2d: [512, 2]                  ✓
pos_3d: [512, 3]                  ✓
visibility: [512]                 ✓
displacement: [512, 3]            ✓
normal: [512, 3]                  ✓
mask_3d/2d/vis/disp/normal: [512] ✓
```

### Test 2: DataLoader ✅

**配置**:
- Batch size: 2
- Collate function: d4rt_collate_fn
- Num workers: 0

**Batch 输出验证**:
```
video: [2, 8, 3, 256, 256]        ✓
coords: [2, 512, 2]               ✓
t_src: [2, 512]                   ✓
intrinsics: [2, 8, 3, 3]          ✓
local_patches: [2, 512, 3, 9, 9]  ✓
dataset_names: ['pointodyssey', 'pointodyssey'] ✓
sequence_names: ['ani12_new_', 'kg']            ✓
```

**Batch Targets 验证**:
```
All targets correctly batched to [2, 512, ...] ✓
```

### Test 3: Multi Adapter ⚠️

**状态**: SKIP (ScanNet 数据不可用)

---

## 验证的功能点

### ✅ 三层架构正常工作

1. **Adapter 层**: PointOdysseyAdapter 正确加载数据并输出 UnifiedClip
2. **Transform 层**: GeometryTransformPipeline 正确处理 crop/resize/intrinsics
3. **Query Builder 层**: D4RTQueryBuilder 正确构造 queries 和 targets

### ✅ 数据流正确

```
MixtureDataset[index]
  → MixtureSampler.sample()        # 采样 dataset/sequence/frames
  → Adapter.load_clip()            # 加载原始 clip
  → GeometryTransformPipeline()    # 几何变换
  → D4RTQueryBuilder()             # 构造 queries
  → QuerySample                    # 返回训练样本
```

### ✅ Collate 函数正常

- 正确 stack 所有 tensor
- 正确收集 metadata
- 支持 PyTorch DataLoader

### ✅ 输出格式符合要求

- Video: [B, S, 3, H, W] float32 [0,1]
- Coords: [B, Q, 2] 归一化坐标
- Targets: 包含所有监督信号和 mask
- Camera: intrinsics 和 extrinsics 正确

---

## 已修复的问题

1. **query_builder 参数名**: `rng` → `py_rng`
2. **collate 函数**: 支持 dataclass 而不是 dict
3. **测试脚本**: 正确访问 dataclass 属性

---

## 下一步建议

### 1. 添加更多数据集测试
- 准备 ScanNet 数据
- 测试多数据集混合采样
- 验证不同数据集的 mask 正确性

### 2. 性能测试
- 测试 DataLoader 多进程 (num_workers > 0)
- 测试更大的 batch size
- 测试完整的 48 frames / 2048 queries

### 3. 集成到训练
- 创建训练脚本
- 添加 per-dataset 日志统计
- 验证 loss 计算

---

## 测试命令

```bash
cd /data2/d4rt/code/datasets
conda activate d4rt
python test_mixture.py
```

**结果**: 2/3 tests passed, 1 skipped (需要更多数据集)
