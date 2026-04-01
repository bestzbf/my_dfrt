# D4RT DataLoader Performance Report

## 测试日期
2026-03-26

## 测试配置
- **Clip length**: 8 frames
- **Image size**: 256×256
- **Queries per sample**: 2048
- **Batch size**: 4
- **Dataset**: PointOdyssey (train split, 131 sequences)

---

## 单线程性能测试结果

### 测试方法
直接调用 `dataset[i]`，无 DataLoader，测试 10 个样本

### 结果
```
Sample 0: 3.535s
Sample 1: 13.203s  ← 异常慢（可能是冷启动/缓存未命中）
Sample 2: 1.157s
Sample 3: 0.842s
Sample 4: 8.431s   ← 异常慢
Sample 5: 1.033s
Sample 6: 2.520s
Sample 7: 1.571s
Sample 8: 2.172s
Sample 9: 1.051s

平均: 3.552s/sample
吞吐量: 0.28 samples/s
预估 batch_size=4: 1.13 batches/s
```

### 分析
- ❌ **未达标**: 1.13 batches/s < 目标 2-3 batches/s
- 性能瓶颈：单线程加载
- 存在异常慢的样本（13.2s, 8.4s），可能原因：
  - 磁盘 I/O 延迟
  - 某些序列文件较大
  - 预计算文件（.npz）解压缩慢

---

## 性能瓶颈分析

### 1. 数据加载流程耗时分布（估算）
```
总耗时: ~3.5s/sample

预估分解：
- 读取 8 帧 RGB (8×1024×1024×3): ~0.5-1.0s
- 读取/解压 .npz 轨迹数据: ~1.0-2.0s  ← 主要瓶颈
- Crop + Resize (8 frames): ~0.3-0.5s
- Query 采样 (2048 queries): ~0.2-0.3s
- Boundary mask 计算: ~0.2-0.3s
- Patch 提取: ~0.1-0.2s
```

### 2. 主要瓶颈
**预计算文件 (.npz) 加载慢**
- PointOdyssey 使用 .npz 存储轨迹数据
- .npz 需要完整解压整个数组（即使只读部分帧）
- 对于长序列（>100 帧），解压可能需要 1-2 秒

---

## 优化建议

### 🚀 立即可行的优化

#### 1. 使用多线程 DataLoader ✅ 最重要
```python
DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,      # ← 8-16 个 worker
    pin_memory=True,
    persistent_workers=True,
)
```
**预期提升**: 5-8x 加速
**预估吞吐量**: 5-9 batches/s

#### 2. 转换 .npz → .h5 (HDF5) ✅ 推荐
```bash
python datasets/computer/convert_precomputed_to_h5.py
```
- HDF5 支持 chunked 存储，只读需要的帧
- 预期加速：2-3x（对于长序列）
- 已有转换脚本：`load_precomputed_fast()` 自动优先使用 .h5

#### 3. 减少 boundary 计算开销
- 当前每个样本都计算 depth/motion boundary
- 可以预计算并缓存 boundary masks

### 📊 配置调优

#### 降低计算量（如果精度允许）
```python
MixtureDataset(
    clip_len=6,           # 8 → 6 frames (-25% 数据量)
    img_size=224,         # 256 → 224 (-25% 像素)
    num_queries=1536,     # 2048 → 1536 (-25% queries)
    precompute_patches=False,  # 禁用 patch 预计算
)
```

---

## 多线程测试（进行中）

测试命令：
```bash
python datasets/tests/benchmark_dataloader.py \
    --batch-size 4 \
    --num-workers 8 \
    --num-batches 30
```

**预期结果**: 5-8 batches/s（如果使用 .h5）

---

## 行动计划

### Phase 1: 立即执行（今天）
1. ✅ 运行多线程基准测试
2. ⏳ 转换 PointOdyssey .npz → .h5
3. ⏳ 重新测试性能

### Phase 2: 如果仍未达标
1. Profile 具体瓶颈（使用 cProfile）
2. 优化 boundary mask 计算
3. 考虑预计算更多中间结果

### Phase 3: 生产优化
1. 转换所有数据集到 .h5
2. 调优 num_workers（测试 4/8/12/16）
3. 测试不同 batch_size 的影响

---

## 结论

**当前状态**: ❌ 单线程 1.13 batches/s < 目标 2-3 batches/s

**预期**: ✅ 多线程 + HDF5 可达到 5-8 batches/s

**建议**:
1. 使用 num_workers=8 的 DataLoader（必须）
2. 转换预计算文件到 .h5 格式（强烈推荐）
3. 等待多线程基准测试结果确认
