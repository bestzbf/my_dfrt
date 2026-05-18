# COS 盘数据集加载指南

## 概述

本指南说明如何使用新修改的 **Planned Mode** 从腾讯云 COS 盘加载数据集进行训练。Planned Mode 通过预先规划采样顺序并使用后台进程预取样本，解决了传统 DataLoader 多 worker 模式下的 CUDA fork 问题和 I/O 瓶颈。

---

## 核心特性

### ✅ 已修复并验证的功能

1. **单 RNG 设计**：每个样本使用独立 RNG，匹配 online mode 的 per-sample RNG 结构
2. **DDP padding**：与 PyTorch DistributedSampler 一致，padding 样本复用早期索引（不创造新样本）
3. **Generation 隔离**：epoch 切换时完全重建 pipeline，零交叉污染
4. **Forkserver 安全**：避免 CUDA fork 问题，builder 进程在独立地址空间运行
5. **端到端验证**：28 个测试覆盖 planning、spool、builder、DataLoader 集成

### 📊 测试覆盖

- 12 个 planning 单元测试（determinism、multi-rank、padding、RNG 语义）
- 10 个 spool 单元测试（generation isolation、disk watermark）
- 3 个 builder 集成测试（forkserver、generation gate、并发）
- 3 个 dataset 端到端测试（DataLoader + set_epoch + 多 epoch）

---

## 快速开始

### 1. 基本用法（单机单卡）

```python
from datasets.factory import create_mixture_dataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader, SequentialSampler

# 创建 planned mode 数据集
dataset = create_mixture_dataset(
    config_path="configs/mixture_4datasets_cos.yaml",
    split="train",
    planned_mode=True,           # 启用 planned mode
    spool_dir="/tmp/d4rt_spool", # 临时文件目录
    builder_workers=2,           # 后台 builder 进程数
    prefetch_depth=32,           # 预取队列深度
)

# 使用 SequentialSampler（planned mode 已内置 shuffle）
loader = DataLoader(
    dataset,
    batch_size=4,
    sampler=SequentialSampler(dataset),
    num_workers=0,               # planned mode 不需要 DataLoader workers
    collate_fn=d4rt_collate_fn,
    pin_memory=True,
)

# 训练循环
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)     # 必须：切换 epoch 并重建 pipeline

    for batch in loader:
        video = batch["video"]       # [B, T, 3, H, W]
        coords = batch["coords"]     # [B, Q, 2]
        targets = batch["targets"]   # dict with pos_2d, pos_3d, etc.

        # 训练逻辑...

# 清理（可选，析构函数会自动调用）
dataset.cleanup()
```

### 2. DDP 多卡训练

```python
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

# 初始化 DDP
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 创建数据集（每个 rank 独立创建）
dataset = create_mixture_dataset(
    config_path="configs/mixture_4datasets_cos.yaml",
    split="train",
    planned_mode=True,
    spool_dir=f"/tmp/d4rt_spool_rank{rank}",  # 每个 rank 独立 spool 目录
    builder_workers=2,
    prefetch_depth=32,
    rank=rank,                    # 传入 rank
    world_size=world_size,        # 传入 world_size
)

# DataLoader（不需要 DistributedSampler，planned mode 已内置分片）
loader = DataLoader(
    dataset,
    batch_size=4,
    sampler=SequentialSampler(dataset),
    num_workers=0,
    collate_fn=d4rt_collate_fn,
    pin_memory=True,
)

# 训练循环
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)

    for batch in loader:
        # 训练逻辑...
        pass

dataset.cleanup()
```

---

## 配置文件示例

### `configs/mixture_4datasets_cos.yaml`

```yaml
# 数据集列表
datasets:
  - name: co3dv2
    root: /data/zbf/openclaw/co3dv2  # COS 挂载路径
    split: train
    weight: 0.4

  - name: scannet
    root: /data/zbf/openclaw/scannet
    split: train
    weight: 0.3

  - name: blendedmvs
    root: /data/zbf/openclaw/blendedmvs
    split: train
    weight: 0.2

  - name: kubric
    root: /data/zbf/openclaw/kubric
    split: train
    weight: 0.1

# 采样参数
clip_len: 48                    # 每个 clip 的帧数
img_size: 256                   # 图像分辨率
num_queries: 2048               # query 点数量
boundary_ratio: 0.3             # 边界过采样比例
t_tgt_eq_t_cam_ratio: 0.4       # t_tgt=t_cam 的比例

# Planned mode 参数
epoch_size: 10000               # 每个 epoch 的样本数
seed: 42                        # 随机种子
reshuffle_each_epoch: true      # 每个 epoch 重新 shuffle
max_spool_bytes: 2147483648     # ��盘缓存上限（2GB）
```

---

## 重要参数说明

### Planned Mode 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `planned_mode` | `False` | 是否启用 planned mode（必须设为 `True`） |
| `spool_dir` | `/tmp/d4rt_spool` | 临时文件目录，用于存储预构建的样本 |
| `builder_workers` | `2` | 后台 builder 进程数，建议 2-4 |
| `prefetch_depth` | `32` | 预取队列深度，越大越平滑但占用更多磁盘 |
| `max_spool_bytes` | `2GB` | 磁盘缓存软上限（实际可能略超） |
| `epoch_size` | 必填 | 每个 epoch 的样本数 |
| `seed` | `42` | 随机种子 |
| `reshuffle_each_epoch` | `True` | 每个 epoch 是否重新 shuffle |

### DDP 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rank` | `0` | 当前进程的 rank |
| `world_size` | `1` | 总进程数 |

**注意**：
- `epoch_size` 不能被 `world_size` 整除时，会自动 padding 到 `ceil(epoch_size / world_size) * world_size`
- Padding 样本会复用早期样本（与 PyTorch DistributedSampler 行为一致）

---

## 与 Online Mode 的对比

| 特性 | Online Mode | Planned Mode |
|------|-------------|--------------|
| **DataLoader workers** | 需要（8-16） | 不需要（`num_workers=0`） |
| **CUDA fork 问题** | 存在 | 无（forkserver 隔离） |
| **I/O 瓶颈** | 可能卡顿 | 预取平滑 |
| **采样顺序** | 隐式（worker 调度） | 显式（预先规划） |
| **Locality** | 隐式保留 | 不保留（per-sample RNG） |
| **DDP 支持** | DistributedSampler | 内置分片 |
| **Epoch 切换** | 热切换 | 冷重启（重建 pipeline） |
| **磁盘占用** | 无 | `max_spool_bytes` |

### 已知差异

**Locality 演化不同**：
- Online mode 的 locality（dataset blocks、sequence blocks）通过 DataLoader worker 隐式调度演化，不可复现
- Planned mode 使用独立 per-sample RNG，采样决策确定性可复现，但 locality 模式不同

**这不影响训练效果**，因为：
1. 每个样本的 RNG seed 语义与 online mode 一致
2. Sampling、transform、query 的随机性结构完全匹配
3. 只是样本间的 locality 关联方式不同（online 隐式，planned 显式）

---

## 性能调优

### 1. Builder Workers 数量

```python
# 推荐配置
builder_workers=2   # 单机单卡
builder_workers=4   # 单机多卡（每卡 2 个）
```

**原则**：
- 太少：预取跟不上消费，训练等待 I/O
- 太多：进程切换开销大，磁盘竞争激烈

### 2. Prefetch Depth

```python
# 推荐配置
prefetch_depth=32   # 标准配置
prefetch_depth=64   # I/O 慢时增大
prefetch_depth=16   # 磁盘空间紧张时减小
```

**原则**：
- 越大越平滑，但占用更多磁盘和内存
- 至少应为 `batch_size * 4`

### 3. Spool 目录

```python
# 推荐配置
spool_dir="/dev/shm/d4rt_spool"  # 使用内存文件系统（最快）
spool_dir="/tmp/d4rt_spool"      # 使用本地磁盘
spool_dir="/data/ssd/d4rt_spool" # 使用 SSD
```

**注意**：
- `/dev/shm` 最快但容量有限（通常 64GB）
- 确保 `max_spool_bytes` 不超过可用空间

### 4. Max Spool Bytes

```python
# 推荐配置
max_spool_bytes=2 * 1024**3   # 2GB（标准）
max_spool_bytes=4 * 1024**3   # 4GB（空间充足时）
max_spool_bytes=1 * 1024**3   # 1GB（空间紧张时）
```

**注意**：
- 这是**软上限**，实际可能超出 `~sample_size × builder_workers`（约 20MB）
- 单个样本约 10MB（clip_len=48, img_size=256）

---

## 常见问题

### Q1: 训练时出现 "Builder process did not terminate, killing..." 警告

**原因**：中途打断训练（Ctrl+C）或 epoch 未消费完就切换时，builder 进程可能来不及优雅退出。

**解决**：
- 这是**正常现象**，不影响训练
- 强杀 builder 进程是安全的（forkserver 隔离，无共享状态）
- 正常训练（完整消费 epoch 再 `set_epoch()`）不会触发

### Q2: 磁盘空间不足

**解决**：
1. 减小 `max_spool_bytes`
2. 使用 `/dev/shm`（内存文件系统）
3. 减小 `prefetch_depth`
4. 清理旧的 spool 目录（`rm -rf /tmp/d4rt_spool_*`）

### Q3: 训练速度慢

**诊断**：
```python
import time
start = time.time()
for i, batch in enumerate(loader):
    if i >= 100:
        break
elapsed = time.time() - start
print(f"Throughput: {100 / elapsed:.2f} batches/s")
```

**解决**：
1. 增加 `builder_workers`（2 → 4）
2. 增加 `prefetch_depth`（32 → 64）
3. 使用更快的 spool 目录（SSD 或 `/dev/shm`）
4. 检查 COS 挂载是否正常（`df -h`）

### Q4: DDP 训练时不同 rank 看到相同样本

**原因**：忘记传入 `rank` 和 `world_size`。

**解决**：
```python
dataset = create_mixture_dataset(
    ...,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
```

### Q5: Epoch 切换后训练卡住

**原因**：忘记调用 `dataset.set_epoch(epoch)`。

**解决**：
```python
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)  # 必须调用
    for batch in loader:
        ...
```

---

## 验证集使用

**重要**：验证集**不支持** planned mode，必须使用 online mode。

```python
# 训练集：planned mode
train_dataset = create_mixture_dataset(
    config_path="configs/mixture_4datasets_cos.yaml",
    split="train",
    planned_mode=True,
    ...
)

# 验证集：online mode（自动强制）
val_dataset = create_mixture_dataset(
    config_path="configs/mixture_4datasets_cos.yaml",
    split="val",
    planned_mode=False,  # 或不传（验证集会自动设为 False）
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    num_workers=8,       # 验证集需要 DataLoader workers
    collate_fn=d4rt_collate_fn,
    shuffle=False,
)
```

---

## 测试验证

运行完整测试套件：

```bash
# 激活环境
conda activate d4rt

# 运行所有测试
python tests/test_planning.py
python tests/test_sample_spool.py
python tests/test_builder_integration.py
python tests/test_planned_dataset_e2e.py

# 或一次性运行
for test in tests/test_*.py; do python $test; done
```

预期输出：
```
All planning tests passed!
All spool tests passed!
All builder integration tests passed!
All PlannedMixtureDataset end-to-end tests passed!
```

---

## 架构说明

### Pipeline 流程

```
1. SamplePlanner
   ↓ 生成 SampleSpec (dataset_idx, sequence, frames, rng_state)

2. Input Queue
   ↓ 分发给 builder 进程

3. SampleBuilder (forkserver 子进程)
   ↓ load_clip → transform → query_builder

4. SampleSpool
   ↓ 写入 .building → 重命名为 .ready

5. PlannedMixtureDataset.__getitem__
   ↓ 等待 .ready → 读取 → 返回

6. DataLoader
   ↓ collate → 返回 batch
```

### Generation 隔离

每个 epoch 有独立的 `generation` 标记：
- Epoch 0: generation=0
- Epoch 1: generation=1
- ...

`set_epoch()` 时：
1. 停止旧 builder 进程
2. 清空旧队列
3. 删除旧 generation 的 spool 文件
4. 创建新 builder 进程
5. 生成新 plan

这保证了**零交叉污染**：epoch N 的样本永远不会被 epoch N+1 读到。

---

## 文件结构

```
datasets/
├── adapters/           # 数据集适配器
│   ├── base.py
│   ├── co3dv2.py
│   ├── scannet.py
│   └── ...
├── planning.py         # SamplePlanner（生成采样计划）
├── sample_builder.py   # SampleBuilder（后台构建样本）
├── sample_spool.py     # SampleSpool（磁盘缓存管理）
├── planned_dataset.py  # PlannedMixtureDataset（主入口）
├── mixture.py          # MixtureDataset（online mode）
├── factory.py          # create_mixture_dataset（统一接口）
└── registry.py         # 数据集注册表

tests/
├── test_planning.py              # Planning 单元测试
├── test_sample_spool.py          # Spool 单元测试
├── test_builder_integration.py   # Builder 集成测试
└── test_planned_dataset_e2e.py   # 端到端测试

configs/
├── mixture_4datasets_cos.yaml    # 4 数据集混合配置
└── ...
```

---

## 更新日志

### 2026-04-27

**修复的问题**：
1. ✅ 单 RNG 设计匹配 online mode 结构
2. ✅ DDP padding 复用样本（不创造新 ID）
3. ✅ Generation 隔离防止 epoch 交叉污染
4. ✅ Forkserver 避免 CUDA fork 问题
5. ✅ 端到端测试覆盖 DataLoader + set_epoch
6. ✅ 文档完善（docstring + 本指南）

**测试覆盖**：28 个测试全部通过

**已知限制**：
- Locality 演化与 online mode 不同（已文档化）
- 中途打断 cleanup 可能超时强杀 builder（正常训练不触发）

---

## 联系方式

如有问题，请联系：
- 代码维护者：zbf
- 项目路径：`/data/zbf/openclaw/d4rt`
- Worktree：`.claude/worktrees/cos-acceleration`

---

## 参考资料

- [datasets/README.md](datasets/README.md) - 数据集加载器总览
- [tests/test_planned_dataset_e2e.py](tests/test_planned_dataset_e2e.py) - 端到端测试示例
- PyTorch DistributedSampler 文档：https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
