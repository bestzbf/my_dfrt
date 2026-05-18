下面这份方案，目标不是“先有个能跑的 patch”，而是把这件事在你这份代码里**怎么才能做对**说清楚。

**先给结论**

在这份仓库里，最合理、最稳的第一版不是“精确文件预取 + 本地 mirror”，而是：

- **先按 rank 生成未来一段训练样本的计划**
- **后台独立构建这些样本的本地 bundle**
- **训练 DataLoader 只读本地 bundle**

也就是我前面说的 `sample-bundle prefetch`。

这条路能最大化复用现有代码；但它有一个很重要的前提：

> **planned 模式不可能严格逐 bit 复现当前 `num_workers>0` 在线采样行为。**

不是做不到，而是**当前行为本身就不是单一确定语义**。

---

## 1. 先把当前代码的真实行为讲清楚

### 1.1 当前 sample 是在哪里决定的

当前训练 sample 的主路径在 [datasets/mixture.py](/data/zbf/openclaw/d4rt/datasets/mixture.py:282)：

1. 根据 `index` 生成 `rng`
2. 用 `mixture_sampler.sample(rng)` 选：
   - dataset
   - sequence
   - frame_indices
3. `adapter.load_clip(...)`
4. `transform(clip, rng=rng)`
5. `query_builder(result, py_rng=rng)`

这里最重要的是：

- **同一个 `rng` 同时驱动采样、数据增强、query 采样**
- 所以如果想离线复现当前 sample，不能只记 `dataset/sequence/frame_indices`
- 还要记住**采样结束后的 `rng` 状态**

---

### 1.2 当前 sample 不是纯函数

虽然 `rng` 初始种子来自 `index`，见 [datasets/mixture.py](/data/zbf/openclaw/d4rt/datasets/mixture.py:295)，但 sample 结果不只由 `index` 决定，因为 sampler 有**短程状态**：

- dataset locality 在 [datasets/mixture.py](/data/zbf/openclaw/d4rt/datasets/mixture.py:79)
- sequence/frame locality 在 [datasets/sampling.py](/data/zbf/openclaw/d4rt/datasets/sampling.py:74)

这意味着：

- 当前第 `k` 个 sample 用哪个 scene、哪 48 帧
- 依赖前面若干 sample 的采样结果

所以不能把现在的逻辑理解成：

`index -> 独立 sample`

它其实更像：

`顺序消费 index 序列 -> 由 sampler 状态产生 sample 序列`

---

### 1.3 更大的问题：`num_workers>0` 时，每个 worker 都有自己一份 sampler 状态

这是最关键的细节。

PyTorch `DataLoader` 在 `num_workers>0` 时，会把 dataset 复制到多个 worker。
也就是说现在不是只有一份 `MixtureSampler`，而是**每个 worker 各有一份**。

于是 locality 实际发生在：

- worker 0 自己的样本子序列里
- worker 1 自己的样本子序列里
- worker 2 自己的样本子序列里
- ...

而不是全 rank 全局唯一一条序列。

这直接带来一个结论：

> 当前 `num_workers>0` 的在线采样语义，本身就依赖 DataLoader 把 index 什么时候派给哪个 worker。
> 这个过程带 `prefetch_factor`，而且是动态的，不存在一个唯一“全局精确序列”。

所以如果你让我做“完全精确复现现在多 worker 在线行为”，这是不严谨的要求，因为当前行为本身就不是单义的。

**这件事必须明确。**

---

### 1.4 当前重试逻辑也会改变未来状态

坏样本重试在 [datasets/mixture.py](/data/zbf/openclaw/d4rt/datasets/mixture.py:300)：

- 某次尝试失败
- `reset_locality_state(dataset_idx)`
- 用新 seed 再采一次

这意味着失败不只是“丢当前样本”，还会改变后续 locality 轨迹。

所以如果 planned 模式还想严格复刻“失败之后未来怎么变”，那 planner 就必须知道前面样本是否真的失败。
这会严重削弱“深 lookahead 预取”的意义。

---

## 2. 因此，planned 模式必须定义一套**新的、明确的语义**

这套语义我建议这样定：

### 2.1 规划语义

- 每个 rank 只有**一条显式 sample plan**
- locality 在这条 rank-local plan 上发生
- 不再依赖 DataLoader worker 隐式状态

也就是说，planned 模式下 locality 从“worker 私有隐状态”变成“rank 显式计划”。

这其实更好：

- 可复现
- 可调试
- 可预取
- 不依赖 worker 调度时序

---

### 2.2 失败语义

planned 模式里，不建议保留当前“任意异常都重新采样”的语义。

建议改成两类：

1. **瞬时 I/O 错误**
   - 重试**同一个计划样本**
   - 不改变数据分布

2. **结构性坏数据**
   - 训练直接报错 / 或显式禁用该 sequence 后重建计划
   - 不要 silently resample

理由很简单：

- 对 planned 系统来说，瞬时网络波动不应该改变训练分布
- 当前 blanket resample 更像历史遗留兜底，不是值得保留的“正确语义”

---

## 3. 为什么我选 `sample-bundle prefetch`，不是先做 raw-file prefetch

### 3.1 因为 adapter 差异太大

有些 adapter 很适合做文件级预取：

- `KubricAdapter` 路径清楚，见 [datasets/adapters/kubric.py](/data/zbf/openclaw/d4rt/datasets/adapters/kubric.py:172)
- `PointOdysseyAdapter` 已经有 fast/encoded cache 体系，见 [datasets/adapters/pointodyssey.py](/data/zbf/openclaw/d4rt/datasets/adapters/pointodyssey.py:283)

但有些不适合：

- `ScanNetAdapter.get_num_frames()` 直接返回 `10000`，见 [datasets/adapters/scannet.py](/data/zbf/openclaw/d4rt/datasets/adapters/scannet.py:95)
- `VKITTI2Adapter.get_num_frames()` 也是 `10000`，见 [datasets/adapters/VirtualKitti.py](/data/zbf/openclaw/d4rt/datasets/adapters/VirtualKitti.py:63)
- `ScanNetAdapter.load_clip()` 每次还会重新 `glob` 图像和深度，见 [datasets/adapters/scannet.py](/data/zbf/openclaw/d4rt/datasets/adapters/scannet.py:156)
- `TartanAirAdapter.load_clip()` 每次也在 `glob`，见 [datasets/adapters/TartanAir.py](/data/zbf/openclaw/d4rt/datasets/adapters/TartanAir.py:202)
- `VKITTI2Adapter.load_clip()` 每次也在 `glob`，见 [datasets/adapters/VirtualKitti.py](/data/zbf/openclaw/d4rt/datasets/adapters/VirtualKitti.py:157)

所以你如果第一步就做“统一 file manifest + COS SDK exact fetch”，会先卡在 adapter 整理。

---

### 3.2 `sample-bundle` 方案可以最大复用现有主链路

后台 builder 直接复用现有链路：

- `adapter.load_clip()`
- `transform()`
- `query_builder()`

然后把结果存本地。

这样第一版只需要改：

- 采样权从 worker 收回到 planner
- 训练读取本地 bundle

而不是先统一 10 多个 dataset 的远端对象结构。

---

## 4. 这份仓库里，首版准确方案应该长什么样

我建议分成 **Phase 0 / 1 / 2**。

---

# Phase 0：先把现有 adapter 修到适合 planning

这是必须做的，不然后面 planned 系统会系统性地产生坏样本。

## 4.1 必修一：把 fake `get_num_frames()` 改成真实值

至少先改：

- [datasets/adapters/scannet.py](/data/zbf/openclaw/d4rt/datasets/adapters/scannet.py:95)
- [datasets/adapters/VirtualKitti.py](/data/zbf/openclaw/d4rt/datasets/adapters/VirtualKitti.py:63)

否则 planner 会以为这些 sequence 很长，结果后面大量 build 失败，planned 系统就失去意义。

---

## 4.2 必修二：把“每次 `load_clip` 重新扫目录”的 adapter 改成索引缓存式

首批至少改：

- `ScanNetAdapter`
- `VKITTI2Adapter`
- `TartanAirAdapter`

目标是：

- index 阶段缓存每个 sequence 的 `image_paths / depth_paths / flow_paths / num_frames`
- `load_clip()` 只按 index 直接取 path
- 不要在 sample build 热路径里再 `glob()` / `iterdir()`

否则即使你把训练改成后台 builder，builder 还是会被远端元数据请求拖死。

---

## 4.3 必修三：planned 模式下减少 DataLoader worker

planned 模式不该继续保留现在这种：

- `NUM_WORKERS=32`
- `prefetch_factor=4`

见 [train_kubric_single_gpu.sh](/data/zbf/openclaw/d4rt/train_kubric_single_gpu.sh:10) 和 [train_mixture.py](/data/zbf/openclaw/d4rt/train_mixture.py:87)

planned 后建议：

- `train_loader.num_workers = 0~2`
- `prefetch_factor = 1`
- `persistent_workers = False`

因为远端预取已经挪到 builder 层了，训练 loader 不该再有第二层深预取。

---

# Phase 1：引入 planned sample-bundle 体系

这是真正的主改动。

---

## 5. 需要新增的核心模块

建议新增 4 个文件：

1. `datasets/planning.py`
2. `datasets/sample_builder.py`
3. `datasets/sample_spool.py`
4. `datasets/planned_dataset.py`

---

## 5.1 `datasets/planning.py`

核心类：

### `SamplePlanEntry`
字段建议：

- `epoch: int`
- `rank: int`
- `ordinal: int`
  这个 rank 本 epoch 第几个样本
- `source_index: int`
  原始 dataset index，用于兼容当前 seed 逻辑
- `dataset_idx: int`
- `dataset_name: str`
- `sequence_name: str`
- `frame_indices: list[int]`
- `py_rng_state: bytes`
- `cache_key: str`

这里的 `py_rng_state` 是重点。
planner 在完成 `mixture_sampler.sample(rng)` 后，存 `pickle.dumps(rng.getstate())`。

后面 builder 用：

```python
rng = random.Random()
rng.setstate(pickle.loads(entry.py_rng_state))
```

再进入：

- `transform(clip, rng=rng)`
- `query_builder(result, py_rng=rng)`

这样 augment 和 query 采样就能精确复现当前逻辑。

---

### `RankSamplePlanner`

它不应该复用运行中 dataset 的 `mixture_sampler`，因为那会和 worker 状态打架。
应该用**一份独立 planner sampler**。

它负责：

- 给定当前 rank 的 `source_indices`
- 顺序生成 `SamplePlanEntry`

---

## 5.2 `datasets/sample_builder.py`

核心类：

### `SampleBuilder`

它内部持有：

- adapters
- transform
- query_builder
- clip_len

方法：

### `build(entry: SamplePlanEntry) -> QuerySample`

流程：

1. `adapter = adapters[entry.dataset_idx]`
2. `clip = adapter.load_clip(entry.sequence_name, entry.frame_indices)`
3. `rng.setstate(entry.py_rng_state)`
4. `result = transform(clip, rng=rng)`
5. `sample = query_builder(result, py_rng=rng)`
6. 校验 `sample.video.shape[0] == clip_len`

这条路径完全复用现有行为。

---

## 5.3 `datasets/sample_spool.py`

它负责本地磁盘协议。

目录建议：

```text
<spool_root>/
  epoch_00012/
    rank_00/
      000/
        00000000.pt
        00000000.ready
        00000001.pt
      001/
        ...
```

为什么分桶：

- 避免单目录几千上万文件

每个样本写法：

1. 写临时文件 `xxx.tmp`
2. `torch.save(...)`
3. `os.replace(tmp, final)`
4. 写 `.ready`

这样训练侧只认 `.ready`

---

## 5.4 `datasets/planned_dataset.py`

核心类：

### `PlannedDataset`

字段：

- `plan_entries`
- `spool_root`
- `wait_timeout_s`
- `poll_interval_s`

`__getitem__(ordinal)` 只做：

1. 等待该 ordinal 的 `.ready`
2. `torch.load(...)`
3. 重建 `QuerySample`
4. 返回给现有 `d4rt_collate_fn`

这样 collate 和训练主逻辑基本不动。

---

## 6. 本地 bundle 到底存什么

第一版建议直接存**最终 `QuerySample` 所需字段**：

- `video`
- `highres_video`
- `depths`
- `normals`
- `coords`
- `t_src`
- `t_tgt`
- `t_cam`
- `intrinsics`
- `extrinsics`
- `targets`
- `local_patches`
- `transform_metadata`
- `aspect_ratio`
- `dataset_name`
- `sequence_name`
- `metadata`

不要存原始 `UnifiedClip`。
因为训练真正消费的是 `QuerySample`，存最终对象最省事。

---

## 7. 训练循环怎么改

当前 `train_mixture.py` 是训练前构建一次 loader，整轮复用，见 [train_mixture.py](/data/zbf/openclaw/d4rt/train_mixture.py:235)。

planned 模式不适合这样。
正确做法是：**每个 epoch 重建一次 train loader**。

---

### 7.1 每个 epoch 的流程

每个 rank：

1. 准备本 epoch 的 `source_indices`
2. planner 生成 `plan_entries`
3. 启动 prefetch manager
4. 先预热若干样本
5. 构建 `PlannedDataset`
6. 构建本 epoch 专用 `DataLoader`
7. 开始训练
8. 边训练边让 prefetch manager 继续往前填
9. epoch 结束，清理该 epoch 的 spool

---

### 7.2 `source_indices` 怎么来

#### DDP
保持现在的 `DistributedSampler.set_epoch(epoch)`，见 [train_mixture.py](/data/zbf/openclaw/d4rt/train_mixture.py:402)

然后直接拿本 rank 的 index 列表：

```python
source_indices = list(iter(train_sampler))
```

#### 单卡
不能继续用当前裸 `RandomSampler`，因为它每次 `iter()` 顺序都不稳定。
planned 模式下要改成一个**显式 per-epoch deterministic random index 生成器**。

注意：这会让单卡 planned 模式的 source-index 顺序比当前 online 单卡更可复现，但会和当前 `RandomSampler` 的偶然顺序不同。
这是一个**有意的、可控的行为变化**。

---

## 8. DDP 里 planned 模式为什么能成立

这个问题要讲清楚。

planned 模式下：

- 每个 rank 自己生成自己的 plan
- 不再使用 `DistributedSampler` 去驱动最终 `PlannedDataset`
- `PlannedDataset` 自身长度已经是“这个 rank 本 epoch 要训练的样本数”

也就是说：

- `DistributedSampler` 只参与**生成 source index 顺序**
- 不参与 planned dataset 的最终迭代

最终 `PlannedDataset` 直接 `shuffle=False` + 顺序读即可。

---

## 9. Prefetch Manager 应该怎么工作

首版不建议用线程，建议用**少量独立 builder 进程**。

原因：

- `load_clip` + PIL/cv2 + numpy + query_builder 都不是纯 I/O
- adapter 内部缓存和字典不是严格线程安全设计
- process 更隔离

但 builder 进程数不要太大。
建议起步：

- 每个 rank `2~4` 个 builder process
- Co3D-heavy mixture 从 `2` 开始

---

### 9.1 为什么不能开很多 builder

看 [datasets/adapters/co3dv2.py](/data/zbf/openclaw/d4rt/datasets/adapters/co3dv2.py:730)：

- 每个 worker 可能缓存多个 category 的 frame annotation
- 单 category 解压后可到 `150~250MB`

如果你 builder process 开太多，RAM 会炸。

所以 planned 模式里要用**少而稳的 builder**，而不是现在 DataLoader 那种 32 worker 暴力并发。

---

### 9.2 预取窗口怎么设

建议同时用两个阈值：

- `lookahead_samples`
- `max_spool_bytes`

不要只按样本数。

因为 sample 大小差异可能很大：

- 普通 `video`：48x3x256x256 float32，大约 36MB
- `depths`：大约 12MB
- 如果 `sampled_highres` 开着，`highres_video` 可能再大一截

所以同样 200 个 sample，可能是 10GB，也可能是 40GB。

建议：

- 先按 byte cap 控制
- sample cap 只是辅助

---

## 10. 这个方案里最容易忽略的几个坑

### 10.1 当前 online 多 worker 行为不能逐字复刻
这是前面说过的根问题。
planned 模式必须接受“语义收敛到 rank-global deterministic plan”。

---

### 10.2 `ScanNet` / `VKITTI` 这种 fake `num_frames` 必须先修
否则 planner 不是在计划，而是在批量制造坏样本。

---

### 10.3 如果 bundle 存 float32 视频，空间会很快变大
第一版可以先接受，但要知道代价。

优化空间：

- `video` 可以存 `uint8`
- 当前 `move_batch_to_device()` 会把 `batch["video"]` 从 `uint8` 转成 `float`，见 [train_mixture.py](/data/zbf/openclaw/d4rt/train_mixture.py:25)

但 `highres_video` 目前不会自动做这个转换，所以如果后续要压缩 `highres_video`，还得顺手改 `move_batch_to_device()`。

---

### 10.4 不能再让 DataLoader worker 参与采样
planned 模式里 worker 只能做“读 ready 文件”。
只要 worker 还持有 sampler 状态，系统就会重新变回在线随机。

---

### 10.5 训练前的 scheduler 步数计算要改
当前 `optimizer_steps_per_epoch` 是基于训练前创建的 `train_loader` 算的，见 [train_mixture.py](/data/zbf/openclaw/d4rt/train_mixture.py:348)

planned 模式下 train loader 是每个 epoch 动态创建。
所以应改成基于：

- `per_rank_num_samples`
- `batch_size`
- `drop_last`
- `grad_accum`

直接静态计算，而不是依赖早建的 loader。

---

## 11. 推荐的精确实现顺序

### 第一步：只修 adapter 基础问题
先改：

- `ScanNetAdapter`
- `VKITTI2Adapter`
- `TartanAirAdapter`

目标：

- 真实 `num_frames`
- 缓存每 sequence 文件列表
- `load_clip` 不再反复 glob

---

### 第二步：加 `planner + planned dataset`，但 builder 先单进程
先不追求极限速度。
先验证：

- 计划生成是否稳定
- offline sample 和 online sample 是否一致
- DDP 是否工作

---

### 第三步：把 builder 提升为 2~4 个进程
再压测：

- 构建速度
- 本地 spool 空间
- GPU 等待时间

---

### 第四步：只对最慢 dataset 再考虑 COS SDK 原生下载
也就是 Phase 2，再做“adapter 不再通过 `/data_cos` 读，而是 builder 直接用 SDK 抓 raw 文件”。

这个不应该作为第一步。

---

## 12. 我认为最准确、最稳的落地方案

如果让我现在定最终方案，不写代码只定设计，我会这样定：

### 语义
- planned 模式定义为“每 rank 单 planner，全局显式 locality”
- 不追求复刻 online 多 worker 隐式 locality

### 存储层
- 本地 spool 存 `QuerySample` bundle
- 用完即删，不做长期 cache

### 失败策略
- 瞬时错误：重试同一 spec
- 结构性错误：直接 fail fast
- 不再 silently resample

### 并发
- builder process：2~4 / rank
- train loader workers：0~2 / rank

### 前置改造
- 先修 fake `num_frames`
- 先消灭 `load_clip` 热路径里的 `glob`

---