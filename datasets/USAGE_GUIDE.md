# D4RT 数据加载器使用指南

## 概述

D4RT 数据加载器采用三层架构设计，支持三种训练模式：

| 模式 | 说明 |
|------|------|
| `single` | 训练某个数据集的**全部序列** |
| `scene`  | 训练某个数据集的**指定场景**（一个或多个序列）|
| `mixture` | 多个数据集**混合训练** |

### 三层架构

1. **Adapter 层** - 统一各数据集的原始格式
2. **Mixture/Sampler 层** - 控制多数据集混合采样
3. **Query Builder 层** - 构造 D4RT 训练监督信号

---

## 快速开始

### 方式1: `single` — 训练整个数据集

```python
from datasets.factory import create_training_dataset
from torch.utils.data import DataLoader
from datasets.collate import d4rt_collate_fn

config = {
    'mode': 'single',
    'name': 'pointodyssey',
    'root': '/data2/d4rt/datasets/PointOdyssey',
    'clip_len': 48,
    'img_size': 256,
    'num_queries': 2048,
}

dataset = create_training_dataset(config, split='train')

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    collate_fn=d4rt_collate_fn,
)
```

### 方式2: `scene` — 只训练指定场景

```python
config = {
    'mode': 'scene',
    'name': 'pointodyssey',
    'root': '/data2/d4rt/datasets/PointOdyssey',
    'sequences': ['ani'],          # 只训练 train/ani 这一个场景
    # 也可以同时指定多个场景：
    # 'sequences': ['ani', 'ani11_new_', 'ani13_new_'],
    'clip_len': 48,
    'img_size': 256,
    'num_queries': 2048,
}

dataset = create_training_dataset(config, split='train')
```

### 方式3: `mixture` — 多数据集混合训练

```python
config = {
    'mode': 'mixture',
    'datasets': [
        {'name': 'pointodyssey', 'root': '/path', 'weight': 0.5},
        {'name': 'scannet',      'root': '/path', 'weight': 0.3},
        {'name': 'kubric',       'root': '/path', 'weight': 0.2},
    ],
    'clip_len': 48,
    'img_size': 256,
    'num_queries': 2048,
}

dataset = create_training_dataset(config, split='train')
```

---

## 配置文件使用

### 指定场景训练（新增）

```bash
python train.py --config configs/scene_pointodyssey_ani.yaml
```

`scene_pointodyssey_ani.yaml` 关键字段：

```yaml
data:
  mode: scene
  name: pointodyssey
  root: /data2/d4rt/datasets/PointOdyssey
  sequences:
    - ani
```

### 单数据集训练（全部序列）

```bash
python train.py --config configs/single_pointodyssey.yaml
python train.py --config configs/single_kubric.yaml
python train.py --config configs/single_dynamic_replica.yaml
python train.py --config configs/single_scannet.yaml
```

### 混合训练

```bash
python train.py --config configs/mixture_dynamic.yaml   # 动态场景组
python train.py --config configs/mixture_full.yaml      # 全部10个数据集
```
