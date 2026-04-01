# D4RT 混合数据集训练配置指南

## 论文原始设置

论文使用 **11个数据集** 混合训练（第5节第12-22行）：

### 有轨迹数据集（3个）
- **PointOdyssey**: 密集轨迹跟踪
- **Kubric**: 合成动态场景  
- **Dynamic Replica**: 动态场景重建

### 静态重建数据集（8个）
- **BlendedMVS**: 多视角立体
- **Co3Dv2**: 常见物体3D
- **MVS-Synth**: 合成MVS
- **ScanNet/ScanNet++**: 室内场景
- **TartanAir**: 无人机导航
- **VirtualKitti**: 自动驾驶合成
- **Waymo Open**: 自动驾驶真实

## 推荐配置

### 1. 完整版（11个数据集）
```bash
CONFIG=configs/mixture_full_11datasets.yaml bash train_mixture.sh
```
- 最接近论文设置
- 需要所有数据集
- 训练时间：500k steps（论文用64 TPU，2天）

### 2. 中等版（6个数据集）⭐ 推荐
```bash
CONFIG=configs/mixture_medium_6datasets.yaml bash train_mixture.sh
```
- 平衡性能和资源
- 包含核心有轨迹数据集 + 主要静态数据集
- 适合大多数场景

### 3. 最小版（3个数据集）
```bash
CONFIG=configs/mixture_minimal_3datasets.yaml bash train_mixture.sh
```
- 快速实验
- PointOdyssey(50%) + ScanNet(30%) + Kubric(20%)
- 资源受限场景

## 权重分配原则

1. **有轨迹数据集权重更高**（50-65%）
   - 提供完整的2D/3D轨迹监督
   - 对动态场景理解至关重要

2. **静态数据集补充**（35-50%）
   - 提供深度/法向量监督
   - 增强静态场景重建能力

3. **数据集多样性**
   - 室内 + 室外
   - 真实 + 合成
   - 动态 + 静态
