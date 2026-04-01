# 混合数据集训练

## 快速开始

```bash
# 使用默认配置
bash train_mixture.sh

# 自定义配置
CONFIG=configs/my_mixture.yaml BATCH_SIZE=2 bash train_mixture.sh
```

## 配置文件格式

参考 `configs/mixture_train.yaml`：

```yaml
mode: mixture  # 或 single / scene
datasets:
  - name: pointodyssey
    root: /path/to/data
    weight: 0.4
  - name: scannet
    root: /path/to/data
    weight: 0.3
```

## 支持的数据集

pointodyssey, scannet, co3dv2, kubric, blendedmvs, mvssynth, dynamic_replica, tartanair, vkitti2, waymo
