# D4RT 环境安装指南

## 快速安装

```bash
# 方法1: 使用安装脚本
bash setup_env.sh

# 方法2: 手动安装
conda env create -f environment.yml
conda activate d4rt
```

## 验证安装

```bash
conda activate d4rt
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import cv2, numpy, yaml; print('所有依赖已安装')"
```

## 环境说明

- Python 3.10
- PyTorch 2.0+ (CUDA 11.8)
- OpenCV
- NumPy
- PyYAML
- TensorBoard

## 如果遇到问题

CUDA版本不匹配时，修改 `environment.yml` 中的 `pytorch-cuda` 版本：
- CUDA 11.8: `pytorch-cuda=11.8`
- CUDA 12.1: `pytorch-cuda=12.1`
