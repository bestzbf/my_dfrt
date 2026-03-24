#!/bin/bash
set -e

echo "创建 D4RT conda 环境..."
conda env create -f environment.yml

echo ""
echo "环境创建完成！"
echo ""
echo "激活环境："
echo "  conda activate d4rt"
echo ""
echo "验证安装："
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"
