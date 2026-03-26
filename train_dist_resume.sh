#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 分布式多 GPU 训练脚本，支持从 checkpoint resume
#   NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=3,4,5 bash train_dist_resume.sh
# cd /data1/zbf/my_dfrt && NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=3,4,5 bash train_dist_resume.sh
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ─── 分布式配置 ───────────────────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 自动找一个空闲端口（用户可用 MASTER_PORT=xxxxx 手动指定）
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$(python3 -c "
import socket
with socket.socket() as s:
    s.bind(('', 0))
    s.listen(1)
    print(s.getsockname()[1])
")
fi
MASTER_PORT="${MASTER_PORT:-29501}"

# ─── 路径 ─────────────────────────────────────────────────────────────────────
export DATA_ROOT="${DATA_ROOT:-/data2/d4rt/datasets/PointOdyssey_fast}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_all_augs/256_lossPaper_load1580}"
PYTHON_ENV_BIN="/root/miniconda3/envs/d4rt/bin"
TORCHRUN_BIN="$PYTHON_ENV_BIN/torchrun"

RESUME_CKPT="${RESUME_CKPT:-/data1/zbf/my_dfrt/outputs_all_augs/256_lossPaper_load1580/full_main/checkpoint_epoch_1620.pth}"

# ─── 模型 / 训练配置 ──────────────────────────────────────────────────────────
export MODE=normal
export STAGE=full_main
export ENCODER=base
export VIDEOMAE_MODEL=/data1/zbf/pretrained/videomae-base
export PATCH_PROVIDER=precomputed_highres
export QUERY_CHUNK_SIZE=0

# per-GPU batch size；effective batch = BATCH_SIZE * NPROC_PER_NODE
export BATCH_SIZE="${BATCH_SIZE:-8}"

export VAL_EVERY_EPOCHS=10

# per-GPU dataloader workers；清理rg后空闲CPU充足
export NUM_WORKERS="${NUM_WORKERS:-20}"  # 3 GPU × 16 workers = 48核
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

# ─── 用 torchrun 替换 python（run_pointodyssey_curriculum_stage.sh 里
#     CMD=( $PYTHON_BIN train.py ... ) 不带引号，会正确 word-split）───────────
export PYTHON_BIN="$TORCHRUN_BIN --nproc_per_node=$NPROC_PER_NODE --master_addr=127.0.0.1 --master_port=$MASTER_PORT"

echo "=========================================================="
echo "  分布式训练：NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  effective batch size = $BATCH_SIZE x $NPROC_PER_NODE = $((BATCH_SIZE * NPROC_PER_NODE))"
echo "  Resume: $RESUME_CKPT"
echo "=========================================================="

bash train.sh \
  --train-split train \
  --val-split val \
  --steps 1000000 \
  --warmup-steps 0 \
  --lambda-3d 1.0 \
  --lambda-raw-3d 0.0 \
  --lambda-conf 0.0 \
  --lambda-2d 0.1 \
  --lambda-vis 0.1 \
  --lambda-disp 0.1 \
  --lambda-normal 0.5 \
  --conf-weighting-start-step 999999 \
  --conf-ramp-steps 0 \
  --img-size 256 \
  --t-tgt-eq-t-cam-ratio 0.4 \
  --amp \
  --no-gradient-checkpointing \
  --skip-pointodyssey-sanity \
  --gradient-accumulation-steps 4 \
  --resume "$RESUME_CKPT"
