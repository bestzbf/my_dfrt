#!/usr/bin/env python3
"""
Experiment 002: Minimal Prototype of Planned Mode

目标：验证 planned mode 逻辑正确性
- 单 GPU 训练
- 只用 1 个数据集（Co3Dv2）
- Builder 单进程
- Lookahead 50 个样本
- 训练 100 steps

验证指标：
1. 数值一致性：Planned sample 和 online sample 的 loss 是否一致？
2. 稳定性：是否有死锁、超时、文件损坏？
3. 性能：GPU 利用率是否提升？
"""

import os
import sys
import time
import random
import pickle
import shutil
import torch
import torch.multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mixture import MixtureDataset
from datasets.sampling import DatasetSampler
from datasets.registry import get_dataset_adapter


@dataclass
class SamplePlanEntry:
    """样本计划条目"""
    epoch: int
    rank: int
    ordinal: int              # 本 rank 本 epoch 第几个样本
    source_index: int         # 原始 dataset index (用于 seed)
    dataset_idx: int
    dataset_name: str
    sequence_name: str
    frame_indices: List[int]
    py_rng_state: bytes       # pickle.dumps(rng.getstate())
    cache_key: str            # 用于 spool 文件名


class RankSamplePlanner:
    """为单个 rank 生成 sample plan"""

    def __init__(self, mixture_sampler, adapters, clip_len, seed):
        self.mixture_sampler = mixture_sampler
        self.adapters = adapters
        self.clip_len = clip_len
        self.seed = seed

    def generate_plan(
        self,
        epoch: int,
        rank: int,
        source_indices: List[int]
    ) -> List[SamplePlanEntry]:
        """生成本 rank 本 epoch 的完整 sample plan"""
        plan = []

        for ordinal, source_idx in enumerate(source_indices):
            # 1. 创建 RNG (与 MixtureDataset.__getitem__ 逻辑一致)
            rng = random.Random(self.seed + source_idx)

            # 2. 采样 dataset/sequence/frames
            dataset_idx, seq_name, frame_indices = \
                self.mixture_sampler.sample(rng)

            # 3. 保存 RNG 状态 (用于后续 transform/query_builder)
            rng_state = pickle.dumps(rng.getstate())

            # 4. 生成 cache_key
            cache_key = f"{epoch:05d}_{rank:02d}_{ordinal:08d}"

            entry = SamplePlanEntry(
                epoch=epoch,
                rank=rank,
                ordinal=ordinal,
                source_index=source_idx,
                dataset_idx=dataset_idx,
                dataset_name=self.adapters[dataset_idx].dataset_name,
                sequence_name=seq_name,
                frame_indices=frame_indices,
                py_rng_state=rng_state,
                cache_key=cache_key
            )
            plan.append(entry)

        return plan


class SampleBuilder:
    """构建单个样本 (复用现有逻辑)"""

    def __init__(self, adapters, transform, query_builder, clip_len):
        self.adapters = adapters
        self.transform = transform
        self.query_builder = query_builder
        self.clip_len = clip_len

    def build(self, entry: SamplePlanEntry):
        """构建单个样本"""
        # 1. Load clip from adapter
        adapter = self.adapters[entry.dataset_idx]
        clip = adapter.load_clip(
            entry.sequence_name,
            entry.frame_indices
        )

        # 2. 恢复 RNG 状态
        rng = random.Random()
        rng.setstate(pickle.loads(entry.py_rng_state))

        # 3. Transform (与 MixtureDataset.__getitem__ 一致)
        result = self.transform(clip, rng=rng)

        # 4. Query builder (与 MixtureDataset.__getitem__ 一致)
        sample = self.query_builder(result, py_rng=rng)

        # 5. 校验
        assert sample.video.shape[0] == self.clip_len, \
            f"Expected {self.clip_len} frames, got {sample.video.shape[0]}"

        return sample


class SpoolManager:
    """本地磁盘 spool 管理"""

    def __init__(self, spool_root: Path, bucket_size: int = 100):
        self.spool_root = Path(spool_root)
        self.bucket_size = bucket_size
        self.spool_root.mkdir(parents=True, exist_ok=True)

    def _parse_cache_key(self, cache_key: str) -> Tuple[int, int, int]:
        """解析 cache_key: epoch_rank_ordinal"""
        parts = cache_key.split('_')
        return int(parts[0]), int(parts[1]), int(parts[2])

    def get_sample_path(self, cache_key: str) -> Tuple[Path, Path]:
        """返回 (data_path, ready_path)"""
        epoch, rank, ordinal = self._parse_cache_key(cache_key)
        bucket = ordinal // self.bucket_size

        dir_path = self.spool_root / f"epoch_{epoch:05d}" / \
                   f"rank_{rank:02d}" / f"{bucket:03d}"
        dir_path.mkdir(parents=True, exist_ok=True)

        data_path = dir_path / f"{ordinal:08d}.pt"
        ready_path = dir_path / f"{ordinal:08d}.ready"

        return data_path, ready_path

    def save_sample(self, cache_key: str, sample):
        """原子写入样本"""
        data_path, ready_path = self.get_sample_path(cache_key)

        # 1. 写临时文件
        tmp_path = data_path.with_suffix('.tmp')
        torch.save(sample, tmp_path)

        # 2. 原子替换
        os.replace(tmp_path, data_path)

        # 3. 写 ready 标记
        ready_path.touch()

    def load_sample(self, cache_key: str, timeout: float = 300):
        """等待并加载样本"""
        data_path, ready_path = self.get_sample_path(cache_key)

        # 等待 ready 标记
        start_time = time.time()
        while not ready_path.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Sample {cache_key} not ready after {timeout}s")
            time.sleep(0.1)

        # 加载样本
        return torch.load(data_path)

    def cleanup_sample(self, cache_key: str):
        """删除已消费的样本"""
        data_path, ready_path = self.get_sample_path(cache_key)
        data_path.unlink(missing_ok=True)
        ready_path.unlink(missing_ok=True)

    def cleanup_epoch(self, epoch: int):
        """清理整个 epoch 的 spool"""
        epoch_dir = self.spool_root / f"epoch_{epoch:05d}"
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)


class PlannedDataset(Dataset):
    """从本地 spool 读取预构建的样本"""

    def __init__(
        self,
        plan_entries: List[SamplePlanEntry],
        spool_manager: SpoolManager,
        wait_timeout: float = 300
    ):
        self.plan_entries = plan_entries
        self.spool_manager = spool_manager
        self.wait_timeout = wait_timeout

    def __len__(self):
        return len(self.plan_entries)

    def __getitem__(self, idx: int):
        entry = self.plan_entries[idx]

        # 等待并加载样本
        sample = self.spool_manager.load_sample(
            entry.cache_key,
            timeout=self.wait_timeout
        )

        return sample


def builder_worker(
    plan_entries: List[SamplePlanEntry],
    sample_builder: SampleBuilder,
    spool_manager: SpoolManager,
    start_idx: int,
    end_idx: int
):
    """Builder worker 进程"""
    print(f"Builder worker: building samples {start_idx} to {end_idx}")

    for i in range(start_idx, end_idx):
        entry = plan_entries[i]

        try:
            # 构建样本
            sample = sample_builder.build(entry)

            # 保存到 spool
            spool_manager.save_sample(entry.cache_key, sample)

            if (i - start_idx + 1) % 10 == 0:
                print(f"  Built {i - start_idx + 1}/{end_idx - start_idx} samples")

        except Exception as e:
            print(f"  Error building sample {i}: {e}")
            raise


def run_experiment(
    config_path: str,
    dataset_name: str = "co3dv2",
    num_samples: int = 100,
    batch_size: int = 4,
    spool_root: str = "/tmp/d4rt_planned_spool"
):
    """
    运行最小原型实验

    Args:
        config_path: 配置文件路径
        dataset_name: 使用的数据集名称
        num_samples: 训练的样本数
        batch_size: batch size
        spool_root: spool 目录
    """
    print("="*80)
    print("Experiment 002: Minimal Prototype of Planned Mode")
    print("="*80)

    # 1. 加载配置和数据集
    print("\n[1/6] 加载配置和数据集...")
    from datasets.factory import create_mixture_dataset

    # 创建 online mode dataset (用于对比)
    online_dataset = create_mixture_dataset(config_path, split='train')

    # 提取组件
    mixture_sampler = online_dataset.mixture_sampler
    adapters = online_dataset.adapters
    transform = online_dataset.transform
    query_builder = online_dataset.query_builder
    clip_len = online_dataset.clip_len
    seed = online_dataset.seed

    # 2. 生成 sample plan
    print("\n[2/6] 生成 sample plan...")
    planner = RankSamplePlanner(mixture_sampler, adapters, clip_len, seed)

    # 模拟 source_indices (简单的顺序索引)
    source_indices = list(range(num_samples))

    plan_entries = planner.generate_plan(
        epoch=0,
        rank=0,
        source_indices=source_indices
    )

    print(f"  生成了 {len(plan_entries)} 个样本计划")
    print(f"  示例: {plan_entries[0].dataset_name} / {plan_entries[0].sequence_name}")

    # 3. 构建样本 (单进程，简化版)
    print("\n[3/6] 构建样本...")
    spool_manager = SpoolManager(spool_root)
    sample_builder = SampleBuilder(adapters, transform, query_builder, clip_len)

    # 清理旧数据
    if Path(spool_root).exists():
        shutil.rmtree(spool_root)
    Path(spool_root).mkdir(parents=True, exist_ok=True)

    # 构建所有样本
    t_build_start = time.time()
    for i, entry in enumerate(plan_entries):
        try:
            sample = sample_builder.build(entry)
            spool_manager.save_sample(entry.cache_key, sample)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_build_start
                rate = (i + 1) / elapsed
                print(f"  Built {i+1}/{len(plan_entries)} samples "
                      f"({rate:.2f} samples/sec)")

        except Exception as e:
            print(f"  Error building sample {i}: {e}")
            raise

    t_build_end = time.time()
    build_time = t_build_end - t_build_start
    print(f"  完成！总耗时 {build_time:.1f}s ({len(plan_entries)/build_time:.2f} samples/sec)")

    # 4. 创建 PlannedDataset
    print("\n[4/6] 创建 PlannedDataset...")
    planned_dataset = PlannedDataset(plan_entries, spool_manager)
    planned_loader = DataLoader(
        planned_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 不需要额外 worker
        pin_memory=True
    )

    # 5. 测试加载速度
    print("\n[5/6] 测试加载速度...")
    t_load_start = time.time()
    num_batches = 0

    for batch in planned_loader:
        num_batches += 1
        if num_batches >= 10:
            break

    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    samples_loaded = num_batches * batch_size

    print(f"  加载 {samples_loaded} 个样本耗时 {load_time:.3f}s")
    print(f"  加载速度: {samples_loaded/load_time:.1f} samples/sec")

    # 6. 数值一致性检查
    print("\n[6/6] 数值一致性检查...")
    print("  对比 planned mode 和 online mode 的样本...")

    # 从 planned dataset 加载第一个样本
    planned_sample = planned_dataset[0]

    # 从 online dataset 加载第一个样本 (相同 index)
    online_sample = online_dataset[0]

    # 对比关键字段
    def compare_tensors(t1, t2, name, rtol=1e-5, atol=1e-8):
        if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
            diff = (t1 - t2).abs().max().item()
            print(f"    ⚠️  {name} 不一致！最大差异: {diff}")
            return False
        else:
            print(f"    ✅ {name} 一致")
            return True

    all_match = True
    all_match &= compare_tensors(planned_sample.video, online_sample.video, "video")
    all_match &= compare_tensors(planned_sample.coords, online_sample.coords, "coords")

    if hasattr(planned_sample, 'depths') and planned_sample.depths is not None:
        all_match &= compare_tensors(planned_sample.depths, online_sample.depths, "depths")

    if all_match:
        print("\n  ✅ 数值一致性检查通过！")
    else:
        print("\n  ⚠️  数值一致性检查失败！")

    # 清理
    print("\n[清理] 删除 spool 目录...")
    spool_manager.cleanup_epoch(0)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

    return {
        'build_time': build_time,
        'build_rate': len(plan_entries) / build_time,
        'load_rate': samples_loaded / load_time,
        'numerical_match': all_match
    }


def main():
    """
    使用方法：

    python experiments/exp002_minimal_prototype.py
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/mixture_5datasets_local.yaml',
                        help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='co3dv2',
                        help='使用的数据集名称')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='训练的样本数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--spool-root', type=str,
                        default='/tmp/d4rt_planned_spool',
                        help='spool 目录')

    args = parser.parse_args()

    results = run_experiment(
        config_path=args.config,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        spool_root=args.spool_root
    )

    print("\n结果摘要:")
    print(f"  构建速度: {results['build_rate']:.2f} samples/sec")
    print(f"  加载速度: {results['load_rate']:.1f} samples/sec")
    print(f"  数值一致: {'✅ 通过' if results['numerical_match'] else '❌ 失败'}")


if __name__ == '__main__':
    main()
