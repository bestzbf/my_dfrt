#!/usr/bin/env python3
"""
Experiment 001: Profile Baseline Training

目标：确认 COS 是训练瓶颈
- 监控 GPU 利用率
- 监控 Data loading 时间 vs Training 时间
- 监控 s3fs I/O 延迟

预期结果：
- GPU 利用率 < 30%
- Data loading 时间 >> Training 时间
- 证明 I/O 是瓶颈
"""

import time
import torch
import subprocess
import threading
from pathlib import Path
from collections import defaultdict
import json

class GPUMonitor:
    """后台监控 GPU 利用率"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.utilizations = []
        self.stop_flag = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        while not self.stop_flag.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                utils = [int(x.strip()) for x in result.stdout.strip().split('\n')]
                self.utilizations.append({
                    'timestamp': time.time(),
                    'gpu_utils': utils
                })
            except Exception as e:
                print(f"GPU monitor error: {e}")

            time.sleep(self.interval)

    def get_stats(self):
        if not self.utilizations:
            return {}

        all_utils = []
        for record in self.utilizations:
            all_utils.extend(record['gpu_utils'])

        return {
            'mean': sum(all_utils) / len(all_utils),
            'min': min(all_utils),
            'max': max(all_utils),
            'samples': len(self.utilizations)
        }


class TrainingProfiler:
    """训练循环性能分析"""
    def __init__(self):
        self.data_loading_times = []
        self.training_times = []
        self.batch_sizes = []

    def record_batch(self, data_time, train_time, batch_size):
        self.data_loading_times.append(data_time)
        self.training_times.append(train_time)
        self.batch_sizes.append(batch_size)

    def get_stats(self):
        if not self.data_loading_times:
            return {}

        total_data_time = sum(self.data_loading_times)
        total_train_time = sum(self.training_times)
        total_time = total_data_time + total_train_time

        return {
            'data_loading': {
                'total_s': total_data_time,
                'mean_s': total_data_time / len(self.data_loading_times),
                'percentage': 100 * total_data_time / total_time
            },
            'training': {
                'total_s': total_train_time,
                'mean_s': total_train_time / len(self.training_times),
                'percentage': 100 * total_train_time / total_time
            },
            'throughput': {
                'samples_per_sec': sum(self.batch_sizes) / total_time,
                'batches_per_sec': len(self.batch_sizes) / total_time
            },
            'num_batches': len(self.data_loading_times)
        }


def profile_training_loop(train_loader, model, optimizer, device, num_steps=100):
    """
    运行训练循环并收集性能数据

    Args:
        train_loader: DataLoader
        model: 模型
        optimizer: 优化器
        device: 设备
        num_steps: 运行的 step 数

    Returns:
        dict: 性能统计
    """
    gpu_monitor = GPUMonitor(interval=1.0)
    profiler = TrainingProfiler()

    model.train()
    gpu_monitor.start()

    print(f"开始 profiling，运行 {num_steps} steps...")

    data_iter = iter(train_loader)
    t_data_start = time.time()

    for step in range(num_steps):
        # 1. 数据加载
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t_data_end = time.time()
        data_time = t_data_end - t_data_start

        # 2. 训练
        t_train_start = time.time()

        # 移动数据到 GPU
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs['loss']

        # 反向传播
        loss.backward()
        optimizer.step()

        t_train_end = time.time()
        train_time = t_train_end - t_train_start

        # 记录
        batch_size = batch['video'].shape[0]
        profiler.record_batch(data_time, train_time, batch_size)

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{num_steps}: "
                  f"data={data_time:.3f}s, train={train_time:.3f}s, "
                  f"ratio={data_time/train_time:.2f}x")

        # 准备下一次数据加载计时
        t_data_start = time.time()

    gpu_monitor.stop()

    # 汇总统计
    stats = {
        'gpu': gpu_monitor.get_stats(),
        'training': profiler.get_stats()
    }

    return stats


def print_report(stats):
    """打印性能报告"""
    print("\n" + "="*80)
    print("性能分析报告")
    print("="*80)

    # GPU 利用率
    gpu_stats = stats['gpu']
    print(f"\nGPU 利用率:")
    print(f"  平均: {gpu_stats['mean']:.1f}%")
    print(f"  最小: {gpu_stats['min']:.1f}%")
    print(f"  最大: {gpu_stats['max']:.1f}%")

    # 训练性能
    train_stats = stats['training']
    print(f"\n时间分布:")
    print(f"  数据加载: {train_stats['data_loading']['total_s']:.1f}s "
          f"({train_stats['data_loading']['percentage']:.1f}%)")
    print(f"  模型训练: {train_stats['training']['total_s']:.1f}s "
          f"({train_stats['training']['percentage']:.1f}%)")

    print(f"\n吞吐量:")
    print(f"  {train_stats['throughput']['samples_per_sec']:.2f} samples/sec")
    print(f"  {train_stats['throughput']['batches_per_sec']:.2f} batches/sec")

    # 瓶颈判断
    print(f"\n瓶颈分析:")
    data_pct = train_stats['data_loading']['percentage']
    gpu_util = gpu_stats['mean']

    if data_pct > 60 and gpu_util < 40:
        print("  ⚠️  确认：I/O 是主要瓶颈")
        print(f"     - 数据加载占用 {data_pct:.1f}% 时间")
        print(f"     - GPU 平均利用率仅 {gpu_util:.1f}%")
        print("  ✅ Planned mode 预期有显著收益")
    elif data_pct > 40:
        print("  ⚠️  I/O 是瓶颈之一")
        print("  ✅ Planned mode 预期有一定收益")
    else:
        print("  ℹ️  I/O 不是主要瓶颈")
        print("  ⚠️  Planned mode 收益可能有限")

    print("="*80 + "\n")


def main():
    """
    使用方法：

    1. 修改 train_mixture.py，在训练循环前调用：
       from experiments.exp001_profile_baseline import profile_training_loop, print_report

       stats = profile_training_loop(
           train_loader, model, optimizer, device, num_steps=100
       )
       print_report(stats)
       exit(0)  # 只做 profiling，不继续训练

    2. 运行训练脚本：
       bash train_mixture_5datasets_3gpu.sh

    3. 查看报告，确认是否需要 planned mode
    """
    print(__doc__)


if __name__ == '__main__':
    main()
