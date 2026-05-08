# Planned Mode 文档中心

## 📖 文档概览

本目录包含 D4RT Planned Mode 的完整实施方案，旨在解决 COS 挂载盘训练时的 I/O 瓶颈问题。

### 核心问题
- 本地存储空间不足，无法完整拷贝数据集
- COS 挂载盘（s3fs）小文件读取慢
- GPU 大部分时间闲置（利用率 <30%）

### 解决方案
**Sample-Bundle Prefetch**: 后台预构建样本到本地 spool，训练时直接读取

### 预期收益
- GPU 利用率：20-30% → 80-90% (**3-4x**)
- 训练吞吐量：5 → 18 samples/sec (**3.6x**)
- 单 epoch 时间：10h → 3h (**节省 70%**)

---

## 📚 文档导航

### 🚀 快速开始

**如果你是第一次接触这个方案，按以下顺序阅读：**

1. **[PLANNED_MODE_QUICK_REFERENCE.md](PLANNED_MODE_QUICK_REFERENCE.md)** ⭐ **从这里开始**
   - 一页纸快速参考
   - 核心配置和命令
   - 常见问题速查

2. **[PLANNED_MODE_SUMMARY.md](PLANNED_MODE_SUMMARY.md)** ⭐ **执行摘要**
   - 方案评估结论
   - 实施路径和时间表
   - 成本效益分析
   - 立即行动指南

### 📖 详细文档

**深入了解技术细节和实施方案：**

3. **[PLANNED_MODE_IMPLEMENTATION.md](PLANNED_MODE_IMPLEMENTATION.md)** 📘 **技术方案**
   - 完整架构设计
   - 核心模块详解
   - 代码实现指南
   - 训练循环改造

4. **[PLANNED_MODE_TUNING_GUIDE.md](PLANNED_MODE_TUNING_GUIDE.md)** 🔧 **调优指南**
   - 参数详解和计算公式
   - 不同场景的推荐配置
   - 性能调优流程
   - 自动调优脚本

5. **[PLANNED_MODE_RISK_ASSESSMENT.md](PLANNED_MODE_RISK_ASSESSMENT.md)** ⚠️ **风险管理**
   - 技术风险评估
   - 运维风险评估
   - 缓解方案详解
   - 应急响应流程

---

## 🎯 使用场景

### 适合你的情况？

**✅ 强烈推荐，如果你：**
- 本地存储空间不足（<1 TB）
- 数据存储在 COS/S3 等对象存储
- GPU 利用率低（<40%）
- 数据加载时间 > 训练时间
- 有 300+ GB 本地 SSD 可用
- 可以接受 3-4 周开发时间

**⚠️ 谨慎考虑，如果你：**
- 数据已在本地 SSD
- GPU 利用率已经很高（>70%）
- 本地存储 <200 GB
- 团队缺乏多进程编程经验

**❌ 不推荐，如果你：**
- 数据加载不是瓶颈
- 只是偶尔训练（ROI 不高）
- 无法接受训练语义变化

---

## 🔬 实验代码

### Phase 0: 确认瓶颈

```bash
cd /data/zbf/openclaw/d4rt
python experiments/exp001_profile_baseline.py
```

**目标：** 验证 COS 确实是瓶颈

**预期结果：**
- GPU 利用率 <30%
- 数据加载时间 > 训练时间
- 报告显示 "I/O 是主要瓶颈"

**时间：** 30 分钟

---

### Phase 1: 最小原型

```bash
python experiments/exp002_minimal_prototype.py \
    --config configs/mixture_5datasets_local.yaml \
    --num-samples 100 \
    --batch-size 4
```

**目标：** 验证 planned mode 逻辑正确性

**验证指标：**
- ✅ 数值一致性（planned vs online）
- ✅ 稳定性（无死锁、超时）
- ✅ 性能提升（加载速度）

**时间：** 3-5 天

---

## 📊 方案评估

### 技术可行性：✅ **可行**

| 维度 | 评分 | 说明 |
|------|------|------|
| 问题理解 | 10/10 | 完全准确 |
| 方案方向 | 10/10 | Sample-bundle prefetch 是最佳选择 |
| 技术细节 | 8/10 | 大部分正确，需调整 builder 数量 |
| 实施顺序 | 9/10 | 合理，建议加入最小原型阶段 |
| 风险评估 | 6/10 | 需补充存储、预热、失败处理 |

### 投资回报：✅ **高 ROI**

**投入：**
- 开发时间：3-4 周
- 本地存储：300 GB SSD
- 学习成本：1-2 周

**收益：**
- 训练时间缩短 70%
- GPU 利用率提升 3-4x
- 每个训练任务节省 $1400（假设 GPU $2/小时）
- 长期收益：所有未来训练任务都受益

**ROI：** 如果你计划训练 10+ 次，ROI > 10x

---

## 🛠️ 实施路径

### 总览（3-4 周）

```
Week 1: Phase 0-1 (确认瓶颈 + 最小原型)
Week 2: Phase 2-3 (前置改造 + 核心实现)
Week 3: Phase 3-4 (测试调优 + 验证)
Week 4: Phase 4   (部署上线)
```

### 详细步骤

| 阶段 | 任务 | 时间 | 产出 |
|------|------|------|------|
| **Phase 0** | 确认瓶颈 | 1 天 | 性能分析报告 |
| **Phase 1** | 最小原型 | 3-5 天 | 可用原型 + 验证结果 |
| **Phase 2** | 前置改造 | 2-3 天 | 优化后的 adapter |
| **Phase 3** | 完整实现 | 7-10 天 | 生产级代码 |
| **Phase 4** | 验证部署 | 3-5 天 | 上线 + 文档 |

**总计：** 16-24 天（3-4 周）

---

## ⚙️ 推荐配置

### 初始配置（保守）

```python
PLANNED_MODE_CONFIG = {
    # Builder 配置
    'num_builder_workers': 12,      # 每个 rank
    'lookahead_window': 1000,       # 提前构建 1000 个样本
    'warmup_samples': 100,          # 预热 100 个样本
    'max_retries': 5,               # 失败重试 5 次

    # Spool 配置
    'spool_root': '/data/local_ssd/d4rt_spool',
    'spool_bucket_size': 100,       # 每目录 100 个文件
    'cleanup_lag': 100,             # 保留 100 个已消费样本

    # DataLoader 配置
    'train_loader_num_workers': 0,  # Planned mode 不需要
    'train_loader_pin_memory': True,
}
```

### 存储需求

```
单个样本: 60 MB
Lookahead: 1000 samples
Per rank: 60 GB
4 GPUs: 240 GB
推荐预留: 300 GB
```

### 性能预期

```
Builder 速度: 4 samples/sec per rank (12 workers / 3s)
训练速度:     2 samples/sec per rank (1 / 0.5s)
结论: Builder 速度 > 训练速度 ✅ GPU 不会等待
```

---

## ⚠️ 关键风险

| 风险 | 等级 | 概率 | 影响 | 缓解后 |
|------|------|------|------|--------|
| 存储空间不足 | 🔴 | 中 | 高 | 🟢 |
| COS 服务中断 | 🔴 | 低 | 高 | 🟡 |
| Builder 进程崩溃 | 🟡 | 中 | 中 | 🟢 |
| 训练结果不一致 | 🟡 | 低 | 中 | 🟢 |
| 死锁或竞态 | 🟡 | 低 | 中 | 🟢 |

**详见：** [PLANNED_MODE_RISK_ASSESSMENT.md](PLANNED_MODE_RISK_ASSESSMENT.md)

---

## 🎓 学习资源

### 前置知识

- Python 多进程编程（`multiprocessing`）
- PyTorch DataLoader 机制
- 文件系统操作和原子写入
- 分布式训练（DDP）

### 推荐阅读

1. **原始方案：** `/data/zbf/openclaw/d4rt/cos/cos优化`
2. **项目文档：** `/data/zbf/openclaw/d4rt/CLAUDE.md`
3. **相关代码：**
   - `datasets/mixture.py` - 当前 online mode
   - `datasets/sampling.py` - Sampler 和 locality
   - `train_mixture.py` - 训练主循环

---

## 📞 获取帮助

### 立即可做

1. **确认瓶颈**
   ```bash
   python experiments/exp001_profile_baseline.py
   ```

2. **验证原型**
   ```bash
   python experiments/exp002_minimal_prototype.py
   ```

3. **阅读文档**
   - 从 [PLANNED_MODE_QUICK_REFERENCE.md](PLANNED_MODE_QUICK_REFERENCE.md) 开始
   - 然后阅读 [PLANNED_MODE_SUMMARY.md](PLANNED_MODE_SUMMARY.md)

### 需要支持

**实现帮助：**
- 核心模块代码编写
- 调试和测试
- 性能调优

**问题排查：**
- 死锁、OOM、性能问题
- 提供错误日志和性能数据
- 详细的调试步骤

**随时告诉我你需要什么！**

---

## 📝 版本历史

### v1.0 (2026-04-26)
- 初始版本
- 完整技术方案
- 参数调优指南
- 风险评估文档
- 实验代码框架

---

## 🎯 成功标准

### 技术指标
- ✅ GPU 利用率 > 80%
- ✅ 训练吞吐量提升 > 3x
- ✅ 本地存储占用 < 250 GB
- ✅ 无死锁、OOM、数据损坏

### 科学指标
- ✅ Validation metrics 与 online mode 相当（±2%）
- ✅ 收敛曲线相似
- ✅ 可复现性好

### 运维指标
- ✅ 监控和告警完善
- ✅ 应急响应流程清晰
- ✅ 文档完整
- ✅ 团队熟悉操作

---

## 🚀 开始行动

### 今天就可以做

```bash
# 1. 确认瓶颈
cd /data/zbf/openclaw/d4rt
python experiments/exp001_profile_baseline.py

# 2. 检查存储
df -h /data/local_ssd

# 3. 阅读文档
cat docs/PLANNED_MODE_QUICK_REFERENCE.md
```

### 本周可以完成

- Phase 0: 确认瓶颈（1 天）
- Phase 1: 最小原型（3-5 天）
- 决策：继续 or 调整 or 放弃

---

## 💡 最终建议

### ✅ 方案可行，建议实施

**理由：**
1. 技术方案正确，架构合理
2. 你的场景高度匹配
3. 预期收益显著（3-4x）
4. 风险可控

**前提条件：**
- ✅ 有 300 GB 本地存储
- ✅ 有 3-4 周开发时间
- ✅ 团队有多进程编程经验
- ✅ 可接受训练语义变化

**如果满足以上条件，强烈建议实施。**

---

## 📧 联系方式

如有任何问题或需要帮助，请随时联系。

**祝你实施顺利！🎉**
