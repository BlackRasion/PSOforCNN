# PSO-CNN架构自动搜索系统使用指南

## 概述

本系统基于粒子群优化（PSO）算法实现CNN架构的自动搜索，能够自动找到适合特定数据集的最优CNN架构。

## 最新更新 (v2.0)

### 数据集优化
- **数据子集比例调整**: 从30%提升至75%（通过PSO_DATA_SUBSET_RATIO配置），提高搜索质量和结果可靠性
- **公平性比较**: 基准模型和PSO优化模型使用相同的数据子集进行训练
- **内存效率**: 避免重复创建数据加载器，提高内存使用效率

### 结果管理优化
- **统一保存位置**: 基准模型训练结果统一保存到PSO文件夹
- **完整训练曲线**: 基准模型训练曲线自动保存到 `PSO/baseline/training_curves.png`
- **TensorBoard集成**: 基准模型日志保存到 `PSO/runs/baseline/` 目录
- **配置自动恢复**: 训练完成后自动恢复原始配置，避免配置污染

## 系统架构

```
PSO-CNN系统
├── config.py              # 配置文件（包含PSO参数）
├── pso_particle.py         # PSO粒子定义
├── pso_model.py           # 动态CNN模型构建
├── pso_trainer.py         # PSO训练器
├── pso_optimizer.py       # PSO优化器主逻辑
├── pso_demo.py           # PSO演示脚本
└── main.py               # 主程序（支持PSO模式）
```

## 快速开始

### 1. 基本PSO架构搜索

```bash
# 运行完整的PSO架构搜索
python main.py --mode pso

# 快速演示模式（减少粒子数和迭代次数）
python main.py --mode pso --pso-quick

# 跳过基准模型比较
python main.py --mode pso --skip-baseline
```

### 2. 直接运行PSO演示

```bash
# 完整演示
python pso_demo.py

# 快速演示
python pso_demo.py --quick

# 跳过基准比较
python pso_demo.py --skip-baseline
```

## 配置参数

在 `config.py` 中可以调整以下PSO相关参数：

```python
# PSO算法参数
PSO_PARTICLE_SIZE = 10      # 粒子群大小
PSO_ITERATIONS = 5          # PSO迭代次数
PSO_TRAIN_EPOCHS = 8        # 每个粒子训练的epochs
PSO_FINAL_EPOCHS = 15       # 最优架构最终训练epochs
PSO_DATA_SUBSET_RATIO = 0.75 # PSO训练时使用的数据子集比例

# PSO算法超参数
PSO_W_MIN = 0.4            # 惯性权重最小值
PSO_W_MAX = 0.9            # 惯性权重最大值
PSO_C1 = 2.0               # 个体学习因子
PSO_C2 = 2.0               # 社会学习因子
```

## 架构搜索空间

PSO算法在以下架构空间中搜索：

- **卷积层数量**: 2-4层
- **卷积核数量**: 每层16-128个
- **卷积核大小**: 3x3, 5x5, 7x7
- **全连接层数量**: 2-4层
- **全连接层大小**: 64-512个神经元

## 输出结果

### 输出文件结构

```
PSO/
├── baseline/                               # 基准模型结果
│   ├── chinese_number_cnn.pth             # 基准模型文件
│   └── training_curves.png               # 基准模型训练曲线
├── results_YYYYMMDD_HHMMSS/
│   ├── optimization_history.json          # 优化历史
│   ├── best_position.json                # 最优架构参数
│   └── optimization_curves.png           # 优化曲线图
├── models/
│   ├── best_model_YYYYMMDD_HHMMSS.pth    # 最优模型
│   └── training_history_YYYYMMDD_HHMMSS.json # 训练历史
├── runs/
│   ├── baseline/                          # 基准模型TensorBoard日志
│   └── best_model_YYYYMMDD_HHMMSS/       # 最优模型TensorBoard日志
├── demo_report_YYYYMMDD_HHMMSS.txt       # 演示报告
└── training_curves.png                    # PSO最优模型训练曲线
```

### 关键指标

- **最优适应度**: PSO找到的最佳准确率
- **收敛情况**: 优化过程中的性能提升
- **架构参数**: 最优CNN架构的详细配置
- **训练时间**: 总优化时间和平均迭代时间

## 参考文献

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
2. Sun, Y., et al. (2019). Particle swarm optimization of deep neural networks architectures for image classification.
3. 相关PSO-CNN优化论文和实现

---

## 重要说明

### 数据一致性
- 基准模型和PSO优化模型现在使用相同的数据子集进行训练（比例由PSO_DATA_SUBSET_RATIO配置）
- 确保了公平的性能比较和可靠的实验结果
- 所有训练结果统一保存在PSO文件夹中，便于对比分析

### 文件管理
- 基准模型结果自动保存到 `PSO/baseline/` 目录
- PSO优化结果保存到 `PSO/models/` 和 `PSO/results_*/` 目录
- TensorBoard日志分别保存，支持同时可视化对比

**注意**: 首次运行建议使用快速模式熟悉系统，然后根据实际需求调整参数进行正式实验。