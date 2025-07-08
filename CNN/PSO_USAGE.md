# PSO-CNN架构自动搜索系统使用指南

## 概述

本系统基于粒子群优化（PSO）算法实现CNN架构的自动搜索，能够自动找到适合特定数据集的最优CNN架构。

## 最新更新 (v2.0)

### 数据集优化
- **数据子集比例调整**: 从30%提升至75%，提高搜索质量和结果可靠性
- **公平性比较**: 基准模型和PSO优化模型使用相同的数据子集进行训练
- **内存效率**: 避免重复创建数据加载器，提高内存使用效率

### 结果管理优化
- **统一保存位置**: 基准模型训练结果统一保存到PSO文件夹
- **完整训练曲线**: 基准模型训练曲线自动保存到 `PSO/baseline/training_curves.png`
- **TensorBoard集成**: 基准模型日志保存到 `PSO/runs/baseline/` 目录
- **配置自动恢复**: 训练完成后自动恢复原始配置，避免配置污染

### 代码质量提升
- **异常处理**: 完善的错误处理机制，确保配置一致性
- **参数验证**: 更严格的参数验证和边界检查
- **文档同步**: 代码注释和文档保持同步更新

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
- **全连接层数量**: 1-3层
- **全连接层大小**: 64-512个神经元

## 输出结果

### 文件结构

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

## 性能优化

### 1. 数据加载优化

- 使用数据子集进行PSO搜索（默认75%）
- 增加数据加载器的工作进程数
- 启用内存固定（pin_memory）
- 使用持久化工作进程

### 2. 训练优化

- PSO搜索阶段：每个粒子只训练8个epochs
- 最终模型训练：使用完整的15个epochs
- 快速评估模式：减少验证频率

### 3. 硬件优化

- 自动检测和使用GPU
- 优化批次大小
- 内存管理优化

## 使用建议

### 1. 首次使用

```bash
# 建议先运行快速演示了解系统
python main.py --mode pso --pso-quick --skip-baseline
```

### 2. 正式实验

```bash
# 运行完整的PSO搜索（包含基准模型比较）
python main.py --mode pso

# 查看基准模型训练曲线
# 文件位置: PSO/baseline/training_curves.png

# 使用TensorBoard比较基准模型和PSO模型
tensorboard --logdir=PSO/runs
```

### 3. 参数调优

- **小数据集**: 减少粒子数量和迭代次数
- **大数据集**: 增加训练epochs和粒子数量
- **时间限制**: 使用快速模式或减少搜索空间

## 结果分析

### 1. 优化曲线分析

- 观察全局最优适应度的收敛趋势
- 分析平均适应度的变化
- 检查是否存在早熟收敛

### 2. 架构分析

- 比较最优架构与基准模型的差异
- 分析架构复杂度与性能的关系
- 验证架构的合理性

### 3. 性能比较

- PSO优化结果 vs 基准模型
- 训练时间 vs 性能提升
- 不同PSO参数设置的影响
- 基准模型和PSO模型的训练曲线对比（均保存在PSO文件夹中）
- TensorBoard可视化对比分析

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小
   - 减少粒子数量
   - 使用数据子集

2. **训练时间过长**
   - 使用快速模式
   - 减少迭代次数
   - 减少每个粒子的训练epochs

3. **收敛效果差**
   - 调整PSO超参数
   - 增加粒子数量
   - 扩大搜索空间

### 调试模式

```bash
# 启用详细输出
python pso_demo.py --quick 2>&1 | tee pso_debug.log
```

## 扩展功能

### 1. 自定义搜索空间

修改 `pso_particle.py` 中的参数范围：

```python
# 自定义卷积层数量范围
self.conv_layers = random.randint(2, 6)  # 2-6层

# 自定义卷积核数量范围
self.conv_filters = [random.randint(32, 256) for _ in range(self.conv_layers)]
```

### 2. 多目标优化

可以扩展适应度函数，同时考虑准确率和模型复杂度：

```python
# 在pso_trainer.py中修改适应度计算
def calculate_fitness(self, accuracy, model_params):
    # 平衡准确率和模型复杂度
    complexity_penalty = model_params / 1000000  # 参数数量惩罚
    return accuracy - complexity_penalty
```

### 3. 集成其他优化算法

系统架构支持轻松集成其他进化算法：
- 遗传算法（GA）
- 差分进化（DE）
- 蚁群优化（ACO）

## 技术细节

### PSO算法实现

- **位置编码**: 直接编码CNN架构参数
- **速度更新**: 考虑惯性、个体最优和全局最优
- **边界处理**: 确保参数在有效范围内
- **适应度评估**: 基于验证集准确率

### 动态模型构建

- **模块化设计**: 支持不同层数和参数组合
- **自适应池化**: 根据输入尺寸调整
- **批归一化**: 提高训练稳定性
- **Dropout**: 防止过拟合

## 参考文献

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
2. Sun, Y., et al. (2019). Particle swarm optimization of deep neural networks architectures for image classification.
3. 相关PSO-CNN优化论文和实现

---

## 重要说明

### 数据一致性
- 基准模型和PSO优化模型现在使用相同的75%数据子集进行训练
- 确保了公平的性能比较和可靠的实验结果
- 所有训练结果统一保存在PSO文件夹中，便于对比分析

### 文件管理
- 基准模型结果自动保存到 `PSO/baseline/` 目录
- PSO优化结果保存到 `PSO/models/` 和 `PSO/results_*/` 目录
- TensorBoard日志分别保存，支持同时可视化对比

**注意**: 首次运行建议使用快速模式熟悉系统，然后根据实际需求调整参数进行正式实验。