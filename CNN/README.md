# 中文数字识别CNN - 模块化版本

这是一个用于识别中文数字的卷积神经网络项目，采用模块化设计，支持动态学习率调整和完整的训练监控功能。

## 项目结构

```
CNN/
├── config.py              # 配置管理模块
├── data_loader.py          # 数据加载和预处理模块
├── model.py               # 基础CNN模型定义模块
├── trainer.py             # 基础CNN训练和评估模块
├── main.py                # 主程序入口
├── check_hardware.py      # 硬件检测工具
├── pso_particle.py        # PSO粒子定义模块
├── pso_model.py           # PSO动态CNN模型模块
├── pso_trainer.py         # PSO训练器模块
├── pso_optimizer.py       # PSO优化算法模块
├── pso_demo.py            # PSO演示程序
├── PSO_USAGE.md           # PSO使用说明文档
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖包列表
├── data/
│   └── minist/            # 训练数据目录
├── models/                # 基础模型保存目录
├── PSO/                   # PSO相关文件目录
└── runs/                  # 基础训练TensorBoard日志目录
```

## 功能特性

### 🔧 核心功能
- **配置管理**: 统一管理所有超参数和设置
- **数据处理**: 独立的数据加载和预处理模块
- **模型定义**: 基于LeNet-5的CNN网络结构
- **训练管理**: 完整的训练、评估和可视化流程
- **动态学习率**: 支持学习率自动调度和衰减
- **硬件检测**: 自动检测和优化硬件配置
- **实时监控**: TensorBoard集成和训练过程可视化

### 📊 数据处理
- 自动图像加载和预处理
- 图像大小调整为28x28
- 数据标准化 (均值0.5, 标准差0.5)
- 训练/测试集划分 (70%/30%)
- PyTorch DataLoader支持
- 支持15个中文数字类别

### 🧠 模型架构 (ChineseNumberCNN)
- **输入层**: 1×28×28 灰度图像
- **卷积层1**: 6个5×5卷积核 + ReLU + 2×2最大池化
- **卷积层2**: 16个5×5卷积核 + ReLU + 2×2最大池化
- **全连接层1**: 120个神经元 + ReLU
- **全连接层2**: 84个神经元 + ReLU
- **输出层**: 15个神经元 (对应15个中文数字类别)

### 📈 训练功能
- **动态学习率**: 从0.017开始，每3个epoch衰减33%
- **TensorBoard集成**: 实时监控训练指标和学习率变化
- **自动模型保存**: 训练完成后自动保存最佳模型
- **精确率-召回率曲线**: 每个类别的详细性能分析
- **分类报告**: 完整的性能评估报告
- **训练曲线可视化**: 自动生成并保存训练曲线图

### 🔍 PSO架构优化
- **粒子群优化**: 自动搜索最优CNN架构
- **动态模型构建**: 根据PSO参数动态创建CNN模型
- **架构参数优化**: 卷积层数、滤波器数量、全连接层配置
- **性能对比**: PSO优化架构与基准模型的性能比较
- **可视化分析**: PSO收敛过程和最优架构可视化

## 环境要求

- Python 3.8+
- PyTorch 1.13.1+ (支持CUDA)
- 其他依赖见requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 硬件检测

运行硬件检测工具，确保环境配置正确：

```bash
python check_hardware.py

# 保存检测结果
python check_hardware.py --save --file hardware_info.json

# 静默模式
python check_hardware.py --quiet
```

## 使用方法

### 1. 基础CNN训练

#### 配置设置

编辑 `config.py` 文件，设置数据路径和其他参数.

#### 训练模型

```bash
# 基本训练模式
python main.py --mode train

# 训练+评估模式
python main.py --mode both

# 仅评估模式
python main.py --mode eval --model_path models/chinese_number_cnn.pth

# 自定义参数训练
python main.py --epochs 20 --batch_size 64 --lr 0.01
```

### 2. 监控训练过程

```bash
# 启动TensorBoard查看训练过程
tensorboard --logdir=runs/mnist_experiment_1
# 可以查看:
# - 训练损失和准确率曲线
# - 学习率变化曲线
# - 每个类别的精确率-召回率曲线
# - 预测结果可视化
```

### 4. 查看结果

训练完成后，检查以下文件：
- `models/chinese_number_cnn.pth`: 训练好的模型
- `models/training_curves.png`: 训练曲线图
- 控制台输出: 详细的分类报告

## 配置参数说明

### 核心配置 (config.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EPOCHS` | 训练轮数 | 15 |
| `BATCH_SIZE` | 批次大小 | 32 |
| `LEARNING_RATE` | 初始学习率 | 0.017 |
| `LR_DECAY_FACTOR` | 学习率衰减因子 | 0.67 |
| `LR_DECAY_STEP` | 学习率衰减步长 | 3 |
| `MOMENTUM` | SGD动量 | 0.9 |
| `NUM_CLASSES` | 分类数量 | 15 |
| `TEST_SIZE` | 测试集比例 | 0.3 |

## 学习率调度策略

项目采用StepLR学习率调度器：

```python
# 学习率变化轨迹
Epoch 1-3:   学习率 = 0.017
Epoch 4-6:   学习率 = 0.011  (0.017 × 0.67)
Epoch 7-9:   学习率 = 0.007  (0.011 × 0.67)
Epoch 10-12: 学习率 = 0.005  (0.007 × 0.67)
Epoch 13-15: 学习率 = 0.003  (0.005 × 0.67)
```

这种策略有助于：
- **前期快速收敛**: 较高学习率加速训练
- **后期精细调优**: 降低学习率提高精度
- **避免震荡**: 防止在最优解附近震荡

## 输出文件

训练完成后，会在以下位置生成文件：

- `models/chinese_number_cnn.pth`: 完整的模型检查点
- `models/training_curves.png`: 训练和测试准确率曲线
- `runs/mnist_experiment_1/`: TensorBoard日志文件
- 控制台输出: 详细的分类报告和性能指标

## 数据集获取

### 数据来源

本项目使用中文数字手写识别数据集，可以从以下来源获取：

- **Kaggle数据集**: [Chinese MNIST](https://www.kaggle.com/datasets/gpreda/chinese-mnist/data)
- **数据集描述**: 包含15个中文数字字符的手写图像数据
- **数据规模**: 每个字符类别包含大量训练样本

### 数据格式要求

数据文件夹 `data/minist/` 应包含以下格式的图像文件：
- **文件格式**: `.jpg`
- **文件命名**: `input_X_Y_Z.jpg`
- **图像要求**: 28×28像素，灰度图像
- **类别标签**: 零、一、二、三、四、五、六、七、八、九、十、百、千、万、亿


## 故障排除

### 调试工具

```bash
# 检查硬件环境
python check_hardware.py

# 测试各个模块
python data_loader.py    # 测试数据加载
python model.py          # 测试模型创建
python config.py         # 查看配置信息
```

## 许可证

本项目仅供学习和交流使用。
