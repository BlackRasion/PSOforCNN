# 中文数字识别CNN - 模块化版本

这是一个用于识别中文数字的卷积神经网络项目，采用模块化设计，支持动态学习率调整和完整的训练监控功能。

## 项目结构

```
CNN/
├── config.py              # 配置管理模块
├── data_loader.py          # 数据加载和预处理模块
├── model.py               # 模型定义模块
├── trainer.py             # 训练和评估模块
├── utils.py               # 工具和可视化模块
├── main.py                # 主程序入口
├── check_hardware.py      # 硬件检测工具
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖包列表
├── data/
│   └── minist/            # 训练数据目录
├── models/                # 模型保存目录
│   ├── chinese_number_cnn.pth
│   └── training_curves.png
├── runs/                  # TensorBoard日志目录
│   └── mnist_experiment_1/
└── results/               # 结果保存目录
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
- **动态学习率**: 从0.02开始，每5个epoch衰减50%
- **TensorBoard集成**: 实时监控训练指标和学习率变化
- **自动模型保存**: 训练完成后自动保存最佳模型
- **精确率-召回率曲线**: 每个类别的详细性能分析
- **分类报告**: 完整的性能评估报告
- **训练曲线可视化**: 自动生成并保存训练曲线图

### 🔍 分析工具
- 混淆矩阵可视化
- 错误分析
- 类别分布统计
- 预测结果可视化
- 训练曲线绘制

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

### 1. 配置设置

编辑 `config.py` 文件，设置数据路径和其他参数：

```python
# 数据配置
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "minist")
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 15
TEST_SIZE = 0.3

# 训练配置
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.02  # 初始学习率
MOMENTUM = 0.9
LR_DECAY_FACTOR = 0.5  # 学习率衰减因子
LR_DECAY_STEP = 5      # 每5个epoch衰减一次
```

### 2. 训练模型

```bash
# 基本训练
python main.py

# 查看训练过程
# 训练会自动显示每个epoch的:
# - 训练准确率
# - 测试准确率  
# - 当前学习率
# - 训练时间
```

### 3. 监控训练过程

```bash
# 启动TensorBoard查看训练过程
tensorboard --logdir=runs/mnist_experiment_1

# 在浏览器中访问: http://localhost:6006
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
| `LEARNING_RATE` | 初始学习率 | 0.02 |
| `LR_DECAY_FACTOR` | 学习率衰减因子 | 0.5 |
| `LR_DECAY_STEP` | 学习率衰减步长 | 5 |
| `MOMENTUM` | SGD动量 | 0.9 |
| `NUM_CLASSES` | 分类数量 | 15 |
| `TEST_SIZE` | 测试集比例 | 0.3 |

## 学习率调度策略

项目采用StepLR学习率调度器：

```python
# 学习率变化轨迹
Epoch 1-5:   学习率 = 0.02
Epoch 6-10:  学习率 = 0.01  (0.02 × 0.5)
Epoch 11-15: 学习率 = 0.005 (0.01 × 0.5)
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

## 数据格式要求

数据文件夹 `data/minist/` 应包含以下格式的图像文件：
- **文件格式**: `.jpg`
- **文件命名**: `input_X_Y_Z.jpg`
  - X: 数据集编号 (1, 2, 3, 4)
  - Y: 类别编号 (1-10对应数字0-9, 11-15对应十、百、千、万、亿)
  - Z: 样本编号 (1-15)
- **图像要求**: 28×28像素，灰度图像
- **类别映射**: 
  ```
  1→零, 2→一, 3→二, 4→三, 5→四, 6→五, 7→六, 8→七, 9→八, 10→九
  11→十, 12→百, 13→千, 14→万, 15→亿
  ```

## 模型性能

基于当前配置的典型性能：
- **预期准确率**: 80%+ (使用动态学习率调整)
- **训练时间**: 约5-10分钟 (取决于硬件)
- **模型大小**: 约1MB
- **参数数量**: ~61,706个可训练参数
- **支持设备**: CPU / CUDA GPU / MPS (Apple Silicon)

## 进一步优化建议

1. **数据增强**: 添加旋转、缩放、噪声等数据增强技术
2. **网络结构**: 尝试ResNet、DenseNet等更深的网络
3. **正则化**: 添加Dropout和BatchNorm层防止过拟合
4. **优化器**: 尝试Adam、AdamW等自适应学习率优化器
5. **学习率调度**: 尝试CosineAnnealingLR、ReduceLROnPlateau等策略
6. **数据扩充**: 增加训练数据量和多样性
7. **模型集成**: 使用多个模型进行集成预测

## 故障排除

### 常见问题

1. **数据路径错误**
   ```
   错误: 找不到数据文件
   解决: 确保 data/minist/ 目录存在且包含图像文件
   ```

2. **CUDA内存不足**
   ```
   解决: 减小BATCH_SIZE (如改为16或8) 或使用CPU训练
   ```

3. **PyTorch版本兼容性**
   ```
   错误: torch.cuda.memory_reserved 不存在
   解决: 已在check_hardware.py中处理版本兼容性
   ```

4. **学习率过高导致训练不稳定**
   ```
   解决: 降低初始学习率或调整衰减参数
   ```

### 调试工具

```bash
# 检查硬件环境
python check_hardware.py

# 测试各个模块
python data_loader.py    # 测试数据加载
python model.py          # 测试模型创建
python config.py         # 查看配置信息
```

## 项目特色

✅ **完全模块化设计** - 清晰的代码结构和职责分离  
✅ **动态学习率调整** - 自动优化训练过程  
✅ **实时训练监控** - TensorBoard集成可视化  
✅ **硬件自适应** - 自动检测和配置最佳设备  
✅ **版本兼容性** - 支持不同PyTorch版本  
✅ **详细性能分析** - 完整的分类报告和曲线图  
✅ **易于扩展** - 清晰的接口设计便于功能扩展  

## 许可证

本项目仅供学习和研究使用。

## 更新日志

- **v2.0**: 动态学习率调整版本
  - 添加StepLR学习率调度器
  - 优化训练监控和可视化
  - 增强硬件检测功能
  - 改进版本兼容性
- **v1.0**: 初始模块化版本
  - 重构原始Jupyter Notebook
  - 实现完整的训练和评估流程
  - 添加TensorBoard集成