# 中文数字识别CNN - 模块化版本

这是一个用于识别中文数字的卷积神经网络项目，采用模块化设计，将原始的Jupyter Notebook重构为多个Python模块。

## 项目结构

```
AI_experiments/
├── config.py          # 配置管理模块
├── data_loader.py      # 数据加载和预处理模块
├── model.py           # 模型定义模块
├── trainer.py         # 训练和评估模块
├── utils.py           # 工具和可视化模块
├── main.py            # 主程序入口
├── README.md          # 项目说明文档
├── requirements.txt   # 依赖包列表
├── models/            # 模型保存目录
├── logs/              # TensorBoard日志目录
└── results/           # 结果保存目录
```

## 功能特性

### 🔧 模块化设计
- **配置管理**: 统一管理所有超参数和设置
- **数据处理**: 独立的数据加载和预处理模块
- **模型定义**: 清晰的CNN网络结构定义
- **训练管理**: 完整的训练、评估和可视化流程
- **工具集合**: 丰富的分析和可视化工具

### 📊 数据处理
- 自动图像加载和预处理
- 图像大小调整为28x28
- One-hot编码
- 训练/测试集划分
- PyTorch数据加载器

### 🧠 模型架构
- 类LeNet-5的CNN结构
- 2个卷积层 + 最大池化
- 3个全连接层
- 支持15个中文数字类别

### 📈 训练功能
- TensorBoard可视化
- 训练过程监控
- 模型自动保存
- 精确率-召回率曲线
- 分类报告生成

### 🔍 分析工具
- 混淆矩阵可视化
- 错误分析
- 类别分布统计
- 预测结果可视化
- 训练曲线绘制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置设置

编辑 `config.py` 文件，设置数据路径和其他参数：

```python
# 数据配置
DATA_PATH = "your/data/path"  # 修改为你的数据路径

# 训练配置
NUM_EPOCHS = 7
BATCH_SIZE = 4
LEARNING_RATE = 0.001
```

### 2. 训练模型

```bash
# 基本训练
python main.py --mode train

# 自定义参数训练
python main.py --mode train --epochs 10 --batch_size 8 --lr 0.01

# 指定数据路径
python main.py --mode train --data_path "path/to/your/data"
```

### 3. 评估模型

```bash
# 评估已训练的模型
python main.py --mode eval --model_path "models/chinese_number_cnn.pth"

# 训练后立即评估
python main.py --mode both
```

### 4. 查看训练过程

```bash
# 启动TensorBoard
tensorboard --logdir=logs
```

然后在浏览器中访问 `http://localhost:6006`

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式 (train/eval/both) | train |
| `--model_path` | 模型文件路径 | models/chinese_number_cnn.pth |
| `--data_path` | 数据路径 | 配置文件中的路径 |
| `--epochs` | 训练轮数 | 配置文件中的设置 |
| `--batch_size` | 批次大小 | 配置文件中的设置 |
| `--lr` | 学习率 | 配置文件中的设置 |

## 配置说明

### 数据配置
- `DATA_PATH`: 数据文件夹路径，包含.jpg格式的图像文件
- `IMAGE_SIZE`: 图像大小，默认(28, 28)
- `TEST_SIZE`: 测试集比例，默认0.2

### 模型配置
- `NUM_CLASSES`: 分类数量，默认15
- `INPUT_CHANNELS`: 输入通道数，默认1
- `CONV1_OUT_CHANNELS`: 第一卷积层输出通道，默认6
- `CONV2_OUT_CHANNELS`: 第二卷积层输出通道，默认16

### 训练配置
- `NUM_EPOCHS`: 训练轮数，默认7
- `BATCH_SIZE`: 批次大小，默认4
- `LEARNING_RATE`: 学习率，默认0.001
- `MOMENTUM`: 动量，默认0.9

## 输出文件

训练完成后，会在以下位置生成文件：

- `models/chinese_number_cnn.pth`: 训练好的模型
- `models/training_curves.png`: 训练曲线图
- `logs/`: TensorBoard日志文件
- `results/`: 分析结果和报告

## 数据格式要求

数据文件夹应包含以下格式的图像文件：
- 文件格式：`.jpg`
- 文件命名：`*_数字.jpg`（例如：`sample_1.jpg`表示数字1）
- 数字范围：1-15（对应15个中文数字类别）

## 模型性能

基于原始配置的典型性能：
- 准确率：约75%
- 训练时间：约几分钟（取决于硬件）
- 模型大小：约1MB

## 优化建议

1. **数据增强**：添加旋转、缩放、噪声等数据增强技术
2. **网络结构**：尝试更深的网络或ResNet结构
3. **正则化**：添加Dropout和BatchNorm层
4. **学习率调度**：使用学习率衰减策略
5. **更多数据**：增加训练数据量

## 故障排除

### 常见问题

1. **数据路径错误**
   ```
   错误: 数据路径不存在
   解决: 检查config.py中的DATA_PATH设置
   ```

2. **CUDA内存不足**
   ```
   解决: 减小BATCH_SIZE或使用CPU训练
   ```

3. **依赖包缺失**
   ```
   解决: pip install -r requirements.txt
   ```

### 调试模式

可以在各个模块的`if __name__ == "__main__":`部分运行单独测试：

```bash
# 测试数据加载
python data_loader.py

# 测试模型创建
python model.py

# 测试工具功能
python utils.py
```

## 扩展功能

项目支持以下扩展：

1. **添加新的数据增强方法**
2. **实现不同的网络架构**
3. **集成其他优化器**
4. **添加模型集成功能**
5. **实现在线推理接口**

## 许可证

本项目仅供学习和研究使用。

## 更新日志

- v1.0: 初始模块化版本
- 重构原始Jupyter Notebook
- 添加配置管理
- 实现完整的训练和评估流程
- 添加丰富的可视化和分析工具