# -*- coding: utf-8 -*-
"""
配置文件 - 管理所有超参数和设置
"""

import torch
import os

class Config:
    """配置类，包含所有超参数和设置"""
    
    # 数据相关配置
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "data", "minist")
    IMAGE_SIZE = (28, 28) # 图像大小
    NUM_CLASSES = 15 # 类别数量
    TEST_SIZE = 0.3 # 测试集比例
    RANDOM_STATE = 42 # 随机种子
    
    # 数据加载器配置
    BATCH_SIZE = 32 # 批次大小
    NUM_WORKERS = 8 # 工作线程数
    SHUFFLE_TRAIN = True # 训练集是否随机洗牌
    SHUFFLE_TEST = False  # 测试集是否随机洗牌
    
    # 训练配置
    EPOCHS = 15
    LEARNING_RATE = 0.017  # 初始学习率
    MOMENTUM = 0.9 # 动量
    LR_DECAY_FACTOR = 0.67  # 学习率衰减因子
    LR_DECAY_STEP = 3  # 每3个epoch衰减一次
    LOG_INTERVAL = 1000  # 每1000个batch记录一次
    MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
    
    # TensorBoard配置
    TENSORBOARD_LOG_DIR = 'runs/mnist_experiment_1'
    
    # PSO配置
    PSO_PARTICLE_SIZE = 10  # 粒子群大小
    PSO_ITERATIONS = 5  # PSO迭代次数
    PSO_TRAIN_EPOCHS = 8  # 每个粒子训练的epochs数
    PSO_FINAL_EPOCHS = 15  # 最优架构的最终训练epochs
    PSO_SAVE_DIR = os.path.join(SCRIPT_DIR, "PSO")  # PSO结果保存目录
    PSO_TENSORBOARD_DIR = 'PSO/runs'  # PSO TensorBoard日志目录
    
    # PSO算法参数
    PSO_W = 0.9  # 惯性权重
    PSO_C1 = 2.0  # 个体学习因子
    PSO_C2 = 2.0  # 社会学习因子
    PSO_W_MIN = 0.4  # 最小惯性权重
    PSO_W_MAX = 0.9  # 最大惯性权重
    
    # 设备配置
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # 类别标签
    CLASSES = [
        "零", "一", "二", "三", "四",
        "五", "六", "七", "八", "九",
        "十", "百", "千", "万", "亿"
    ]
    
    # 数据预处理配置
    NORMALIZE_MEAN = (0.5,)
    NORMALIZE_STD = (0.5,)
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 50)
        print("当前配置:")
        print(f"设备: {cls.DEVICE}")
        print(f"数据集路径: {cls.DATA_PATH}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.EPOCHS}")
        print(f"类别数量: {cls.NUM_CLASSES}")
        print("=" * 50)
if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
