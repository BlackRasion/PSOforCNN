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
    BATCH_SIZE = 4 # 批次大小
    NUM_WORKERS = 2 # 工作线程数
    SHUFFLE_TRAIN = True # 训练集是否随机洗牌
    SHUFFLE_TEST = False  # 测试集是否随机洗牌
    
    # 训练配置
    EPOCHS = 7
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    LOG_INTERVAL = 1000  # 每1000个batch记录一次
    MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
    
    # TensorBoard配置
    TENSORBOARD_LOG_DIR = 'runs/mnist_experiment_1'
    
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
