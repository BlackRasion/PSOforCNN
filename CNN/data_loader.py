# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""

import os
import numpy as np
import matplotlib.image as mpig
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

from config import Config


class ChineseNumberDataset:
    """
    中文数字数据集加载和预处理类
    功能:
    1. 读取图像文件
    2. 调整图像大小为28x28
    3. 处理标签
    4. 生成one-hot编码
    5. 划分训练集和测试集
    """
    
    def __init__(self, data_path):
        """
        初始化数据集
        
        Args:
            data_path (str): 数据文件夹路径
        """
        self.data_path = data_path
        self.images = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """加载数据"""
        print(f"正在从 {self.data_path} 加载数据...")
        
        for filename in os.listdir(self.data_path):
            if filename.endswith('.jpg'):
                # 读取图像
                img_path = os.path.join(self.data_path, filename)
                image = mpig.imread(img_path, 0)
                image = resize(image, Config.IMAGE_SIZE)
                
                # 提取标签
                label_str = filename.replace('.jpg', '')
                label = int(label_str.split('_')[-1]) - 1  # 标签从0开始
                
                self.images.append(image)
                self.labels.append(label)
        
        # 转换为numpy数组
        self.data = np.array(self.images)[:, :, :, np.newaxis]  # 添加通道维度
        self.targets = np.array(self.labels)
        self.targets_hot = self._onehot_encode(self.targets, Config.NUM_CLASSES)
        
        print(f"数据加载完成: {len(self.data)} 个样本")
        print(f"图像形状: {self.data.shape}")
        print(f"标签形状: {self.targets.shape}")
    
    def _onehot_encode(self, targets, num_classes):
        """One-hot编码"""
        return np.eye(num_classes)[targets]
    
    def split_data(self, test_size=None, random_state=None):
        """
        划分训练集和测试集
        
        Args:
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        Returns:
            tuple: (X_train, y_train, y_train_hot, X_test, y_test, y_test_hot)
        """
        if test_size is None:
            test_size = Config.TEST_SIZE
        if random_state is None:
            random_state = Config.RANDOM_STATE
            
        X_train, X_test, y_train, y_test, y_train_hot, y_test_hot = train_test_split(
            self.data, self.targets, self.targets_hot, 
            test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        return X_train, y_train, y_train_hot, X_test, y_test, y_test_hot


def create_data_loaders(X_train, y_train, X_test, y_test):
    """
    创建PyTorch数据加载器
    
    Args:
        X_train, y_train: 训练数据和标签
        X_test, y_test: 测试数据和标签
        
    Returns:
        tuple: (trainloader, testloader)
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 创建数据集
    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.SHUFFLE_TRAIN,
        num_workers=Config.NUM_WORKERS,
    )
    
    testloader = DataLoader(
        testset,
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.SHUFFLE_TEST,
        num_workers=Config.NUM_WORKERS,
    )
    
    print(f"数据加载器创建完成:")
    print(f"训练批次数: {len(trainloader)}")
    print(f"测试批次数: {len(testloader)}")
    
    return trainloader, testloader


def get_transforms():
    """
    获取数据预处理变换
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),
    ])
    return transform


if __name__ == "__main__":
    # 测试数据加载
    dataset = ChineseNumberDataset(Config.DATA_PATH)
    X_train, y_train, y_train_hot, X_test, y_test, y_test_hot = dataset.split_data()
    trainloader, testloader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    print("\n数据加载测试完成!")