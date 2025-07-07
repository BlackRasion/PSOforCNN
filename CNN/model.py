# -*- coding: utf-8 -*-
"""
模型定义模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ChineseNumberCNN(nn.Module):
    """
    中文数字识别CNN模型
    
    网络结构:
    - 输入: 1通道 28x28 图像
    - Conv1: 1->6 通道, 5x5卷积核
    - MaxPool1: 2x2池化
    - Conv2: 6->16 通道, 5x5卷积核  
    - MaxPool2: 2x2池化
    - Flatten
    - FC1: 256->120
    - FC2: 120->84
    - FC3: 84->15 (输出层)
    """
    
    # 模型配置参数
    INPUT_CHANNELS = 1
    CONV1_OUT_CHANNELS = 6
    CONV2_OUT_CHANNELS = 16
    CONV_KERNEL_SIZE = 5
    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2
    FC1_SIZE = 120
    FC2_SIZE = 84
    
    def __init__(self, num_classes=None):
        """
        初始化模型
        
        Args:
            num_classes (int): 分类数量，默认使用配置文件中的值
        """
        super(ChineseNumberCNN, self).__init__()
        
        if num_classes is None:
            num_classes = Config.NUM_CLASSES
            
        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=self.INPUT_CHANNELS,
            out_channels=self.CONV1_OUT_CHANNELS,
            kernel_size=self.CONV_KERNEL_SIZE
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=self.CONV1_OUT_CHANNELS,
            out_channels=self.CONV2_OUT_CHANNELS,
            kernel_size=self.CONV_KERNEL_SIZE
        )
        
        # 池化层
        self.max_pool2d = nn.MaxPool2d(
            kernel_size=self.POOL_KERNEL_SIZE,
            stride=self.POOL_STRIDE
        )
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 全连接层
        # 计算展平后的特征数量: 16 * 4 * 4 = 256
        # (28-5+1)/2 = 12, (12-5+1)/2 = 4
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.CONV2_OUT_CHANNELS * 4 * 4, self.FC1_SIZE),
            nn.ReLU(),
            nn.Linear(self.FC1_SIZE, self.FC2_SIZE),
            nn.ReLU(),
            nn.Linear(self.FC2_SIZE, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量 (N, 1, 28, 28)
            
        Returns:
            torch.Tensor: 输出张量 (N, num_classes)
        """
        # 第一个卷积块: Conv1 + ReLU + MaxPool
        # (N, 1, 28, 28) -> (N, 6, 24, 24) -> (N, 6, 12, 12)
        x = self.max_pool2d(F.relu(self.conv1(x)))
        
        # 第二个卷积块: Conv2 + ReLU + MaxPool
        # (N, 6, 12, 12) -> (N, 16, 8, 8) -> (N, 16, 4, 4)
        x = self.max_pool2d(F.relu(self.conv2(x)))
        
        # 展平: (N, 16, 4, 4) -> (N, 256)
        x = self.flatten(x)
        
        # 全连接层: (N, 256) -> (N, num_classes)
        x = self.linear_relu_stack(x)
        
        return x
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("=" * 50)
        print("模型信息:")
        print(f"总参数数量: {info['total_parameters']:,}")
        print(f"可训练参数数量: {info['trainable_parameters']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        print("=" * 50)


def create_model(device=None):
    """
    创建模型实例
    
    Args:
        device (str): 设备类型，默认使用配置文件中的设备
        
    Returns:
        ChineseNumberCNN: 模型实例
    """
    if device is None:
        device = Config.DEVICE
        
    model = ChineseNumberCNN()
    model = model.to(device)
    
    print(f"模型已创建并移动到设备: {device}")
    model.print_model_info()
    
    return model


def save_model(model, filepath):
    """
    保存模型
    
    Args:
        model (nn.Module): 要保存的模型
        filepath (str): 保存路径
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info(),
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'input_channels': Config.INPUT_CHANNELS,
        }
    }, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(filepath, device=None):
    """
    加载模型
    
    Args:
        filepath (str): 模型文件路径
        device (str): 设备类型
        
    Returns:
        ChineseNumberCNN: 加载的模型
    """
    if device is None:
        device = Config.DEVICE
        
    checkpoint = torch.load(filepath, map_location=device)
    
    model = ChineseNumberCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"模型已从 {filepath} 加载")
    
    # 检查是否存在model_info字段
    if 'model_info' in checkpoint:
        print(f"模型信息: {checkpoint['model_info']}")
    else:
        # 如果没有保存的模型信息，重新计算
        model_info = model.get_model_info()
        print(f"模型信息: {model_info}")
    
    return model


if __name__ == "__main__":
    # 测试模型创建
    model = create_model()
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28).to(Config.DEVICE)
    output = model(dummy_input)
    print(f"\n测试输入形状: {dummy_input.shape}")
    print(f"测试输出形状: {output.shape}")
    print(f"输出值范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\n模型测试完成!")