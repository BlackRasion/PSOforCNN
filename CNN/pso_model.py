# -*- coding: utf-8 -*-
"""
PSO动态CNN模型构建器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class PSODynamicCNN(nn.Module):
    """
    基于PSO粒子参数动态构建的CNN模型
    """
    
    def __init__(self, particle_position, num_classes=None):
        """
        初始化动态CNN模型
        
        Args:
            particle_position (dict): PSO粒子的位置参数
            num_classes (int): 分类数量
        """
        super(PSODynamicCNN, self).__init__()
        
        if num_classes is None:
            num_classes = Config.NUM_CLASSES
        
        self.particle_position = particle_position
        self.num_classes = num_classes
        
        # 构建卷积层
        self.conv_layers = self._build_conv_layers()
        
        # 计算卷积层输出尺寸
        self.conv_output_size = self._calculate_conv_output_size()
        
        # 构建全连接层
        self.fc_layers = self._build_fc_layers()
        
        # 展平层
        self.flatten = nn.Flatten()
        
    def _build_conv_layers(self):
        """
        构建卷积层
        
        Returns:
            nn.ModuleList: 卷积层列表
        """
        layers = nn.ModuleList()
        in_channels = 1  # 输入通道数（灰度图像）
        
        for i in range(self.particle_position['conv_layers']):
            out_channels = self.particle_position['conv_filters'][i]
            kernel_size = self.particle_position['conv_kernels'][i]
            
            # 卷积层
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2  # 保持尺寸
            )
            
            layers.append(conv_layer)
            in_channels = out_channels
        
        return layers
    
    def _calculate_conv_output_size(self):
        """
        计算卷积层输出的特征图尺寸
        
        Returns:
            int: 展平后的特征数量
        """
        # 创建一个测试输入来计算输出尺寸
        with torch.no_grad():
            x = torch.randn(1, 1, *Config.IMAGE_SIZE)
            
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            
            return x.numel()  # 总元素数量
    
    def _build_fc_layers(self):
        """
        构建全连接层
        
        Returns:
            nn.ModuleList: 全连接层列表
        """
        layers = nn.ModuleList()
        
        # 第一个全连接层的输入尺寸是卷积层的输出
        in_features = self.conv_output_size
        
        # 构建隐藏的全连接层
        for i in range(self.particle_position['fc_layers']):
            out_features = self.particle_position['fc_sizes'][i]
            
            fc_layer = nn.Linear(in_features, out_features)
            layers.append(fc_layer)
            in_features = out_features
        
        # 输出层
        output_layer = nn.Linear(in_features, self.num_classes)
        layers.append(output_layer)
        
        return layers
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        # 卷积层前向传播
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 展平
        x = self.flatten(x)
        
        # 全连接层前向传播
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            # 最后一层不使用激活函数
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
        
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
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'conv_layers': self.particle_position['conv_layers'],
            'conv_filters': self.particle_position['conv_filters'],
            'conv_kernels': self.particle_position['conv_kernels'],
            'fc_layers': self.particle_position['fc_layers'],
            'fc_sizes': self.particle_position['fc_sizes'],
            'conv_output_size': self.conv_output_size
        }
    
    def print_model_info(self):
        """
        打印模型信息
        """
        info = self.get_model_info()
        print("=" * 60)
        print("PSO动态CNN模型信息:")
        print(f"总参数数量: {info['total_parameters']:,}")
        print(f"可训练参数数量: {info['trainable_parameters']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        print(f"卷积层数量: {info['conv_layers']}")
        print(f"卷积核数量: {info['conv_filters']}")
        print(f"卷积核大小: {info['conv_kernels']}")
        print(f"全连接层数量: {info['fc_layers']}")
        print(f"全连接层大小: {info['fc_sizes']}")
        print(f"卷积输出尺寸: {info['conv_output_size']}")
        print("=" * 60)
    
    def get_architecture_string(self):
        """
        获取架构描述字符串
        
        Returns:
            str: 架构描述
        """
        arch_str = "CNN Architecture:\n"
        
        # 卷积层描述
        in_channels = 1
        for i in range(self.particle_position['conv_layers']):
            out_channels = self.particle_position['conv_filters'][i]
            kernel_size = self.particle_position['conv_kernels'][i]
            arch_str += f"  Conv{i+1}: {in_channels}->{out_channels}, kernel={kernel_size}x{kernel_size}\n"
            arch_str += f"  ReLU + MaxPool2d(2x2)\n"
            in_channels = out_channels
        
        arch_str += f"  Flatten: -> {self.conv_output_size}\n"
        
        # 全连接层描述
        in_features = self.conv_output_size
        for i in range(self.particle_position['fc_layers']):
            out_features = self.particle_position['fc_sizes'][i]
            arch_str += f"  FC{i+1}: {in_features}->{out_features} + ReLU\n"
            in_features = out_features
        
        # 输出层
        arch_str += f"  Output: {in_features}->{self.num_classes}\n"
        
        return arch_str


def create_pso_model(particle_position, device=None, num_classes=None):
    """
    根据PSO粒子位置创建动态CNN模型
    
    Args:
        particle_position (dict): PSO粒子位置参数
        device (str): 计算设备
        num_classes (int): 分类数量
        
    Returns:
        PSODynamicCNN: 动态CNN模型
    """
    if device is None:
        device = Config.DEVICE
    
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    
    model = PSODynamicCNN(particle_position, num_classes)
    model = model.to(device)
    
    return model


def save_pso_model(model, particle_position, filepath):
    """
    保存PSO模型
    
    Args:
        model (PSODynamicCNN): PSO模型
        particle_position (dict): 粒子位置参数
        filepath (str): 保存路径
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'particle_position': particle_position,
        'model_info': model.get_model_info(),
        'architecture_string': model.get_architecture_string()
    }, filepath)
    print(f"PSO模型已保存到: {filepath}")


def load_pso_model(filepath, device=None):
    """
    加载PSO模型
    
    Args:
        filepath (str): 模型文件路径
        device (str): 计算设备
        
    Returns:
        PSODynamicCNN: 加载的PSO模型
    """
    if device is None:
        device = Config.DEVICE
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = PSODynamicCNN(checkpoint['particle_position'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"PSO模型已从 {filepath} 加载")
    
    return model, checkpoint['particle_position']