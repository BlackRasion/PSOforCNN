# -*- coding: utf-8 -*-
"""
PSO粒子类 - 用于表示CNN架构
"""

import numpy as np
from config import Config


class PSOParticle:
    """
    PSO粒子类，用于表示CNN架构
    
    粒子编码方案:
    - conv_layers: 卷积层数量 (1-4)
    - conv_filters: 每层卷积核数量 [6, 16, 32, 64]
    - conv_kernels: 每层卷积核大小 [3, 5, 7]
    - fc_layers: 全连接层数量 (1-3)
    - fc_sizes: 每层全连接层大小 [64, 128, 256, 512]
    """
    
    def __init__(self, particle_id=None):
        """
        初始化粒子
        
        Args:
            particle_id (int): 粒子ID
        """
        self.particle_id = particle_id
        
        # 架构参数范围
        self.conv_layers_range = (2, 4)  # 卷积层数量范围
        self.conv_filters_options = [6, 16, 32, 64]  # 卷积核数量选项
        self.conv_kernels_options = [3, 5, 7]  # 卷积核大小选项
        self.fc_layers_range = (2, 4)  # 全连接层数量范围
        self.fc_sizes_options = [64, 128, 256, 512]  # 全连接层大小选项
        
        # 初始化位置和速度
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()
        
        # 个体最优
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        
        # 当前适应度
        self.fitness = float('-inf')
        
    def _initialize_position(self):
        """
        初始化粒子位置（架构参数）
        
        Returns:
            dict: 架构参数字典
        """
        position = {
            'conv_layers': np.random.randint(*self.conv_layers_range),
            'conv_filters': [],
            'conv_kernels': [],
            'fc_layers': np.random.randint(*self.fc_layers_range),
            'fc_sizes': []
        }
        
        # 随机初始化卷积层参数
        for _ in range(position['conv_layers']):
            position['conv_filters'].append(
                np.random.choice(self.conv_filters_options)
            )
            position['conv_kernels'].append(
                np.random.choice(self.conv_kernels_options)
            )
        
        # 随机初始化全连接层参数
        for _ in range(position['fc_layers']):
            position['fc_sizes'].append(
                np.random.choice(self.fc_sizes_options)
            )
        
        return position
    
    def _initialize_velocity(self):
        """
        初始化粒子速度
        
        Returns:
            dict: 速度字典
        """
        velocity = {
            'conv_layers': np.random.uniform(-1, 1),
            'conv_filters': [],
            'conv_kernels': [],
            'fc_layers': np.random.uniform(-1, 1),
            'fc_sizes': []
        }
        
        # 初始化卷积层速度
        for _ in range(len(self.position['conv_filters'])):
            velocity['conv_filters'].append(np.random.uniform(-1, 1))
            velocity['conv_kernels'].append(np.random.uniform(-1, 1))
        
        # 初始化全连接层速度
        for _ in range(len(self.position['fc_sizes'])):
            velocity['fc_sizes'].append(np.random.uniform(-1, 1))
        
        return velocity
    
    def update_velocity(self, global_best_position, w, c1, c2):
        """
        更新粒子速度
        
        Args:
            global_best_position (dict): 全局最优位置
            w (float): 惯性权重
            c1 (float): 个体学习因子
            c2 (float): 社会学习因子
        """
        r1, r2 = np.random.random(), np.random.random()
        
        # 更新层数速度
        self.velocity['conv_layers'] = (
            w * self.velocity['conv_layers'] +
            c1 * r1 * (self.best_position['conv_layers'] - self.position['conv_layers']) +
            c2 * r2 * (global_best_position['conv_layers'] - self.position['conv_layers'])
        )
        
        self.velocity['fc_layers'] = (
            w * self.velocity['fc_layers'] +
            c1 * r1 * (self.best_position['fc_layers'] - self.position['fc_layers']) +
            c2 * r2 * (global_best_position['fc_layers'] - self.position['fc_layers'])
        )
        
        # 更新卷积层参数速度
        max_conv_layers = max(
            len(self.position['conv_filters']),
            len(self.best_position['conv_filters']),
            len(global_best_position['conv_filters'])
        )
        
        for i in range(max_conv_layers):
            if i < len(self.velocity['conv_filters']):
                best_filter = self.best_position['conv_filters'][i] if i < len(self.best_position['conv_filters']) else self.conv_filters_options[0]
                global_filter = global_best_position['conv_filters'][i] if i < len(global_best_position['conv_filters']) else self.conv_filters_options[0]
                current_filter = self.position['conv_filters'][i] if i < len(self.position['conv_filters']) else self.conv_filters_options[0]
                
                self.velocity['conv_filters'][i] = (
                    w * self.velocity['conv_filters'][i] +
                    c1 * r1 * (best_filter - current_filter) +
                    c2 * r2 * (global_filter - current_filter)
                )
                
                best_kernel = self.best_position['conv_kernels'][i] if i < len(self.best_position['conv_kernels']) else self.conv_kernels_options[0]
                global_kernel = global_best_position['conv_kernels'][i] if i < len(global_best_position['conv_kernels']) else self.conv_kernels_options[0]
                current_kernel = self.position['conv_kernels'][i] if i < len(self.position['conv_kernels']) else self.conv_kernels_options[0]
                
                self.velocity['conv_kernels'][i] = (
                    w * self.velocity['conv_kernels'][i] +
                    c1 * r1 * (best_kernel - current_kernel) +
                    c2 * r2 * (global_kernel - current_kernel)
                )
        
        # 更新全连接层参数速度
        max_fc_layers = max(
            len(self.position['fc_sizes']),
            len(self.best_position['fc_sizes']),
            len(global_best_position['fc_sizes'])
        )
        
        for i in range(max_fc_layers):
            if i < len(self.velocity['fc_sizes']):
                best_size = self.best_position['fc_sizes'][i] if i < len(self.best_position['fc_sizes']) else self.fc_sizes_options[0]
                global_size = global_best_position['fc_sizes'][i] if i < len(global_best_position['fc_sizes']) else self.fc_sizes_options[0]
                current_size = self.position['fc_sizes'][i] if i < len(self.position['fc_sizes']) else self.fc_sizes_options[0]
                
                self.velocity['fc_sizes'][i] = (
                    w * self.velocity['fc_sizes'][i] +
                    c1 * r1 * (best_size - current_size) +
                    c2 * r2 * (global_size - current_size)
                )
    
    def update_position(self):
        """
        更新粒子位置
        """
        # 更新层数
        self.position['conv_layers'] = int(np.clip(
            self.position['conv_layers'] + self.velocity['conv_layers'],
            *self.conv_layers_range
        ))
        
        self.position['fc_layers'] = int(np.clip(
            self.position['fc_layers'] + self.velocity['fc_layers'],
            *self.fc_layers_range
        ))
        
        # 调整卷积层参数数组长度
        current_conv_layers = self.position['conv_layers']
        while len(self.position['conv_filters']) < current_conv_layers:
            self.position['conv_filters'].append(np.random.choice(self.conv_filters_options))
            self.position['conv_kernels'].append(np.random.choice(self.conv_kernels_options))
            self.velocity['conv_filters'].append(np.random.uniform(-1, 1))
            self.velocity['conv_kernels'].append(np.random.uniform(-1, 1))
        
        while len(self.position['conv_filters']) > current_conv_layers:
            self.position['conv_filters'].pop()
            self.position['conv_kernels'].pop()
            self.velocity['conv_filters'].pop()
            self.velocity['conv_kernels'].pop()
        
        # 更新卷积层参数
        for i in range(len(self.position['conv_filters'])):
            new_filter = self.position['conv_filters'][i] + self.velocity['conv_filters'][i]
            self.position['conv_filters'][i] = min(self.conv_filters_options, 
                                                  key=lambda x: abs(x - new_filter))
            
            new_kernel = self.position['conv_kernels'][i] + self.velocity['conv_kernels'][i]
            self.position['conv_kernels'][i] = min(self.conv_kernels_options,
                                                  key=lambda x: abs(x - new_kernel))
        
        # 调整全连接层参数数组长度
        current_fc_layers = self.position['fc_layers']
        while len(self.position['fc_sizes']) < current_fc_layers:
            self.position['fc_sizes'].append(np.random.choice(self.fc_sizes_options))
            self.velocity['fc_sizes'].append(np.random.uniform(-1, 1))
        
        while len(self.position['fc_sizes']) > current_fc_layers:
            self.position['fc_sizes'].pop()
            self.velocity['fc_sizes'].pop()
        
        # 更新全连接层参数
        for i in range(len(self.position['fc_sizes'])):
            new_size = self.position['fc_sizes'][i] + self.velocity['fc_sizes'][i]
            self.position['fc_sizes'][i] = min(self.fc_sizes_options,
                                              key=lambda x: abs(x - new_size))
    
    def update_best(self):
        """
        更新个体最优
        """
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
    
    def get_architecture_info(self):
        """
        获取架构信息字符串
        
        Returns:
            str: 架构描述
        """
        info = f"Particle {self.particle_id}:\n"
        info += f"  Conv layers: {self.position['conv_layers']}\n"
        info += f"  Conv filters: {self.position['conv_filters']}\n"
        info += f"  Conv kernels: {self.position['conv_kernels']}\n"
        info += f"  FC layers: {self.position['fc_layers']}\n"
        info += f"  FC sizes: {self.position['fc_sizes']}\n"
        info += f"  Fitness: {self.fitness:.4f}\n"
        return info
    
    def copy(self):
        """
        复制粒子
        
        Returns:
            PSOParticle: 粒子副本
        """
        new_particle = PSOParticle(self.particle_id)
        new_particle.position = self.position.copy()
        new_particle.velocity = self.velocity.copy()
        new_particle.best_position = self.best_position.copy()
        new_particle.best_fitness = self.best_fitness
        new_particle.fitness = self.fitness
        return new_particle