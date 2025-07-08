# -*- coding: utf-8 -*-
"""
PSO优化器 - 实现粒子群优化算法
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from pso_particle import PSOParticle
from pso_trainer import PSOTrainer


class PSOOptimizer:
    """
    PSO优化器类
    
    功能:
    1. 初始化粒子群
    2. 迭代优化
    3. 结果分析和可视化
    """
    
    def __init__(self, trainloader, testloader, device=None):
        """
        初始化PSO优化器
        
        Args:
            trainloader: 训练数据加载器
            testloader: 测试数据加载器
            device: 计算设备
        """
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device else Config.DEVICE
        
        # 创建PSO训练器
        self.trainer = PSOTrainer(trainloader, testloader, device)
        
        # PSO参数
        self.particle_size = Config.PSO_PARTICLE_SIZE
        self.iterations = Config.PSO_ITERATIONS
        self.w_min = Config.PSO_W_MIN
        self.w_max = Config.PSO_W_MAX
        self.c1 = Config.PSO_C1
        self.c2 = Config.PSO_C2
        
        # 初始化粒子群
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # 优化历史
        self.history = {
            'global_best_fitness': [],
            'avg_fitness': [],
            'best_positions': [],
            'iteration_times': []
        }
        
        # 结果保存目录
        self.save_dir = Config.PSO_SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'runs'), exist_ok=True)
        
        print(f"PSO优化器初始化完成")
        print(f"粒子数量: {self.particle_size}")
        print(f"迭代次数: {self.iterations}")
        print(f"惯性权重范围: [{self.w_min}, {self.w_max}]")
        print(f"个体学习因子: {self.c1}")
        print(f"社会学习因子: {self.c2}")
    
    def initialize_particles(self):
        """
        初始化粒子群
        """
        print("\n初始化粒子群...")
        self.particles = []
        
        for i in range(self.particle_size):
            particle = PSOParticle(particle_id=i)
            self.particles.append(particle)
            print(f"初始化粒子 {i+1}/{self.particle_size}")
        
        print(f"粒子群初始化完成，共 {len(self.particles)} 个粒子")
    
    def evaluate_particles(self, particles=None, quick_eval=False):
        """
        评估粒子群
        
        Args:
            particles (list): 要评估的粒子列表，默认为所有粒子
            quick_eval (bool): 是否使用快速评估
        """
        if particles is None:
            particles = self.particles
        
        print(f"\n评估 {len(particles)} 个粒子...")
        start_time = time.time()
        
        for i, particle in enumerate(particles):
            print(f"\n评估粒子 {particle.particle_id+1}/{len(particles)}")
            
            if quick_eval:
                # 快速评估
                fitness = self.trainer.quick_evaluate(particle.position)
            else:
                # 完整训练和评估
                fitness = self.trainer.train_particle(
                    particle, 
                    epochs=Config.PSO_TRAIN_EPOCHS,
                    verbose=True
                )
            
            particle.fitness = fitness
            
            # 更新个体最优
            particle.update_best()
            
            # 更新全局最优
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
                print(f"发现新的全局最优! 准确率: {fitness:.2f}%")
        
        eval_time = time.time() - start_time
        print(f"粒子评估完成，用时: {eval_time:.2f}s")
        print(f"当前全局最优适应度: {self.global_best_fitness:.2f}%")
    
    def update_particles(self, iteration):
        """
        更新粒子群
        
        Args:
            iteration (int): 当前迭代次数
        """
        # 计算当前惯性权重（线性递减）
        w = self.w_max - (self.w_max - self.w_min) * iteration / self.iterations
        
        for particle in self.particles:
            # 更新速度
            particle.update_velocity(
                self.global_best_position,
                w=w,
                c1=self.c1,
                c2=self.c2
            )
            
            # 更新位置
            particle.update_position()
    
    def optimize(self):
        """
        执行PSO优化
        
        Returns:
            dict: 优化结果
        """
        print("\n" + "=" * 60)
        print("开始PSO优化")
        print("=" * 60)
        
        # 初始化粒子群
        self.initialize_particles()
        
        # 初始评估
        print("\n初始评估粒子群...")
        self.evaluate_particles()
        
        # 记录初始全局最优
        self.history['global_best_fitness'].append(self.global_best_fitness)
        self.history['best_positions'].append(self.global_best_position.copy())
        avg_fitness = np.mean([p.fitness for p in self.particles])
        self.history['avg_fitness'].append(avg_fitness)
        
        # 迭代优化
        total_start_time = time.time()
        
        for iteration in range(self.iterations):
            iter_start_time = time.time()
            
            print(f"\n开始第 {iteration+1}/{self.iterations} 次迭代")
            
            # 更新粒子
            self.update_particles(iteration)
            
            # 评估粒子
            self.evaluate_particles()
            
            # 记录历史
            self.history['global_best_fitness'].append(self.global_best_fitness)
            self.history['best_positions'].append(self.global_best_position.copy())
            avg_fitness = np.mean([p.fitness for p in self.particles])
            self.history['avg_fitness'].append(avg_fitness)
            
            iter_time = time.time() - iter_start_time
            self.history['iteration_times'].append(iter_time)
            
            print(f"第 {iteration+1} 次迭代完成，用时: {iter_time:.2f}s")
            print(f"全局最优适应度: {self.global_best_fitness:.2f}%")
            print(f"平均适应度: {avg_fitness:.2f}%")
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("PSO优化完成")
        print(f"总用时: {total_time:.2f}s")
        print(f"最优适应度: {self.global_best_fitness:.2f}%")
        print("=" * 60)
        
        # 保存优化结果
        self.save_results()
        
        # 训练最终模型
        self.train_final_model()
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'history': self.history,
            'total_time': total_time
        }
    
    def save_results(self):
        """
        保存优化结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.save_dir, f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存优化历史
        history_file = os.path.join(results_dir, "optimization_history.json")
        
        # 将numpy数组转换为列表
        serializable_history = {
            'global_best_fitness': self.history['global_best_fitness'],
            'avg_fitness': self.history['avg_fitness'],
            'iteration_times': self.history['iteration_times'],
            # 不保存位置历史，因为它包含复杂的嵌套字典
        }
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        # 保存最优位置
        best_position_file = os.path.join(results_dir, "best_position.json")
        with open(best_position_file, 'w') as f:
            # 将numpy数组转换为列表
            serializable_position = self.global_best_position.copy()
            for key in serializable_position:
                if isinstance(serializable_position[key], np.ndarray):
                    serializable_position[key] = serializable_position[key].tolist()
            json.dump(serializable_position, f, indent=2)
        
        # 绘制优化曲线
        self.plot_optimization_curves(results_dir)
        
        print(f"优化结果已保存到: {results_dir}")
    
    def plot_optimization_curves(self, save_dir):
        """
        绘制优化曲线
        
        Args:
            save_dir (str): 保存目录
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制适应度曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.history['global_best_fitness'], 'r-', label='Global Best')
        plt.plot(self.history['avg_fitness'], 'b--', label='Avg Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (%)')
        plt.title('PSO Optimization Fitness Curve')
        plt.legend()
        plt.grid(True)
        
        # 绘制迭代时间曲线
        plt.subplot(2, 1, 2)
        plt.bar(range(len(self.history['iteration_times'])), self.history['iteration_times'])
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.title('PSO Iteration Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "optimization_curves.png"))
        plt.close()
    
    def train_final_model(self):
        """
        训练最终的最优模型
        
        Returns:
            tuple: (模型, 测试准确率, 训练历史)
        """
        print("\n" + "=" * 60)
        print("训练最终最优模型")
        print("=" * 60)
        
        # 创建保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.save_dir, "models", f"best_model_{timestamp}.pth")
        tensorboard_dir = os.path.join(self.save_dir, "runs", f"best_model_{timestamp}")
        
        # 训练最优模型
        model, accuracy, history = self.trainer.train_final_model(
            self.global_best_position,
            epochs=Config.PSO_FINAL_EPOCHS,
            save_path=model_path,
            tensorboard_dir=tensorboard_dir
        )
        
        # 保存训练历史
        history_file = os.path.join(self.save_dir, "models", f"training_history_{timestamp}.json")
        with open(history_file, 'w') as f:
            # 将numpy数组转换为列表
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_history[key] = [v.tolist() for v in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(history, self.save_dir)
        
        return model, accuracy, history
    
    def plot_training_curves(self, history, save_dir):
        """
        绘制训练曲线
        
        Args:
            history (dict): 训练历史
            save_dir (str): 保存目录
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(history['train_losses'], 'b-', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(history['train_accuracies'], 'g-', label='Train Accuracy')
        plt.plot(history['test_accuracies'], 'r-', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"))
        plt.close()
    
    def get_best_architecture_info(self):
        """
        获取最优架构信息
        
        Returns:
            str: 架构信息
        """
        if self.global_best_position is None:
            return "尚未找到最优架构"
        
        info = "最优CNN架构:\n"
        info += f"卷积层数量: {self.global_best_position['conv_layers']}\n"
        info += f"卷积核数量: {self.global_best_position['conv_filters']}\n"
        info += f"卷积核大小: {self.global_best_position['conv_kernels']}\n"
        info += f"全连接层数量: {self.global_best_position['fc_layers']}\n"
        info += f"全连接层大小: {self.global_best_position['fc_sizes']}\n"
        info += f"适应度(准确率): {self.global_best_fitness:.2f}%\n"
        
        return info