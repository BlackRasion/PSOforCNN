# -*- coding: utf-8 -*-
"""
PSO训练器 - 用于训练和评估PSO粒子对应的CNN模型
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import Config
from pso_model import create_pso_model


class PSOTrainer:
    """
    PSO训练器类
    
    功能:
    1. 训练PSO粒子对应的CNN模型
    2. 评估模型性能
    3. 记录训练日志
    """
    
    def __init__(self, trainloader, testloader, device=None):
        """
        初始化PSO训练器
        
        Args:
            trainloader: 训练数据加载器
            testloader: 测试数据加载器
            device: 计算设备
        """
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device else Config.DEVICE
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"PSO训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(trainloader.dataset)}")
        print(f"测试样本数: {len(testloader.dataset)}")
    
    def train_particle(self, particle, epochs=None, verbose=False):
        """
        训练单个粒子对应的CNN模型
        
        Args:
            particle: PSO粒子
            epochs (int): 训练轮数
            verbose (bool): 是否显示详细信息
            
        Returns:
            float: 测试准确率（适应度）
        """
        if epochs is None:
            epochs = Config.PSO_TRAIN_EPOCHS
        
        # 创建模型
        model = create_pso_model(particle.position, self.device)
        
        # 优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM
        )
        
        # 学习率调度器 - 统一使用配置参数
        # 对于短期训练，调整step_size以适应较少的epochs
        step_size = min(Config.LR_DECAY_STEP, max(1, epochs // 3))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=Config.LR_DECAY_FACTOR
        )
        
        if verbose:
            print(f"\n开始训练粒子 {particle.particle_id + 1}")
            model.print_model_info()
        
        model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 更新学习率
            scheduler.step()
            
            if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
                train_acc = 100 * correct / total
                avg_loss = running_loss / len(self.trainloader)
                print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.2f}%")
        
        # 评估模型
        test_accuracy = self.evaluate_particle(model, verbose=verbose)
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"粒子 {particle.particle_id + 1} 训练完成")
            print(f"训练时间: {training_time:.2f}s")
            print(f"测试准确率: {test_accuracy:.2f}%")
            print("="*60)
        
        return test_accuracy
    
    def evaluate_particle(self, model, verbose=False):
        """
        评估粒子对应的模型
        
        Args:
            model: 要评估的模型
            verbose (bool): 是否显示详细信息
            
        Returns:
            float: 测试准确率
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        if verbose:
            print(f"测试准确率: {accuracy:.2f}%")
        
        return accuracy
    
    def train_final_model(self, particle_position, epochs=None, save_path=None, tensorboard_dir=None):
        """
        训练最终的最优模型
        
        Args:
            particle_position (dict): 最优粒子位置
            epochs (int): 训练轮数
            save_path (str): 模型保存路径
            tensorboard_dir (str): TensorBoard日志目录
            
        Returns:
            tuple: (模型, 测试准确率, 训练历史)
        """
        if epochs is None:
            epochs = Config.PSO_FINAL_EPOCHS
        
        # 创建模型
        model = create_pso_model(particle_position, self.device)
        
        print("\n" + "=" * 60)
        print("训练最优PSO架构")
        print("=" * 60)
        model.print_model_info()
        print(model.get_architecture_string())
        
        # 优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=Config.LR_DECAY_STEP,
            gamma=Config.LR_DECAY_FACTOR
        )
        
        # TensorBoard
        writer = None
        if tensorboard_dir:
            writer = SummaryWriter(tensorboard_dir)
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        print(f"\n开始训练最优模型，共 {epochs} 个epoch")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 计算训练指标
            train_loss = running_loss / len(self.trainloader)
            train_acc = 100 * correct / total
            
            # 评估
            test_acc = self.evaluate_particle(model)
            
            # 记录历史
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # TensorBoard记录
            if writer:
                writer.add_scalar('Training/Loss', train_loss, epoch)
                writer.add_scalar('Training/Accuracy', train_acc, epoch)
                writer.add_scalar('Test/Accuracy', test_acc, epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'训练准确率: {train_acc:.2f}%, '
                  f'测试准确率: {test_acc:.2f}%, '
                  f'学习率: {current_lr:.6f}, '
                  f'用时: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        final_accuracy = test_accuracies[-1]
        
        print(f"\n最优模型训练完成！")
        print(f"总用时: {total_time:.2f}s")
        print(f"最终测试准确率: {final_accuracy:.2f}%")
        print(f"最佳测试准确率: {max(test_accuracies):.2f}%")
        
        # 保存模型
        if save_path:
            from pso_model import save_pso_model
            save_pso_model(model, particle_position, save_path)
        
        # 关闭TensorBoard
        if writer:
            writer.close()
            print(f"TensorBoard日志已保存到: {tensorboard_dir}")
        
        training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'final_accuracy': final_accuracy,
            'best_accuracy': max(test_accuracies),
            'training_time': total_time
        }
        
        return model, final_accuracy, training_history
    
    def quick_evaluate(self, particle_position, sample_ratio=0.1):
        """
        快速评估粒子（使用部分数据）
        
        Args:
            particle_position (dict): 粒子位置
            sample_ratio (float): 采样比例
            
        Returns:
            float: 估计的准确率
        """
        model = create_pso_model(particle_position, self.device)
        
        # 简单训练几个batch - 使用统一的学习率配置
        optimizer = optim.SGD(
            model.parameters(), 
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM
        )
        model.train()
        
        train_batches = max(1, int(len(self.trainloader) * sample_ratio))
        for i, (inputs, labels) in enumerate(self.trainloader):
            if i >= train_batches:
                break
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 快速评估
        model.eval()
        correct = 0
        total = 0
        test_batches = max(1, int(len(self.testloader) * sample_ratio))
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.testloader):
                if i >= test_batches:
                    break
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0