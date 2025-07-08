# -*- coding: utf-8 -*-
"""
PSO-CNN演示脚本

基于粒子群优化算法的CNN架构自动搜索系统演示
"""

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from config import Config
from pso_optimizer import PSOOptimizer
from pso_model import create_pso_model
from data_loader import create_data_loaders, ChineseNumberDataset

def create_optimized_data_loaders():
    """
    创建优化的数据加载器，减少训练时间
    
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """
    print("创建优化的数据加载器...")
    
    # 加载数据集
    dataset = ChineseNumberDataset(Config.DATA_PATH)
    X_train, y_train, y_train_hot, X_test, y_test, y_test_hot = dataset.split_data()
    
    # 为了减少训练时间，使用数据集的子集
    subset_ratio = 0.75  # 使用75%的数据进行PSO搜索
    
    # 计算子集大小
    train_size = len(X_train)
    test_size = len(X_test)
    
    train_subset_size = int(train_size * subset_ratio)
    test_subset_size = int(test_size * subset_ratio)
    
    # 随机采样索引
    np.random.seed(Config.RANDOM_STATE)
    train_indices = np.random.choice(train_size, train_subset_size, replace=False)
    test_indices = np.random.choice(test_size, test_subset_size, replace=False)
    
    # 创建子集数据
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    # 使用create_data_loaders函数创建数据加载器
    optimized_train_loader, optimized_test_loader = create_data_loaders(
        X_train_subset, y_train_subset, X_test_subset, y_test_subset
    )
    
    print(f"训练数据: {len(X_train_subset)} 样本 (原始: {train_size})")
    print(f"测试数据: {len(X_test_subset)} 样本 (原始: {test_size})")
    print(f"数据子集比例: {subset_ratio*100:.1f}%")
    
    return optimized_train_loader, optimized_test_loader


def run_baseline_comparison_with_data(train_loader, test_loader):
    """
    运行基准模型比较
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    
    Returns:
        float: 基准模型准确率
    """
    print("\n" + "=" * 60)
    print("运行基准模型比较（使用相同的75%子集数据）")
    print("=" * 60)
    
    try:
        # 导入基准模型
        from model import ChineseNumberCNN
        from trainer import CNNTrainer
        
        # 确保PSO保存目录存在
        os.makedirs(Config.PSO_SAVE_DIR, exist_ok=True)
        baseline_save_dir = os.path.join(Config.PSO_SAVE_DIR, "baseline")
        os.makedirs(baseline_save_dir, exist_ok=True)
        
        # 创建基准模型
        baseline_model = ChineseNumberCNN().to(Config.DEVICE)
        
        # 临时修改配置，将基准模型结果保存到PSO文件夹
        original_model_save_dir = Config.MODEL_SAVE_DIR
        original_tensorboard_dir = Config.TENSORBOARD_LOG_DIR
        
        Config.MODEL_SAVE_DIR = baseline_save_dir
        Config.TENSORBOARD_LOG_DIR = os.path.join(Config.PSO_TENSORBOARD_DIR, "baseline")
        
        # 创建训练器
        trainer = CNNTrainer(
            model=baseline_model,
            trainloader=train_loader,  # 修正参数名称
            testloader=test_loader,    # 修正参数名称
            device=Config.DEVICE
        )
        
        # 不使用预训练模型，从头开始训练基准模型以公平比较
        print("从头开始训练基准模型以公平比较PSO优化效果")
        print(f"基准模型结果将保存到: {baseline_save_dir}")
        trainer.train(num_epochs=Config.PSO_FINAL_EPOCHS)
        
        # 评估基准模型
        baseline_accuracy, _, _ = trainer.evaluate()  # 修正返回值解包
        print(f"基准模型准确率: {baseline_accuracy:.2f}%")
        
        # 保存基准模型训练曲线
        trainer._plot_training_curves()
        
        # 恢复原始配置
        Config.MODEL_SAVE_DIR = original_model_save_dir
        Config.TENSORBOARD_LOG_DIR = original_tensorboard_dir
        
        print(f"基准模型训练曲线已保存到: {baseline_save_dir}")
        
        return baseline_accuracy
        
    except Exception as e:
        # 确保在异常情况下也恢复配置
        try:
            Config.MODEL_SAVE_DIR = original_model_save_dir
            Config.TENSORBOARD_LOG_DIR = original_tensorboard_dir
        except:
            pass
        print(f"基准模型比较失败: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def analyze_results(pso_results, baseline_accuracy=None):
    """
    分析PSO优化结果
    
    Args:
        pso_results (dict): PSO优化结果
        baseline_accuracy (float): 基准模型准确率
    """
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)
    
    # 使用最终模型准确率，如果没有则使用PSO搜索过程中的适应度
    final_accuracy = pso_results.get('final_model_accuracy', pso_results['best_fitness'])
    best_fitness = pso_results['best_fitness']
    total_time = pso_results['total_time']
    history = pso_results['history']
    
    print(f"PSO优化结果:")
    print(f"  最优准确率: {final_accuracy:.2f}%")
    print(f"  优化总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    print(f"  平均每次迭代时间: {sum(history['iteration_times'])/len(history['iteration_times']):.2f}s")
    
    if baseline_accuracy is not None:
        improvement = final_accuracy - baseline_accuracy
        print(f"\n与基准模型比较:")
        print(f"  基准模型准确率: {baseline_accuracy:.2f}%")
        print(f"  PSO优化准确率: {final_accuracy:.2f}%")
        print(f"  性能提升: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"  ✅ PSO优化成功，性能提升 {improvement:.2f}%")
        elif improvement > -1:
            print(f"  ⚖️ PSO优化结果与基准模型相当")
        else:
            print(f"  ❌ PSO优化结果低于基准模型")
    
    # 分析收敛情况
    initial_fitness = history['global_best_fitness'][0]
    final_fitness = history['global_best_fitness'][-1]
    convergence = final_fitness - initial_fitness
    
    print(f"\n收敛分析:")
    print(f"  初始最优适应度: {initial_fitness:.2f}%")
    print(f"  最终最优适应度: {final_fitness:.2f}%")
    print(f"  收敛提升: {convergence:.2f}%")
    
    # 找到最大提升的迭代
    max_improvement_iter = 0
    max_improvement = 0
    for i in range(1, len(history['global_best_fitness'])):
        improvement = history['global_best_fitness'][i] - history['global_best_fitness'][i-1]
        if improvement > max_improvement:
            max_improvement = improvement
            max_improvement_iter = i
    
    if max_improvement > 0:
        print(f"  最大单次提升: {max_improvement:.2f}% (第{max_improvement_iter}次迭代)")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='PSO-CNN架构搜索演示')
    parser.add_argument('--skip-baseline', action='store_true', 
                       help='跳过基准模型比较')
    parser.add_argument('--quick', action='store_true',
                       help='快速演示模式（减少粒子数和迭代次数）')
    args = parser.parse_args()
    
    print("PSO-CNN架构自动搜索系统演示")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    # 快速演示模式
    if args.quick:
        print("\n启用快速演示模式")
        Config.PSO_PARTICLE_SIZE = 3 # 粒子数
        Config.PSO_ITERATIONS = 2 # 迭代次数
        Config.PSO_TRAIN_EPOCHS = 3 # 训练轮数
        Config.PSO_FINAL_EPOCHS = 5 # 最终训练轮数
        print(f"调整参数: 粒子数={Config.PSO_PARTICLE_SIZE}, 迭代次数={Config.PSO_ITERATIONS}")
    
    # 创建优化的数据加载器（基准模型和PSO优化器共用）
    train_loader, test_loader = create_optimized_data_loaders()
    
    # 运行基准模型比较
    baseline_accuracy = None
    if not args.skip_baseline:
        baseline_accuracy = run_baseline_comparison_with_data(train_loader, test_loader)
    
    # 创建PSO优化器
    print("\n创建PSO优化器...")
    optimizer = PSOOptimizer(
        trainloader=train_loader,
        testloader=test_loader,
        device=Config.DEVICE
    )
    
    # 执行PSO优化
    print("\n开始PSO优化...")
    start_time = time.time()
    pso_results = optimizer.optimize()
    total_time = time.time() - start_time
    
    # 更新总时间
    pso_results['total_time'] = total_time
    
    # 更新最终模型准确率到结果中
    if optimizer.final_model_accuracy is not None:
        pso_results['final_model_accuracy'] = optimizer.final_model_accuracy
    
    # 显示最优架构信息
    print("\n" + "=" * 60)
    print("最优架构信息")
    print("=" * 60)
    print(optimizer.get_best_architecture_info())
    
    # 分析结果
    analyze_results(pso_results, baseline_accuracy)
    
    # 保存演示报告
    save_demo_report(pso_results, baseline_accuracy, args)
    
    print(f"\n演示完成! 结果已保存到: {Config.PSO_SAVE_DIR}")
    print(f"总用时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def save_demo_report(pso_results, baseline_accuracy, args):
    """
    保存演示报告
    
    Args:
        pso_results (dict): PSO优化结果
        baseline_accuracy (float): 基准模型准确率
        args: 命令行参数
    """
    # 确保保存目录存在
    os.makedirs(Config.PSO_SAVE_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(Config.PSO_SAVE_DIR, f"demo_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("PSO-CNN架构自动搜索系统演示报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 配置信息
        f.write("配置信息:\n")
        f.write(f"  粒子群大小: {Config.PSO_PARTICLE_SIZE}\n")
        f.write(f"  迭代次数: {Config.PSO_ITERATIONS}\n")
        f.write(f"  每个粒子训练epochs: {Config.PSO_TRAIN_EPOCHS}\n")
        f.write(f"  最终模型训练epochs: {Config.PSO_FINAL_EPOCHS}\n")
        f.write(f"  设备: {Config.DEVICE}\n")
        f.write(f"  快速模式: {'是' if args.quick else '否'}\n\n")
        
        # PSO结果
        final_accuracy = pso_results.get('final_model_accuracy', pso_results['best_fitness'])
        f.write("PSO优化结果:\n")
        f.write(f"  最优准确率: {final_accuracy:.2f}%\n")
        f.write(f"  优化总时间: {pso_results['total_time']:.2f}s\n")
        
        # 基准比较
        if baseline_accuracy is not None:
            improvement = final_accuracy - baseline_accuracy
            f.write(f"\n基准模型比较:\n")
            f.write(f"  基准模型准确率: {baseline_accuracy:.2f}%\n")
            f.write(f"  性能提升: {improvement:+.2f}%\n")
        
        # 最优架构
        f.write(f"\n最优架构信息:\n")
        best_pos = pso_results['best_position']
        f.write(f"  卷积层数量: {best_pos['conv_layers']}\n")
        f.write(f"  卷积核数量: {best_pos['conv_filters']}\n")
        f.write(f"  卷积核大小: {best_pos['conv_kernels']}\n")
        f.write(f"  全连接层数量: {best_pos['fc_layers']}\n")
        f.write(f"  全连接层大小: {best_pos['fc_sizes']}\n")
    
    print(f"演示报告已保存: {report_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断演示")
        sys.exit(0)
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)