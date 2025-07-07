# -*- coding: utf-8 -*-
"""
工具模块

包含各种辅助功能:
- 数据可视化
- 模型分析
- 结果处理
- 文件操作
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torchvision.utils as vutils

from config import Config


def visualize_dataset_samples(dataset, num_samples=16, save_path=None):
    """
    可视化数据集样本
    
    Args:
        dataset: 数据集对象
        num_samples (int): 显示的样本数量
        save_path (str): 保存路径，如果为None则不保存
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('数据集样本展示', fontsize=16)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset.data), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        row, col = i // 4, i % 4
        
        # 显示图像
        image = dataset.data[idx].squeeze()
        label = dataset.targets[idx]
        class_name = Config.CLASSES[label]
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'{class_name} (标签: {label})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"样本可视化已保存到: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path (str): 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASSES,
                yticklabels=Config.CLASSES)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()
    
    return cm


def analyze_class_distribution(dataset, save_path=None):
    """
    分析类别分布
    
    Args:
        dataset: 数据集对象
        save_path (str): 保存路径
    """
    # 统计每个类别的样本数量
    unique, counts = np.unique(dataset.targets, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # 创建可视化
    plt.figure(figsize=(14, 6))
    
    # 柱状图
    plt.subplot(1, 2, 1)
    class_names = [Config.CLASSES[i] for i in unique]
    bars = plt.bar(class_names, counts, color='skyblue', alpha=0.7)
    plt.title('类别分布 - 柱状图')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # 饼图
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.title('类别分布 - 饼图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别分布图已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n类别分布统计:")
    print("-" * 40)
    total_samples = sum(counts)
    for i, (class_idx, count) in enumerate(class_counts.items()):
        class_name = Config.CLASSES[class_idx]
        percentage = count / total_samples * 100
        print(f"{class_name:>8}: {count:>4} 样本 ({percentage:>5.1f}%)")
    print("-" * 40)
    print(f"{'总计':>8}: {total_samples:>4} 样本")
    
    return class_counts


def visualize_model_predictions(model, dataloader, device, num_samples=8, save_path=None):
    """
    可视化模型预测结果
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
        num_samples (int): 显示的样本数量
        save_path (str): 保存路径
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('模型预测结果展示', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        row, col = i // 4, i % 4
        
        # 显示图像
        image = images[i].cpu().squeeze()
        true_label = labels[i].cpu().item()
        pred_label = predictions[i].cpu().item()
        confidence = probabilities[i][pred_label].cpu().item()
        
        true_class = Config.CLASSES[true_label]
        pred_class = Config.CLASSES[pred_label]
        
        axes[row, col].imshow(image, cmap='gray')
        
        # 设置标题颜色（正确预测为绿色，错误预测为红色）
        color = 'green' if true_label == pred_label else 'red'
        title = f'真实: {true_class}\n预测: {pred_class}\n置信度: {confidence:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果可视化已保存到: {save_path}")
    
    plt.show()


def analyze_model_errors(y_true, y_pred, save_path=None):
    """
    分析模型错误
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path (str): 保存路径
    """
    # 找出错误预测的样本
    errors = np.where(np.array(y_true) != np.array(y_pred))[0]
    
    print(f"\n模型错误分析:")
    print("=" * 50)
    print(f"总样本数: {len(y_true)}")
    print(f"错误预测数: {len(errors)}")
    print(f"错误率: {len(errors)/len(y_true)*100:.2f}%")
    
    if len(errors) == 0:
        print("没有错误预测！")
        return
    
    # 统计每个类别的错误
    error_by_class = {}
    for i in errors:
        true_class = y_true[i]
        pred_class = y_pred[i]
        
        if true_class not in error_by_class:
            error_by_class[true_class] = {'total': 0, 'errors': {}}
        
        error_by_class[true_class]['total'] += 1
        
        if pred_class not in error_by_class[true_class]['errors']:
            error_by_class[true_class]['errors'][pred_class] = 0
        error_by_class[true_class]['errors'][pred_class] += 1
    
    # 打印详细错误信息
    print("\n各类别错误详情:")
    print("-" * 50)
    for true_class, error_info in error_by_class.items():
        true_class_name = Config.CLASSES[true_class]
        print(f"\n{true_class_name} 的错误预测:")
        for pred_class, count in error_info['errors'].items():
            pred_class_name = Config.CLASSES[pred_class]
            print(f"  误认为 {pred_class_name}: {count} 次")
    
    # 保存错误分析报告
    if save_path:
        error_report = {
            'total_samples': len(y_true),
            'total_errors': len(errors),
            'error_rate': len(errors)/len(y_true)*100,
            'error_by_class': {}
        }
        
        for true_class, error_info in error_by_class.items():
            true_class_name = Config.CLASSES[true_class]
            error_report['error_by_class'][true_class_name] = {
                'total_errors': error_info['total'],
                'confused_with': {}
            }
            for pred_class, count in error_info['errors'].items():
                pred_class_name = Config.CLASSES[pred_class]
                error_report['error_by_class'][true_class_name]['confused_with'][pred_class_name] = count
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n错误分析报告已保存到: {save_path}")


def save_classification_report(y_true, y_pred, save_path):
    """
    保存分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path (str): 保存路径
    """
    report = classification_report(
        y_true, y_pred,
        target_names=Config.CLASSES,
        output_dict=True
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"分类报告已保存到: {save_path}")


def create_results_summary(accuracy, save_dir):
    """
    创建结果摘要
    
    Args:
        accuracy (float): 模型准确率
        save_dir (str): 保存目录
    """
    summary = {
        'model_info': {
            'architecture': 'CNN',
            'input_size': Config.IMAGE_SIZE,
            'num_classes': Config.NUM_CLASSES,
            'classes': Config.CLASSES
        },
        'training_config': {
            'epochs': Config.EPOCHS,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'optimizer': 'SGD',
            'momentum': Config.MOMENTUM
        },
        'results': {
            'accuracy': accuracy,
            'device': Config.DEVICE
        }
    }
    
    summary_path = os.path.join(save_dir, 'results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"结果摘要已保存到: {summary_path}")
    return summary


def check_data_path(data_path):
    """
    检查数据路径是否有效
    
    Args:
        data_path (str): 数据路径
        
    Returns:
        bool: 路径是否有效
    """
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在 - {data_path}")
        return False
    
    # 检查是否包含图像文件
    jpg_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    if len(jpg_files) == 0:
        print(f"错误: 数据路径中没有找到.jpg文件 - {data_path}")
        return False
    
    print(f"数据路径有效: {data_path} (包含 {len(jpg_files)} 个.jpg文件)")
    return True


if __name__ == "__main__":
    print("工具模块测试")
    print("请在主程序中使用此模块的功能")