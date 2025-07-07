# -*- coding: utf-8 -*-
"""
中文数字识别CNN - 主程序

模块化设计:
- config.py: 配置管理
- data_loader.py: 数据加载和预处理
- model.py: 模型定义
- trainer.py: 训练和评估
- main.py: 主程序入口
"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import ChineseNumberDataset, create_data_loaders
from model import create_model, save_model, load_model
from trainer import CNNTrainer


def setup_environment():
    """
    设置运行环境
    """
    # 设置随机种子
    torch.manual_seed(Config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_STATE)
    
    # 创建必要的目录
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)
    
    print("环境设置完成")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"设备: {Config.DEVICE}")
    print(f"随机种子: {Config.RANDOM_STATE}")


def load_data():
    """
    加载和预处理数据
    
    Returns:
        tuple: (trainloader, testloader)
    """
    print("\n" + "=" * 60)
    print("数据加载阶段")
    print("=" * 60)
    
    # 检查数据路径
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"数据路径不存在: {Config.DATA_PATH}")
    
    # 加载数据集
    dataset = ChineseNumberDataset(Config.DATA_PATH)
    
    # 划分训练集和测试集
    X_train, y_train, y_train_hot, X_test, y_test, y_test_hot = dataset.split_data()
    
    # 创建数据加载器
    trainloader, testloader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    print("数据加载完成")
    return trainloader, testloader


def create_and_setup_model():
    """
    创建和设置模型
    
    Returns:
        torch.nn.Module: 创建的模型
    """
    print("\n" + "=" * 60)
    print("模型创建阶段")
    print("=" * 60)
    
    model = create_model(Config.DEVICE)
    return model


def train_model(model, trainloader, testloader):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
    """
    print("\n" + "=" * 60)
    print("模型训练阶段")
    print("=" * 60)
    
    # 创建训练器
    trainer = CNNTrainer(model, trainloader, testloader, Config.DEVICE)
    
    # 开始训练
    trainer.train(Config.EPOCHS)
    
    return trainer


def evaluate_model(model_path, testloader):
    """
    评估已训练的模型
    
    Args:
        model_path (str): 模型文件路径
        testloader: 测试数据加载器
    """
    print("\n" + "=" * 60)
    print("模型评估阶段")
    print("=" * 60)
    
    # 加载模型
    model = load_model(model_path, Config.DEVICE)
    
    # 创建训练器进行评估
    trainer = CNNTrainer(model, None, testloader, Config.DEVICE)
    
    # 评估模型
    accuracy, predictions, true_labels = trainer.evaluate()
    
    return accuracy, predictions, true_labels


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='中文数字识别CNN训练程序')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'both'],
                       help='运行模式: train(训练), eval(评估), both(训练+评估)')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(Config.MODEL_SAVE_DIR, 'chinese_number_cnn.pth'),
                       help='模型文件路径(用于评估模式)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据路径(覆盖配置文件中的路径)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数(覆盖配置文件中的设置)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小(覆盖配置文件中的设置)')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率(覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 动态更新配置
    if args.data_path:
        Config.DATA_PATH = args.data_path
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    print("中文数字识别CNN - 模块化版本")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"数据路径: {Config.DATA_PATH}")
    print(f"模型保存目录: {Config.MODEL_SAVE_DIR}")
    print(f"TensorBoard日志目录: {Config.TENSORBOARD_LOG_DIR}")
    
    try:
        # 设置环境
        setup_environment()
        
        if args.mode in ['train', 'both']:
            # 训练模式
            print("\n开始训练流程...")
            
            # 加载数据
            trainloader, testloader = load_data()
            
            # 创建模型
            model = create_and_setup_model()
            
            # 训练模型
            trainer = train_model(model, trainloader, testloader)
            
            print("\n训练流程完成!")
            
            if args.mode == 'both':
                print("\n开始评估流程...")
                # 使用刚训练的模型进行评估
                accuracy, predictions, true_labels = trainer.evaluate()
                print(f"最终测试准确率: {accuracy:.2f}%")
        
        elif args.mode == 'eval':
            # 评估模式
            print("\n开始评估流程...")
            
            # 检查模型文件是否存在
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
            
            # 加载测试数据
            _, testloader = load_data()
            
            # 评估模型
            accuracy, predictions, true_labels = evaluate_model(args.model_path, testloader)
            
            print(f"模型评估完成，准确率: {accuracy:.2f}%")
        
        print("\n程序执行完成!")
        print("\n使用以下命令查看TensorBoard:")
        print(f"tensorboard --logdir={Config.TENSORBOARD_LOG_DIR}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("程序执行失败!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)