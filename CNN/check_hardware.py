# -*- coding: utf-8 -*-
"""
硬件配置检测脚本
用于检测训练环境的硬件配置信息
"""

import torch
import psutil
import platform
import subprocess
import sys
import os
from datetime import datetime


def get_cpu_info():
    """获取CPU信息"""
    print("=" * 60)
    print("CPU 信息")
    print("=" * 60)
    
    # CPU基本信息
    print(f"处理器: {platform.processor()}")
    print(f"架构: {platform.machine()}")
    print(f"物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"逻辑核心数: {psutil.cpu_count(logical=True)}")
    
    # CPU频率
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"当前频率: {cpu_freq.current:.2f} MHz")
        print(f"最大频率: {cpu_freq.max:.2f} MHz")
        print(f"最小频率: {cpu_freq.min:.2f} MHz")
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"当前CPU使用率: {cpu_percent}%")
    
    # 每个核心的使用率
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    for i, percentage in enumerate(cpu_per_core):
        print(f"核心 {i}: {percentage}%")


def get_memory_info():
    """获取内存信息"""
    print("\n" + "=" * 60)
    print("内存信息")
    print("=" * 60)
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / (1024**3):.2f} GB")
    print(f"可用内存: {memory.available / (1024**3):.2f} GB")
    print(f"已使用内存: {memory.used / (1024**3):.2f} GB")
    print(f"内存使用率: {memory.percent}%")
    
    # 交换内存
    swap = psutil.swap_memory()
    print(f"交换内存总量: {swap.total / (1024**3):.2f} GB")
    print(f"交换内存使用: {swap.used / (1024**3):.2f} GB")
    print(f"交换内存使用率: {swap.percent}%")


def get_gpu_info():
    """获取GPU信息"""
    print("\n" + "=" * 60)
    print("GPU 信息")
    print("=" * 60)
    
    # 检查CUDA是否可用
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        # 遍历所有GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            
            # GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            print(f"  总显存: {memory_total:.2f} GB")
            print(f"  已分配显存: {memory_allocated:.2f} GB")
            print(f"  已保留显存: {memory_reserved:.2f} GB")
            print(f"  可用显存: {memory_total - memory_reserved:.2f} GB")
            
            # GPU属性
            props = torch.cuda.get_device_properties(i)
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数量: {props.multi_processor_count}")
            print(f"  最大线程数/块: {props.max_threads_per_block}")
            print(f"  最大块维度: {props.max_block_dims}")
            print(f"  最大网格维度: {props.max_grid_dims}")
    
    # 检查MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"\nMPS (Apple Silicon) 可用: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("MPS 设备已启用")


def get_pytorch_info():
    """获取PyTorch相关信息"""
    print("\n" + "=" * 60)
    print("PyTorch 环境信息")
    print("=" * 60)
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # 推荐设备
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"推荐训练设备: {device}")
    
    # 检查是否支持混合精度训练
    if torch.cuda.is_available():
        print(f"支持自动混合精度 (AMP): {torch.cuda.amp.autocast is not None}")


def get_disk_info():
    """获取磁盘信息"""
    print("\n" + "=" * 60)
    print("磁盘信息")
    print("=" * 60)
    
    # 获取当前工作目录的磁盘使用情况
    current_path = os.getcwd()
    disk_usage = psutil.disk_usage(current_path)
    
    print(f"当前路径: {current_path}")
    print(f"总空间: {disk_usage.total / (1024**3):.2f} GB")
    print(f"已使用: {disk_usage.used / (1024**3):.2f} GB")
    print(f"可用空间: {disk_usage.free / (1024**3):.2f} GB")
    print(f"使用率: {(disk_usage.used / disk_usage.total) * 100:.1f}%")


def get_network_info():
    """获取网络信息"""
    print("\n" + "=" * 60)
    print("网络信息")
    print("=" * 60)
    
    # 网络接口信息
    network_stats = psutil.net_io_counters()
    print(f"总发送字节: {network_stats.bytes_sent / (1024**2):.2f} MB")
    print(f"总接收字节: {network_stats.bytes_recv / (1024**2):.2f} MB")
    
    # 网络接口详情
    interfaces = psutil.net_if_addrs()
    print(f"\n网络接口数量: {len(interfaces)}")
    for interface_name in interfaces:
        print(f"接口: {interface_name}")


def check_dependencies():
    """检查重要依赖包"""
    print("\n" + "=" * 60)
    print("依赖包版本检查")
    print("=" * 60)
    
    packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'scikit-learn', 'pillow', 'tensorboard'
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '未知版本')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: 未安装")


def generate_training_recommendations():
    """生成训练建议"""
    print("\n" + "=" * 60)
    print("训练环境建议")
    print("=" * 60)
    
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True)
    
    # 批次大小建议
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"基于 {gpu_memory:.1f}GB GPU显存:")
        if gpu_memory >= 8:
            print("  - 可以使用较大的批次大小 (64-128)")
        elif gpu_memory >= 4:
            print("  - 建议中等批次大小 (32-64)")
        else:
            print("  - 建议较小批次大小 (16-32)")
    
    # 数据加载器工作线程建议
    print(f"\n基于 {cpu_count} 个CPU核心:")
    recommended_workers = min(cpu_count, 8)
    print(f"  - 建议数据加载器工作线程数: {recommended_workers}")
    
    # 内存建议
    memory_gb = memory.total / (1024**3)
    print(f"\n基于 {memory_gb:.1f}GB 系统内存:")
    if memory_gb >= 16:
        print("  - 内存充足，可以进行大规模训练")
    elif memory_gb >= 8:
        print("  - 内存适中，注意监控内存使用")
    else:
        print("  - 内存较少，建议减少批次大小或使用数据流式加载")


def main():
    """主函数"""
    print("CNN训练环境硬件配置检测")
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        get_cpu_info()
        get_memory_info()
        get_gpu_info()
        get_pytorch_info()
        get_disk_info()
        get_network_info()
        check_dependencies()
        generate_training_recommendations()
        
        print("\n" + "=" * 80)
        print("硬件配置检测完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n检测过程中出现错误: {str(e)}")
        print("请确保已安装所需的依赖包")


if __name__ == "__main__":
    main()