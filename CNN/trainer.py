# -*- coding: utf-8 -*-
"""
训练模块
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import average_precision_score
import torchvision.utils as vutils

from config import Config


class CNNTrainer:
    """
    CNN训练器类
    
    功能:
    1. 模型训练
    2. 模型评估
    3. TensorBoard日志记录
    4. 模型保存和加载
    5. 可视化预测结果
    """
    
    def __init__(self, model, trainloader, testloader, device=None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            trainloader: 训练数据加载器
            testloader: 测试数据加载器
            device: 计算设备
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device else Config.DEVICE
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM
        )
        
        # TensorBoard
        self.writer = SummaryWriter(Config.TENSORBOARD_LOG_DIR)
        
        # 训练统计
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        print(f"训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"损失函数: {self.criterion}")
        print(f"优化器: {self.optimizer}")
        print(f"TensorBoard日志目录: {Config.TENSORBOARD_LOG_DIR}")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch (int): 当前epoch数
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每1000个mini-batch记录一次
            if i % Config.LOG_INTERVAL == Config.LOG_INTERVAL - 1:
                avg_loss = running_loss / Config.LOG_INTERVAL
                accuracy = 100 * correct / total
                
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] '
                      f'Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
                
                # TensorBoard记录
                global_step = epoch * len(self.trainloader) + i
                self.writer.add_scalar('Training/Loss', avg_loss, global_step)
                self.writer.add_scalar('Training/Accuracy', accuracy, global_step)
                
                # 可视化预测结果
                if i % (Config.LOG_INTERVAL * 5) == Config.LOG_INTERVAL * 5 - 1:
                    self._log_predictions(inputs, labels, outputs, global_step)
                
                running_loss = 0.0
        
        epoch_accuracy = 100 * correct / total
        return running_loss / len(self.trainloader), epoch_accuracy
    
    def evaluate(self, epoch=None):
        """
        评估模型
        
        Args:
            epoch (int): 当前epoch数，用于TensorBoard记录
            
        Returns:
            tuple: (accuracy, all_predictions, all_labels)
        """
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # 获取预测结果
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集所有预测和标签用于详细分析
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        # TensorBoard记录
        if epoch is not None:
            self.writer.add_scalar('Test/Accuracy', accuracy, epoch)
            
            # 记录每个类别的精确率-召回率曲线
            self._log_precision_recall_curves(all_labels, all_probs, epoch)
        
        print(f'测试集准确率: {accuracy:.2f}%')
        
        return accuracy, all_predictions, all_labels
    
    def train(self, num_epochs=None):
        """
        完整训练过程
        
        Args:
            num_epochs (int): 训练轮数
        """
        if num_epochs is None:
            num_epochs = Config.EPOCHS
            
        print(f"开始训练，共 {num_epochs} 个epoch")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 评估
            test_acc, _, _ = self.evaluate(epoch)
            
            # 记录统计信息
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{num_epochs} 完成 - '
                  f'训练准确率: {train_acc:.2f}%, '
                  f'测试准确率: {test_acc:.2f}%, '
                  f'用时: {epoch_time:.2f}s')
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time:.2f}s")
        
        # 保存模型
        self._save_final_model()
        
        # 生成最终报告
        self._generate_final_report()
    
    def _log_predictions(self, inputs, labels, outputs, global_step):
        """
        记录预测结果到TensorBoard
        
        Args:
            inputs: 输入图像
            labels: 真实标签
            outputs: 模型输出
            global_step: 全局步数
        """
        # 获取预测概率和类别
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        # 创建图像网格
        img_grid = vutils.make_grid(inputs[:4], normalize=True, scale_each=True)
        
        # 创建预测文本
        pred_texts = []
        for i in range(min(4, len(labels))):
            true_label = Config.CLASSES[labels[i]]
            pred_label = Config.CLASSES[preds[i]]
            confidence = probs[i][preds[i]].item()
            pred_texts.append(f'True: {true_label}, Pred: {pred_label} ({confidence:.2f})')
        
        # 记录到TensorBoard
        self.writer.add_image('Predictions/Images', img_grid, global_step)
        self.writer.add_text('Predictions/Details', '\n'.join(pred_texts), global_step)
    
    def _log_precision_recall_curves(self, labels, probs, epoch):
        """
        记录精确率-召回率曲线到TensorBoard
        
        Args:
            labels: 真实标签列表
            probs: 预测概率列表
            epoch: 当前epoch
        """
        labels_array = np.array(labels)
        probs_array = np.array(probs)
        
        for i, class_name in enumerate(Config.CLASSES):
            # 创建二分类标签
            binary_labels = (labels_array == i).astype(int)
            class_probs = probs_array[:, i]
            
            # 计算精确率-召回率曲线
            precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
            ap = average_precision_score(binary_labels, class_probs)
            
            # 记录到TensorBoard
            self.writer.add_pr_curve(f'PR_Curve/{class_name}', 
                                   torch.tensor(binary_labels), 
                                   torch.tensor(class_probs), 
                                   epoch)
            self.writer.add_scalar(f'AP/{class_name}', ap, epoch)
    
    def _save_final_model(self):
        """
        保存最终模型
        """
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        model_path = os.path.join(Config.MODEL_SAVE_DIR, 'chinese_number_cnn.pth')
        
        # 只保存可序列化的配置参数
        serializable_config = {
            'NUM_CLASSES': Config.NUM_CLASSES,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'NUM_EPOCHS': Config.EPOCHS,
            'LEARNING_RATE': Config.LEARNING_RATE,
            'MOMENTUM': Config.MOMENTUM,
            'IMAGE_SIZE': Config.IMAGE_SIZE,
            'CLASSES': Config.CLASSES
        }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'config': serializable_config
        }, model_path)
        
        print(f"模型已保存到: {model_path}")
    
    def _generate_final_report(self):
        """
        生成最终训练报告
        """
        print("\n" + "=" * 60)
        print("最终训练报告")
        print("=" * 60)
        
        # 最终评估
        final_accuracy, predictions, true_labels = self.evaluate()
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            true_labels, predictions, 
            target_names=Config.CLASSES,
            digits=4
        ))
        
        # 训练曲线
        self._plot_training_curves()
        
        print(f"\n最佳测试准确率: {max(self.test_accuracies):.2f}%")
        print(f"最终测试准确率: {final_accuracy:.2f}%")
        
        # 关闭TensorBoard writer
        self.writer.close()
        print(f"\nTensorBoard日志已保存到: {Config.TENSORBOARD_LOG_DIR}")
        print("运行以下命令查看TensorBoard:")
        print(f"tensorboard --logdir={Config.TENSORBOARD_LOG_DIR}")
    
    def _plot_training_curves(self):
        """
        绘制训练曲线
        """
        plt.figure(figsize=(12, 4))
        
        # 准确率曲线
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_accuracies) + 1)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training accuracy')
        plt.plot(epochs, self.test_accuracies, 'r-', label='Testing accuracy')
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_losses, 'b-', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        plt.savefig(os.path.join(Config.MODEL_SAVE_DIR, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练曲线已保存")


def matplotlib_imshow(img, one_channel=False):
    """
    显示图像的辅助函数
    
    Args:
        img: 图像张量
        one_channel: 是否为单通道图像
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    print("训练模块测试")
    print("请在主程序中使用此模块")