import torch
import torch.nn as nn
from torchvision.models import resnet18
import os
import time
import logging

# 设置日志级别
logging.getLogger().setLevel(logging.INFO)

# 检查是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置工作目录
os.chdir('/home/cx/Desktop/CrypTen-main/tutorials')

class FinalLayersResNet(nn.Module):
    def __init__(self, original_model):
        super(FinalLayersResNet, self).__init__()
        # 提取 layer4 和全连接层（fc）
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)  # 应用自适应平均池化层
        x = torch.flatten(x, 1)  # 展平，用于全连接层输入
        x = self.fc(x)
        return x


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


# 加载标签数据
labels = torch.load('bm-labels.pth')

batch_size = 1  # 可以根据内存情况调整批次大小
num_samples = 1
total_time = 0.0
total_accuracy = 0.0
num_batches = num_samples // batch_size


# 正常推理函数
def normal_inference():
    global total_time, total_accuracy

    # 加载预训练的模型
    model = torch.load('model_50cpu.pth')
    model = model.to(device)  # 将模型移动到设备

    # 加载数据
    data = torch.load('bob_test.pth').to(device)

    model.eval()  # 设置模型为评估模式

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        data_batch = data[batch_start:batch_end]
        label_batch = labels[batch_start:batch_end]

        # 对数据进行推理
        start_time = time.time()
        output = model(data_batch)

        # 计算准确率
        accuracy = compute_accuracy(output, label_batch)
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_time += elapsed_time
        total_accuracy += accuracy.item()

        logging.info(
            f"Batch {batch_start // batch_size + 1}/{num_batches} - Time: {elapsed_time:.2f}s, Accuracy: {accuracy.item():.4f}")

    # 输出总时间和平均准确率
    average_accuracy = total_accuracy / num_batches
    logging.info(f"\nTotal Time Elapsed: {total_time:.2f} seconds")
    logging.info(f"Average Accuracy: {average_accuracy:.4f}")


# 调用正常推理函数
normal_inference()
