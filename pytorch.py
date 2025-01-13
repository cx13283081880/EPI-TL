import torch
import torch.nn as nn
from torchvision.models import resnet18
import os
import time
import logging


# 检查是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


# 加载标签数据
labels = torch.load('bob_test_labels.pth')

batch_size = 1  # 可以根据内存情况调整批次大小
num_samples = 128
total_time = 0.0
total_accuracy = 0.0
num_batches = num_samples // batch_size


# 正常推理函数
def normal_inference():
    global total_time, total_accuracy

    # 加载预训练的模型
    model = torch.load('model_pre_cpu.pth')
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
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_time += elapsed_time
        total_accuracy += accuracy.item()

        logging.info(
            f"Batch {batch_start // batch_size + 1}/{num_batches} - Time: {elapsed_time:.2f}s, Accuracy: {accuracy.item():.4f}")

    # 输出总时间和平均准确率
    logging.info(f"\nTotal Time Elapsed: {total_time:.2f} seconds")



# 调用正常推理函数
normal_inference()
