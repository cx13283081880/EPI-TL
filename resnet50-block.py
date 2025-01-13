import torch.nn as nn
import crypten
import torch
from torchvision.models import resnet18
import crypten.mpc as mpc
import os
import time
import crypten.communicator as comm
import logging
crypten.init()
# torch.set_num_threads(1)
os.chdir('/home/cx/Desktop/CrypTen-main/tutorials')

ALICE = 0
BOB = 1
logging.getLogger().setLevel(logging.INFO)
# 检查是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = torch.load('model_50cpu.pth')
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


final_layers_model = FinalLayersResNet(model)

# crypten.common.serial.register_safe_class(ResNet)
crypten.common.serial.register_safe_class(torch.nn.modules.container.Sequential)
crypten.common.serial.register_safe_class(torch.nn.modules.pooling.AdaptiveAvgPool2d)
crypten.common.serial.register_safe_class(FinalLayersResNet)
# crypten.common.serial.register_safe_class(ResNet)
crypten.common.serial.register_safe_class(torch.nn.modules.container.Sequential)
crypten.common.serial.register_safe_class(torch.nn.modules.pooling.AdaptiveAvgPool2d)


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


labels = torch.load('bm-labels.pth')

batch_size = 1  # 可以根据内存情况调整批次大小
num_samples = 128
total_time = 0.0
total_accuracy = 0.0
num_batches = num_samples // batch_size


@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    global total_time, total_accuracy
    # Load pre-trained model to Alice
    model = crypten.load_from_party('finalmodel50-cpu.pth', src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 1024, 14, 14))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=ALICE)
    private_model = private_model.to(device)

    # Load data to Bob
    data_enc = crypten.load_from_party('bm-data.pth', src=BOB)

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        data_batch = data_enc[batch_start:batch_end]
        data_flatten = data_batch.view(-1, 1024, 14, 14)
        label_batch = labels[batch_start:batch_end]

        # Classify the encrypted data
        private_model.eval()
        start_time = time.time()
        output_enc = private_model(data_flatten)

        # Compute the accuracy
        output = output_enc.get_plain_text()
        accuracy = compute_accuracy(output, label_batch)
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_time += elapsed_time
        total_accuracy += accuracy.item()

        crypten.print(
            f"Batch {batch_start // batch_size + 1}/{num_batches} - Time: {elapsed_time:.2f}s, Accuracy: {accuracy.item():.4f}")

    # 输出总时间和平均准确率
    average_accuracy = total_accuracy / num_batches
    comm.get().print_communication_stats()
    crypten.print(f"\nTotal Time Elapsed: {total_time:.2f} seconds")
    crypten.print(f"Average Accuracy: {average_accuracy:.4f}")


encrypt_model_and_data()
