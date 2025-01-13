import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torch.optim import lr_scheduler
import time
import os
from PIL import Image

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理和数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, covid_dir, noncovid_dir, transform=None):
        self.covid_images = [os.path.join(covid_dir, img) for img in os.listdir(covid_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.noncovid_images = [os.path.join(noncovid_dir, img) for img in os.listdir(noncovid_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.images = self.covid_images + self.noncovid_images
        self.labels = [0] * len(self.covid_images) + [1] * len(self.noncovid_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


train_ben_dir = r'C:\Users\Admin\Desktop\train\benign'
train_mal_dir = r'C:\Users\Admin\Desktop\train\malignant'
val_ben_dir = r'C:\Users\Admin\Desktop\test\benign'
val_mal_dir = r'C:\Users\Admin\Desktop\test\malignant'

train_0_dir = r'C:\Users\Admin\Desktop\train\0'
train_1_dir = r'C:\Users\Admin\Desktop\train\1'
val_0_dir = r'C:\Users\Admin\Desktop\test\0'
val_1_dir = r'C:\Users\Admin\Desktop\test\1'




trainset = CustomDataset(train_ben_dir, train_mal_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

valset = CustomDataset(val_ben_dir, val_mal_dir, transform=transform_test)
valloader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

testset = CustomDataset(val_ben_dir, val_mal_dir, transform=transform_test)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
#****************************************************************************************#
# trainset = CustomDataset(train_mal_dir, train_ben_dir, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
#
# valset = CustomDataset(val_mal_dir, val_ben_dir, transform=transform_test)
# valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
#
# testset = CustomDataset(val_mal_dir, val_ben_dir, transform=transform_test)
# testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

#****************************************************************************************#
# trainset = CustomDataset(train_covid_dir, train_noncovid_dir, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
#
# valset = CustomDataset(val_covid_dir, val_noncovid_dir, transform=transform)
# valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
#
# testset = CustomDataset(val_covid_dir, val_noncovid_dir, transform=transform)
# testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)



# 定义模型并使用预训练的权重
# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)  # 分类数改为2
)

# model = ResNet()

# for name, param in model.named_parameters():
#     if "layer4" in name or "fc" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

model = model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# 训练和评价函数
def train_and_evaluate_model(model, trainloader, valloader, testloader, criterion, optimizer, scheduler, epochs):
    # try:
    #     model.load_state_dict(torch.load('model.pth'))
    #     print("Loaded existing model weights from model.pth")
    # except FileNotFoundError:
    #     print("No existing model weights found, starting training from scratch")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()  # 记录开始时间

        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time  # 记录结束时间并计算这个epoch的时间
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.3f} Accuracy: {train_accuracy:.2f}% Time: {epoch_time:.2f} seconds')

        # 保存模型结构和参数
        torch.save(model, f'model_50.pth')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1} Validation Loss: {avg_val_loss:.3f} Accuracy: {val_accuracy:.2f}%')

        # 更新学习率
        scheduler.step()

    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {avg_test_loss:.3f} Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    # 训练和测试模型
    train_and_evaluate_model(model, trainloader, valloader, testloader, criterion, optimizer, scheduler, epochs=100)
