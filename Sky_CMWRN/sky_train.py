import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sky_model import get_model
from sky_dataset import get_dataloaders
import torch.nn as nn
from tqdm import tqdm  # 导入tqdm库
import time

def train_model():
    # 加载数据集
    train_loader, val_loader, _ = get_dataloaders()

    # 获取模型
    model = get_model()

    # 使用device来判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 存储每轮的损失和准确率
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # 变量用于保存最好的模型
    best_val_acc = 0.0
    best_model_wts = None
    start_time = time.time()
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_n_accuracy = (train_accuracy - 1 / 2) / (1 - 1 / 2)
        # 计算验证集的损失和准确率
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_n_accuracy = (val_accuracy - 1 / 2) / (1 - 1 / 2)

        # 保存效果更好的模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = model.state_dict()
            print('当前最优模型参数已记录')

        # 存储每轮的损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train N_Accuracy: {train_n_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val N_Accuracy: {val_n_accuracy:.4f}")
        end_time = time.time()
        user_time = end_time - start_time
        print(f'训练时间: {int(user_time // 60)} m {int(user_time%60)} s ')
        # 调整学习率
        scheduler.step(val_loss)

    # 加载最佳模型
    model.load_state_dict(best_model_wts)

    # 保存最佳模型
    torch.save(model.state_dict(), 'sky_best_model.pth')

    # 绘制损失和准确率变化曲线
    epochs = range(1, num_epochs+1)
    # plt.figure(figsize=(12, 6))
    # 损失变化
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_losses, label='Train Loss')
    # plt.plot(epochs, val_losses, label='Val Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_result/sky_train_loss_curve.png')
    plt.close()

    # 准确率变化
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accuracies, label='Train Accuracy')
    # plt.plot(epochs, val_accuracies, label='Val Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_result/sky_train_accuracy_curve.png')
    plt.close()

if __name__ == "__main__":
    train_model()
