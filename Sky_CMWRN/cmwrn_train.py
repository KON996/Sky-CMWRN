import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from cmwrn_config import Config
from cmwrn_model import WeatherResNet
from cmwrn_dataset import get_dataloaders
from cmwrn_metrics import calculate_metrics, AverageMeter
import utils
import time
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
# from model_5 import WeatherResNet
# from Alexnet import WeatherAlexNet


# 新增DWA权重计算模块
class DynamicWeightAveraging:
    def __init__(self, warmup_epochs=0, temp=2.0):
        self.warmup_epochs = warmup_epochs  # 新增预热轮数
        self.temperature = temp  # 控制权重分布平滑度
        self.history = {'sunny': [], 'rain': []}  # 损失历史记录

    def __call__(self, global_epoch):
        if global_epoch < self.warmup_epochs:  # 预热阶段固定权重
            return 1.0, 1.0

        else:
            # 确保有足够的历史损失计算变化率
            if len(self.history['sunny']) >= 2 and len(self.history['rain']) >= 2:
                # 使用预热阶段+正式阶段的全部历史数据
                r_sunny = (self.history['sunny'][-1] + 1e-7) / (self.history['sunny'][-2] + 1e-7)
                r_rain = (self.history['rain'][-1] + 1e-7) / (self.history['rain'][-2] + 1e-7)
                T_sunny = torch.exp(torch.tensor(r_sunny / self.temperature))
                T_rain = torch.exp(torch.tensor(r_rain / self.temperature))
                total = T_sunny + T_rain
                return T_sunny.item()/total.item(), T_rain.item()/total.item()
            else:
                return 1.0, 1.0  # 默认固定权重
def train_one_epoch(model, train_loader, criterion_sunny, criterion_rain, optimizer, device, lambda_sunny, lambda_rain):
    model.train()
    sunny_losses = AverageMeter()
    rain_losses = AverageMeter()
    all_preds_sunny = []
    all_targets_sunny = []
    all_correct_rain = 0
    total_samples = 0

    with tqdm(train_loader, desc='Training') as pbar:
        for images, sunny_targets, rain_targets in pbar:
            images, sunny_targets, rain_targets = images.to(device), sunny_targets.to(device), rain_targets.to(device)

            optimizer.zero_grad()
            sunny_outputs, rain_outputs = model(images)
            sunny_loss = criterion_sunny(sunny_outputs, sunny_targets)
            rain_loss = criterion_rain(rain_outputs, rain_targets)
            loss = lambda_sunny * sunny_loss + lambda_rain * rain_loss  # 动态加权

            loss.backward()
            optimizer.step()

            sunny_losses.update(sunny_loss.item(), images.size(0))
            rain_losses.update(rain_loss.item(), images.size(0))
            pbar.set_postfix({'sunny_loss': f'{sunny_losses.avg:.4f}', 'rain_loss': f'{rain_losses.avg:.4f}'})

            all_preds_sunny.extend(sunny_outputs.detach().cpu().numpy())
            all_targets_sunny.extend(sunny_targets.cpu().numpy())

            _, predicted_rain = torch.max(rain_outputs, 1)
            all_correct_rain += (predicted_rain == rain_targets).sum().item()
            total_samples += rain_targets.size(0)

    all_preds_sunny = np.array(all_preds_sunny)
    all_targets_sunny = np.array(all_targets_sunny)
    sunny_metrics = calculate_metrics(all_preds_sunny.flatten(), all_targets_sunny.flatten())
    rain_accuracy = all_correct_rain / total_samples

    return sunny_losses.avg, rain_losses.avg, sunny_metrics, rain_accuracy

def validate(model, val_loader, criterion_sunny, criterion_rain, device):
    model.eval()
    sunny_losses = AverageMeter()
    rain_losses = AverageMeter()
    all_preds_sunny = []
    all_targets_sunny = []
    all_correct_rain = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(val_loader, desc='Val') as pbar:
            for images, sunny_targets, rain_targets in pbar:
                images, sunny_targets, rain_targets = images.to(device), sunny_targets.to(device), rain_targets.to(device)
                sunny_outputs, rain_outputs = model(images)
                sunny_loss = criterion_sunny(sunny_outputs, sunny_targets)
                rain_loss = criterion_rain(rain_outputs, rain_targets)

                sunny_losses.update(sunny_loss.item(), images.size(0))
                rain_losses.update(rain_loss.item(), images.size(0))

                all_preds_sunny.extend(sunny_outputs.cpu().numpy())
                all_targets_sunny.extend(sunny_targets.cpu().numpy())

                _, predicted_rain = torch.max(rain_outputs, 1)
                all_correct_rain += (predicted_rain == rain_targets).sum().item()
                total_samples += rain_targets.size(0)

    all_preds_sunny = np.array(all_preds_sunny)
    all_targets_sunny = np.array(all_targets_sunny)
    sunny_metrics = calculate_metrics(all_preds_sunny.flatten(), all_targets_sunny.flatten())
    rain_accuracy = all_correct_rain / total_samples

    return sunny_losses.avg, rain_losses.avg, sunny_metrics, rain_accuracy

def main():
    train_loader, val_loader, _ = get_dataloaders()

    model = WeatherResNet(pretrained=True).to(Config.DEVICE)
    # model = WeatherAlexNet(pretrained=True).to(Config.DEVICE)
    # (归一化加权法)初始化基准损失记录
    init_epochs = 5  # 前5个epoch用于计算基准
    dwa = DynamicWeightAveraging(warmup_epochs=init_epochs, temp=1.5)
    train_sunny_history, train_rain_history = [], []
    # 损失函数
    criterion_sunny = nn.MSELoss()
    # criterion_rain = nn.CrossEntropyLoss()


    # 从训练集DataFrame中提取所有雨天类别标签 (第3列是rain_class)逆频率加权
    rain_labels = pd.read_csv(Config.DATA_CSV).iloc[train_loader.dataset.df.index, 2].values
    # 计算类别权重 (处理可能存在的缺失类别)
    all_possible_classes = np.unique(pd.read_csv(Config.DATA_CSV).iloc[:, 2])  # 从完整数据集中获取所有可能类别
    class_weights = compute_class_weight(
        'balanced',
        classes=all_possible_classes,  # 确保覆盖所有可能类别
        y=rain_labels
    )
    # 转换为Tensor并发送到设备
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)

    # 初始化损失函数
    criterion_rain = nn.CrossEntropyLoss(weight=class_weights)

    # 验证类别数量与模型输出匹配
    # 获取rain_head 中最后一个层的out_features
    last_layer = model.rain_head[-1]
    assert isinstance(last_layer, nn.Linear), "rain_head 最后一层必须是nn.Linear"
    num_classes = len(all_possible_classes)
    assert model.rain_head[-1].out_features == num_classes, \
        f"模型输出维度({model.rain_head.out_features})与类别数量({num_classes})不匹配"


    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    train_losses_sunny = []
    val_losses_sunny = []
    train_losses_rain = []
    val_losses_rain = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()
    # 预热阶段：收集初始训练损失
    print('预热阶段，前5个epoch用于计算基准')
    for init_epoch in range(init_epochs):
        train_sunny, train_rain, _, _ = train_one_epoch(
            model, train_loader, criterion_sunny, criterion_rain,
            optimizer, Config.DEVICE, 1.0, 1.0
        )
        dwa.history['sunny'].append(train_sunny)
        dwa.history['rain'].append(train_rain)
        train_sunny_history.append(train_sunny)
        train_rain_history.append(train_rain)
        # 计算归一化基准
    base_sunny = np.mean(train_sunny_history)  # 晴天任务基准（如0.25）
    base_rain = np.mean(train_rain_history)    # 雨天任务基准（如1.83）
    # 修改模型保存逻辑
    best_norm_loss = float('inf')
    # 初始化DWA模块
    print('')
    print('训练阶段')
    for epoch in range(Config.EPOCHS):
        global_epoch = init_epochs + epoch   # 计算全局的epoch
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        lambda_sunny, lambda_rain = dwa(global_epoch)
        train_sunny_loss, train_rain_loss, train_sunny_metrics, train_rain_accuracy = train_one_epoch(
            model, train_loader, criterion_sunny, criterion_rain, optimizer, Config.DEVICE, lambda_sunny, lambda_rain
        )
        # 更新损失历史记录
        dwa.history['sunny'].append(train_sunny_loss)
        dwa.history['rain'].append(train_rain_loss)
        val_sunny_loss, val_rain_loss, val_sunny_metrics, val_rain_accuracy = validate(
            model, val_loader, criterion_sunny, criterion_rain, Config.DEVICE
        )
        # 归一化加权验证损失
        norm_sunny = val_sunny_loss / base_sunny  # 晴天归一化系数
        norm_rain = val_rain_loss / base_rain     # 雨天归一化系数
        weighted_loss = 0.6 * norm_sunny + 0.4 * norm_rain  # 可调权重
        scheduler.step(weighted_loss)
        if weighted_loss < best_norm_loss:
            best_norm_loss = weighted_loss
            utils.save_checkpoint(model, optimizer, epoch, best_norm_loss, Config.MODEL_SAVE_PATH)

        train_losses_sunny.append(train_sunny_loss)
        val_losses_sunny.append(val_sunny_loss)
        train_losses_rain.append(train_rain_loss)
        val_losses_rain.append(val_rain_loss)
        train_accuracies.append(train_rain_accuracy)
        val_accuracies.append(val_rain_accuracy)
        end_time = time.time()
        user_time = end_time - start_time
        print(f'DWA_sunny:{lambda_sunny}, DWA_rain:{lambda_rain}')
        print(f'Train Sunny Loss: {train_sunny_loss:.4f}, Train Rain Loss: {train_rain_loss:.4f}')
        print(f'Val Sunny Loss: {val_sunny_loss:.4f}, Val Rain Loss: {val_rain_loss:.4f}')
        # 日志输出归一化指标
        print(f'[NormWeight] Sunny: {norm_sunny:.2f}x | Rain: {norm_rain:.2f}x | Total: {weighted_loss:.4f}')
        print(f'Train Sunny Metrics - RMSE: {train_sunny_metrics[0]:.4f}, CORR: {train_sunny_metrics[1]:.4f}, '
              f'SAGR: {train_sunny_metrics[2]:.4f}, CCC: {train_sunny_metrics[3]:.4f}')
        print(f'Val Sunny Metrics - RMSE: {val_sunny_metrics[0]:.4f}, CORR: {val_sunny_metrics[1]:.4f}, '
              f'SAGR: {val_sunny_metrics[2]:.4f}, CCC: {val_sunny_metrics[3]:.4f}')
        print(f'Train Rain Accuracy: {train_rain_accuracy:.4f}, Val Rain Accuracy: {val_rain_accuracy:.4f}')
        train_n_accuracy = (train_rain_accuracy - 1 / 3) / (1 - 1 / 3)
        val_n_accuracy = (val_rain_accuracy - 1 / 3) / (1 - 1 / 3)
        print(f'Train Rain N-Accuracy: {train_n_accuracy:.4f}, Val Rain N-Accuracy: {val_n_accuracy:.4f}')
        print(f'训练时间: {int(user_time // 60)} m {int(user_time%60)} s ')
    utils.plot_losses(train_losses_sunny, val_losses_sunny, 'sunny train loss')
    utils.plot_losses(train_losses_rain, val_losses_rain, 'rain train Loss')
    utils.plot_accuracies(train_accuracies, val_accuracies)

if __name__ == '__main__':
    main()