import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from cmwrn_config import Config
from cmwrn_model import WeatherResNet
from cmwrn_dataset import get_dataloaders
from cmwrn_metrics import calculate_metrics
# from model_5 import WeatherResNet
# from Alexnet import WeatherAlexNet

def test():
    model = WeatherResNet(pretrained=False).to(Config.DEVICE)
    # model = WeatherAlexNet(pretrained=False).to(Config.DEVICE)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, _, test_loader = get_dataloaders()

    num_classes = 3
    all_preds_sunny = []
    all_targets_sunny = []
    all_correct_rain = 0
    total_samples = 0
    all_preds_rain = []
    all_targets_rain = []

    with torch.no_grad():
        for images, sunny_targets, rain_targets in tqdm(test_loader, desc='Testing'):
            images = images.to(Config.DEVICE)
            sunny_targets = sunny_targets.to(Config.DEVICE)  # 移动 sunny_targets 到指定设备
            rain_targets = rain_targets.to(Config.DEVICE)    # 移动 rain_targets 到指定设备
            sunny_outputs, rain_outputs = model(images)
            all_preds_sunny.extend(sunny_outputs.cpu().numpy())
            all_targets_sunny.extend(sunny_targets.cpu().numpy())

            _, predicted_rain = torch.max(rain_outputs, 1)
            all_preds_rain.extend(predicted_rain.cpu().numpy())
            all_targets_rain.extend(rain_targets.cpu().numpy())
            all_correct_rain += (predicted_rain == rain_targets).sum().item()
            total_samples += rain_targets.size(0)

    all_preds_sunny = np.array(all_preds_sunny)
    all_targets_sunny = np.array(all_targets_sunny)
    sunny_metrics = calculate_metrics(all_preds_sunny.flatten(), all_targets_sunny.flatten())
    rain_accuracy = all_correct_rain / total_samples
    normalized_accuracy = (rain_accuracy - 1 / num_classes) / (1 - 1 / num_classes)

    results_df = pd.DataFrame({
        'Predicted Sunny Degree': all_preds_sunny,
        'True Sunny Degree': all_targets_sunny,
        'Predicted Rain Level': all_preds_rain,
        'True Rain Level': all_targets_rain
    })

    results_df.to_csv('test_result/cmwrn_test_results.csv', index=False)
    classes = ['无雨', '中小雨', '大雨']

    for i in range(len(results_df)):
        print(f'第 {i+1} 个样本:')
        print(f'  预测晴阴程度: {results_df.at[i, "Predicted Sunny Degree"]:.4f},'
              f'  真实晴阴程度: {results_df.at[i, "True Sunny Degree"]:.4f}')
        print(f'  预测降雨类别: {classes[int(results_df.at[i, "Predicted Rain Level"])]},'
              f'  真实降雨类别: {classes[int(results_df.at[i, "True Rain Level"])]}')

    print(f'\n评估指标:')
    print(f'Sunny RMSE: {sunny_metrics[0]:.4f}')
    print(f'Sunny CORR: {sunny_metrics[1]:.4f}')
    print(f'Sunny SAGR: {sunny_metrics[2]:.4f}')
    print(f'Sunny CCC: {sunny_metrics[3]:.4f}')
    print(f'Rain Accuracy: {rain_accuracy:.4f}')
    print(f'Rain N-Accuracy: {normalized_accuracy:.4f}')

if __name__ == '__main__':
    test()