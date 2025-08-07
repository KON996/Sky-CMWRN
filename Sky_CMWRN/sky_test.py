import torch
from sky_model import get_model
from sky_dataset import get_dataloaders
from tqdm import tqdm  # 导入tqdm库
import pandas as pd
def test_model(model_path='sky_best_model.pth'):
    # 加载数据集
    _, _, test_loader = get_dataloaders()

    # 获取模型
    model = get_model()

    # 使用device来判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path))

    # 测试模型
    model.eval()
    correct = 0
    total = 0

    # 用于存储预测结果和真实结果
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 记录预测结果和真实结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    num_classes = 2
    normalized_accuracy = (test_accuracy - 1 / num_classes) / (1 - 1 / num_classes)
    results_df = pd.DataFrame({
        'Predicted Sky': all_predictions,
        'True Sky': all_labels
    })

    results_df.to_csv('test_result/sky_test_results.csv', index=False)
    classes = ['没有天空', '有天空']

    for i in range(len(results_df)):
        print(f'第 {i+1} 个样本:')
        print(f'  预测情况: {classes[int(results_df.at[i, "Predicted Sky"])]},'
              f'  真实情况: {classes[int(results_df.at[i, "True Sky"])]}')
    print(f"Test Accuracy: {test_accuracy:.4f}, Test N_Accuracy: {normalized_accuracy:.4f}")




if __name__ == "__main__":
    test_model('sky_best_model.pth')
