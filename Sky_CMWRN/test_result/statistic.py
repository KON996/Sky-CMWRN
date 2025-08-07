import pandas as pd

# 读取CSV文件
df = pd.read_csv('cmwrn_test_results.csv')  # 替换为你的文件路径

# 确保列名正确（处理可能的空格）
true_col = 'True Rain Level'
pred_col = 'Predicted Rain Level'

# 获取所有类别（自动识别）
classes = sorted(df[true_col].unique().tolist())

# 计算每个类别的准确率（召回率）
accuracy_dict = {}

for cls in classes:
    # 获取真实标签为当前类别的样本
    true_samples = df[df[true_col] == cls]

    # 计算正确预测数
    correct = true_samples[true_samples[pred_col] == cls].shape[0]

    # 计算总数
    total = true_samples.shape[0]

    # 计算准确率（避免除零错误）
    accuracy = correct / total if total != 0 else 0.0
    accuracy_dict[f'Class {cls}'] = accuracy

# # 格式化输出结果
# print("各类别预测准确率：")
# for cls, acc in accuracy_dict.items():
#     print(f"{cls}: {acc:.2%} ({correct}/{total})".format(
#         correct=correct, total=total)) if '=' in cls else None  # 移除此行注释如果需要显示具体数字

# 可选：转换为DataFrame输出
result_df = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['Accuracy'])
print("\n表格格式结果：")
print(result_df)