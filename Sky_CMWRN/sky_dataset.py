import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from cmwrn_config import Config

class SkyDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图片相对路径
        relative_path = self.df.iloc[idx, 0]
        image_path = os.path.join(self.base_dir, relative_path.replace('\\', '/'))

        # 打开图片并转换为RGB格式
        image = Image.open(image_path).convert('RGB')

        # 如果有transform，应用转换
        if self.transform:
            image = self.transform(image)

        # 获取标签（0：有天空，1：没有天空）
        label = torch.tensor(self.df.iloc[idx, 1], dtype=torch.long)

        return image, label

def get_transforms():
    """
    返回训练集和验证集的预处理变换操作
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4652, 0.4709, 0.4592], std=[0.2219, 0.2152, 0.2203])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4652, 0.4709, 0.4592], std=[0.2219, 0.2152, 0.2203])
    ])

    return train_transform, val_transform

def get_dataloaders():
    """
    获取训练集、验证集、测试集的DataLoader
    """
    # # 读取CSV文件
    # df = pd.read_csv(r'D:\Study\Project\weather_new\datasets\sky_labels.csv')
    #
    # # 按照8:1:1的比例划分数据集（训练集、验证集、测试集）
    # train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    df = pd.read_csv(r'xxxx').sample(frac=1, random_state=42)  # 先全局打乱

    test_dfs = []
    val_dfs = []
    train_dfs = []

    # 按类别分层抽取
    for class_id, group in df.groupby('Sky Label'):
        # 从每个类别中抽取测试集
        test_class = group.sample(n=300, random_state=42)
        remaining = group.drop(test_class.index)

        # 从剩余数据中抽取验证集
        val_class = remaining.sample(n=300, random_state=42)
        train_class = remaining.drop(val_class.index)

        test_dfs.append(test_class)
        val_dfs.append(val_class)
        train_dfs.append(train_class)

    # 合并所有类别
    test_df = pd.concat(test_dfs)
    val_df = pd.concat(val_dfs)
    train_df = pd.concat(train_dfs)

    # 验证数据集无交集
    assert len(set(test_df.index) & set(val_df.index)) == 0, "测试集和验证集有交集"
    assert len(set(train_df.index) & set(test_df.index)) == 0, "训练集和测试集有交集"
    assert len(set(train_df.index) & set(val_df.index)) == 0, "训练集和验证集有交集"



    # 获取数据转换（预处理）
    train_transform, val_transform = get_transforms()

    # 创建数据集对象
    train_dataset = SkyDataset(train_df, base_dir=r'xxx', transform=train_transform)
    val_dataset = SkyDataset(val_df, base_dir=r'xxx', transform=val_transform)
    test_dataset = SkyDataset(test_df, base_dir=r'xxx', transform=val_transform)

    # 创建DataLoader对象
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
