import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from cmwrn_config import Config

class WeatherDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        relative_path = self.df.iloc[idx, 0]
        image_path = os.path.join(self.base_dir, relative_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sunny_degree = torch.tensor(self.df.iloc[idx, 1], dtype=torch.float32)
        rain_class = torch.tensor(self.df.iloc[idx, 2], dtype=torch.long)
        return image, sunny_degree, rain_class

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
    ])

    return train_transform, val_transform

def get_dataloaders():
    # # 读取CSV文件
    # df = pd.read_csv(Config.DATA_CSV)
    #
    # # 数据集分离
    # train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    df = pd.read_csv(Config.DATA_CSV).sample(frac=1, random_state=42)  # 先全局打乱

    test_dfs = []
    val_dfs = []
    train_dfs = []

    # 按类别分层抽取
    for class_id, group in df.groupby('Rainfall'):
        # 从每个类别中抽取测试集
        test_class = group.sample(n=130, random_state=42)
        remaining = group.drop(test_class.index)

        # 从剩余数据中抽取验证集
        val_class = remaining.sample(n=100, random_state=42)
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


    # 获取数据预处理方式
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = WeatherDataset(train_df, base_dir=Config.IMAGE_DIR, transform=train_transform)
    val_dataset = WeatherDataset(val_df, base_dir=Config.IMAGE_DIR, transform=val_transform)
    test_dataset = WeatherDataset(test_df, base_dir=Config.IMAGE_DIR, transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    return train_loader, val_loader, test_loader
def transform_image(image):
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
    ])

    # 应用预处理变换
    image = transform(image)

    return image