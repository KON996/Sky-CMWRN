import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

def one_calculate_mean_std(image_dir, image_size=(224, 224)):
    # 初始化变量
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    # 定义变换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            # 重塑张量为 (channels, height*width)
            image = image.view(image.size(0), -1)  # (channels, height*width)
            mean += image.mean(dim=1)  # mean over spatial dimensions
            std += image.std(dim=1)  # std over spatial dimensions
            nb_samples += 1

    # 计算最终的均值和标准差
    mean /= nb_samples
    std /= nb_samples

    return mean, std
def two_calculate_mean_std(image_dirs, image_size=(224, 224)):
    # 初始化变量
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    # 定义变换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # 遍历每个文件夹
    for image_dir in image_dirs:
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                # 重塑张量为 (channels, height*width)
                image = image.view(image.size(0), -1)  # (channels, height*width)
                mean += image.mean(dim=1)  # mean over spatial dimensions
                std += image.std(dim=1)  # std over spatial dimensions
                nb_samples += 1

    # 计算最终的均值和标准差
    mean /= nb_samples
    std /= nb_samples

    return mean, std
if __name__ == "__main__":
    # # 指定图像文件夹路径
    # image_dir = r'D:\Study\Project\Create_Data\weather'
    #
    # # 计算均值和标准差
    # mean, std = one_calculate_mean_std(image_dir)
    # 指定图像文件夹路径
    image_dirs = [
        r'xxxx',
        r'xxxx'
    ]

    # 计算均值和标准差
    mean, std = two_calculate_mean_std(image_dirs)

    print(f"Mean: {mean}")
    print(f"Std: {std}")
