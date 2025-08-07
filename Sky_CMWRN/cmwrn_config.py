import torch

class Config:
    # 数据集配置
    DATA_CSV = r'xxxxx'
    IMAGE_DIR = r'xxxxx'
    IMAGE_SIZE = 224
    BATCH_SIZE = 28
    NUM_WORKERS = 4

    # 训练配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # 模型配置
    MODEL_SAVE_PATH = 'cmwrn_best_model.pth'

    # 数据归一化参数
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    # NORMALIZE_MEAN = [0.4747, 0.4945, 0.5067]
    # NORMALIZE_STD = [0.2291, 0.2248, 0.2340]