import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSkyClassifier(nn.Module):
    def __init__(self):
        super(ResNetSkyClassifier, self).__init__()
        # 加载预训练的ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # 修改最后的全连接层，输出2类（有天空/没有天空）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

def get_model():
    model = ResNetSkyClassifier()
    return model
