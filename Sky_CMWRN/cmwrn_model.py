import torch
import torch.nn as nn
import torchvision.models as models
import copy

class WeatherResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(WeatherResNet, self).__init__()

        # 加载预训练ResNet-101
        resnet = models.resnet101(pretrained=pretrained)

        # 共享层定义 (conv1 ~ conv4_x)
        self.shared_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # conv2_x
            resnet.layer2,  # conv3_x
            resnet.layer3   # conv4_x
        )

        # 任务特定层克隆 (conv5_x)
        self.sunny_conv5 = copy.deepcopy(resnet.layer4)  # 阴晴回归分支
        self.rain_conv5 = copy.deepcopy(resnet.layer4)   # 降雨分类分支

        # 任务特定头部
        self._build_task_heads()

    def _build_task_heads(self):
        """为两个任务分别构建头部"""
        # 共用池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 多层结构
        # 阴晴回归头
        self.sunny_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()  # 输出范围[-1,1]
        )

        # 降雨分类头
        self.rain_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 三分类
        )
        # # 简化结构
        # # 阴晴回归头
        # self.sunny_head = nn.Sequential(
        #     nn.Linear(2048, 1),
        #     nn.Tanh()  # 输出范围[-1,1]
        # )
        #
        # # 降雨分类头
        # self.rain_head = nn.Sequential(
        #     nn.Linear(2048, 3)  # 三分类
        # )

    def forward(self, x, run_sunny=True):
        # 共享特征提取
        shared_features = self.shared_layers(x)

        # ========== 降雨分类分支（始终执行） ==========
        rain_feat = self.rain_conv5(shared_features)
        rain_out = self.avgpool(rain_feat)
        rain_out = torch.flatten(rain_out, 1)
        rain_out = self.rain_head(rain_out)

        # ========== 阴晴回归分支（条件执行） ==========
        sunny_out = None
        if run_sunny:
            sunny_feat = self.sunny_conv5(shared_features)
            sunny_out = self.avgpool(sunny_feat)
            sunny_out = torch.flatten(sunny_out, 1)
            sunny_out = self.sunny_head(sunny_out).squeeze(1)

        return sunny_out, rain_out