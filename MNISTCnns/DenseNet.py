#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造密集网络DenseNet的网络结构
"""


class DenseLayer(nn.Module):
    """
        DenseBlock的基本结构，由BN-Conv-relu-Conv构成
    """

    def __init__(self, in_channels, growth_rate):
        """
        :param in_channels: 输入通道数
        :param growth_rate: 增长率
        """
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=1, padding=0,
                               stride=1)
        self.Conv2 = nn.Conv2d(in_channels=growth_rate, out_channels=growth_rate, kernel_size=3, padding=1,
                               stride=1)

    def forward(self, x):
        new_features = self.Conv2(f.relu(self.Conv1(self.bn(x))))
        return torch.cat((x, new_features), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, size):
        """
        :param in_channels: 输入通道数
        :param growth_rate: 增长率
        :param size: 包含的DenseLayer数
        """
        super(DenseBlock, self).__init__()
        self.layer_list = []
        for i in range(size):
            _layer = DenseLayer(in_channels=in_channels+i*growth_rate, growth_rate=growth_rate)
            self.layer_list.append(_layer)

    def forward(self, x):
        for _layer in self.layer_list:
            x = _layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, _in_channels):
        super(Transition, self).__init__()
        # Transition layer基本结构
        self.Conv = nn.Conv2d(in_channels=_in_channels, out_channels=_in_channels, kernel_size=1, padding=0, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

    def forward(self, x):
        return self.pool(self.Conv(x))


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        # DenseNet基本结构
        self.net = nn.Sequential(
            # C1 7*7卷积层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),

            # P1 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # D2 DenseBlock块
            DenseBlock(in_channels=64, growth_rate=32, size=6),

            # T3 Transition layer
            Transition(256),

            # D4 DenseBlock块
            DenseBlock(in_channels=256, growth_rate=32, size=12),

            # T5 Transition layer
            Transition(640),

            # D6 DenseBlock块
            DenseBlock(in_channels=640, growth_rate=32, size=24),

            # T7 Transition layer
            Transition(1408),

            # D8 DenseBlock块
            DenseBlock(in_channels=1408, growth_rate=32, size=16),

            # P9 7*7池化层
            nn.AvgPool2d(kernel_size=7, stride=1),

            # 展平
            nn.Flatten(),

            # 全连接层
            nn.Linear(1920, 10)
        )

    def forward(self, x):
        return self.net(x)


# 制作数据集
def Get_dataset():
    # 创建数据集

    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=None),
        transforms.Normalize(0.1307, 0.3801)
    ])

    # 制作数据集

    train_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple CNN',
        train=True,
        download=True,
        transform=transform
    )

    test_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple CNN',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_Data, shuffle=True, batch_size=64)

    return train_Data, test_Data, train_loader, test_loader
