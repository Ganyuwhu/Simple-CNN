#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构建LeNet5卷积网络模型
"""


class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.net = nn.Sequential(

            # 输入数据为 batch_size * 1 * 28 * 28的灰度图像
            # C1 卷积层 + Tanh
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # 6 * 28 * 28
            nn.Tanh(),

            # P2 平均池化层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 6 * 14 * 14

            # C3 卷积层 + Tanh
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 16 * 10 * 10
            nn.Tanh(),

            # P4 平均池化层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 16 * 5 * 5
            nn.Tanh(),

            # C5 卷积层 + Tanh
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),  # # 120 * 1 * 1
            nn.Tanh(),

            # 铺平
            nn.Flatten(),

            # F6 全连接层
            nn.Linear(120, 84),
            nn.Tanh(),

            # Out 输出层
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


def Get_dataset():
    # 创建数据集

    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
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

    train_loader = DataLoader(train_Data, shuffle=True, batch_size=256)
    test_loader = DataLoader(test_Data, shuffle=True, batch_size=256)

    return train_Data, test_Data, train_loader, test_loader

