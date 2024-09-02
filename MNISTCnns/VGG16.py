#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造VGG16卷积神经网络
"""


class vgg_16(nn.Module):

    def __init__(self):
        super(vgg_16, self).__init__()

        # 定义网格结构
        self.net = nn.Sequential(
            # input 1 * 224 * 224
            # C1 卷积层 + ReLU
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C2 卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P2 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C3 卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C4 卷积层 + ReLU
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P4 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C5 卷积层 + ReLU
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C6 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C7 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P7 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C8 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C9 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C10 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P10 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C11 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C12 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C13 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P13 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 铺平
            nn.Flatten(),

            # FC14 全连接层
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # FC15 全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # FC16 全连接层
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        return self.net(x)


def Get_dataset():
    # 创建数据集

    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(0.1307, 0.3801)
    ])

    # 制作数据集

    train_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple-CNN',
        train=True,
        download=True,
        transform=transform
    )

    test_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple-CNN',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_Data, shuffle=True, batch_size=64)

    return train_Data, test_Data, train_loader, test_loader
