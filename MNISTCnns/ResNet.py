#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造残差网络ResNet-34的结构
"""


class ResBlock(nn.Module):
    # 构造补偿为2的ResBlock块
    def __init__(self, _in_channels, _out_channels, _stride=1):
        super(ResBlock, self).__init__()

        # 定义ResBlock内部结构
        # ResBlock块由两个3*3卷积核串联而成，如果通道数翻倍，那么矩阵的维度将要降低一半
        # 因此残差的维数分为两种情况：
        # 1、若输入通道数和输出通道数相同，则可以直接进行残差连接
        # 2、若输入通道数和输出通道数不同，则需要匹配维度和通道数
        self.descent = _stride - 1  # 表示是否需要对x进行处理

        self.p1 = nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, kernel_size=3, stride=_stride,
                            padding=1)  # 若stride=2，则需要卷积核匹配
        self.Bth1 = nn.BatchNorm2d(_out_channels)

        self.p2 = nn.Conv2d(in_channels=_out_channels, out_channels=_out_channels, kernel_size=3, stride=1,
                            padding=1)
        self.Bth2 = nn.BatchNorm2d(_out_channels)
        self.relu = nn.LeakyReLU()

        # 若通道数翻倍，则对x进行修改操作：
        # 1、通道数匹配
        self.px1 = nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, kernel_size=1)
        # 2、维度下降
        self.px2 = nn.Conv2d(in_channels=_out_channels, out_channels=_out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        output = self.Bth2(self.p2(f.relu(self.Bth1(self.p1(x)))))
        if self.descent != 0:
            identity = self.px2(self.px1(x))
        else:
            identity = x
        return self.relu(output+identity)


class ResNet(nn.Module):
    # 构建ResNet基本网络结构
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            # C1 7*7卷积层 + ReLU
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 7), padding=(0, 3), stride=(1, 2)),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 1), padding=(3, 0), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # R2 ResBlock块，通道数不变(64)，维度下降至原本的1/2
            ResBlock(_in_channels=64, _out_channels=64, _stride=2),

            # R3 ResBlock块，通道数不变(64)，维度不变
            ResBlock(_in_channels=64, _out_channels=64, _stride=1),

            # R4 ResBlock块，通道数不变(64)，维度不变
            ResBlock(_in_channels=64, _out_channels=64, _stride=1),

            # R5 ResBlock块，通道数变为原来的2倍(128)，维度下降至原本的1/2
            ResBlock(_in_channels=64, _out_channels=128, _stride=2),

            # R6 ResBlock块，通道数不变(128)，维度不变
            ResBlock(_in_channels=128, _out_channels=128, _stride=1),

            # R7 ResBlock块，通道数不变(128)，维度不变
            ResBlock(_in_channels=128, _out_channels=128, _stride=1),

            # R8 ResBlock块，通道数不变(128)，维度不变
            ResBlock(_in_channels=128, _out_channels=128, _stride=1),

            # R9 ResBlock块，通道数变为原来的2倍(256)，维度下降至原本的1/2
            ResBlock(_in_channels=128, _out_channels=256, _stride=2),

            # R10 ResBlock块，通道数不变(256)，维度不变
            ResBlock(_in_channels=256, _out_channels=256, _stride=1),

            # R11 ResBlock块，通道数不变(256)，维度不变
            ResBlock(_in_channels=256, _out_channels=256, _stride=1),

            # R12 ResBlock块，通道数不变(256)，维度不变
            ResBlock(_in_channels=256, _out_channels=256, _stride=1),

            # R13 ResBlock块，通道数不变(256)，维度不变
            ResBlock(_in_channels=256, _out_channels=256, _stride=1),

            # R14 ResBlock块，通道数不变(256)，维度不变
            ResBlock(_in_channels=256, _out_channels=256, _stride=1),

            # R15 ResBlock块，通道数变为原来的2倍(512)，维度下降至原本的1/2
            ResBlock(_in_channels=256, _out_channels=512, _stride=2),

            # R16 ResBlock块，通道数不变(512)，维度不变
            ResBlock(_in_channels=512, _out_channels=512, _stride=1),

            # R17 ResBlock块，通道数不变(512)，维度不变
            ResBlock(_in_channels=512, _out_channels=512, _stride=1),

            # P18 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),

            # 展平
            nn.Flatten(),

            # dropout
            nn.Dropout(0.9),

            # F19 全连接层
            nn.Linear(512, 10)

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

    train_loader = DataLoader(train_Data, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_Data, shuffle=True, batch_size=128)

    return train_Data, test_Data, train_loader, test_loader
