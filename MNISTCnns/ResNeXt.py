#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造改良版的残差网络ResNeXt
"""


class ResNeXtBlock(nn.Module):
    """
        构造ResNeXt中的残差块，相比原版ResNet，ResNeXt对残差块的输入项进行分组卷积，从而拓展了网络的宽度；
        此外，由于论文中是以ResNet-50作为模板，而ResNet-50和ResNet-34中使用的ResBlock有一些差别，因此这里
        将修改部分ResBlock的结构
    """
    def __init__(self, _in_channels, _mid_channels, _out_channels, cardinality, _stride=1):
        super(ResNeXtBlock, self).__init__()
        # 1、1*1卷积层，用于降低维度减少运算次数，一般将维度降低至输出通道数的1/2(?)
        self.descent = (_in_channels != _out_channels)  # 表示是否需要对x进行处理

        self.Conv1 = nn.Conv2d(in_channels=_in_channels, out_channels=_mid_channels, kernel_size=1, padding=0,
                               stride=_stride)
        self.Bth1 = nn.BatchNorm2d(_mid_channels)

        # 2、3*3卷积层
        self.Conv2 = nn.Conv2d(in_channels=_mid_channels, out_channels=_mid_channels, kernel_size=3, padding=1,
                               stride=1, groups=cardinality)
        self.Bth2 = nn.BatchNorm2d(_mid_channels)

        # 3、1*1卷积层，用于还原维度
        self.Conv3 = nn.Conv2d(in_channels=_mid_channels, out_channels=_out_channels, kernel_size=1, padding=0,
                               stride=1)
        self.Bth3 = nn.BatchNorm2d(_out_channels)

        # 若通道数翻倍，则对x进行修改操作：
        # 1、通道数匹配
        self.px1 = nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, kernel_size=1)
        # 2、维度下降
        self.px2 = nn.Conv2d(in_channels=_out_channels, out_channels=_out_channels, kernel_size=_stride, stride=_stride)

    def forward(self, x):
        output1 = f.relu(self.Bth1(self.Conv1(x)))
        output2 = f.relu(self.Bth2(self.Conv2(output1)))
        output = self.Bth3(self.Conv3(output2))
        if self.descent != 0:
            identity = self.px2(self.px1(x))
        else:
            identity = x
        return f.relu(output+identity)


class ResNeXt(nn.Module):
    # 构造ResNeXt网络结构
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.net = nn.Sequential(
            # C1、7*7卷积层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),

            # P1、最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # R2、3*ResNeXt块
            ResNeXtBlock(_in_channels=64, _mid_channels=128, _out_channels=256, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=256, _mid_channels=128, _out_channels=256, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=256, _mid_channels=128, _out_channels=256, cardinality=32, _stride=1),

            # R3、4*ResNeXt块
            ResNeXtBlock(_in_channels=256, _mid_channels=256, _out_channels=512, cardinality=32, _stride=2),
            ResNeXtBlock(_in_channels=512, _mid_channels=256, _out_channels=512, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=512, _mid_channels=256, _out_channels=512, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=512, _mid_channels=256, _out_channels=512, cardinality=32, _stride=1),

            # R4、6*ResNeXt块
            ResNeXtBlock(_in_channels=512, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=2),
            ResNeXtBlock(_in_channels=1024, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=1024, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=1024, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=1024, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=1024, _mid_channels=512, _out_channels=1024, cardinality=32, _stride=1),

            # R5、3*ResNeXt块
            ResNeXtBlock(_in_channels=1024, _mid_channels=1024, _out_channels=2048, cardinality=32, _stride=2),
            ResNeXtBlock(_in_channels=2048, _mid_channels=1024, _out_channels=2048, cardinality=32, _stride=1),
            ResNeXtBlock(_in_channels=2048, _mid_channels=1024, _out_channels=2048, cardinality=32, _stride=1),

            # P6 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),

            # 展平
            nn.Flatten(),

            # dropout
            nn.Dropout(0.9),

            # 全连接层
            nn.Linear(2048, 10)
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
