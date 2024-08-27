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


class DenseBlock(nn.Module):
    def __init__(self, _in_channels, loop):
        super(DenseBlock, self).__init__()
        # DenseBlock块基本结构
        self.Conv1 = nn.Conv2d(in_channels=_in_channels, out_channels=_in_channels, kernel_size=1, padding=0, stride=1)
        self.Conv2 = nn.Conv2d(in_channels=_in_channels, out_channels=_in_channels, kernel_size=3, padding=1, stride=1)
        self.Compute = nn.LeakyReLU()
        self.Loop = loop

    def forward(self, x):
        X = x
        for i in range(self.Loop):
            x_temp = self.Compute(self.Conv2(self.Conv1(X)))
            X = torch.cat((X, x_temp), dim=1)
        return X


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
            DenseBlock(_in_channels=64, loop=6)

            # T3 Transition layer

        )
