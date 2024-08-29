#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import recall_score


def Get_Precision(model, loader, data_type):
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            predict = model(x)
            _, predict = torch.max(predict.data, dim=1)
            correct += torch.sum((predict == y))
            total += y.size(0)

    if data_type == 'train':
        print(f'对训练集的精度为:{correct * 100 / total}%')

    elif data_type == 'check':
        print(f'对验证集的精度为:{correct * 100 / total}%')

    else:
        print(f'对测试集的精度为:{correct * 100 / total}%')


def Get_callback(model, loader, data_type):
    with torch.no_grad():
        for (x, y) in loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            predict = model(x)

    recall = recall_score(y, predict)

    if data_type == 'train':
        print(f'对训练集的召回率为:{recall}')

    elif data_type == 'check':
        print(f'对验证集的召回率为:{recall}')

    else:
        print(f'对测试集的召回率为:{recall}')
