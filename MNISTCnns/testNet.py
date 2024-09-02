#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as opt
import time
from matplotlib_inline import backend_inline
from MNISTCnns import Schedulers
from MNISTCnns import Precisions
backend_inline.set_matplotlib_formats('svg')


def test(_model, _train_loader, _test_loader, _learning_rate=0.001, _loss_fn=nn.CrossEntropyLoss(), _momentum=1.0,
         batch_size=64, scheduler_type='Origin'):
    losses = []

    loss_temp = [0] * batch_size
    result_temp = 1e-5
    epochs = 100
    lr = _learning_rate
    start_time = time.time()

    optimizer = opt.Adam(_model.parameters(), lr=lr)
    scheduler = Schedulers.init_scheduler(base_lr=lr, Type=scheduler_type)
    print('使用的学习率下降调度器为：', scheduler.__class__.__name__)
    result = 0

    for _epochs in range(epochs):
        loss_epoch = []

        if scheduler_type == 'Origin':
            lr = scheduler(result)
        elif scheduler_type == 'Factor':
            lr = scheduler()
        else:
            lr = scheduler(_epochs+1)

        for (x, y) in _train_loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            optimizer.zero_grad()
            predict = _model(x)
            loss = _loss_fn(predict, y)
            losses.append(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        diff = [(a - b) ** 2 for a, b in zip(loss_epoch, loss_temp)]
        result = sum(diff)

        print(
            f'第 {_epochs+1} 次训练: loss = {max(loss_epoch)}, lr = {lr}, loss_error = {result}, '
            f'result change rate = {abs(result - result_temp) / result_temp}'
        )

        if max(loss_epoch) < 0.1:
            print(f'训练次数: {_epochs}')
            break

        Precisions.Get_Precision(_model, _train_loader, 'train')
        Precisions.Get_Precision(_model, _test_loader, 'test')
        Precisions.Get_callback(_model, _train_loader, 'train')
        Precisions.Get_callback(_model, _test_loader, 'test')
        Precisions.Get_F1score(_model, _train_loader, 'train')
        Precisions.Get_F1score(_model, _test_loader, 'test')

        loss_temp = loss_epoch
        result_temp = result

    end_time = time.time()

    print('程序运行时间:\t', end_time - start_time)

    return _model, losses
