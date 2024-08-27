#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as opt
import time
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')


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


def test(_model, _train_loader, _test_loader, _learning_rate=0.001, _loss_fn=nn.CrossEntropyLoss(), _momentum=1.0,
         batch_size=64):
    print(_model.__class__.__name__, '训练结果:')
    losses = []
    loss_temp = [0] * batch_size
    result_temp = 1e-5
    epochs = 100
    start_time = time.time()

    optimizer = opt.Adam(_model.parameters(), lr=_learning_rate)

    for _epochs in range(epochs):
        loss_epoch = []
        for (x, y) in _train_loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            optimizer.zero_grad()
            predict = _model(x)
            loss = _loss_fn(predict, y)
            losses.append(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

        diff = [(a - b) ** 2 for a, b in zip(loss_epoch, loss_temp)]
        result = sum(diff)

        print(
            f'第 {_epochs} 次训练: loss = {max(loss_epoch)}, lr = {_learning_rate}, loss_error = {result}, '
            f'result change rate = {abs(result - result_temp) / result_temp}'
        )

        if abs(result - result_temp) / result < 0.02:
            _learning_rate /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = _learning_rate

        if max(loss_epoch) < 0.15:
            print(f'训练次数: {_epochs}')
            break

        loss_temp = loss_epoch
        result_temp = result

    end_time = time.time()
    # fig = plt.figure()
    # plt.plot(range(len(losses)), losses)
    # plt.show()

    Get_Precision(_model, _train_loader, 'train')
    Get_Precision(_model, _test_loader, 'test')
    print('程序运行时间:\t', end_time - start_time)

    return _model, losses
