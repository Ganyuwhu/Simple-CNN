#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class FactorScheduler:
    # 构建单因子调度器
    def __init__(self, factor=0.9, stop_factor=1e-7, base_lr=0.1):
        """
        :param self: 指向调度器本身
        :param factor: 每次衰减的倍率
        :param stop_factor: 学习率的下界，保证学习率不会过低导致训练过慢
        :param base_lr: 初始学习率
        """
        self.factor = factor
        self.stop_factor = stop_factor
        self.current_lr = base_lr

    def __call__(self):
        self.current_lr = max(self.stop_factor, self.current_lr * self.factor)
        return self.current_lr


class OriginFactor:
    # 构建普通调度器
    def __init__(self, result=0, result_temp=0, stop_factor=1e-7, base_lr=0.1):
        """
        :param result: 存放当前epoch的损失函数值
        :param result_temp: 存放上一epoch的损失函数值
        :param stop_factor: 学习率的下界，保证学习率不会过低导致训练过慢
        :param base_lr: 当前学习率
        """
        self.result = result
        self.result_temp = result_temp
        self.stop_factor = stop_factor
        self.current_lr = base_lr

    def __call__(self, result):
        """
        :param result: 当前epoch的损失函数值
        :return: self.current_lr
        """
        # 更新scheduler中的损失函数值
        self.result_temp = self.result
        self.result = result

        # 判断是否降低学习率
        if abs((self.result_temp-self.result)/(self.result_temp+1e-10)) < 0.02:
            self.current_lr /= 10
        return self.current_lr


def init_scheduler(base_lr, Type='Origin'):
    if Type == 'Origin':
        scheduler = OriginFactor(base_lr=base_lr)
    elif Type == 'Factor':
        scheduler = FactorScheduler(base_lr=base_lr)
    else:
        raise ValueError('Type must be chosen from {Origin, Factor}')
    return scheduler
