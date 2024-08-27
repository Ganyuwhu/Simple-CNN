#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def save_picture(x, y, name):
    plt.figure()
    plt.plot(x, y)
    plt.savefig(name)
    plt.close()
