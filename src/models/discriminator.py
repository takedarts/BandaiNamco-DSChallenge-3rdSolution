#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn


class ResBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel, stride):
    super().__init__()

    self.op = nn.Sequential(
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel,
        stride=stride, padding=kernel // 2, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels))

    if in_channels != out_channels or stride != 1:
      self.down = nn.Sequential(
        nn.Conv2d(
          in_channels, out_channels, kernel_size=1,
          stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels))
    else:
      self.down = nn.Identity()

  def forward(self, x):
    return self.down(x) + self.op(x)


class Discriminator(nn.Module):

  def __init__(self, channels):
    super().__init__()

    self.op = nn.Sequential(
      nn.Conv2d(1, channels * 2, kernel_size=7, padding=3, stride=2, bias=False),
      nn.BatchNorm2d(channels * 2),
      ResBlock(channels * 2, channels * 4, kernel=3, stride=2),
      ResBlock(channels * 4, channels * 8, kernel=3, stride=2),
      ResBlock(channels * 8, channels * 16, kernel=3, stride=2),
      ResBlock(channels * 16, channels * 16, kernel=3, stride=1),
      ResBlock(channels * 16, channels * 16, kernel=3, stride=1),
      ResBlock(channels * 16, channels * 16, kernel=3, stride=1),
      nn.BatchNorm2d(channels * 16),
      nn.ReLU(inplace=True),
      nn.Conv2d(channels * 16, 1, kernel_size=1, padding=0, bias=True),
      nn.AdaptiveAvgPool2d(1),
      nn.Sigmoid())

  def forward(self, x):
    return self.op(x).view(-1)

