#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.utils.data
import numpy
import os

MAX_VALUE = 413.5


class Dataset(torch.utils.data.Dataset):

  def __init__(self, path):
    super().__init__()

    self.data = []
    self.indexes = []

    for name in sorted(os.listdir(path)):
      if not name.endswith('.npy'):
        continue

      x = numpy.load(os.path.join(path, name))
      x = (x / MAX_VALUE) ** 0.5
      x = numpy.pad(x, ((0, 0), (8, 8)))
      i = len(self.data)

      self.data.append(x)
      self.indexes.extend((i, j) for j in range(x.shape[1] - 63))

  def __len__(self):
    return len(self.indexes)

  def __getitem__(self, idx):
    n, o = self.indexes[idx]

    return self.data[n][None, :, o:o + 64].astype(numpy.float32)

