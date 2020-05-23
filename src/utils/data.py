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


def generate(model, src, dst, gpu):
  print('{}->{}'.format(os.path.normpath(src), os.path.normpath(dst)), flush=True)

  s = numpy.load(src)
  x = (s / MAX_VALUE) ** 0.5
  x = numpy.pad(x, ((0, 0), (8, 8)))
  x = [x[:, i:i + 64] for i in range(x.shape[1] - 63)]
  x = numpy.stack(x, axis=0)[:, None]
  x = torch.tensor(x.astype(numpy.float32))

  if gpu is not None:
    x = x.cuda(gpu)

  model.eval()

  with torch.no_grad():
    y = model(x).cpu().detach().numpy()[:, 0]

  z = numpy.zeros((s.shape[0], s.shape[1] + 16), dtype=s.dtype)
  c = numpy.zeros(z.shape[1], dtype=s.dtype)

  for i, v in enumerate(y):
    z[:, i:i + 64] += v
    c[i:i + 64] += 1

  z /= c[None, :]
  z = z ** 2 * MAX_VALUE
  z = z[:, 8:-8]

  numpy.save(dst, z)

