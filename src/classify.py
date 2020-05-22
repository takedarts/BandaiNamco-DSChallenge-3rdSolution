#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import Classifier
from utils import Dataset

import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy
import argparse
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(description='train a classifier model')
parser.add_argument('file', help='file name')
parser.add_argument('--channels', type=int, default=32, help='number of channels')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--gpu', type=int, default=None, help='gpu id')


def mixup(images, targets, beta=1.0):
  lam = numpy.random.beta(beta, beta)
  rand_index = torch.randperm(images.shape[0], device=images.device)  # @UndefinedVariable

  # generate mixed images
  images = lam * images + (1 - lam) * images[rand_index, :]

  # generate mixed targets
  targets = targets * lam + targets[rand_index] * (1 - lam)

  return images, targets


def main():
  args = parser.parse_args()

  # model
  model = Classifier(args.channels)
  optimizer = optim.SGD(
    model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

  if args.gpu is not None:
    model.cuda(args.gpu)

  # dataset
  raw_loader = torch.utils.data.DataLoader(
    Dataset(os.path.join(DATA_DIR, 'raw')),
    args.batch // 2, shuffle=True, drop_last=True)
  noised_loader = torch.utils.data.DataLoader(
    Dataset(os.path.join(DATA_DIR, 'noised_tgt')),
    args.batch // 2, shuffle=True, drop_last=True)

  # train
  for epoch in range(args.epoch):
    loss = 0
    accuracy = 0
    count = 0

    for x0, x1 in zip(noised_loader, raw_loader):
      if args.gpu is not None:
        x0 = x0.cuda(args.gpu)
        x1 = x1.cuda(args.gpu)

      # train
      model.train()

      x = torch.cat((x0, x1), dim=0)  # @UndefinedVariable
      t = torch.zeros((x.shape[0], 2), device=x.device).float()  # @UndefinedVariable

      t[:x0.shape[0], 0] = 1
      t[x0.shape[0]:, 1] = 1

      x, t = mixup(x, t)
      y = model(x)
      e = (-1 * nn.functional.log_softmax(y, dim=1) * t).sum(dim=1).mean()

      optimizer.zero_grad()
      e.backward()
      optimizer.step()

      # validate
      model.eval()

      with torch.no_grad():
        y0 = (model(x0).max(dim=1)[1] == 0).float()
        y1 = (model(x1).max(dim=1)[1] == 1).float()

      a = torch.cat((y0, y1), dim=0).mean()  # @UndefinedVariable

      loss += float(e) * len(x)
      accuracy += float(a) * len(x)
      count += len(x)

    print('[{}] lr={:.7f}, loss={:.4f}, accuracy={:.4f}'.format(
      epoch, float(optimizer.param_groups[0]['lr']), loss / count, accuracy / count),
      flush=True)

    scheduler.step()

    snapshot = {'channels': args.channels, 'model': model.state_dict()}
    torch.save(snapshot, '{}.tmp'.format(args.file))
    os.rename('{}.tmp'.format(args.file), args.file)


if __name__ == '__main__':
  main()
