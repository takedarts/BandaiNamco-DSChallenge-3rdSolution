#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import Classifier, Discriminator, Generator
from utils import Dataset

import torch.utils.data
import torch.optim as optim

import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(description='train a generator model')
parser.add_argument('file', help='file name')
parser.add_argument('--classifier', default=None, help='classifier file')
parser.add_argument('--channels', type=int, default=16, help='number of channels')
parser.add_argument('--epoch', type=int, default=500, help='number of epochs')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=None, help='gpu id')


class Trainer(object):

  def __init__(self, generator_f, generator_r, discriminator_f, discriminator_r,
               classifier, gpu):
    self.generator_f = generator_f
    self.generator_r = generator_r
    self.discriminator_f = discriminator_f
    self.discriminator_r = discriminator_r
    self.classifier = classifier
    self.gpu = gpu

    self.optim_discriminator_f = optim.Adam(
      discriminator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))
    self.optim_discriminator_r = optim.Adam(
      discriminator_r.parameters(), lr=0.0002, betas=(0.5, 0.999))
    self.optim_generator_f = optim.Adam(
      generator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))
    self.optim_generator_r = optim.Adam(
      generator_r.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if self.gpu is not None:
      self.discriminator_f.cuda(self.gpu)
      self.discriminator_r.cuda(self.gpu)
      self.generator_f.cuda(self.gpu)
      self.generator_r.cuda(self.gpu)

    if self.gpu is not None and self.classifier is not None:
      self.classifier.cuda(self.gpu)

    self.reset()

  def reset(self):
    self.discriminator_count = 0
    self.discriminator_f_loss = 0
    self.discriminator_r_loss = 0
    self.generator_count = 0
    self.generator_f_loss = 0
    self.generator_r_loss = 0
    self.constraint_loss = 0
    self.classifier_score = 0

  def _update_discriminator(self, x, t):
    self.generator_f.eval()
    self.generator_r.eval()
    self.discriminator_f.train()
    self.discriminator_r.train()

    # forward
    with torch.no_grad():
      f = self.generator_f(x)

    e = -1 * (1 - self.discriminator_f(f)).log().mean() * 0.5
    e += -1 * self.discriminator_f(t).log().mean() * 0.5

    self.optim_discriminator_f.zero_grad()
    e.backward()
    self.optim_discriminator_f.step()
    self.discriminator_f_loss += float(e) * len(x)

    # reverse
    with torch.no_grad():
      s = self.generator_r(t)

    e = -1 * (1 - self.discriminator_r(s)).log().mean() * 0.5
    e += -1 * self.discriminator_r(x).log().mean() * 0.5

    self.optim_discriminator_r.zero_grad()
    e.backward()
    self.optim_discriminator_r.step()
    self.discriminator_r_loss += float(e) * len(x)

  def _update_generator(self, x, t):
    self.generator_f.train()
    self.generator_r.train()
    self.discriminator_f.eval()
    self.discriminator_r.eval()

    # forward
    e = -1 * self.discriminator_f(self.generator_f(x)).log().mean()

    self.optim_generator_f.zero_grad()
    e.backward()
    self.optim_generator_f.step()
    self.generator_f_loss += float(e) * len(x)

    # reverse
    e = -1 * self.discriminator_r(self.generator_r(t)).log().mean()

    self.optim_generator_r.zero_grad()
    e.backward()
    self.optim_generator_r.step()
    self.generator_r_loss += float(e) * len(x)

  def _update_constraint(self, x, t, force_direction):
    self.generator_f.train()
    self.generator_r.train()

    e = (self.generator_r(self.generator_f(x)) - x).abs().mean()
    e += (self.generator_f(self.generator_r(t)) - t).abs().mean()

    self.optim_generator_f.zero_grad()
    self.optim_generator_r.zero_grad()
    (e * 10).backward()
    self.optim_generator_f.step()
    self.optim_generator_r.step()
    self.constraint_loss += float(e) * len(x)

    if force_direction:
      self.generator_f.train()
      self.generator_r.train()

      e = (self.generator_f(t) - t).abs().mean()
      e += (self.generator_r(x) - x).abs().mean()

      self.optim_generator_f.zero_grad()
      self.optim_generator_r.zero_grad()
      (e * 5).backward()
      self.optim_generator_f.step()
      self.optim_generator_r.step()

  def _update_score(self, x, t):
    if self.classifier is None:
      return

    self.generator_f.eval()
    self.classifier.eval()

    with torch.no_grad():
      s = (self.classifier(self.generator_f(x)).max(dim=1)[1] == 1).float().mean()

    self.classifier_score += float(s) * len(x)

  def train(self, noised_loader, raw_loader, force_direction):
    self.reset()

    for x, t in zip(noised_loader, raw_loader):
      if self.gpu is not None:
        x = x.cuda(self.gpu)
        t = t.cuda(self.gpu)

      self._update_discriminator(x, t)
      self.discriminator_count += len(x)

      self._update_generator(x, t)
      self._update_constraint(x, t, force_direction)
      self._update_score(x, t)

      self.generator_count += len(x)

  def __str__(self):
    discirminator_names = [
      ('discriminator_f', 'discriminator_f_loss'),
      ('discriminator_r', 'discriminator_r_loss')]
    generator_names = [
      ('generator_f', 'generator_f_loss'),
      ('generator_r', 'generator_r_loss'),
      ('constraint', 'constraint_loss'),
      ('score', 'classifier_score')]

    values = [(n, getattr(self, k) / self.discriminator_count) for n, k in discirminator_names]
    values += [(n, getattr(self, k) / self.generator_count) for n, k in generator_names]

    return ', '.join('{}={:.4f}'.format(k, v) for k, v in values)


def main():
  args = parser.parse_args()

  # classifier
  if args.classifier is not None:
    snapshot = torch.load(args.classifier, map_location=lambda s, _: s)
    classifier = Classifier(snapshot['channels'])
    classifier.load_state_dict(snapshot['model'])
  else:
    classifier = None

  # dataset
  raw_loader = torch.utils.data.DataLoader(
    Dataset(os.path.join(DATA_DIR, 'raw')),
    batch_size=args.batch, shuffle=True, drop_last=True)
  noised_loader = torch.utils.data.DataLoader(
    Dataset(os.path.join(DATA_DIR, 'noised_tgt')),
    batch_size=args.batch, shuffle=True, drop_last=True)

  # model
  generator_f = Generator(args.channels)
  generator_r = Generator(args.channels)
  discriminator_f = Discriminator(args.channels)
  discriminator_r = Discriminator(args.channels)

  # train
  trainer = Trainer(
    generator_f, generator_r, discriminator_f, discriminator_r, classifier, args.gpu)

  for epoch in range(args.epoch):
    trainer.train(noised_loader, raw_loader, epoch < args.epoch // 10)
    print('[{}] {}'.format(epoch, trainer), flush=True)

    snapshot = {'channels': args.channels, 'model': generator_f.state_dict()}
    torch.save(snapshot, '{}.tmp'.format(args.file))
    os.rename('{}.tmp'.format(args.file), args.file)


if __name__ == '__main__':
  main()

