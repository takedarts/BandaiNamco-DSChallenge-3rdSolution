#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import Generator

import torch
import numpy
import argparse
import os
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')
MAX_VALUE = 413.5

parser = argparse.ArgumentParser(description='make a submission file')
parser.add_argument('file', help='file name')
parser.add_argument('--gpu', type=int, default=None, help='gpu id')


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

  print(' - mean={:.2f}->{:.2f}, max={:.2f}->{:.2f}, min={:.2f}->{:.2f}'.format(
    numpy.mean(s), numpy.mean(z),
    numpy.max(s), numpy.max(z),
    numpy.min(s), numpy.min(z)), flush=True)


def main():
  args = parser.parse_args()

  src_dir = os.path.join(DATA_DIR, 'noised_tgt')
  dst_dir = os.path.join(DATA_DIR, 'tmp')

  # clear
  for name in os.listdir(dst_dir):
    if name.endswith('.npy'):
      os.remove(os.path.join(dst_dir, name))

  # model
  snapshot = torch.load(args.file, map_location=lambda s, _: s)
  model = Generator(snapshot['channels'])
  model.load_state_dict(snapshot['model'])

  if args.gpu is not None:
    model.cuda(args.gpu)

  # generate
  for name in sorted(os.listdir(os.path.join(src_dir))):
    if not name.endswith('.npy'):
      continue

    src = os.path.join(src_dir, name)
    dst = os.path.join(dst_dir, name[7:])
    generate(model, src, dst, args.gpu)

  # archive
  with zipfile.ZipFile('submission.zip', 'w') as zip_writer:
    for name in sorted(os.listdir(dst_dir)):
      zip_writer.write(os.path.join(dst_dir, name), name)


if __name__ == '__main__':
  main()
