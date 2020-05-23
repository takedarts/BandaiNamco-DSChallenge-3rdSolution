#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import Generator
from utils import generate

import torch
import argparse
import os
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(description='make a submission file')
parser.add_argument('file', help='file name')
parser.add_argument('--gpu', type=int, default=None, help='gpu id')


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
