import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.evaluator import AccEvaluator

from mvnet.env import BACKEND, LAZY
from mvnet.nn.layers import Dense, ReLU
from mvnet.nn.loss import SoftmaxCrossEntropyLoss
from mvnet.nn.net import SequentialNet
from mvnet.nn.optimizer import Adam
from mvnet.tensor import Tensor
from mvnet.utils.misc import kernelstat


def get_one_hot(targets, nb_classes):
  return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def prepare_dataset(data_dir):
  url = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
  save_path = os.path.join(data_dir, url.split("/")[-1])
  print("Preparing MNIST dataset ...")
  try:
    download_url(url, save_path)
  except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)
  with gzip.open(save_path, "rb") as f:
    return pickle.load(f, encoding="latin1")

"""
from mvnet.nn.initializer import XavierUniformInit, ZerosInit
layer_params = []
hidden_units = [int(i) for i in args.hidden_units.split(",")]
units = [784] + hidden_units + [10]
for i in range(len(units) - 1):
  layer_params.append({
    "w": XavierUniformInit()(shape=(units[i], units[i+1])).numpy(),
    "b": XavierUniformInit()(shape=(1, units[i+1])).numpy()
  })
def numpy_forward(x):
  for i, params in enumerate(layer_params):
    x = x.dot(params["w"]) + params["b"]
    if i != len(layer_params) - 1:
      x = np.maximum(x, 0)
  return x
"""

def main(args):
  if args.seed >= 0:
    np.random.seed(args.seed)

  train_set, _, test_set = prepare_dataset(args.data_dir)
  train_x, train_y = train_set
  test_x, test_y = test_set
  train_y = get_one_hot(train_y, 10)
  train_x = Tensor(train_x)
  train_y = Tensor(train_y)
  test_x = Tensor(test_x).to(args.device)
  test_y = Tensor(test_y)

  hidden_units = [int(i) for i in args.hidden_units.split(",")]
  units = [784] + hidden_units
  layers = []
  for i in range(len(units) - 1):
    layers.append(Dense(units[i], units[i+1]))
    layers.append(ReLU())
  layers.append(Dense(units[-1], 10))

  net = SequentialNet(*layers).to(args.device)
  optim = Adam(net.get_parameters(), lr=args.lr)
  loss_fn = SoftmaxCrossEntropyLoss()

  iterator = BatchIterator(batch_size=args.batch_size)
  evaluator = AccEvaluator()
  from mvnet.backend.opencl import cl
  for epoch in range(args.num_ep):
    t_start = time.monotonic()
    for batch in iterator(train_x, train_y):
      net.zero_grad()
      x, y = batch.inputs.to(args.device), batch.targets.to(args.device)
      pred = net.forward(x)
      #pred = numpy_forward(x.numpy())
      if args.forward_only:
        continue
      loss = loss_fn(pred, y)
      if args.profile_forward:
        if LAZY: print(loss.numpy())
        print(kernelstat.info)
        print("total kernel call: ", kernelstat.total())
        print(f"opencl info: {cl.info}")
        print(f"time cost: {time.monotonic() - t_start:.4f}")
        sys.exit()
      loss.backward()
      optim.step()
      if args.profile_backward:
        print(kernelstat.info)
        print("total kernel call: ", kernelstat.total())
        print(f"opencl info: {cl.info}")
        print(f"time cost: {time.monotonic() - t_start:.4f}")
        sys.exit()

    print(f"Epoch {epoch} time cost: {time.monotonic() - t_start:.4f}")
    print(f"opencl info: {cl.info}")
    if args.eval:
      test_pred = net.forward(test_x).numpy()
      test_pred_idx = np.argmax(test_pred, axis=1)
      test_y_idx = test_y.numpy()
      print(evaluator.evaluate(test_pred_idx, test_y_idx))

  print(kernelstat.info)
  print("total kernel call: ", kernelstat.total())

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_ep", default=10, type=int)
  parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
  parser.add_argument("--lr", default=1e-3, type=float)
  parser.add_argument("--batch_size", default=4096, type=int)
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--hidden_units", default="512,256,128,64", type=str)

  parser.add_argument("--forward_only", default=0, type=int)
  parser.add_argument("--profile_forward", default=0, type=int)
  parser.add_argument("--profile_backward", default=0, type=int)
  parser.add_argument("--eval", default=0, type=int)
  default_device = "gpu" if BACKEND in ("opencl", "cuda") else "cpu"
  parser.add_argument("--device", default=default_device, type=str)
  args = parser.parse_args()
  main(args)
