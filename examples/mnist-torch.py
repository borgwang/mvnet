import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.evaluator import AccEvaluator


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

class Dense(nn.Module):

  def __init__(self, hidden_units):
    super(Dense, self).__init__()
    units = [784] + hidden_units
    self.layers = nn.ModuleList()
    for i in range(len(units) - 1):
      self.layers.append(nn.Linear(units[i], units[i+1]))
    self.proj = nn.Linear(units[-1], 10)
    for layer in self.layers:
      torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.xavier_uniform_(self.proj.weight)

  def forward(self, x):
    for layer in self.layers:
      x = F.relu(layer(x))

    x = self.proj(x)
    x = F.log_softmax(x, dim=1)
    return x

def main():
  if args.seed >= 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

  train_set, _, test_set = prepare_dataset(args.data_dir)
  train_x, train_y = train_set
  test_x, test_y = test_set

  hidden_units = [int(i) for i in args.hidden_units.split(",")]
  model = Dense(hidden_units)
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  iterator = BatchIterator(batch_size=args.batch_size)
  evaluator = AccEvaluator()
  for epoch in range(args.num_ep):
    t_start = time.monotonic()
    for batch in iterator(train_x, train_y):
      x = torch.tensor(batch.inputs).to(device)
      y = torch.tensor(batch.targets).to(device)
      optimizer.zero_grad()
      pred = model(x)
      if args.forward_only:
        continue
      loss = F.nll_loss(pred, y)
      loss.backward()
      optimizer.step()
    print(f"Epoch {epoch} time cost: {time.monotonic() - t_start}")
    if args.eval:
      x, y = torch.tensor(test_x).to(device), torch.tensor(test_y).to(device)
      with torch.no_grad():
        pred = model(x)
        test_pred_idx = pred.argmax(dim=1).cpu().numpy()
        test_y_idx = test_y
        print(evaluator.evaluate(test_pred_idx, test_y_idx))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_ep", default=10, type=int)
  parser.add_argument("--data_dir", type=str, default="./examples/mnist/data")
  parser.add_argument("--lr", default=1e-3, type=float)
  parser.add_argument("--batch_size", default=4096, type=int)
  parser.add_argument("--seed", default=31, type=int)
  parser.add_argument("--hidden_units", default="512,256,128,64", type=str)

  parser.add_argument("--forward_only", default=0, type=int)

  parser.add_argument("--eval", default=0, type=int)
  parser.add_argument("--device", default="cuda", type=str)
  args = parser.parse_args()
  device = torch.device(args.device)
  main()
