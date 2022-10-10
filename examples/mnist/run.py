import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.nn.net import SequentialNet
from core.nn.layers import Dense, ReLU
from core.nn.loss import SoftmaxCrossEntropyLoss
from core.nn.optimizer import Adam, SGD
from core.tensor import Tensor
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.evaluator import AccEvaluator
from utils.helper import kernelstat
from env import DEBUG, GRAPH, LAZY, BACKEND

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def prepare_dataset(data_dir):
    url = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

"""
import line_profiler, signal, sys, atexit
profile = line_profiler.LineProfiler()
def handle_exit(*args):
    profile.print_stats()
    sys.exit()
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)
atexit.register(handle_exit)

@profile
"""
def main(args):
    if args.seed >= 0:
        np.random.seed(args.seed)

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)
    train_x = Tensor(train_x)
    train_y = Tensor(train_y)
    test_x = Tensor(test_x).to(args.device)
    test_y = Tensor(test_y)

    net = SequentialNet(
            Dense(256), ReLU(),
            Dense(128), ReLU(),
            Dense(64), ReLU(),
            Dense(32), ReLU(),
            Dense(10)).to(args.device)
    optim = Adam(net.get_parameters(), lr=args.lr)
    #optim = SGD(net.get_parameters(), lr=args.lr)
    loss_fn = SoftmaxCrossEntropyLoss()

    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    for epoch in range(args.num_ep):
        t_start = time.monotonic()
        for batch in iterator(train_x, train_y):
            net.zero_grad()
            x, y = batch.inputs.to(args.device), batch.targets.to(args.device)
            pred = net.forward(x)
            loss = loss_fn(pred, y)
            if args.profile_forward:
                print(loss.array.eager())
                print(kernelstat.info)
                print("total kernel call: ", kernelstat.total())
                sys.exit()
            loss.backward()
            optim.step()
        print("Epoch %d tim cost: %.4f" % (epoch, time.monotonic() - t_start))
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
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--profile_forward", default=0, type=int)
    parser.add_argument("--eval", default=0, type=int)
    default_device = "gpu" if BACKEND in ("opencl", "cuda") else "cpu"
    parser.add_argument("--device", default=default_device, type=str)
    args = parser.parse_args()
    main(args)
