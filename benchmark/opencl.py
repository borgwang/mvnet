#!/usr/bin/env python3
import numpy as np
import torch
from utils import Timer

from mvnet.backend.opencl import CLArray, cl
from mvnet.env import BACKEND, LAZY

FS = {
  "UNDERLINE": "\033[4m",
  "BLUE": "\033[94m",
  "HIGHLIGHT": "\x1b[6;30;42m",  # highlight green
  "HIGHLIGHT_RED": "\x1b[6;30;41m",  # highlight red
  "DARKGREY": "\033[90m",
  "LIGHTGREY": "\033[37m",
  "RED": "\033[91m",
  "YELLOW": "\033[93m",
  "CYAN": "\033[96m",
  "GREEN": "\033[92m",
  "BOLD": "\033[1m",
  "WHITE": "\033[1;37m",
  "STRIKETHROUGH": "\033[9m",
  "ENDC": "\033[0m"
}

def rnd(shape):
  #return np.random.randint(0, 10, shape).astype(np.float32)
  return np.random.uniform(0, 1, shape).astype(np.float32)

# GEMM
# Tesla T4 vGPU(1/2) has 4TFLOPs for FP23
assert BACKEND == "opencl"
assert LAZY == 0

"""
import speedscope
with speedscope.track("/tmp/speedscope.json"):
  for size in (2**11,):
    M = K = N = size
    np_arr1, np_arr2 = rnd((M, K)), rnd((K, N))
    cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
    for _ in range(10):
      cl_arr1 @ cl_arr2
    cl.queue.finish()
"""

def benchmark_matmul():
  for e in range(10, 13):
    M = K = N = 2**e
    np_arr1, np_arr2 = rnd((M, K)), rnd((K, N))
    cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
    torch_arr1, torch_arr2 = torch.from_numpy(np_arr1.copy()).cuda(), torch.from_numpy(np_arr2.copy()).cuda()

    n_warmup, n_measure = 5, 20
    flops = M*N*K*2*n_measure

    cl_fn = lambda: cl_arr1 @ cl_arr2
    #cl_fn = lambda: (cl_arr1.reshape((N,1,N)) * cl_arr2.permute((1,0)).reshape((1,N,N))).sum(axis=2)
    torch_fn = lambda: torch_arr1 @ torch_arr2

    for _ in range(n_warmup): cl_fn()
    cl.queue.finish()
    with Timer("cl") as t1:
      for _ in range(n_measure): cl_fn()
      cl.queue.finish()

    for _ in range(n_warmup): torch_fn()
    torch.cuda.synchronize()
    with Timer("torch") as t2:
      for _ in range(n_measure): torch_fn()
      torch.cuda.synchronize()

    factor = t2.ms/t1.ms
    color = FS["GREEN"] if t1.ms < t2.ms else FS["RED"]
    print(f"MatmulOp({M}x{K})\t"
          f"cl_backend: {t1.ms:.2f}ms, {1e-9*flops/t1.ms:.2f}TFLOPS\t"
          f"pytorch: {t2.ms:.2f}ms, {1e-9*flops/t2.ms:.2f}TFLOPS\t"
          f"factor: {color}{factor:.3f}{FS['ENDC']}")
  print("-"*50)

def benchmark_reduce():
  for N in (2**10, 2**11, 2**12):
    np_arr = rnd((N, N))
    cl_arr = CLArray(np_arr)
    torch_arr = torch.from_numpy(np_arr.copy()).cuda()

    n_warmup, n_measure = 10, 500
    flops = N*N*n_measure
    axis = None

    for _ in range(n_warmup):
      cl_arr.sum(axis=axis)
    cl.queue.finish()
    with Timer("cl") as t1:
      for _ in range(n_measure):
        cl_arr.sum(axis=axis)
      cl.queue.finish()

    for _ in range(n_warmup):
      torch_arr.sum(axis=axis)
    torch.cuda.synchronize()
    with Timer("torch") as t2:
      for _ in range(n_measure):
        torch_arr.sum(axis=axis)
      torch.cuda.synchronize()

    factor = t2.ms/t1.ms
    color = FS["GREEN"] if t1.ms < t2.ms else FS["RED"]
    print(f"ReduceOp({N}x{N})\t"
          f"cl_backend: {t1.ms:.2f}ms, {1e-6*flops/t1.ms:.2f}GFLOPS\t"
          f"pytorch: {t2.ms:.2f}ms, {1e-6*flops/t2.ms:.2f}GFLOPS\t"
          f"factor: {color}{factor:.3f}{FS['ENDC']}")
  print("-"*50)

def benchmark_elemwise():
  for e in range(5, 13):
    N = 2**e
    np_arr1, np_arr2 = rnd((N, N)), rnd((N, N))
    cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
    torch_arr1, torch_arr2 = torch.from_numpy(np_arr1.copy()).cuda(), torch.from_numpy(np_arr2.copy()).cuda()

    n_warmup, n_measure = 50, 500
    flops = N*N*n_measure

    cl_fn = lambda: cl_arr1 + cl_arr2
    torch_fn = lambda: torch_arr1 + torch_arr2

    for _ in range(n_warmup): cl_fn()
    cl.queue.finish()
    with Timer("cl") as t1:
      for _ in range(n_measure): cl_fn()
      cl.queue.finish()

    for _ in range(n_warmup): torch_fn()
    torch.cuda.synchronize()
    with Timer("torch") as t2:
      for _ in range(n_measure): torch_fn()
      torch.cuda.synchronize()

    factor = t2.ms/t1.ms
    color = FS["GREEN"] if t1.ms < t2.ms else FS["RED"]
    print(f"ElemwiseOp({N}x{N})\t"
          f"cl_backend: {t1.ms:.2f}ms, {1e-6*flops/t1.ms:.2f}GFLOPS\t"
          f"pytorch: {t2.ms:.2f}ms, {1e-6*flops/t2.ms:.2f}GFLOPS\t"
          f"factor: {color}{factor:.3f}{FS['ENDC']}")
  print("-"*50)


if __name__ == "__main__":
  benchmark_matmul()
  #benchmark_reduce()
  #benchmark_elemwise()
