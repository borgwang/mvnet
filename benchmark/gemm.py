import datetime
import time

import numpy as np

from mvnet.backend.opencl import CLArray
from mvnet.tensor import Tensor

#import torch


np.random.seed(0)

class Timer:
  def __init__(self, name:str):
    self.name = name
    self.__seconds = 0.0

  @property
  def seconds(self) -> float:
    return self.__seconds

  def __enter__(self):
    self.st = time.monotonic()
    return self

  def __exit__(self, *args, **kwargs):
    self.__seconds += time.monotonic() - self.st
    print(f"`{self.name}` time_cost={datetime.timedelta(seconds=self.__seconds)}")

  def reset(self) -> None:
    self.__seconds = 0.0

def rnd(shape):
  return np.random.normal(0, 1, shape).astype(np.float32)

def check_array(myarr, nparr, atol=0, rtol=1e-3):
  assert myarr.shape == nparr.shape, f"shape {myarr.shape} != {nparr.shape}"
  assert myarr.dtype == nparr.dtype, f"dtype {myarr.dtype} != {nparr.dtype}"
  assert np.allclose(myarr.numpy(), nparr, atol=atol, rtol=rtol)

def benchmark_opencl():
  a, b = 4096, 8192
  np_arr1, np_arr2 = rnd((a, b)), rnd((b, a))
  cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
  #mv_tensor1, mv_tensor2 = Tensor(np_arr1), Tensor(np_arr2)

  with Timer(f"numpy [{a},{b}] [{b}, {a}]"):
    np_res = np_arr1 @ np_arr2
  with Timer(f"opencl [{a},{b}] [{b}, {a}]"):
    cl_res = cl_arr1 @ cl_arr2
  #with Timer(f"mvnet tensor {shape}"):
  #  mv_res = mv_tensor1 @ mv_tensor2

benchmark_opencl()
