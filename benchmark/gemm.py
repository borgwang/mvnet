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
  #return np.random.normal(0, 1, shape).astype(np.float32)
  return np.random.randint(0, 10, shape).astype(np.float32)

def check_array(myarr, nparr, atol=0, rtol=1e-3):
  assert myarr.shape == nparr.shape, f"shape {myarr.shape} != {nparr.shape}"
  assert myarr.dtype == nparr.dtype, f"dtype {myarr.dtype} != {nparr.dtype}"
  a, b = myarr.numpy(), nparr
  #print("result"); print(a.astype(int))
  #print("ground truth"); print(b.astype(int))
  #print((a-b).astype(int))
  assert np.allclose(myarr.numpy(), nparr, atol=atol, rtol=rtol)

def benchmark_opencl():
  a, b = 4096, 4096
  #a = b = int(2**4)
  np_arr1, np_arr2 = rnd((a, b)), rnd((b, a))
  cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
  #mv_tensor1, mv_tensor2 = Tensor(np_arr1), Tensor(np_arr2)

  with Timer(f"numpy {np_arr1.shape} {np_arr2.shape}"):
    np_res = np_arr1 @ np_arr2
  with Timer(f"opencl {cl_arr1.shape} {cl_arr2.shape}"):
    cl_res = cl_arr1 @ cl_arr2

  #print("nparr1")
  #print(np_arr1.astype(int))
  #print("nparr2")
  #print(np_arr2.astype(int))
  check_array(cl_res, np_res, atol=1e-3)
  #with Timer(f"mvnet tensor {shape}"):
  #  mv_res = mv_tensor1 @ mv_tensor2

benchmark_opencl()
