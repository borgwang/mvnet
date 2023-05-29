import datetime
import time

import numpy as np
import torch

from mvnet.backend.opencl import CLArray
from mvnet.env import DEBUG
from mvnet.tensor import Tensor

np.random.seed(31)

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
  return np.random.randint(0, 10, shape).astype(np.float32)

def check_array(myarr, nparr, atol=0, rtol=1e-3):
  assert myarr.shape == nparr.shape, f"shape {myarr.shape} != {nparr.shape}"
  assert myarr.dtype == nparr.dtype, f"dtype {myarr.dtype} != {nparr.dtype}"
  a, b = myarr.numpy(), nparr
  if DEBUG:
    print("result"); print(a.astype(int))
    print("ground truth"); print(b.astype(int))
    #print((a-b).astype(int))
  assert np.allclose(myarr.numpy(), nparr, atol=atol, rtol=rtol), "check failed!!"
  print("check pass!!")

def benchmark_opencl():
  BS = 1
  #m=k=n=5120
  #m=k=n=512
  m=k=n=512
  np_arr1, np_arr2 = rnd((BS, m, k)), rnd((BS, k, n))
  cl_arr1, cl_arr2 = CLArray(np_arr1), CLArray(np_arr2)
  #torch_arr1, torch_arr2 = torch.from_numpy(np_arr1.copy()).cuda(), torch.from_numpy(np_arr2.copy()).cuda()

  with Timer(f"numpy {np_arr1.shape} {np_arr2.shape}"):
    np_res = np_arr1 @ np_arr2
  with Timer(f"opencl {cl_arr1.shape} {cl_arr2.shape}"):
    cl_res = cl_arr1 @ cl_arr2
  #with Timer(f"pytorch {cl_arr1.shape} {cl_arr2.shape}"):
  #  torch_res = torch_arr1 @ torch_arr2
  if DEBUG:
    print("A"); print(np_arr1.astype(int))
    print("B"); print(np_arr2.astype(int))
    print("B.T"); print(np_arr2.transpose(0, 2, 1).astype(int))
  check_array(cl_res, np_res, atol=1e-3)

benchmark_opencl()
