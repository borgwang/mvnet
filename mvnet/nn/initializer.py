import numpy as np

from mvnet.backend.numpy import NPArray as CPUArray
from mvnet.backend.opencl import CLArray as GPUArray
from mvnet.tensor import Tensor


def get_fans(shape):
  fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
  fan_out = shape[1] if len(shape) == 2 else shape[0]
  return fan_in, fan_out

class Initializer:
  def __call__(self, shape, dtype=np.float32, device="cpu", name=""):
    array = self.init(tuple(shape), dtype=dtype, device=device)
    return Tensor(array, requires_grad=True, dtype=dtype, name=name)

  def init(self, shape, dtype, device):
    raise NotImplementedError

class NormalInit(Initializer):
  def __init__(self, mean=0.0, std=1.0):
    self._mean = mean
    self._std = std

  def init(self, shape, dtype, device):
    if device == "cpu":
      return CPUArray.normal(loc=self._mean, scale=self._std, shape=shape, dtype=dtype)
    if device == "gpu":
      return GPUArray.normal(loc=self._mean, scale=self._std, shape=shape, dtype=dtype)
    raise ValueError(f"Invalid device type {device}")

class UniformInit(Initializer):
  def __init__(self, a=0.0, b=1.0):
    self._a = a
    self._b = b

  def init(self, shape, dtype, device):
    if device == "cpu":
      return CPUArray.uniform(a=self._a, b=self._b, shape=shape, dtype=dtype)
    if device == "gpu":
      return GPUArray.uniform(a=self._a, b=self._b, shape=shape, dtype=dtype)
    raise ValueError(f"Invalid device type {device}")

class ConstantInit(Initializer):
  def __init__(self, val):
    self._val = val

  def init(self, shape, dtype, device):
    if device == "cpu":
      return CPUArray.full(shape, self._val, dtype)
    if device == "gpu":
      return GPUArray.full(shape, self._val, dtype)
    raise ValueError(f"Invalid device type {device}")

class ZerosInit(ConstantInit):
  def __init__(self):
    super().__init__(np.float32(0.0))

class XavierUniformInit(Initializer):
  def __init__(self, gain=1.0):
    self._gain = gain

  def init(self, shape, dtype, device):
    fan_in, fan_out = get_fans(shape)
    a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
    if device == "cpu":
      return CPUArray.uniform(a=-a, b=a, shape=shape, dtype=dtype)
    if device == "gpu":
      return GPUArray.uniform(a=-a, b=a, shape=shape, dtype=dtype)
    raise ValueError(f"Invalid device type {device}")

class XavierNormalInit(Initializer):
  def __init__(self, gain=1.0):
    self._gain = gain

  def init(self, shape, dtype, device):
    fan_in, fan_out = get_fans(shape)
    std = self._gain * np.sqrt(2.0 / (fan_in + fan_out))
    if device == "cpu":
      return CPUArray.normal(loc=0.0, scale=std, shape=shape, dtype=dtype)
    if device == "gpu":
      return GPUArray.normal(loc=0.0, scale=std, shape=shape, dtype=dtype)
    raise ValueError(f"Invalid device type {device}")
