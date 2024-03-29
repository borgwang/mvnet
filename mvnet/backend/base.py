from enum import Enum

from mvnet.dtype import float32

ElemwiseOps = Enum("ElemwiseOps", ["NEG", "EXP", "LOG", "ADD", "SUB", "DIV", "MUL", "POW", "EQ", "GE", "GT", "NOOP", "RELU", "DRELU"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
ProcessingOps = Enum("ProcessingOps", ["MATMUL", "CONV"])
ViewOps = Enum("ViewOps", ["SLICE", "RESHAPE", "PERMUTE", "EXPAND"])
CreationOps = Enum("CreationOps", ["EMPTY", "FULL", "UNIFORM", "NORMAL"])

class Array:
  for op in ("add", "sub", "mul", "div", "pow", "matmul"):
    magic = "truediv" if op == "div" else op
    exec(f"def __{magic}__(self, other): return self.{op}(self.asarray(other))")
    exec(f"def __i{magic}__(self, other): return self.{op}(self.asarray(other), out=self)")
    exec(f"def __r{magic}__(self, other): return self.asarray(other).{op}(self)")
  for op in ("eq", "ge", "gt"):
    exec(f"def __{op}__(self, other): return self.{op}(self.asarray(other))")
  exec("def __neg__(self): return self.neg()")

  def __init__(self, shape=None, dtype=float32, op_info=None, is_lazy=False):
    self.shape, self.dtype = shape, dtype
    self.op_info = op_info
    self.is_lazy = is_lazy
    self.constant_value = None
    self.strides = None

  def __repr__(self):
    clsname = self.__class__.__name__
    if self.is_lazy: clsname = "Lazy" + clsname
    return (f"<{clsname} dtype={self.dtype} shape={self.shape} strides={self.strides}>")

  @property
  def ndim(self):
    return len(self.shape)

  @classmethod
  def asarray(cls, obj):
    if not isinstance(obj, cls):
      obj = cls(obj.numpy()) if issubclass(obj.__class__, Array) else cls(obj)
    return obj

  def numpy(self):
    raise NotImplementedError

  # ##### Elemwise Ops #####
  for op in ("neg", "exp", "log"):
    exec(f"def {op}(self, out=None): raise NotImplementedError")
  for op in ("add", "sub", "div", "mul", "pow", "eq", "ge", "gt"):
    exec(f"def {op}(self, other, out=None): raise NotImplementedError")

  # ##### Reduce Ops #####
  def sum(self, axis=None, keepdims=False): raise NotImplementedError
  def max(self, axis=None, keepdims=False): raise NotImplementedError

  # ##### View Ops #####
  def reshape(self, shape): raise NotImplementedError
  def expand(self, shape): raise NotImplementedError
  def squeeze(self, axis=None): raise NotImplementedError
  def permute(self, axes): raise NotImplementedError

  @property
  def T(self):
    return self.permute(axes=tuple(range(self.ndim)[::-1]))

  # ##### Slice Ops #####
  def __getitem__(self, key): raise NotImplementedError
  def __setitem__(self, key, value): raise NotImplementedError

  # #### Creation Ops #####
  @classmethod
  def uniform(cls, a, b, shape, dtype=float32): raise NotImplementedError
  @classmethod
  def normal(cls, loc, scale, shape, dtype=float32): raise NotImplementedError
  @classmethod
  def empty(cls, shape, dtype=float32): raise NotImplementedError
  @classmethod
  def full(cls, shape, value, dtype=float32): raise NotImplementedError
