import numpy as np
import core.autograd.ops as ops
from env import GRAPH, BACKEND, LAZY
from core.dtype import float32

from core.backend.numpy import NpArray as CPUArray
GPUArray = type(None)
if BACKEND == "opencl":
    from core.backend.opencl import ClArray as GPUArray
elif BACKEND == "cuda":
    from core.backend.cuda import CuArray as GPUArray

class Tensor:
    def __init__(self, array, requires_grad=False, dependency=(), dtype=float32, name=None):
        self._gpu = isinstance(array, GPUArray)
        self.array = array if isinstance(array, (CPUArray, GPUArray)) else CPUArray(array, dtype=dtype)
        self.dtype = dtype

        self.grad = None
        self.requires_grad = requires_grad
        self.dependency = dependency

        self.name = name
        self.array.name = name
        self.outdegree = 0
        self.bwdcost = 0

    def astensor(self, obj):
        if not isinstance(obj, self.__class__):
            if not isinstance(obj, self.array.__class__):
                obj = self.array.__class__(obj, dtype=self.dtype)
            obj = Tensor(obj)
        return obj

    def to(self, device):
        assert device in ("cpu", "gpu"), f"Device {device} not support yet."
        return getattr(self, device)()

    def gpu(self):
        assert GPUArray != type(None), f"backend {BACKEND} not support gpu device"
        if not self._gpu:
            return Tensor(array=GPUArray(self.array.numpy()),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def cpu(self):
        if self._gpu:
            return Tensor(array=CPUArray(self.array.numpy()),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def numpy(self):
        return self.array.numpy()

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self):
        return (f"Tensor(name={self.name}, shape={self.shape}, "
                f"requires_grad={self.requires_grad}, "
                f"gpu={self._gpu}, array={self.array.__class__.__name__})")

    for op in ("add", "sub", "mul", "div", "pow", "matmul"):
        op_ = "truediv" if op == "div" else op
        exec(f"def __{op_}__(self, other): return ops.{op}(self, self.astensor(other))")
        exec(f"def __i{op_}__(self, other): self.array = self.array.__i{op_}__(self.astensor(other).array); return self")
        exec(f"def __r{op_}__(self, other): return ops.{op}(self.astensor(other), self)")

    for op in ("eq", "ge", "gt"):
        exec(f"def __{op}__(self, other): return ops.{op}(self, self.astensor(other))")

    exec(f"def __neg__(self): return ops.neg(self)")

    for op in ("sum", "max", "log", "exp", "relu", "reshape", "flatten", "permute"):
        exec(f"def {op}(self, *args, **kwargs): return ops.{op}(self, *args, **kwargs)")

    @property
    def T(self):
        return ops.permute(self, axes=None)

    def __getitem__(self, key):
        return ops.getitem(self, key)

    def __len__(self):
        assert self.shape, "Error getting length of a 0-d tensor"
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        self.outdegree -= 1
        if grad is None:
            grad = GPUArray([1.0]) if self._gpu else CPUArray([1.0])
            self.outdegree = 0
        if self._gpu and not isinstance(grad, GPUArray):
            grad = GPUArray(grad, dtype=self.dtype)
        if not self._gpu and not isinstance(grad, CPUArray):
            grad = CPUArray(grad, dtype=self.dtype)

        if self.requires_grad:
            self.grad = self.grad + grad if self.grad is not None else grad

        if self.outdegree <= 0:
            for dep in self.dependency:
                grad_for_dep = dep["grad_fn"](self.grad)
                if GRAPH:
                    grad_for_dep, cost = grad_for_dep
                    self.bwdcost += cost
                dep["tensor"].backward(grad_for_dep)

    def zero_grad(self):
        self.grad = None

