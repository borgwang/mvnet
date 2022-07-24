from core.dtype import float32

from enum import Enum

ElemwiseOps = Enum("ElemwiseOps",
    ["NEG", "EXP", "LOG", "ADD", "SUB", "DIV", "MUL", "POW", "EQ", "GE", "GT" , "NOOP", "RELU", "DRELU"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
ProcessingOps = Enum("ProcessingOps", ["MATMUL", "CONV"])
ViewOps = Enum("ViewOps", ["SLICE", "RESHAPE", "PERMUTE", "EXPAND"])
CreationOps = Enum("CreationOps", ["EMPTY", "FULL", "UNIFORM", "NORMAL"])

class Array:
    for op in ("add", "sub", "mul", "div", "pow", "matmul"):
        op_ = "truediv" if op == "div" else op
        exec(f"def __{op_}__(self, other): return self.{op}(self.asarray(other))")
        exec(f"def __i{op_}__(self, other): return self.{op}(self.asarray(other), out=self)")
        exec(f"def __r{op_}__(self, other): return self.asarray(other).{op}(self)")
    for op in ("eq", "ge", "gt"):
        exec(f"def __{op}__(self, other): return self.{op}(self.asarray(other))")
    exec(f"def __neg__(self): return self.neg()")

    def __init__(self, shape=None, dtype=float32, op_info=None, is_lazy=False):
        self.shape, self.dtype = shape, dtype
        self.op_info = op_info
        self.is_lazy = is_lazy

    def __repr__(self):
        clsname = self.__class__.__name__
        if self.is_lazy: clsname = "Lazy" + clsname
        return (f"<{clsname} dtype={self.dtype} shape={self.shape} strides={self.strides}>")

    @property
    def size(self):
        raise NotImplementedError

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

    @staticmethod
    def broadcast(*arrs):
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        if len(set([arr.shape for arr in arrs])) == 1:
            return arrs
        reverted_shapes = [arr.shape[::-1] for arr in arrs]
        min_ndim = min([arr.ndim for arr in arrs])
        for i in range(min_ndim):
            unique = set([shape[i] for shape in reverted_shapes])
            if len(unique) > 2 or (len(unique) == 2 and 1 not in unique):
                raise ValueError(f"Error broadcasting for {arrs}")
        ndim = max([arr.ndim for arr in arrs])
        arrs = [a.reshape([1] * (ndim - a.ndim) + list(a.shape)) if a.ndim != ndim else a for a in arrs]
        broadcast_shape = tuple([max(*s) for s in zip(*[a.shape for a in arrs])])
        arrs = [a.expand(broadcast_shape) if a.shape != broadcast_shape else a for a in arrs]
        return arrs

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
