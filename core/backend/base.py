from core.dtype import float32

from enum import Enum

ElemwiseOps = Enum("ElemwiseOps",
    ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "DIV", "MUL", "POW", "EQ", "GE", "GT", "DRELU", "NOOP"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
ProcessingOps = Enum("ProcessingOps", ["MATMUL", "CONV"])
ViewOps = Enum("ViewOps", ["SLICE", "RESHAPE", "PERMUTE", "EXPAND", "SQUEEZE"])
CreationOps = Enum("CreationOps", ["EMPTY", "FULL", "UNIFORM", "NORMAL"])

class Array:
    for op in ("add", "sub", "mul", "div", "pow", "matmul"):
        op_ = "truediv" if op is "div" else op
        exec(f"def __{op_}__(self, other): return getattr(self, '{op}')(self.asarray(other))")
        exec(f"def __i{op_}__(self, other): return getattr(self, '{op}')(self.asarray(other), out=self)")
        exec(f"def __r{op_}__(self, other): return getattr(self.asarray(other), '{op}')(self)")
    for op in ("eq", "ge", "gt"):
        exec(f"def __{op}__(self, other): return getattr(self, '{op}')(self.asarray(other))")
    exec(f"def __neg__(self): return self.neg()")

    def __init__(self, shape=None, dtype=float32, op_info=None):
        self.op_info = op_info
        self.shape, self.dtype = shape, dtype

    def __repr__(self):
        clsname = self.__class__.__name__
        if self.is_lazy:
            clsname = "Lazy" + clsname
        return (f"<{clsname} dtype={self.dtype} shape={self.shape} "
                f"strides={self.strides}>")

    @property
    def size(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    @classmethod
    def asarray(cls, obj):
        from core.backend.opencl import ClArray
        if not isinstance(obj, cls):
            obj = cls(obj.numpy()) if issubclass(obj.__class__, Array) else cls(obj)
        return obj

    def numpy(self):
        raise NotImplementedError

    @staticmethod
    def broadcast(a, b):
        # rule: https://numpy.org/doc/stable/user/basics.broadcasting.html
        if a.shape == b.shape:
            return a, b
        for i, j in zip(a.shape[::-1], b.shape[::-1]):
            if i != j and (i != 1) and (j != 1):
                raise ValueError(f"Error broadcasting for {a.shape} and {b.shape}")
        ndim = max(a.ndim, b.ndim)
        if a.ndim != ndim:
            a = a.reshape([1] * (ndim - a.ndim) + list(a.shape))
        if b.ndim != ndim:
            b = b.reshape([1] * (ndim - b.ndim) + list(b.shape))
        broadcast_shape = tuple([max(i, j) for i, j in zip(a.shape, b.shape)])
        if a.shape != broadcast_shape:
            a = a.expand(broadcast_shape)
        if b.shape != broadcast_shape:
            b = b.expand(broadcast_shape)
        return a, b

    """
    def broadcast2(*arrs):
        if len(set([arr.shape for arr in arrs])) == 1:
            return arrs
        reveted_shapes = [arr.shape[::-1] for arr in arrs])
        min_ndim = min([arr.ndim for arr in arrs])
        for i in range(min_ndim):
            unique = [shape[i] for shape in reverted_shapes]
            if set(unique) > 2 or (set(unique) == 2 and 1 not in unique):
                raise ValueError(f"Error broadcasting for {arrs}")
        ndim = min([arr.ndim for arr in arrs])
        retarrs = []
    """

    # ##### Unary Ops #####
    def neg(self): raise NotImplementedError
    def exp(self): raise NotImplementedError
    def log(self): raise NotImplementedError
    def relu(self): raise NotImplementedError

    # ##### Binary Ops #####
    def add(self, other, out=None): raise NotImplementedError
    def sub(self, other, out=None): raise NotImplementedError
    def mul(self, other, out=None): raise NotImplementedError
    def div(self, other, out=None): raise NotImplementedError
    def pow(self, other, out=None): raise NotImplementedError
    def eq(self, other, out=None): raise NotImplementedError
    def gt(self, other): raise NotImplementedError
    def ge(self, other): raise NotImplementedError
    def matmul(self, other): raise NotImplementedError
    def drelu(self, other): raise NotImplementedError

    # ##### Reduce Ops #####
    def sum(self, axis=None, keepdims=False): raise NotImplementedError
    def max(self, axis=None, keepdims=False): raise NotImplementedError

    # ##### Movement Ops #####
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
    @classmethod
    def zeros(cls, shape, dtype=float32): return cls.full(shape, 0, dtype)
    @classmethod
    def ones(cls, shape, dtype=float32): return cls.full(shape, 1, dtype)
