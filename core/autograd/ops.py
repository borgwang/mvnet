import numpy as np

from env import GRAPH
from utils.helper import timer, genname
from utils.math import argsort

def unbroadcast(func, shape):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret.shape == shape:
            return ret
        ndim = len(shape)
        for i in range(ret.ndim - ndim):
            ret = ret.sum(axis=0)
        for i in range(ndim):
            if shape[i] == 1:
                ret = ret.sum(axis=i, keepdims=True)
        return ret
    return wrapper

def autograd_ops(func):
    def wrapper(*args, **kwargs):
        from core.tensor import Tensor
        tss = [a for a in args if isinstance(a, Tensor)]
        arr, *grad_fns = func(*[ts.array for ts in tss], *args[len(tss):], **kwargs)
        grad_fns = [unbroadcast(grad_fn, ts.shape) for ts, grad_fn in zip(tss, grad_fns)]
        requires_grad = False
        for ts, grad_fn in zip(tss, grad_fns):
            requires_grad = requires_grad or (ts.requires_grad and grad_fn)
        dependency = []
        for i, (ts, grad_fn) in enumerate(zip(tss, grad_fns)):
            if ts.requires_grad and grad_fn:
                if GRAPH: grad_fn = timer(grad_fn)
                grad_fn.__name__ = f"grad_fn_{i+1} for {func.__name__}"
                dependency.append(dict(tensor=ts, grad_fn=grad_fn))
                ts.outdegree += 1
        name = genname(func.__name__, *tss)
        return Tensor(arr, requires_grad, dependency, name=name)
    return wrapper

@autograd_ops
def add(arr1, arr2):
    grad_fn = lambda g: g
    return arr1 + arr2, grad_fn, grad_fn

@autograd_ops
def sub(arr1, arr2):
    grad_fn1 = lambda g: g
    grad_fn2 = lambda g: -g
    return arr1 - arr2, grad_fn1, grad_fn2

@autograd_ops
def mul(arr1, arr2):
    grad_fn1 = lambda g: arr2 * g
    grad_fn2 = lambda g: arr1 * g
    return arr1 * arr2, grad_fn1, grad_fn2

@autograd_ops
def div(arr1, arr2):
    result = arr1 / arr2
    grad_fn1 = lambda g: g / arr2
    grad_fn2 = lambda g: -g * result / arr2
    return result, grad_fn1, grad_fn2

@autograd_ops
def pow(arr1, arr2):
    result = arr1 ** arr2
    grad_fn1 = lambda g: g * (arr2 * arr1**(arr2 - 1.0))
    grad_fn2 = lambda g: g * (result * arr1.log())
    return result, grad_fn1, grad_fn2

@autograd_ops
def matmul(arr1, arr2):
    grad_fn1 = lambda g: g @ arr2.T
    grad_fn2 = lambda g: arr1.T @ g
    return arr1 @ arr2, grad_fn1, grad_fn2

@autograd_ops
def gt(arr1, arr2):
    return arr1 > arr2, None, None

@autograd_ops
def eq(arr1, arr2):
    return arr1 == arr2, None, None

@autograd_ops
def ge(arr1, arr2):
    return arr1 >= arr2, None, None

@autograd_ops
def sum(arr, axis=None, keepdims=False):
    result = arr.sum(axis=axis, keepdims=keepdims)
    def grad_fn(g):
        shape = arr.shape
        if axis is None:
            assert not keepdims, "keepdims must be False when axis is None"
            return g.reshape([1] * arr.ndim).expand(shape)
        if not keepdims:
            g = g.reshape((*shape[:axis], 1, *shape[axis+1:]))
        return g.expand(shape)
    return result, grad_fn

@autograd_ops
def max(arr, axis=None, keepdims=False):
    result = arr.max(axis=axis, keepdims=keepdims)
    grad_fn = lambda g: g * (result == arr)
    return result, grad_fn

@autograd_ops
def min(arr, axis, keepdims):
    result = arr.min(axis=axis, keepdims=keepdims)
    grad_fn = lambda g: g * (result == arr)
    return result, grad_fn

@autograd_ops
def neg(arr):
    grad_fn = lambda g: -g
    return -arr, grad_fn

@autograd_ops
def exp(arr):
    result = arr.exp()
    grad_fn = lambda g: g * result
    return result, grad_fn

@autograd_ops
def log(arr):
    result = arr.log()
    grad_fn = lambda g: g / arr
    return result, grad_fn

@autograd_ops
def relu(arr):
    mask = arr > 0
    result = mask * arr
    grad_fn = lambda g: mask * g
    return mask * arr, grad_fn

@autograd_ops
def reshape(arr, shape):
    result = arr.reshape(shape)
    grad_fn = lambda g: g.reshape(arr.shape)
    return result, grad_fn

@autograd_ops
def permute(arr, axes=None):
    if axes is None:
        axes = range(arr.ndim)[::-1]
    axes = list(axes)
    result = arr.permute(axes)
    grad_fn = lambda g: g.permute(argsort(axes))
    return result, grad_fn

@autograd_ops
def getitem(arr, key):
    result = arr[key]
    def grad_fn(g):
        ret = g.__class__.zeros(arr.shape)
        ret[key] = g  # TODO
        return ret
    return result, grad_fn

