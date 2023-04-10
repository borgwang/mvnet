from mvnet.utils.helper import genname
from mvnet.utils.math import argsort


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
    from mvnet.tensor import Tensor
    tss = [a for a in args if isinstance(a, Tensor)]
    arr, *grad_fns = func(*[ts.array for ts in tss], *args[len(tss):], **kwargs)
    grad_fns = [unbroadcast(grad_fn, ts.shape) for ts, grad_fn in zip(tss, grad_fns)]
    requires_grad = False
    for ts, grad_fn in zip(tss, grad_fns):
      requires_grad = requires_grad or (ts.requires_grad and grad_fn)
    dependency = []
    for i, (ts, grad_fn) in enumerate(zip(tss, grad_fns)):
      if ts.requires_grad and grad_fn:
        ts.degree += 1
        grad_fn.__name__ = f"grad_fn_{i+1} for {func.__name__}"
        dependency.append({"tensor": ts, "grad_fn": grad_fn})
    return Tensor(arr, requires_grad, dependency, name=genname(func.__name__, *tss))
  return wrapper

class Ops:

  @staticmethod
  @autograd_ops
  def add(arr1, arr2):
    return arr1 + arr2, lambda g: g, lambda g: g

  @staticmethod
  @autograd_ops
  def sub(arr1, arr2):
    return arr1 - arr2, lambda g: g, lambda g: -g

  @staticmethod
  @autograd_ops
  def mul(arr1, arr2):
    return arr1 * arr2, lambda g: arr2 * g, lambda g: arr1 * g

  @staticmethod
  @autograd_ops
  def div(arr1, arr2):
    result = arr1 / arr2
    return result, lambda g: g / arr2, lambda g: -g * result / arr2

  @staticmethod
  @autograd_ops
  def pow(arr1, arr2):
    result = arr1 ** arr2
    return result, lambda g: g * (arr2 * arr1**(arr2 - 1.0)), lambda g: g * (result * arr1.log())

  @staticmethod
  @autograd_ops
  def matmul(arr1, arr2):
    return arr1 @ arr2, lambda g: g @ arr2.T, lambda g: arr1.T @ g

  @staticmethod
  @autograd_ops
  def gt(arr1, arr2):
    return arr1 > arr2, None, None

  @staticmethod
  @autograd_ops
  def eq(arr1, arr2):
    return arr1 == arr2, None, None

  @staticmethod
  @autograd_ops
  def ge(arr1, arr2):
    return arr1 >= arr2, None, None

  @staticmethod
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

  @staticmethod
  @autograd_ops
  def max(arr, axis=None, keepdims=False):
    result = arr.max(axis=axis, keepdims=keepdims)
    return result, lambda g: g * (result == arr)

  @staticmethod
  @autograd_ops
  def min(arr, axis, keepdims):
    result = arr.min(axis=axis, keepdims=keepdims)
    return result, lambda g: g * (result == arr)

  @staticmethod
  @autograd_ops
  def neg(arr):
    return -arr, lambda g: -g

  @staticmethod
  @autograd_ops
  def exp(arr):
    result = arr.exp()
    return result, lambda  g: g * result

  @staticmethod
  @autograd_ops
  def log(arr):
    return arr.log(), lambda g: g / arr

  #@staticmethod
  #@autograd_ops
  #def relu(arr):
  #  mask = arr > 0
  #  return mask * arr, lambda g: mask * g

  @staticmethod
  @autograd_ops
  def relu(arr):
    return arr.relu(), lambda g: g.drelu(arr)

  @staticmethod
  @autograd_ops
  def expand(arr, shape):
    # TODO: test it
    expanded_axes = [i for i, (s1, s2) in enumerate(zip(arr.shape, shape)) if s1 == 1 and s2 > 1]
    return arr.expand(shape), lambda g: g.squeeze(expanded_axes)

  @staticmethod
  @autograd_ops
  def squeeze(arr, axis):
    # TODO: implement it
    pass

  @staticmethod
  @autograd_ops
  def reshape(arr, shape):
    return arr.reshape(shape), lambda g: g.reshape(arr.shape)

  @staticmethod
  @autograd_ops
  def permute(arr, axes=None):
    if axes is None:
      axes = range(arr.ndim)[::-1]
    axes = list(axes)
    result = arr.permute(axes)
    return result, lambda g: g.permute(argsort(axes))

  @staticmethod
  @autograd_ops
  def getitem(arr, key):
    def grad_fn(g):
      ret = g.__class__.zeros(arr.shape)
      ret[key] = g  # TODO
      return ret
    return arr[key], grad_fn
