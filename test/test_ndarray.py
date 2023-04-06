import runtime_path  # isort:skip

import numpy as np

from mvnet.backend.opencl import CLArray
from mvnet.env import DEBUG

np.random.seed(0)

rnd = lambda shape: np.random.normal(0, 1, shape).astype(np.float32)

def check_array(myarr, nparr, atol=0, rtol=1e-3, ignore=()):
  assert myarr.shape == nparr.shape
  assert myarr.dtype == nparr.dtype
  assert np.allclose(myarr.numpy(), nparr, atol=atol, rtol=rtol)
  if "stride" not in ignore:
    np_strides = tuple(s // myarr.dtype().itemsize for s in nparr.strides)
    assert myarr.strides == np_strides
  if "contig" not in ignore:
    assert myarr.c_contiguous == nparr.flags.c_contiguous
    assert myarr.f_contiguous == nparr.flags.f_contiguous

def test_reshape():
  shape = (2, 3, 4)
  nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  arr = CLArray(nparr)
  check_array(arr, nparr)
  #for s in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
  #    check_array(arr.reshape(s), nparr.reshape(s))
  #for s in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
  #    check_array(arr.T.reshape(s), nparr.T.reshape(s, order="A"))
  for s in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
    check_array(arr.permute((0, 2, 1)).reshape(s), nparr.transpose((0, 2, 1)).reshape(s, order="A"))

def test_contiguous():
  shape = (2, 3, 4)
  nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  arr = CLArray(nparr)
  check_array(arr, nparr)

  arr = arr.permute((0, 2, 1))
  nparr = nparr.transpose((0, 2, 1))
  check_array(arr, nparr)

  arr = arr.contiguous()
  nparr = np.ascontiguousarray(nparr)
  check_array(arr, nparr)

def test_expand():
  shape = (3, 1, 1)
  nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  arr = CLArray(nparr)

  arr_expand = arr.expand((3, 3, 1))
  nparr_expand = np.tile(nparr, (1, 3, 1))
  assert np.allclose(arr_expand.numpy(), nparr_expand)

  arr_expand = arr.expand((3, 3, 3))
  nparr_expand = np.tile(nparr, (1, 3, 3))
  assert np.allclose(arr_expand.numpy(), nparr_expand)

def test_permute():
  shape = (2, 3, 4)
  nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
  arr = CLArray(nparr)
  check_array(arr.T, nparr.T)
  check_array(arr.permute((0, 2, 1)), nparr.transpose((0, 2, 1)))

def test_squeeze():
  shape = (1, 2, 3, 1)
  nparr = rnd(shape)
  arr = CLArray(nparr)
  check_array(arr.squeeze(), nparr.squeeze())
  check_array(arr.squeeze(axis=0), nparr.squeeze(axis=0))
  check_array(arr.squeeze(axis=-1), nparr.squeeze(axis=-1))
  check_array(arr.squeeze(axis=(0, -1)), nparr.squeeze(axis=(0, -1)))
  shape = (1, 1)
  nparr = rnd(shape)
  arr = CLArray(nparr)
  check_array(arr.squeeze(), nparr.squeeze())

def test_elemwise_op():
  # inplace
  shape = (2, 4, 5)
  nparr = rnd(shape)
  nparr_copy = nparr.copy()
  arr = CLArray(nparr)
  arr += nparr_copy
  nparr += nparr_copy
  check_array(arr, nparr)
  arr += 1
  nparr += 1
  check_array(arr, nparr)

  # on broadcasted array
  shape = (1,)
  nparr = rnd(shape)
  arr = CLArray(nparr)
  arr = CLArray(nparr).reshape((1, 1, 1)).expand((3, 4, 5))
  nparr = np.broadcast_to(nparr.reshape((1, 1, 1)), (3, 4, 5))
  check_array(arr, nparr)
  check_array(arr.exp(), np.exp(nparr))

  shape = (2, 4, 5)
  nparr = rnd(shape)
  arr = CLArray(nparr)
  check_array(-arr, -nparr)
  check_array((arr+1e8).log(), np.log(nparr+1e8))
  check_array(arr.exp(), np.exp(nparr))

def test_reduce_op():
  for name in ("sum", "max"):
    for shape in [
            (1,),
            (2**6+1,),
            (2**6, 2**6+1),
            (2**6, 2**6+1, 2, 2),
            (1, 1, 1, 1)]:
      nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
      arr = CLArray(nparr)
      op1, op2 = getattr(arr, name), getattr(nparr, name)
      check_array(op1(), op2())
      for axis in range(nparr.ndim):
        check_array(op1(axis=axis), op2(axis=axis))
        check_array(op1(axis=axis, keepdims=True), op2(axis=axis, keepdims=True), ignore=("stride",))

      nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
      arr = CLArray(nparr)
      arr, nparr = arr.T, nparr.T
      op1, op2 = getattr(arr, name), getattr(nparr, name)
      for axis in range(nparr.ndim):
        check_array(op1(axis=axis), op2(axis=axis), ignore=("stride", "contig"))
        check_array(op1(axis=axis, keepdims=True), op2(axis=axis, keepdims=True), ignore=("stride", "contig"))

def test_random():
  arr = CLArray.uniform(-1, 1, (100000,))
  data = arr.numpy()
  assert np.abs(data.mean() - 0) < 1e-2

def test_broadcast():
  for shape1, shape2 in (
          [(), (1, 2, 3, 4)],
          [(1,), (1, 2, 3, 4)],
          [(1, 1, 1, 1), (1, 2, 3, 4)],
          [(4,), (1, 2, 3, 4)],
          [(3, 1), (1, 2, 3, 4)],
          [(1, 3, 1), (1, 2, 3, 4)],
          [(1, 2, 1, 1), (1, 2, 3, 4)],
          [(1,), (1,)]):
    arr1, arr2 = CLArray.empty(shape1), CLArray.empty(shape2)
    assert (arr1+arr2).shape == arr2.shape

def test_comparison_operators():
  rndint = lambda s: np.random.randint(0, 10, size=s).astype(np.float32)
  shape = (64, 64)
  nparr1, nparr2 = rndint(shape), rndint(shape)
  arr1, arr2 = CLArray(nparr1), CLArray(nparr2)
  check_array(arr1==arr2, (nparr1==nparr2).astype(np.float32))
  check_array(arr1>arr2, (nparr1>nparr2).astype(np.float32))
  check_array(arr1>=arr2, (nparr1>=nparr2).astype(np.float32))
  check_array(arr1<arr2, (nparr1<nparr2).astype(np.float32))
  check_array(arr1<=arr2, (nparr1<=nparr2).astype(np.float32))

def test_matmul_op():
  rnd = lambda s: np.random.randint(0, 10, s).astype(np.float32)
  shape_pairs = [
    [(4, 5), (5, 3)],
    [(5,), (5, 3)],
    [(4, 5), (5,)],
    [(5,), (5,)],
    [(2, 4, 5), (2, 5, 3)],
    [(2, 4, 5), (1, 5, 3)],
    [(2, 4, 5), (5, 3)],
    [(2, 4, 5), (5,)],
    [(2, 3, 4, 5), (2, 3, 5, 3)],
    [(2, 3, 4, 5), (1, 1, 5, 3)],
    [(2, 3, 4, 5), (5,)],
    [(1, 128, 256), (1, 256, 8)],
    [(1, 32, 32), (1, 32, 32)],
    [(1, 2**6, 2**6), (1, 2**6, 2**6)],
    [(1, 4096, 256), (1, 256, 512)]
  ]
  for s1, s2 in shape_pairs:
    nparr1, nparr2 = rnd(s1), rnd(s2)
    arr1, arr2 = CLArray(nparr1), CLArray(nparr2)
    check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)

  s1, s2 = (4, 5), (3, 5)
  nparr1, nparr2 = rnd(s1), rnd(s2)
  arr1, arr2 = CLArray(nparr1), CLArray(nparr2)
  arr2, nparr2 = arr2.T, nparr2.T
  check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)

  s1, s2 = (4, 5), (1, 3)
  nparr1, nparr2 = rnd(s1), rnd(s2)
  arr1, arr2 = CLArray(nparr1), CLArray(nparr2)
  arr2 = arr2.expand((5, 3))
  nparr2 = np.ascontiguousarray(np.broadcast_to(nparr2, (5, 3)))
  check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)
