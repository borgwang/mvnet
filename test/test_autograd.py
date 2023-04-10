import numpy as np

from mvnet.env import BACKEND, LAZY
from mvnet.tensor import Tensor


def test_add_op():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
    t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
    t3 = t1 + t2
    assert np.allclose(t3.numpy(), [6, 1, -4])
    t3.backward([2, 2, 2])
    assert np.allclose(t1.grad.numpy(), [2, 2, 2])
    assert np.allclose(t2.grad.numpy(), [2, 2, 2])
    # broadcast (2, 3) + (3,) -> (2, 3)
    t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True).to(device)
    t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
    t3 = t1 + t2
    assert np.allclose(t3.numpy(), [[6, 1, -4], [7, 1, -9]])
    """
    t3.backward([[1, 1, 1], [2, 2, 2]])
    assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [2, 2, 2]])
    assert np.allclose(t2.grad.numpy(), [3, 3, 3])
    # broadcast (2, 3) + (1, 3) -> (2, 3)
    t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True).to(device)
    t2 = Tensor([[5, -2, -9]], requires_grad=True).to(device)
    t3 = t1 + t2
    assert np.allclose(t3.numpy(), [[6, 1, -4], [7, 1, -9]])
    t3.backward([[1, 1, 1], [2, 2, 2]])
    assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [2, 2, 2]])
    assert np.allclose(t2.grad.numpy(), [3, 3, 3])
    """

def test_mul_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
    t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
    t3 = t1 * t2
    assert np.allclose(t3.numpy(), [5, -6, -45])
    t3.backward([2, 2, 2])
    assert np.allclose(t1.grad.numpy(), [2*5, 2*(-2), 2*(-9)])
    assert np.allclose(t2.grad.numpy(), [2*1, 2*3, 2*5])
    t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
    t2 = Tensor([1], requires_grad=True).to(device)
    t3 = t1 * t2
    t3.backward()
    assert np.allclose(t1.grad.numpy(), [1, 1, 1])
    assert np.allclose(t2.grad.numpy(), [9])


def test_div_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, 2, 5], requires_grad=True).to(device)
    t2 = Tensor([8, -2, -10], requires_grad=True).to(device)
    t3 = t1 / t2
    assert np.allclose(t3.numpy(), [0.125, -1, -0.5])
    t3.backward([1, 1, 1])
    assert np.allclose(t1.grad.numpy(), [0.125, -0.5, -0.1])
    assert np.allclose(t2.grad.numpy(), [-0.015625, -0.5, -0.05])

def test_pow_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, -3, 5], requires_grad=True).to(device)
    t2 = t1 ** 3
    assert np.allclose(t2.numpy(), [1, -27, 125])
    t2.backward([2, 2, 2])
    assert np.allclose(t1.grad.numpy(), [2 * 3 * 1 ** 2, 2 * 3 * (-3) ** 2, 2 * 3 * 5 ** 2])

def test_dot_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([[1, 3, 5], [5, -2, 9]], requires_grad=True).to(device)
    t2 = Tensor([[9, 8, 9, 7], [4, 0, 3, 0], [0, 8, 2, 7]], requires_grad=True).to(device)
    t3 = t1 @ t2
    assert np.allclose(t3.numpy(), [[21, 48, 28, 42], [37, 112, 57, 98]])
    t3.backward([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert np.allclose(t1.grad.numpy(), [[80, 13, 50], [85, 22, 35]])
    assert np.allclose(t2.grad.numpy(), [[21, 17, 13, 9], [-5, 0, 5, 10], [41, 37, 33, 29]])

def test_sum_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
    t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
    t3 = (t1 + t2).sum()
    assert t3.numpy() == 3
    t3.backward(2)
    assert np.allclose(t1.grad.numpy(), [2, 2, 2])
    assert np.allclose(t2.grad.numpy(), [2, 2, 2])

def test_epx_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    data = [1, 3, 4]
    t1 = Tensor(data, requires_grad=True).to(device)
    t2 = t1.exp()
    assert np.allclose(t2.numpy(), np.exp(data))
    t2.backward([1, 2, 3])
    assert np.allclose(t1.grad.numpy(), np.exp(data) * np.array([1, 2, 3]))

def test_neg_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
    t2 = -t1
    assert np.allclose(t2.numpy(), [-1, -3, -5])
    t2.backward([1, 2, 3])
    assert np.allclose(t1.grad.numpy(), [-1, -2, -3])

def test_permute_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    shape = [2, 4, 6]
    data = np.random.randn(*shape)
    t1 = Tensor(data, requires_grad=True).to(device)
    t2 = t1.T
    assert list(t2.shape) == shape[::-1]
    t2.backward(2 * np.ones(t2.shape))
    assert list(t1.grad.shape) == shape
    assert np.allclose(t1.grad.numpy(), 2 * np.ones(t1.shape))

def test_max_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([[1, 3, 5], [3, 7, -2]], requires_grad=True).to(device)
    t2 = t1.max()
    t3 = t1.max(axis=0)
    assert t2.numpy() == 7
    assert np.allclose(t3.numpy(), [3, 7, 5])
    t2.backward()
    assert np.allclose(t1.grad.numpy(), [[0, 0, 0], [0, 1, 0]])
    t1.zero_grad()
    t3.backward([1, 1, 1])
    assert np.allclose(t1.grad.numpy(), [[0, 0, 1], [1, 1, 0]])

def test_log_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    data = np.array([1, 3, 5])
    t1 = Tensor(data, requires_grad=True).to(device)
    t2 = t1.log()
    assert np.allclose(t2.numpy(), np.log(data))

    grad = np.array([1, 2, 3])
    t2.backward(grad)
    assert np.allclose(t1.grad.numpy(), grad / np.array([1, 3, 5]))

def test_reshape_ops():
  devices = ("gpu", "cpu")
  for device in devices:
    t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True).to(device)
    t2 = t1.reshape((6,))
    assert np.allclose(t2.numpy(), [1, 2, 3, 4, 5, 6])

    t2.backward(np.ones(6))
    assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [1, 1, 1]])

"""
def test_slice():
  devices = ("gpu", "cpu")
  for device in devices:
    data = np.arange(1, 25).reshape((2, 3, 4)).astype(np.float32)
    t1 = Tensor(data, requires_grad=True).to(device)
    for s in ((1,),
              (1, 2),
              (1, 2, 3),
              slice(1,2),
              (slice(1,2), slice(1, 2)),
              (slice(1,2), slice(1, 2), slice(2, 3)),
              (1, slice(1,2), slice(1, 2)),
              (0, slice(1,2)),
              slice(None, None)
              ):
      t2 = t1[s]
      assert t2.shape == data[s].shape
      assert np.allclose(t2.numpy(), data[s])
      # unary ops
      assert np.allclose(t2.exp().numpy(), np.exp(data[s]))
      assert np.allclose(t2.log().numpy(), np.log(data[s]))
      # binary ops
      t3 = t1[s]
      assert np.allclose((t2+t3).numpy(), data[s]+data[s])
      assert np.allclose((t2*t3).numpy(), data[s]*data[s])
    # matmul ops
    data = np.arange(16).reshape((4, 4)).astype(np.float32)
    t = Tensor(data, requires_grad=True).to(device)
    s = (slice(1,3), slice(1,3))
    assert np.allclose((t[s] @ t[s]).numpy(), data[s] @ data[s])
    # reduce ops
    data = np.arange(48).reshape((4, 4, 3)).astype(np.float32)
    t = Tensor(data, requires_grad=True).to(device)
    s = (slice(1,3), slice(1,3), 0)
    assert np.allclose(t[s].sum().numpy(), data[s].sum())
    assert np.allclose(t[s].sum(axis=1).numpy(), data[s].sum(axis=1))
    assert np.allclose(t[s].max(axis=0, keepdims=True).numpy(), data[s].max(axis=0, keepdims=True))
    # backprop
    data = np.arange(16).reshape((4, 4)).astype(np.float32)
    t = Tensor(data, requires_grad=True).to(device)
    s = (slice(1,3), slice(1,3))
    t2 = t[s]
    t2.backward()
    grad = np.zeros(t.shape).astype(np.float32)
    grad[s] = 1.0
    print(t.grad.numpy())
    print(grad)
    assert np.allclose(t.grad.numpy(), grad)
    import pdb; pdb.set_trace()

"""

def test_minimal():
  from mvnet.utils.helper import kernelstat
  np.random.seed(0)
  n_epoch = 300
  lr = 0.0001

  BS = 2**6
  idim = 2**8
  odim = 2**6
  x_np = np.random.normal(0, 1, (BS, idim)).astype(np.float32)  # (64, 256)
  y_np = np.random.normal(0, 1, (BS, odim)).astype(np.float32)  # (64, 64)
  w_np = np.random.normal(0, 1, (idim, odim)).astype(np.float32)  # (256, 64)
  b_np = np.zeros((1, odim)).astype(np.float32)  # (1, 64)

  x, y, w, b = x_np.copy(), y_np.copy(), w_np.copy(), b_np.copy()
  for epoch in range(n_epoch):
    pred = x @ w + b
    err = pred - y
    loss = (err**2).sum()
    dw = x.T @ (2 * err)
    db = (2 * err).sum(axis=0, keepdims=True)
    w -= lr * dw
    b -= lr * db
  loss_final, w_final, b_final = loss, w, b

  devices = ("gpu", "cpu")
  for device in devices:
    x = Tensor(x_np).to(device)
    y = Tensor(y_np).to(device)
    w = Tensor(w_np, requires_grad=True).to(device)
    b = Tensor(b_np, requires_grad=True).to(device)
    for epoch in range(n_epoch):
      w.zero_grad()
      b.zero_grad()
      pred = x @ w + b
      err = pred - y
      loss = (err ** 2).sum()
      loss.backward()
      w -= lr * w.grad
      b -= lr * b.grad
      if LAZY and device == "gpu":
          w.array = w.array.eager()
          b.array = b.array.eager()
    assert np.allclose(loss.numpy(), loss_final, rtol=1e-3)
    assert np.allclose(w.numpy(), w_final, rtol=1e-3)
    assert np.allclose(b.numpy(), b_final, rtol=1e-3)
