import runtime_path  # isort:skip

import numpy as np

from env import LAZY
from core.tensor import Tensor

def test_lazy_binary():
    np_w = np.array([[1, 2, 3]]).astype(np.float32)
    np_x = np.array([[3, 4, 5]]).astype(np.float32)
    np_b = np.array([[0, 1, 0]]).astype(np.float32)
    np_m = np.array([[3, 4, 5]]).astype(np.float32)
    np_n = np.array([[0, 1, 0]]).astype(np.float32)
    w = Tensor(np_w, name="w").to("gpu")
    x = Tensor(np_x, name="x").to("gpu")
    b = Tensor(np_b, name="b").to("gpu")
    m = Tensor(np_m, name="m").to("gpu")
    n = Tensor(np_n, name="n").to("gpu")

    y = w * x + b
    gt = np_w * np_x + np_b
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x)
    gt = np_b * (np_w + np_x)
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + (w + b) * x
    gt = np_b * (np_w + np_x) + (np_w + np_b) * np_x
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + (m + n)
    gt = np_b * (np_w + np_x) + (np_m + np_n)
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + b * (w + x)
    gt = np_b * (np_w + np_x) + np_b * (np_w + np_x)
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

def test_lazy_unary():
    pass

def test_lazy_backward():
    pass
