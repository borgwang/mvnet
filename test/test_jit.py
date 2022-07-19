import runtime_path  # isort:skip

import numpy as np

from env import LAZY, GRAPH, OPT1
from core.tensor import Tensor
from utils.helper import get_tensor_graph, get_array_graph, kernelstat

def check_tensor(a, b, atol=0, rtol=1e-4):
    assert a.shape == b.shape
    assert np.allclose(a.numpy(), b, atol=atol, rtol=rtol, equal_nan=True)

def test_lazy_elemwise():
    npa = np.array([[1, 2, 3]]).astype(np.float32)
    a = Tensor(npa, name="a").to("gpu")
    check_tensor(-a, -npa)
    check_tensor(a.log(), np.log(npa))
    check_tensor(a.exp(), np.exp(npa))
    check_tensor(a.relu(), npa*(npa>0))
    check_tensor((a>0), (npa>0).astype(np.float32))

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
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x)
    gt = np_b * (np_w + np_x)
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + (w + b) * x
    gt = np_b * (np_w + np_x) + (np_w + np_b) * np_x
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + (m + n)
    gt = np_b * (np_w + np_x) + (np_m + np_n)
    assert np.allclose(y.numpy(), gt)

    y = b * (w + x) + b * (w - x)
    gt = np_b * (np_w + np_x) + np_b * (np_w - np_x)
    assert np.allclose(y.numpy(), gt)

    np_a = np.array([1]).astype(np.float32)
    np_b = np.array([3]).astype(np.float32)
    np_c = np.array([[3, 4, 5]]).astype(np.float32)
    a = Tensor(np_a, name="a").to("gpu")
    b = Tensor(np_b, name="b").to("gpu")
    c = Tensor(np_c, name="c").to("gpu")
    y = -((a + b) * c)
    gt = -((np_a + np_b) * np_c)
    assert np.allclose(y.numpy(), gt)

def test_lazy_forward():
    kernelstat.reset()
    BS = 64
    idim = 2569
    odim = 10
    x_np = np.random.normal(10, 1, (BS, idim)).astype(np.float32)
    y_np = np.random.normal(10, 1, (BS, odim)).astype(np.float32)
    w_np = np.random.normal(10, 1, (idim, odim)).astype(np.float32)
    b_np = np.zeros((1, odim)).astype(np.float32)

    device = "gpu"
    x = Tensor(x_np, name="x").to(device)
    y = Tensor(y_np, name="y").to(device)
    w = Tensor(w_np, requires_grad=True, name="w").to(device)
    b = Tensor(b_np, requires_grad=True, name="b").to(device)
    w.zero_grad(); b.zero_grad()

    kernelstat.reset()
    pred_tmp = x @ w + b
    pred = pred_tmp / pred_tmp.sum()
    loss = ((pred - y)**2).log().exp().sum()

    pred_tmp_np = x_np @ w_np + b_np
    pred_np = pred_tmp_np / pred_tmp_np.sum()
    loss_np = np.sum(np.exp(np.log((pred_np - y_np)** 2)))

    if LAZY:
        assert kernelstat.total() == 0  # not invoke yet, it' lazy

    kernelstat.reset()
    check_tensor(pred_tmp, pred_tmp_np, rtol=1e-3)
    if LAZY:
        assert kernelstat.get("matmul") == 1
        assert kernelstat.get("elementwise") == 1

    kernelstat.reset()
    check_tensor(pred, pred_np, rtol=1e-3)
    if LAZY:
        # matmul has been invoked before
        assert kernelstat.get("matmul") == 0
        assert kernelstat.get("elementwise") == 1

    kernelstat.reset()
    check_tensor(loss, loss_np, rtol=1e-3)
    if LAZY:
        assert kernelstat.get("matmul") == 0
        if not OPT1:
            assert kernelstat.get("elementwise") == 4
        else:
            assert kernelstat.get("elementwise") == 1

    kernelstat.reset()
    check_tensor(loss, loss_np, rtol=1e-3)
    if LAZY:
        # loss has been invoked before
        assert kernelstat.get("matmul") == 0
        assert kernelstat.get("elementwise") == 0
        assert kernelstat.get("reeduce") == 0
        assert kernelstat.get("contiguous") == 1

def test_lazy_backward():
    BS = 64
    idim = 2569
    odim = 10
    x_np = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
    y_np = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
    w_np = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
    b_np = np.zeros((1, odim)).astype(np.float32)

    device = "gpu"
    x = Tensor(x_np, name="x").to(device)
    y = Tensor(y_np, name="y").to(device)
    w = Tensor(w_np, requires_grad=True, name="w").to(device)
    b = Tensor(b_np, requires_grad=True, name="b").to(device)
    w.zero_grad(); b.zero_grad()

    pred = x @ w + b
    cost = (pred - y) ** 2
    loss = cost.sum()

    loss.backward() # TODO: trigger the actual computation duration training

def test_graph_optimizer():
    BS = 64
    idim = 2569
    odim = 10
    x_np = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
    y_np = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
    w_np = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
    b_np = np.zeros((1, odim)).astype(np.float32)

    device = "gpu"
    x = Tensor(x_np, name="x").to(device)
    y = Tensor(y_np, name="y").to(device)
    w = Tensor(w_np, requires_grad=True, name="w").to(device)
    b = Tensor(b_np, requires_grad=True, name="b").to(device)
    w.zero_grad(); b.zero_grad()

    pred_tmp = x @ w + b
    pred = pred_tmp / pred_tmp.sum()
    cost = (pred - y) ** 2
    loss = cost.log().exp().sum()
    loss = cost.sum()

    pred_tmp_np = x_np @ w_np + b_np
    pred_np = pred_tmp_np / pred_tmp_np.sum()
    loss_np = np.sum(np.exp(np.log((pred_np - y_np)** 2)))
    loss_np = np.sum((pred_np - y_np)**2)
    check_tensor(loss, loss_np)

