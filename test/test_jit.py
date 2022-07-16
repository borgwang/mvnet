import runtime_path  # isort:skip

import numpy as np

from env import LAZY, GRAPH
from core.tensor import Tensor
from utils.helper import plot_graph, kernelstat

def check_tensor(a, b, atol=0, rtol=1e-4):
    assert a.shape == b.shape
    assert np.allclose(a.numpy(), b, atol=atol, rtol=rtol, equal_nan=True)

def test_lazy_unary():
    npa = np.array([[1, 2, 3]]).astype(np.float32)
    a = Tensor(npa, name="a").to("gpu")
    if LAZY:
        check_tensor((-a).resolve(), -npa)
        check_tensor(a.log().resolve(), np.log(npa))
        check_tensor(a.exp().resolve(), np.exp(npa))
        check_tensor(a.relu().resolve(), npa*(npa>0))
        #check_tensor((a>0).resolve(), (npa>0).astype(np.float32))

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

    w.zero_grad()
    b.zero_grad()

    kernelstat.reset()
    pred_tmp = x @ w + b
    pred = pred_tmp / pred_tmp.sum()
    #if GRAPH: plot_graph(pred)
    assert kernelstat.total() == 0  # not invoke yet

    # patial run
    kernelstat.reset()
    pred_tmp_np = x_np @ w_np + b_np
    check_tensor(pred_tmp, pred_tmp_np, rtol=1e-3)
    assert kernelstat.get("matmul") == 1
    assert kernelstat.get("reduce") == 0
    assert kernelstat.total() == 2 + 1

    kernelstat.reset()
    pred_np = x_np @ w_np + b_np
    pred_np = pred_np / pred_np.sum()
    check_tensor(pred, pred_np, rtol=1e-3)
    assert kernelstat.get("matmul") == 1
    assert kernelstat.total() == 6

def test_lazy_backward():
    kernelstat.reset()
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

    w.zero_grad()
    b.zero_grad()
    pred = x @ w + b
    loss = ((pred - y)**2).sum()
    if GRAPH: plot_graph(loss)
    loss_np = (((x_np @ w_np + b_np) - y_np) ** 2).sum()
    check_tensor(loss, loss_np, rtol=1e-3)
    print(kernelstat.counter)
    #loss.backward()
    #w -= 0.0001 * w.grad
    #b -= 0.0001 * b.grad

