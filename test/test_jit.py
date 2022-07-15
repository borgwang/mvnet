import runtime_path  # isort:skip

import numpy as np

from env import LAZY
from core.tensor import Tensor

def test_lazy():
    data_w = np.array([[1, 2, 3]]).astype(np.float32)
    data_x = np.array([[3, 4, 5]]).astype(np.float32)
    data_b = np.array([[0, 1, 0]]).astype(np.float32)
    w = Tensor(data_w, requires_grad=True).to("gpu")
    x = Tensor(data_x, requires_grad=True).to("gpu")
    b = Tensor(data_b, requires_grad=True).to("gpu")
    gt = data_w * data_x + data_b

    y = w * x + b
    if LAZY:
        y = y.resolve()
    assert np.allclose(y.numpy(), gt)

