from core.nn.initializer import XavierUniformInit
from core.nn.initializer import ZerosInit

class Layer:
  def __init__(self):
    self.params, self.grads = {}, {}
    self.is_training = True
    self.device = "cpu"

  def forward(self, inputs):
    raise NotImplementedError

  def to(self, device):
    self.device = device
    return self

class Dense(Layer):
  def __init__(self, num_out, num_in=None, w_init=XavierUniformInit(), b_init=ZerosInit()):
    super().__init__()
    self.name = f"dense_(?,{num_out})"
    self.initializers = {"w": w_init, "b": b_init}
    self.shapes = {"w": [num_in, num_out], "b": [1, num_out]}
    self.params = {"w": None, "b": None}

    self.is_init = False
    if num_in is not None:
        self._init_parameters(num_in)

  def forward(self, inputs):
    if not self.is_init:
      self._init_parameters(inputs.shape[-1])
    return inputs @ self.params["w"] + self.params["b"]

  def _init_parameters(self, input_size):
    self.shapes["w"][0] = input_size
    self.params["w"] = self.initializers["w"](shape=self.shapes["w"], device=self.device, name=self.name+"_w")
    self.params["b"] = self.initializers["b"](shape=self.shapes["b"], device=self.device, name=self.name+"_b")
    self.is_init = True

class Activation(Layer):
  def forward(self, inputs):
    raise NotImplementedError

class ReLU(Activation):
  def forward(self, x):
    return x.relu()
