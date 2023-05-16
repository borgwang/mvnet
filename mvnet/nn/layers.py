from mvnet.nn.initializer import XavierUniformInit, ZerosInit


class Layer:
  def __init__(self):
    self.params, self.grads = {}, {}
    self.is_training = True
    self.device = "cpu"

  def forward(self, inputs):
    raise NotImplementedError

  def to(self, device):
    for k in self.params:
      self.params[k] = self.params[k].to(device)
    return self

class Dense(Layer):
  def __init__(self, num_in, num_out, w_init=XavierUniformInit(), b_init=ZerosInit()):
    super().__init__()
    self.name = f"dense_({num_in}, {num_out})"
    self.initializers = {"w": w_init, "b": b_init}
    self.shapes = {"w": [num_in, num_out], "b": [1, num_out]}
    self.params = {
      "w": self.initializers["w"](shape=self.shapes["w"], device=self.device, name=self.name+"_w"),
      "b": self.initializers["b"](shape=self.shapes["b"], device=self.device, name=self.name+"_b")
    }

  def forward(self, inputs):
    return inputs @ self.params["w"] + self.params["b"]

class Activation(Layer):
  def forward(self, inputs):
    raise NotImplementedError

class ReLU(Activation):
  def forward(self, inputs):
    return inputs.relu()
