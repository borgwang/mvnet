## mvnet

mvnet is a small but fully functional deep learning framework built on top of [tinynn](https://github.com/borgwang/tinynn) and [tinynn-autograd](https://github.com/borgwang/tinynn-autograd)

### Features
- automatic differentiation
- support numpy/opencl backends

### TODOs
- ops
  - conv/tconv op
  - slice op
- unit testing
- speedup
  - kernel-level optimization
  - graph-level optimization
- backend
  - support cuda backend

### Laziness

- A lazy node attributes: shape, strides, operator, operands, extra args
- Recursive invoke the computation when we call eager on a lazy node

### License

MIT

