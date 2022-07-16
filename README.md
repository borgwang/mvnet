## mvnet

mvnet is a small but fully functional deep learning framework built on top of [tinynn](https://github.com/borgwang/tinynn) and [tinynn-autograd](https://github.com/borgwang/tinynn-autograd)

### Features
- automatic differentiation
- support numpy/opencl backends

### jit
- [x] rename variable in the experssion
- rules of node merge
  - [x] merge inplace ops
- [] support unary ops
- [] backward?

### TODOs
- ops
  - conv/tconv op
  - slice op
- unit testing
- speedup
  - ops optimization
  - jit
- backend
  - support cuda backend
- readability

### License

MIT

