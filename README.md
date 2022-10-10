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
    - relu & drelu
  - graph-level optimization
- backend
  - support cuda backend

### Laziness

- A lazy node attributes: shape, strides, operator, operands, extra args
- Recursive invoke the computation when we call eager on a lazy node

### Benchmark
```
# profile forward
GRAPH=1 LAZY=1 BACKEND=opencl python3 examples/mnist/run.py --batch_size 4096 --eval 1 --profile_forward 1
```

```
# backend numpy: ~0.65s per epoch
GRAPH=0 LAZY=0 BACKEND=numpy python3 examples/mnist/run.py --batch_size 4096 --eval 1

# opencl backend (eager): ~1.9s per epoch
GRAPH=0 LAZY=1 BACKEND=opencl python3 examples/mnist/run.py --batch_size 4096 --eval 1

# opencl backend (lazy): ~2s per epoch
GRAPH=0 LAZY=1 BACKEND=opencl python3 examples/mnist/run.py --batch_size 4096 --eval 1
```

### License

MIT

