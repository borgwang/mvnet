## mvnet

mvnet is a small but fully functional deep learning framework built on top of [tinynn](https://github.com/borgwang/tinynn) and [tinynn-autograd](https://github.com/borgwang/tinynn-autograd)

### Features
- automatic differentiation
- support numpy/opencl backends
- pytorch-like api

### TODOs
- ops
  - conv/tconv op
- speedup
  - kernel-level optimization
    - relu & drelu
    - gemm
  - graph-level optimization
- backend
  - cuda
  - llvm
  - metal

### Laziness
- A lazy node attributes: shape, strides, operator, operands, extra args
- Recursive invoke the computation when we call eager on a lazy node

### Benchmark
```
# profile forward
GRAPH=1 LAZY=1 BACKEND=opencl python3 examples/mnist.py --batch_size 4096 --eval 1 --profile_forward 1
```
```
# backend numpy: ~0.65s per epoch
LAZY=0 BACKEND=numpy python3 examples/mnist.py --batch_size 4096 --eval 1

# opencl backend (eager): ~0.75s per epoch
LAZY=0 BACKEND=opencl python3 examples/mnist.py --batch_size 4096 --eval 1

# opencl backend (lazy): ~0.78s per epoch
LAZY=1 BACKEND=opencl python3 examples/mnist.py --batch_size 4096 --eval 1

# lazy with optimization: ~0.62s per epoch
OPT_CONSTANT_FOLDING=1 OPT_ELEMWISE_FUSION=1 LAZY=1 BACKEND=opencl python3 examples/mnist.py --batch_size 4096 --eval 1
```

### Test
```
DEBUG=0 GRAPH=0 LAZY=1 BACKEND=opencl pytest -s
```

### Install

```
pip3 install -e .
```

### License

MIT
