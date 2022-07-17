from functools import lru_cache
import copy
from collections import defaultdict
import numpy as np
import pyopencl

from env import DEBUG, GRAPH, LAZY
from core.backend.base import Array
from core.jit.graph import GraphOptimizer
from core.dtype import int32, float32
from utils.math import prod
from utils.helper import varnamegetter, kernelstat

class ClContext:
    def __init__(self):
        self.ctx, self.queue = None, None
        platform = pyopencl.get_platforms()[0]
        devices = platform.get_devices(device_type=pyopencl.device_type.GPU)
        if len(devices) == 0:
            devices = platform.get_devices(device_type=pyopencl.device_type.CPU)
        self.ctx = pyopencl.Context(devices)
        self.queue = pyopencl.CommandQueue(self.ctx)
        import pyopencl.clrandom as clrandom
        self.rng = clrandom.PhiloxGenerator(self.ctx, seed=0)

    @lru_cache(maxsize=None)
    def build(self, name, program):
        if DEBUG > 1: print(f"[DEBUG] program {name}: \n {program}")
        kernel = pyopencl.Program(self.ctx, program).build().__getattr__(name)
        return lambda *args: kernel(self.queue, *args)

    def alloc_local_memory(self, size):
        return pyopencl.LocalMemory(size)

    def alloc_buffer(self, shape, dtype, hostbuf=None):
        size = int(dtype().itemsize * prod(shape))
        flags = pyopencl.mem_flags.READ_WRITE
        if hostbuf is not None:
            flags |= pyopencl.mem_flags.COPY_HOST_PTR
        return pyopencl.Buffer(self.ctx, flags, size, hostbuf=hostbuf)

    def enqueue(self, task, *args, **kwargs):
        getattr(pyopencl, f"enqueue_{task}")(self.queue, *args, **kwargs)

cl = ClContext()

def unary_op(name, a, ret=None, **kwargs):
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)  # TODO
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    code_map = {"noop": "A", "neg": "-A", "log": "log(A)", "exp": "exp(A)", "relu": "max(A, 0.0f)", "sign": "sign(A)"}
    op = cl.build("unary_op", f"""__kernel void unary_op(
        {''.join([f'int a_s{i}, int res_s{i}, ' for i in range(a.ndim)])}
        int a_ofst, __global const float *inp, __global float *ret) {{
      int a_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join([f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i};' for i in range(a.ndim)])}
      float A=inp[a_i+a_ofst];
      ret[gl_id]={code_map[name]};
    }}""")
    args = [int32(s) for ss in zip(a.strides, ret.strides) for s in ss] + [int32(a.offset)]
    e = op((prod(a.shape),), None, *args, a.buffer, ret.buffer)
    kernelstat.log("unary")
    if GRAPH: e.wait()
    return ret

def elementwise_op(code, inp):
    assert len(set([tuple(x.shape) for x in inp.values()])) == 1, \
        f"Invalid input shape for elementwise op {inp}"
    a = inp[list(inp.keys())[0]]
    ret = a.__class__(shape=a.shape, dtype=a.dtype)
    args = "".join("".join(f"int {name}_s{i}, " for name in inp) + f"int res_s{i}," for i in range(ret.ndim))
    args += "".join(f"int {name}_ofst, " for name in inp)
    args += "".join(f"__global const float *inp_{name}, " for name in inp)
    update = ""
    for i in range(ret.ndim):
        update += f"idx=ptr/res_s{i}; ptr%=res_s{i};" + "".join(f"{name}_i+=idx*{name}_s{i};" for name in inp)
    assign = ";".join(f"float {name}=inp_{name}[{name}_i+{name}_ofst]" for name in inp)
    src = f"""__kernel void elementwise_op({args} __global float *ret) {{
      {";".join(f"int {name}_i=0" for name in inp)};
      int idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {update};
      {assign};
      ret[gl_id] = {code};
    }}"""
    op = cl.build("elementwise_op", src)
    args = [int32(s) for ss in zip(*[x.strides for x in (list(inp.values())+[ret])]) for s in ss]
    args += [int32(x.offset) for x in inp.values()]
    args += [x.buffer for x in inp.values()]
    e = op((prod(a.shape),), None, *args, ret.buffer)
    kernelstat.log("elementwise")
    if GRAPH: e.wait()
    return ret

def binary_op(name, a, b, ret=None):
    a, b = Array.broadcast(a, b)
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    ret_strides = (1,) if not ret.strides else ret.strides
    code_map = {"add": "A+B", "sub": "A-B", "div": "A/B", "mul": "A*B", "pow": "pow(A,B)",
                "eq": "(float)isequal(A,B)", "gt": "(float)isgreater(A,B)", "ge": "(float)isgreaterequal(A,B)",
                "drelu": "B>0?A:0.0f"}
    op = cl.build("binary_op", f"""__kernel void binary_op(
        {''.join([f'int a_s{i}, int b_s{i}, int res_s{i}, ' for i in range(a.ndim)])}
        int a_ofst, int b_ofst, __global const float *inp_1, __global const float *inp_2, __global float *ret) {{
      int a_i=0, b_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join(f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i}; b_i+=idx*b_s{i};' for i in range(a.ndim))}
      float A=inp_1[a_i+a_ofst], B=inp_2[b_i+b_ofst];
      ret[gl_id] = {code_map[name]};
    }}""")
    args = [int32(s) for ss in zip(a.strides, b.strides, ret_strides) for s in ss]
    args += [int32(s) for s in [a.offset, b.offset]]
    e = op((prod(a.shape),), None, *args, a.buffer, b.buffer, ret.buffer)
    kernelstat.log("binary")
    if GRAPH: e.wait()
    return ret

def matmul_op(a, b):
    # rule: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    squeezes = []
    if a.ndim == 1: a = a.reshape((1, *a.shape)); squeezes.append(0)
    if b.ndim == 1: b = b.reshape((*b.shape, 1)); squeezes.append(-1)
    ret_shape = tuple((*a.shape[:-1], b.shape[-1]))

    if a.ndim > 3: a = a.reshape((prod(a.shape[:-2]), *a.shape[2:]))
    if b.ndim > 3: b = b.reshape((prod(b.shape[:-2]), *b.shape[2:]))
    if a.ndim == 2: a = a.reshape((1, *a.shape))
    if b.ndim == 2: b = b.reshape((1, *b.shape))
    if a.shape[0] != b.shape[0]:  # broadcasting
        assert a.shape[0] == 1 or b.shape[0] == 1
        if a.shape[0] == 1 and b.shape[0] != 1: a = a.expand((b.shape[0], *a.shape[1:]))
        if b.shape[0] == 1 and a.shape[0] != 1: b = b.expand((a.shape[0], *b.shape[1:]))
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], \
            f"invalid shape for matmul {a.shape} @ {b.shape}"
    ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    BS, M, K, N = prod(a.shape[:-2]), a.shape[-2], a.shape[-1], b.shape[-1]
    gs = 1
    while gs <= 8 and M % gs == 0 and N % gs == 0 and K % gs == 0 and gs <= K and gs <= M and gs <= N:
        gs *= 2
    gs //= 2
    if DEBUG > 1: print(f"[DEBUG] BS:{BS} M:{M} K:{K} N:{N} grp_size:{gs}")
    src = f"""#define GS {gs}
    __kernel void matmul_op(int BS, int M, int N, int K,
        {''.join(f'int A_s{i}, int B_s{i}, ' for i in range(3))} int a_ofst, int b_ofst,
        __global const float *A, __global const float *B, __global float *C) {{
      int bs=get_global_id(0), m=get_global_id(1), n=get_global_id(2), i=get_local_id(1), j=get_local_id(2);
      __local float Alcl[GS][GS], Blcl[GS][GS];
      float acc = 0.0f;
      for (int t=0; t<K/GS; t++) {{
        Alcl[i][j] = A[bs*A_s0+m*A_s1+(t*GS+j)*A_s2+a_ofst];
        Blcl[i][j] = B[bs*B_s0+(t*GS+i)*B_s1+n*B_s2+b_ofst];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<GS; k++) acc += Alcl[i][k] * Blcl[k][j];
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      C[bs*M*N+m*N+n] = acc;
    }}"""
    op = cl.build("matmul_op", src)
    strides = [s for ss in zip(a.strides, b.strides) for s in ss]
    args = [int32(x) for x in [BS, M, N, K] + strides + [a.offset, b.offset]]
    e = op((BS, M, N), (1, gs, gs), *args, a.buffer, b.buffer, ret.buffer)
    kernelstat.log("matmul")
    if GRAPH: e.wait()
    for axis in squeezes:
        ret = ret.squeeze(axis)
    return ret

def contiguous_op(x):
    ret = x.__class__(shape=x.shape, dtype=x.dtype)
    x_ndim, x_shape, x_strides = x.ndim, x.shape, x.strides
    if not x_ndim:
        x_ndim, x_shape, x_strides = 1, (1,), (1,)
    def_args = "".join([f"int a{i}, int b{i}, " for i in range(x_ndim)])
    def_args += "int ofst"
    def_strides = "".join([f"int _s{i}="+"*".join(f"a{j}" for j in range(i+1, x_ndim)) + ";" for i in range(x_ndim-1)])
    def_strides += f"int _s{x_ndim-1}=1;"
    def_indices = "".join(f"int _i{i}=curr/_s{i}; curr%=_s{i}; " for i in range(x_ndim))
    addr = "+".join([f"b{i}*_i{i}" for i in range(x_ndim)])
    src = f"""__kernel void contiguous_op({def_args}, __global const float *A, __global float *B) {{
      int curr = get_global_id(0);
      {def_strides} {def_indices}
      B[get_global_id(0)] = A[{addr}+ofst];
    }}"""
    op = cl.build("contiguous_op", src)
    args = [int32(s) for ss in zip(x_shape, x_strides) for s in ss]
    e = op((prod(x_shape),), None, *args, int32(x.offset), x.buffer, ret.buffer)
    kernelstat.log("contiguous")
    if GRAPH: e.wait()
    return ret

def reduce_op(name, x, axis=None, keepdims=True):
    x = contiguous_op(x) if not x.c_contiguous else x
    code_map = {"sum": "A+B", "max": "max(A,B)"}
    padval_map = {"sum": "0.0f", "max": "-INFINITY"}
    x_shp = x.shape
    if axis is None:
        axis, x_shp = 0, (prod(x.shape),)
    size = x_shp[axis]

    def cal_ret_shape(x_shp, axis, keepdims, grp_size, n_grps):
        if n_grps <= 1:
            ret_shape = [d for i, d in enumerate(x_shp) if i != axis]
            if keepdims: ret_shape.insert(axis, 1)
            return tuple(ret_shape)
        return tuple(n_grps if i == axis else d for i, d in enumerate(x_shp))

    grp_size = 2
    max_work_group_size = cl.queue.device.max_work_group_size
    while grp_size != max_work_group_size and grp_size < size:
        grp_size *= 2
    n_grps = (size + grp_size - 1) // grp_size
    ret_shape = cal_ret_shape(x_shp, axis, keepdims, grp_size, n_grps)
    ret = x.__class__(shape=ret_shape, dtype=x.dtype)

    # merge non-target axes
    p1 = [prod(x_shp[:axis])] if axis!=0 else []
    p2 = [prod(x_shp[axis+1:])] if axis!=len(x_shp)-1 else []
    global_size = (*p1, grp_size*n_grps, *p2)
    axis, ndim = len(p1), len(global_size)

    a = [f"gl_id_{i}" for i in range(ndim)]
    b = [f"gl_s_{i}" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    gl2lcl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    a = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(ndim)]
    b = [f"(gl_s_{i}/grp_s_{i})" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    lcl2gl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    # NOTE: calculate offset to get the proper global index
    offset = f"gl_id_0*{'0' if axis==0 else '1' if axis==ndim-1 else 'gl_s_2'}*(gl_s_{axis}-size)"
    op = cl.build("reduce_op", f"""__kernel void reduce_op(int size, int ofst,
        __global const float *inp, __local float *lcl, __global float *ret) {{
      {''.join([f'int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});' for i in range(ndim)])}
      int lcl_id=get_local_id({axis});
      lcl[lcl_id] = gl_id_{axis}<size?inp[{gl2lcl}-{offset}+ofst]:{padval_map[name]};
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1) {{
        float A = lcl[lcl_id], B = lcl[lcl_id+stride];
        if (lcl_id<stride) lcl[lcl_id]={code_map[name]};
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      if (lcl_id == 0) ret[{lcl2gl}]=lcl[0];
    }}""")
    local_mem = cl.alloc_local_memory(x.dtype().itemsize * grp_size)
    local_size = tuple(grp_size if i == axis else 1 for i in range(ndim))
    e = op(global_size, local_size, int32(size), int32(x.offset), x.buffer, local_mem, ret.buffer)
    kernelstat.log("reduce")
    if GRAPH: e.wait()
    if DEBUG > 1: print(f"[DEBUG] x_shp: {x_shp} ret_shape: {ret_shape} grp_size: {grp_size} n_grps: {n_grps} size: {size} global_size: {global_size} local_size: {local_size} axis={axis} ndim={ndim} offset={offset}")
    if n_grps > 1:
        ret = reduce_op(name, ret, axis=axis, keepdims=keepdims)
    return ret


class ClArray(Array):
    """Pyopencl's multidimension array class only implement limited functionality. So we write our own."""
    def __init__(self, data=None, shape=None, dtype=float32, is_lazy=False, **kwargs):
        super().__init__(shape, dtype, is_lazy)
        if is_lazy:
            # graph related attributes
            self.operator = kwargs["operator"]
            self.operands = kwargs["operands"]
            self.indegree, self.outdegree = len(self.operands), 0
            self.is_visited = False
        else:
            if isinstance(data, pyopencl.Buffer):
                self.buffer = data
                assert self.shape is not None, "cannot infer shape when initialize using clbuffer"
            else:
                if data is not None:
                    data = np.asarray(data, dtype=self.dtype)
                    self.shape = data.shape
                else:
                    assert self.shape is not None, "cannot infer shape when without data"
                self.buffer = cl.alloc_buffer(self.shape, self.dtype, data)
            # graph related attributes
            self.indegree, self.outdegree = 0, 0
            self.operator = None
            self.operands = {}
            self.is_visited = False
        # meta infos (https://numpy.org/doc/stable/dev/internals.html#numpy-internals)
        self.strides = tuple(prod(self.shape[i+1:]) for i in range(len(self.shape)))
        self.offset = 0  # offset relative to the beginning of the buffer
        self.c_contiguous, self.f_contiguous = True, False
        self.__update_contiguousness()

    @property
    def size(self):
        return self.buffer.size

    @property
    def ndim(self):
        return len(self.shape)

    # ##### Unary Ops #####
    def neg(self):
        if not LAZY: return unary_op("neg", self)
        return ClArray(shape=self.shape, dtype=self.dtype, operator={"type": "elementwise", "code": "-A"},
                       operands={"A": self}, is_lazy=True)
    def exp(self):
        if not LAZY: return unary_op("exp", self)
        return ClArray(shape=self.shape, dtype=self.dtype, operator={"type": "elementwise", "code": "exp(A)"},
                       operands={"A": self}, is_lazy=True)
    def log(self):
        if not LAZY: return unary_op("log", self)
        return ClArray(shape=self.shape, dtype=self.dtype, operator={"type": "elementwise", "code": "log(A)"},
                       operands={"A": self}, is_lazy=True)
    def relu(self):
        if not LAZY: return unary_op("relu", self)
        return ClArray(shape=self.shape, dtype=self.dtype, operator={"type": "elementwise", "code": "max(A,0.0f)"},
                       operands={"A": self}, is_lazy=True)

    @staticmethod
    def invoke(operator, operands):
        # invoke the actual computation
        if operator["type"] == "elementwise":
            return elementwise_op(operator["code"], operands, **operator.get("args", {}))
        elif operator["type"] == "reduce":
            return reduce_op(operator["code"], *operands.values(), **operator.get("args", {}))
        elif operator["type"] == "matmul":
            return matmul_op(*operands.values(), **operator.get("args", {}))

    def resolve(self):
        def _resolve(node=None):
            if node is None: node = self
            for name, dep_node in node.operands.items():
                if dep_node.is_lazy:
                    dep_node.buffer = _resolve(dep_node).buffer
                    dep_node.is_lazy = False
            return self.invoke(node.operator, node.operands)
        graphoptimizer = GraphOptimizer(target_node=self)
        graphoptimizer.build()
        if GRAPH == 2: graphoptimizer.visualize("before")
        graphoptimizer.optimize()
        if GRAPH == 2: graphoptimizer.visualize("after")
        self.buffer = _resolve().buffer
        self.is_lazy = False
        return self

    # ##### Binary Ops #####
    def add(self, other, out=None):
        if not LAZY: return binary_op("add", self, other, ret=out)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "A+B"},
                       operands={"A": a, "B": b}, is_lazy=True)

    def sub(self, other, out=None):
        if not LAZY: return binary_op("sub", self, other, ret=out)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "A-B"},
                       operands={"A": a, "B": b}, is_lazy=True)
    def div(self, other, out=None):
        if not LAZY: return binary_op("div", self, other, ret=out)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "A/B"},
                       operands={"A": a, "B": b}, is_lazy=True)
    def mul(self, other, out=None):
        if not LAZY: return binary_op("mul", self, other, ret=out)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "A*B"},
                       operands={"A": a, "B": b}, is_lazy=True)
    def pow(self, other, out=None):
        if not LAZY:
            return binary_op("pow", self, other, ret=out)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "pow(A,B)"},
                       operands={"A": a, "B": b}, is_lazy=True)

    def matmul(self, other):
        if not LAZY: return matmul_op(self, other)
        a, b = self, other
        if a.ndim == 1: a = a.reshape((1, *a.shape))
        if b.ndim == 1: b = b.reshape((*b.shape, 1))
        ret_shape = tuple((*a.shape[:-1], b.shape[-1]))
        return ClArray(shape=ret_shape, dtype=a.dtype, operator={"type": "matmul"},
                       operands={"A": a, "B": b}, is_lazy=True)

    def eq(self, other):
        if not LAZY: return binary_op("eq", self, other)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "(float)isequal(A,B)"},
                       operands={"A": a, "B": b}, is_lazy=True)

    def ge(self, other):
        if not LAZY: return binary_op("ge", self, other)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "(float)isgreaterequal(A,B)"},
                       operands={"A": a, "B": b}, is_lazy=True)
    def gt(self, other):
        if not LAZY: return binary_op("gt", self, other)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "(float)isgreater(A,B)"},
                       operands={"A": a, "B": b}, is_lazy=True)

    def drelu(self, other):
        if not LAZY: return binary_op("drelu", self, other)
        a, b = Array.broadcast(self, other)
        return ClArray(shape=a.shape, dtype=a.dtype, operator={"type": "elementwise", "code": "B>0?A:0.0f"},
                       operands={"A": a, "B": b}, is_lazy=True)

    # ##### Reduce Ops #####
    def sum(self, axis=None, keepdims=False):
        if not LAZY: return reduce_op("sum", self, axis=axis, keepdims=keepdims)
        operator = {"type": "reduce", "code": "sum", "args": {"axis": axis, "keepdims": keepdims}}
        ret_shape = () if axis is None else [d for i, d in enumerate(self.shape) if i != axis]
        if keepdims: ret_shape.insert(axis, 1)
        return ClArray(shape=ret_shape, dtype=self.dtype, operator=operator,
                       operands={"A": self}, is_lazy=True)

    def max(self, axis=None, keepdims=False):
        if not LAZY: return reduce_op("max", self, axis=axis, keepdims=keepdims)
        operator = {"type": "reduce", "code": "max", "args": {"axis": axis, "keepdims": keepdims}}
        ret_shape = () if axis is None else [d for i, d in enumerate(self.shape) if i != axis]
        if keepdims: ret_shape.insert(axis, 1)
        return ClArray(shape=ret_shape, dtype=self.dtype, operator=operator,
                       operands={"A": self}, is_lazy=True)

    # ##### Slice Ops #####
    def __getitem__(self, key):
        # TODO: handle step
        is_basic = lambda k: isinstance(k, (slice, int))
        assert is_basic(key) or all(is_basic(k) for k in key), \
                f"Advantage indexing not supported yet. {key}"
        key = (key,) if is_basic(key) else key
        inst = copy.copy(self)
        reduce = []
        shape = list(inst.shape)
        for i, k in enumerate(key):
            if isinstance(k, int):  # indexing
                if k < 0: k += inst.shape[i]
                assert 0 <= k < inst.shape[i], f"Invalid indexing {key[i]} for tensor {inst.shape}"
                inst.offset += inst.strides[i] * k
                reduce.append(i)
            if isinstance(k, slice):  # slicing
                start = 0 if k.start is None else k.start
                if start < 0: start += inst.shape[i]
                stop = inst.shape[i] if k.stop is None else k.stop
                if stop < 0: stop += inst.shape[i]
                assert 0 <= start < stop <= inst.shape[i], f"Invalid slicing {key[i]} for tensor {inst.shape}"
                shape[i] = stop - start
                inst.offset += inst.strides[i] * start
                inst.c_contiguous, inst.f_contiguous = False, False  # TODO: is still contiguous under certain conditions
        inst.shape = tuple(s for i, s in enumerate(shape) if i not in reduce)
        inst.strides = tuple(s for i, s in enumerate(inst.strides) if i not in reduce)
        return inst

    def __setitem__(self, key, value):
        item = self[key]
        # unary_op("noop", value, ret=item)
        assert False, "TODO: implement assign ops"

    # ##### Movement Ops #####
    def reshape(self, shape):
        if -1 in shape:
            size = prod(self.shape)
            assert shape.count(-1) <= 1, "Only one dimension can be inferred"
            axis = shape.index(-1)
            infer = prod([s for s in shape if s != -1])
            assert size % infer == 0, f"Shape {shape} invalid for size {size}"
            shape = (*shape[:axis], size // infer, *shape[axis+1:])

        assert prod(shape) == prod(self.shape), f"Can not reshape {self.shape} to {shape}"
        if self.c_contiguous or self.f_contiguous:
            inst = copy.copy(self)
            if self.c_contiguous:
                strides = (prod(shape[i+1:]) for i in range(len(shape)))
            else:
                strides = (prod(shape[:i]) for i in range(len(shape)))
            inst.shape, inst.strides = tuple(shape), tuple(strides)
            inst.__update_contiguousness()
        else:
            inst = self.contiguous().reshape(shape)
        return inst

    def expand(self, shape):
        inst = copy.copy(self)
        assert len(shape) == inst.ndim
        strides = []
        for i, (s1, s2) in enumerate(zip(inst.shape, shape)):
            if s1 < s2:
                assert s1 == 1
            strides.append(0 if s1 < s2 else inst.strides[i])
        inst.shape, inst.strides = tuple(shape), tuple(strides)
        inst.c_contiguous, inst.f_contiguous = False, False
        return inst

    def squeeze(self, axis=None):
        if axis is None:
            axis = [i for i, s in enumerate(self.shape) if s == 1]
        elif isinstance(axis, int):
            axis = [axis]
        assert isinstance(axis, (list, tuple))
        axis = [a if a != -1 else self.ndim - 1 for a in axis]
        shape = [s for i, s in enumerate(self.shape) if i not in axis or self.shape[i] != 1]
        if shape == self.shape:
            return self
        return self.reshape(shape)

    def permute(self, axes):
        inst = copy.copy(self)
        inst.strides = tuple(inst.strides[a] for a in axes)
        inst.shape = tuple(inst.shape[a] for a in axes)
        inst.__update_contiguousness()
        return inst

    # ##### Creation Ops #####
    @classmethod
    def empty(cls, shape, dtype=float32):
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def full(cls, shape, value, dtype=float32):
        inst = cls(shape=shape, dtype=dtype)
        cl.enqueue("fill_buffer", inst.buffer, inst.dtype(value), 0, inst.size)
        return inst

    @classmethod
    def uniform(cls, a, b, shape, dtype=float32):
        buffer = cl.rng.uniform(a=a, b=b, shape=shape, dtype=dtype, cq=cl.queue).data
        return cls(data=buffer, shape=shape, dtype=dtype)

    @classmethod
    def normal(cls, loc, scale, shape, dtype=float32):
        buffer = cl.rng.normal(mu=loc, sigma=scale, shape=shape, dtype=dtype, cq=cl.queue).data
        return cls(data=buffer, shape=shape, dtype=dtype)

    def numpy(self):
        arr = self.resolve() if self.is_lazy else self
        data = np.empty(arr.shape, dtype=arr.dtype)
        cl.enqueue("copy", data, arr.contiguous().buffer, is_blocking=True)
        return data

    def contiguous(self):
        assert not self.is_lazy, "LazyArray should be resolve first."
        return contiguous_op(self)

    def __update_contiguousness(self):
        strides = [self.strides[i] for i in range(self.ndim) if self.shape[i] != 1]
        sorted_strides = sorted(strides)
        self.f_contiguous = sorted_strides == strides
        self.c_contiguous = sorted_strides[::-1] == strides

