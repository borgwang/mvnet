import time
from ctypes import CFUNCTYPE, POINTER, c_float, c_int64, c_void_p

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir
from utils import FS, Timer

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.set_option('', '--x86-asm-syntax=intel')

module = ir.Module()
func_type = ir.FunctionType(ir.VoidType(),
        [ir.FloatType().as_pointer(), ir.FloatType().as_pointer(), ir.IntType(64), ir.IntType(64)])

func = ir.Function(module, func_type, name="elemwise_op")

bentry = ir.IRBuilder(func.append_basic_block(name="entry"))
bloop0 = ir.IRBuilder(func.append_basic_block(name="loop0"))
bloopexit0 = ir.IRBuilder(func.append_basic_block(name="loopexit0"))
bloop1 = ir.IRBuilder(func.append_basic_block(name="loop1"))
bloopexit1 = ir.IRBuilder(func.append_basic_block(name="loopexit1"))
bexit = ir.IRBuilder(func.append_basic_block(name="exit"))

bentry.branch(bloop0._block)
i = bloop0.phi(ir.IntType(64), name="i")
i.add_incoming(ir.Constant(i.type, 0), bentry._block)
i_p1 = bloopexit0.add(i, ir.Constant(i.type, 1))
i.add_incoming(i_p1, bloopexit0._block)
bloopexit0.cbranch(
  bloopexit0.icmp_unsigned("<", i_p1, func.args[2]),
  bloop0._block, bexit._block
)

bloop0.branch(bloop1._block)
j = bloop1.phi(ir.IntType(64), name="j")
j.add_incoming(ir.Constant(j.type, 0), bloop0._block)
index = bloop1.add(bloop1.mul(i, func.args[2]), j)
ptr0 = bloop1.gep(func.args[0], [index], name="ptr0")
ptr1 = bloop1.gep(func.args[1], [index], name="ptr1")
value = bloop1.load(ptr0, name="val")
value1 = bloop1.fmul(value, ir.Constant(ir.FloatType(), 1.1))
bloop1.store(value1, ptr1)

j_p1 = bloopexit1.add(j, ir.Constant(j.type, 1))
j.add_incoming(j_p1, bloopexit1._block)
bloopexit1.cbranch(
  bloopexit1.icmp_unsigned("<", j_p1, func.args[3]),
  bloop1._block, bloopexit0._block
)
bloop1.branch(bloopexit1._block)
bexit.ret_void()

print("-------- IR original ---------")
print(str(module))
llmod = llvm.parse_assembly(str(module))
llmod.verify()

llmod.triple = llvm.get_process_triple()
pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 3
pm = llvm.create_module_pass_manager()
pmb.populate(pm)
pm.run(llmod)

print("-------- IR optimized ---------")
print(str(llmod))

def run():
  target_machine = llvm.Target.from_triple(llvm.get_process_triple()).create_target_machine(opt=3)
  with llvm.create_mcjit_compiler(llmod, target_machine) as engine:
    engine.finalize_object()
    print("-------- Assembly ---------")
    print(target_machine.emit_assembly(llmod))

    func_ptr = engine.get_function_address("elemwise_op")
    func = CFUNCTYPE(c_void_p, POINTER(c_float), POINTER(c_float), c_int64, c_int64)(func_ptr)

    np.random.seed(0)
    N = 4096
    arr = np.random.uniform(0, 1, (N, N)).astype(np.float32)
    n_warmup, n_measure = 10, 50

    def llvm_call(arr):
      res = np.empty((N, N), dtype=np.float32)
      func(arr.ctypes.data_as(POINTER(c_float)), res.ctypes.data_as(POINTER(c_float)), N, N)
      return res

    def np_call(arr):
      return arr * 1.1

    flops = N*N*n_measure

    for _ in range(n_warmup): llvm_call(arr)
    with Timer("llvm") as t1:
      for _ in range(n_measure):
        actual = llvm_call(arr)

    for _ in range(n_warmup): np_call(arr)
    with Timer("numpy") as t2:
      for _ in range(n_measure):
        expect = np_call(arr)

    factor = t2.ms/t1.ms
    color = FS["GREEN"] if t1.ms < t2.ms else FS["RED"]
    print(f"ElemwiseOp({N}x{N})\t"
          f"llvm: {t1.ms:.2f}ms, {1e-6*flops/t1.ms:.2f}GFLOPS\t"
          f"numpy: {t2.ms:.2f}ms, {1e-6*flops/t2.ms:.2f}GFLOPS\t"
          f"factor: {color}{factor:.3f}{FS['ENDC']}")

    assert np.allclose(actual, expect)

run()
