from llvmlite import ir

import numpy as np
import llvmlite.binding as llvm
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_int64, c_float

"""
int_type = ir.IntType(64)
func_type = ir.FunctionType(int_type, [int_type])

module = ir.Module(name="fibonacci_example")
fn_fib = ir.Function(module, func_type, name="fn_fib")
fn_fib_block = fn_fib.append_basic_block("fn_fib_entry")
builder = ir.IRBuilder(fn_fib_block)

# access to function argument
fn_fib_n, = fn_fib.args
const_1, const_2 = ir.Constant(int_type, 1), ir.Constant(int_type, 2)
fn_fib_n_lteq_1 = builder.icmp_signed(cmpop="<=", lhs=fn_fib_n, rhs=const_1)

with builder.if_then(fn_fib_n_lteq_1):
    builder.ret(const_1)

fn_fib_n_minus_1 = builder.sub(fn_fib_n, const_1)
fn_fib_n_minus_2 = builder.sub(fn_fib_n, const_2)

call_fn_fib_n_minus_1 = builder.call(fn_fib, [fn_fib_n_minus_1])
call_fn_fib_n_minus_2 = builder.call(fn_fib, [fn_fib_n_minus_2])

fn_fib_rec_res =  builder.add(call_fn_fib_n_minus_1, call_fn_fib_n_minus_2)

builder.ret(fn_fib_rec_res)

#print(module)

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create engine and attach the generated module
# Create a target machine representing the host
#target = llvm.Target.from_default_triple()
target = llvm.Target.from_triple(llvm.get_process_triple())
target_machine = target.create_target_machine(opt=0)
target_machine.set_asm_verbosity(True)
# And an execution engine with an empty backing module
backing_mod = llvm.parse_assembly("")
engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

# parse generate module
mod = llvm.parse_assembly( str( module ) )
mod.verify()
print(str(mod))
print(target_machine.emit_assembly(mod))

engine.add_module(mod)
engine.finalize_object()

func_ptr = engine.get_function_address("fn_fib")
c_fn_fib = CFUNCTYPE(c_int64, c_int64)(func_ptr)

for n in range(5):
    result = c_fn_fib(n)
    #print(f"c_fn_fib({n}) = {result}")
"""


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# simple reduce
func_type = ir.FunctionType(ir.FloatType(), [ir.FloatType().as_pointer(), ir.IntType(64), ir.IntType(64)])
module = ir.Module()

func = ir.Function(module, func_type, name="reduce_op")

bentry = ir.IRBuilder(func.append_basic_block(name="entry"))
bloop0 = ir.IRBuilder(func.append_basic_block(name="loop0"))
bloopexit0 = ir.IRBuilder(func.append_basic_block(name="loopexit0"))
bloop1 = ir.IRBuilder(func.append_basic_block(name="loop1"))
bloopexit1 = ir.IRBuilder(func.append_basic_block(name="loopexit1"))
bexit = ir.IRBuilder(func.append_basic_block(name="exit"))

bentry.branch(bloop0._block)
acc0 = bloop0.phi(ir.FloatType(), name="acc0")
acc0.add_incoming(ir.Constant(acc0.type, 0.0), bentry._block)

i = bloop0.phi(ir.IntType(64), name="i")
i.add_incoming(ir.Constant(i.type, 0), bentry._block)
i_p1 = bloopexit0.add(i, ir.Constant(i.type, 1))
i.add_incoming(i_p1, bloopexit0._block)
bloopexit0.cbranch(
  bloopexit0.icmp_unsigned("<", i_p1, func.args[1]),
  bloop0._block, bexit._block
)

bloop0.branch(bloop1._block)

acc1 = bloop1.phi(ir.FloatType(), name="acc1")
acc1.add_incoming(ir.Constant(acc1.type, 0.0), bloop0._block)
j = bloop1.phi(ir.IntType(64), name="j")
j.add_incoming(ir.Constant(j.type, 0), bloop0._block)
index = bloop1.add(bloop1.mul(i, func.args[1]), j)
ptr = bloop1.gep(func.args[0], [index], name="ptr")
value = bloop1.load(ptr, name="val")
added1 = bloop1.fadd(acc1, value, name="added1", flags=("fast",))
acc1.add_incoming(added1, bloopexit1._block)

added0 = bloop1.fadd(acc0, added1, name="added0", flags=("fast",))
acc0.add_incoming(added0, bloopexit0._block)

j_p1 = bloopexit1.add(j, ir.Constant(j.type, 1))
j.add_incoming(j_p1, bloopexit1._block)
bloopexit1.cbranch(
  bloopexit1.icmp_unsigned("<", j_p1, func.args[2]),
  bloop1._block, bloopexit0._block
)

bloop1.branch(bloopexit1._block)
bexit.ret(added0)

print("-------- IR original ---------")
print(str(module))
llmod = llvm.parse_assembly(str(module))
llmod.verify()

llmod.triple = llvm.get_process_triple()
pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 0
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

        func_ptr = engine.get_function_address("reduce_op")
        func = CFUNCTYPE(c_float, POINTER(c_float), c_int64, c_int64)(func_ptr)

        import time
        XS, YS = 2048, 2048
        np.random.seed(0)
        arr = np.random.uniform(0, 1, (XS, YS)).astype(np.float32)
        print(arr)
        # from numpy to buffer
        buffer = (ctypes.c_float * XS*YS)()
        st = time.monotonic()
        actual = func(arr.ctypes.data_as(POINTER(c_float)), XS, YS)
        et = time.monotonic()
        print(f"actual: {actual:.4f} cost: {(et-st)*100:.2f}ms")

        st = time.monotonic()
        expect = arr.sum()
        et = time.monotonic()
        print(f"expect: {expect:.4f} cost: {(et-st)*100:.2f}ms")

run()
