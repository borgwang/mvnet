import numpy as np
import pyopencl
import pyopencl.array
import pyopencl_blas

pyopencl_blas.setup()  # initialize the library

ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx)

dtype = 'float32'  # also supports 'float64', 'complex64' and 'complex128'
x = np.array([1, 2, 3, 4], dtype=dtype)
y = np.array([4, 3, 2, 1], dtype=dtype)

clx = pyopencl.array.to_device(queue, x)
cly = pyopencl.array.to_device(queue, y)

# call a BLAS function on the arrays
pyopencl_blas.axpy(queue, clx, cly, alpha=0.8)
print("Expected: %s" % (0.8 * x + y))
print("Actual:   %s" % (cly.get()))
