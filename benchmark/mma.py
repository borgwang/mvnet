import numpy as np
import pyopencl as cl

# https://forums.developer.nvidia.com/t/opencl-can-call-tensor-core/246596/2
platform = cl.get_platforms()[0]
devices = platform.get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
  devices = platform.get_devices(device_type=cl.device_type.CPU)
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx)

debug_tid = 1

kernel_src = rf"""
  __kernel void matmul(__global float *out) {{
    float c[8] = {{0., 0., 0., 0., 0., 0., 0., 0.}};
    float d[8] = {{0., 0., 0., 0., 0., 0., 0., 0.}};
    unsigned short a[4] = {{15360, 15360, 15360, 15360}};
    unsigned short b[4] = {{15360, 15360, 15360, 15360}};
    int tidx = get_global_id(0);
    int laneidx = get_local_id(0);
    bool debug = get_global_id(0) == {debug_tid};

    for (int i=0; i<4; i++) {{
      switch (laneidx) {{
        case 0: a[i] = 16384; break;
        case 1: a[i] = 16384; break;
        case 2: a[i] = 16384; break;
        case 3: a[i] = 16384; break;
      }}
    }}
    const unsigned *A = (const unsigned *)&a;
    const unsigned *B = (const unsigned *)&b;
    const float *C = (const float *)&c;
    float *D = (float *)&d;
    asm(
      "mma.sync.aligned.row.col.m8n8k4.f32.f16.f16.f32"
      "{{%0,%1,%2,%3,%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%11}}, {{%12,%13,%14,%15,%16,%17,%18,%19}};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
    );
    //if(debug) printf("tid=%d\n", tidx);
    for (int i = 0; i < 8; i++) {{
      //if(debug) printf("out[%d]=%f\n", tidx*8+i, D[i]);
      out[tidx*8+i] = D[i];
    }}
  }}
"""

prg = cl.Program(ctx, kernel_src).build()
print(prg.binaries[0].decode())

N = 32
#C = np.empty((8*N), dtype=np.float32)
C = np.empty((16, 16), dtype=np.float32)
C_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, C.nbytes)
prg.matmul(queue, (N,), (N,), C_buf)

# Copy the result back to the host
cl.enqueue_copy(queue, C, C_buf)

# Verify the result
print(C)
