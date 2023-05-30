from mvnet.backend.opencl import CLArray, cl

cl.build("matmul_op", f"""
  __kernel void matmul_op(
    int BS, int M, int N, int K,
    {''.join(f'int A_s{i}, int B_s{i}, ' for i in range(3))}
    int a_ofst, int b_ofst,
    __global const float *A, __global const float *B, __global float *C
  ) {{
    //uint grpid1=get_group_id(1), grpid2=get_group_id(2);
    uint s=get_local_size(0);
    //C[bs*M*N + (grpid1*64+i)*N + (grpid2*64+j)] = 0.0f;
    C[s] = 0.0f;
  }}""")

"""

.entry matmul_op(
        .param .u32 matmul_op_param_0,
        .param .u32 matmul_op_param_1,
        .param .u32 matmul_op_param_2,
        .param .u32 matmul_op_param_3,
        .param .u32 matmul_op_param_4,
        .param .u32 matmul_op_param_5,
        .param .u32 matmul_op_param_6,
        .param .u32 matmul_op_param_7,
        .param .u32 matmul_op_param_8,
        .param .u32 matmul_op_param_9,
        .param .u32 matmul_op_param_10,
        .param .u32 matmul_op_param_11,
        .param .u64 .ptr .global .align 4 matmul_op_param_12,
        .param .u64 .ptr .global .align 4 matmul_op_param_13,
        .param .u64 .ptr .global .align 4 matmul_op_param_14
)
{
        .reg .b32       %r<24>;
        .reg .b64       %rd<4>;

        ld.param.u32    %r1, [matmul_op_param_1];  // M
        ld.param.u32    %r2, [matmul_op_param_2];  // N
        ld.param.u64    %rd1, [matmul_op_param_14]; // C
        mov.b32 %r3, %envreg1;
        mov.u32         %r4, %ctaid.y;
        add.s32         %r5, %r4, %r3;
        mov.b32 %r6, %envreg2;
        mov.u32         %r7, %ctaid.z;
        add.s32         %r8, %r7, %r6;
        mov.b32 %r9, %envreg3;
        mov.u32         %r10, %ntid.x;
        mov.u32         %r11, %ctaid.x;
        mad.lo.s32      %r12, %r11, %r10, %r9;
        mov.u32         %r13, %tid.x;
        add.s32         %r14, %r12, %r13;
        mov.u32         %r15, %tid.y;
        mov.u32         %r16, %tid.z;
        shl.b32         %r17, %r5, 6;
        add.s32         %r18, %r15, %r17;
        mad.lo.s32      %r19, %r14, %r1, %r18;   %r14*M+%r18
        shl.b32         %r20, %r8, 6;
        add.s32         %r21, %r16, %r20;
        mad.lo.s32      %r22, %r19, %r2, %r21;
        mul.wide.u32    %rd2, %r22, 4;
        add.s64         %rd3, %rd1, %rd2;
        mov.u32         %r23, 0;
        st.global.u32   [%rd3], %r23;
        ret;
}
"""

# block_id (get_group_id) -> ctaid
# thread_id (get_local_id) -> tid
# group_size (get_local_size)-> ntid
# global_id (get_global_id) -> ctaid * ntid + tid
