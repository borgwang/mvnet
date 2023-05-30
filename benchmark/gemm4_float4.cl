    __kernel void matmul_op(
      int BS, int M, int N, int K,
      uint A_s0, uint B_s0, uint A_s1, uint B_s1, uint A_s2, uint B_s2,
      uint a_ofst, uint b_ofst,
      __global const float4 *A, __global const float4 *B, __global float4 *C
    ) {
      uint grpid1=get_group_id(1), grpid2=get_group_id(2);
      uint bs=get_global_id(0), i=get_local_id(1), j=get_local_id(2);

      float4 acc[2] = {(float4)(0),(float4)(0)};
      __local float4 Alcl[32][8], Blcl[32][8];
      for (uint t=0; t<K/32; t++) {  // loop over groups
        for (uint h=0; h<2; h++) {
          Alcl[i*2+h][j] = A[bs*A_s0/4 + (grpid1*32+(i*2+h))*A_s1/4 + (t*8+j)*A_s2 + a_ofst];
          Blcl[i*2+h][j] = B[bs*B_s0/4 + (grpid2*32+(i*2+h))*B_s1/4 + (t*8+j)*B_s2 + b_ofst];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k=0; k<8; k++) {   // k: loop over vecs inside a group
          for (uint w=0; w<4; w++) {  // w: loop over elems inside a vec
            for (uint h=0; h<2; h++) {  // h: loop over WPT
              ((float*)&acc[h])[w] += dot(Alcl[i*2+h][k], Blcl[j*4+w][k]);
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      for (uint h=0; h<2; h++) C[bs*M*N/4 + (grpid1*32+i*2+h)*N/4 + (grpid2*8+j)] = acc[h];
    }
