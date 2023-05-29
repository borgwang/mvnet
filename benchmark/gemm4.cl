__kernel void matmul_op(
  int BS, int M, int N, int K,
  uint A_s0, uint B_s0, uint A_s1, uint B_s1, uint A_s2, uint B_s2,
  uint a_ofst, uint b_ofst,
  __global const float16 *A, __global const float16 *B, __global float16 *C
) {
  uint grpid1=get_group_id(1), grpid2=get_group_id(2);
  uint bs=get_global_id(0), i=get_local_id(1), j=get_local_id(2);

  float16 acc[2] = {(float16)(0),(float16)(0)};
  __local float16 Alcl[64][4], Blcl[64][4];
  for (uint t=0; t<K/64; t++) {  // loop over groups
    for (uint h=0; h<2; h++) {
      Alcl[i*2+h][j] = A[bs*A_s0/16 + (grpid1*64+(i*2+h))*A_s1/16 + (t*4+j)*A_s2 + a_ofst];
      Blcl[i*2+h][j] = B[bs*B_s0/16 + (grpid2*64+(i*2+h))*B_s1/16 + (t*4+j)*B_s2 + b_ofst];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float16 vecT; float *p_vecT;
    for (uint k=0; k<4; k++) {   // k: loop over vecs inside a group
      for (uint w=0; w<16; w++) {  // w: loop over elems inside a vec
        for (uint h=0; h<2; h++) {  // h: loop over WPT
          float16 vecT = Alcl[i*2+h][k] * Blcl[j*16+w][k]; float *p_vecT = &vecT;
          ((float*)&acc[h])[w] += p_vecT[0]+p_vecT[1]+p_vecT[2]+p_vecT[3]+p_vecT[4]+p_vecT[5]+p_vecT[6]+p_vecT[7]+p_vecT[8]+p_vecT[9]+p_vecT[10]+p_vecT[11]+p_vecT[12]+p_vecT[13]+p_vecT[14]+p_vecT[15];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (uint h=0; h<2; h++) C[bs*M*N/16 + (grpid1*64+i*2+h)*N/16 + (grpid2*4+j)] = acc[h];
}
