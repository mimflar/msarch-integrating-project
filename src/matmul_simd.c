#include "matmul.h"

/*
handwritten arm64 neon + cache tiling implementation
*/
void matmul_simd_tiled(float *A, float *B, float *C, int size)
{
    int TILE = 64; // L1 cache block size
    int size4 = (size / 4) * 4; // largest multiple of 4

    // init C to zero
    for (int i = 0; i < size * size; i++) C[i] = 0;

    // cache tiling - outer loops traverse matrices in blocks
    for (int rb = 0; rb < size4; rb += TILE) {
        int r_end = (rb + TILE < size4) ? rb + TILE : size4;
        
        for (int cb = 0; cb < size4; cb += TILE) {
            int c_end = (cb + TILE < size4) ? cb + TILE : size4;
            
            for (int ib = 0; ib < size; ib += TILE) {
                int i_end = (ib + TILE < size) ? ib + TILE : size;
                
                // 4x4 register-blocked assembly micro-kernel
                for (int row = rb; row < r_end; row += 4) {
                    for (int col = cb; col < c_end; col += 4) {
                        
                        // pointers for a 4x4 block of C
                        float *c_ptr0 = &C[(row + 0) * size + col];
                        float *c_ptr1 = &C[(row + 1) * size + col];
                        float *c_ptr2 = &C[(row + 2) * size + col];
                        float *c_ptr3 = &C[(row + 3) * size + col];

                        // pointers for 4 rows of A
                        float *a_ptr0 = &A[(row + 0) * size + ib];
                        float *a_ptr1 = &A[(row + 1) * size + ib];
                        float *a_ptr2 = &A[(row + 2) * size + ib];
                        float *a_ptr3 = &A[(row + 3) * size + ib];

                        // pointer for 1 row of B
                        float *b_ptr = &B[ib * size + col];

                        long k_iters = i_end - ib;
                        long b_stride = size * sizeof(float); // bytes to jump down one row in B

                        __asm__ volatile (
                            // load the 4x4 block of C into accumulators v16-v19
                            "ld1 {v16.4s}, [%[c0]] \n"
                            "ld1 {v17.4s}, [%[c1]] \n"
                            "ld1 {v18.4s}, [%[c2]] \n"
                            "ld1 {v19.4s}, [%[c3]] \n"

                            "1: \n" // inner loop start (register reuse)
                            
                            // load 4 elements of B into v0, then move pointer down one row
                            "ld1 {v0.4s}, [%[b]] \n"
                            "add %[b], %[b], %[b_stride] \n" 

                            // load 1 scalar of A, broadcast to 4 lanes, increment pointer by 4 bytes
                            "ld1r {v1.4s}, [%[a0]], #4 \n"
                            "ld1r {v2.4s}, [%[a1]], #4 \n"
                            "ld1r {v3.4s}, [%[a2]], #4 \n"
                            "ld1r {v4.4s}, [%[a3]], #4 \n"

                            // multipl-add (accum into C registers)
                            "fmla v16.4s, v0.4s, v1.4s \n"
                            "fmla v17.4s, v0.4s, v2.4s \n"
                            "fmla v18.4s, v0.4s, v3.4s \n"
                            "fmla v19.4s, v0.4s, v4.4s \n"

                            // decrement tile counter, branch if not zero
                            "subs %[iters], %[iters], #1 \n"
                            "b.ne 1b \n"

                            // store accumulated 4x4 C block back to memory
                            "st1 {v16.4s}, [%[c0]] \n"
                            "st1 {v17.4s}, [%[c1]] \n"
                            "st1 {v18.4s}, [%[c2]] \n"
                            "st1 {v19.4s}, [%[c3]] \n"

                            // inputs & outputs
                            : [a0] "+r" (a_ptr0), [a1] "+r" (a_ptr1), [a2] "+r" (a_ptr2), [a3] "+r" (a_ptr3),
                              [b] "+r" (b_ptr), [iters] "+r" (k_iters)
                            : [c0] "r" (c_ptr0), [c1] "r" (c_ptr1), [c2] "r" (c_ptr2), [c3] "r" (c_ptr3),
                              [b_stride] "r" (b_stride)
                            : "v0", "v1", "v2", "v3", "v4", "v16", "v17", "v18", "v19", "memory", "cc"
                        );
                    }
                }
            }
        }
    }

    // scalar cleanup for edges and sizes not a multiple of 4
    for (int row = 0; row < size; row++) {
        for (int inner = 0; inner < size; inner++) {
            float a_value = A[row * size + inner];
            // skip the already computed 4x4 blocks
            int col_start = (row < size4) ? size4 : 0;
            for (int col = col_start; col < size; col++) {
                C[row * size + col] += a_value * B[inner * size + col];
            }
        }
    }
}
