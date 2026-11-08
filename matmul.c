#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
C = A x B, all size x size, row-major.
i-k-j order (row, inner, col): A[row][inner] multiplies the whole row
"inner" of B and adds it to row "row" of C.
*/
void matmul_seq(float *A, float *B, float *C, int size)
{
    // Initialize C to zero
    for (int i = 0; i < size * size; i++)
        C[i] = 0;

    // Perform matrix multiplication
    for (int row = 0; row < size; row++) {
        for (int inner = 0; inner < size; inner++) {
            float a_value = A[row * size + inner];
            for (int col = 0; col < size; col++)
                C[row * size + col] += a_value * B[inner * size + col];
        }
    }
}

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

/*
Small example that we can check by hand

[1 2]   [5 6]   [19 22]
[3 4] x [7 8] = [43 50]
*/
int test_small()
{
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float result[4] = {19, 22, 43, 50};
    float C_seq[4];
    float C_simd[4];

    matmul_seq(A, B, C_seq, 2);
    matmul_simd_tiled(A, B, C_simd, 2);

    for (int i = 0; i < 4; i++) {
        if (C_seq[i] != result[i]) {
            printf("seq test failed at C[%d]: got %f, want %f\n", i, C_seq[i], result[i]);
            return 0;
        }
        if (C_simd[i] != result[i]) {
            printf("simd test failed at C[%d]: got %f, want %f\n", i, C_simd[i], result[i]);
            return 0;
        }
    }
    printf("Small test passed\n");
    return 1;
}

/*
benchmarking below
*/

// space for one size x size matrix.
float *new_matrix(int size)
{
    return malloc(sizeof(float) * size * size);
}

// random values between -1 and 1.
void fill_random(float *matrix, int count)
{
    for (int i = 0; i < count; i++)
        matrix[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);
}

int verify_correctness(float *C_seq, float *C_simd, int size) 
{
    for (int i = 0; i < size * size; i++) {
        if (fabs(C_seq[i] - C_simd[i]) > 1e-3) {
            printf("Mismatch at index %d: Seq=%f, SIMD=%f\n", i, C_seq[i], C_simd[i]);
            return 0;
        }
    }
    return 1;
}

int main()
{
    if (!test_small())
        return 1;

    // sweep across multiple sizes (including non-multiples of 4 and TILE)
    int test_sizes[] = {64, 67, 128, 255, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int runs = 5; // timed runs, after one warm-up run

    for (int s = 0; s < num_sizes; s++) {
        int size = test_sizes[s];

        float *A = new_matrix(size);
        float *B = new_matrix(size);
        float *C_seq = new_matrix(size);
        float *C_simd = new_matrix(size);

        srand(67);
        fill_random(A, size * size);
        fill_random(B, size * size);

        // warm-up runs
        matmul_seq(A, B, C_seq, size);
        matmul_simd_tiled(A, B, C_simd, size);

        if (!verify_correctness(C_seq, C_simd, size)) {
            printf("Correctness check failed at size %d\n", size);
            free(A); free(B); free(C_seq); free(C_simd);
            return 1; 
        }

        // benchmark for sequential
        double total_time_seq = 0;
        for (int run = 0; run < runs; run++) {
            clock_t start = clock();
            matmul_seq(A, B, C_seq, size);
            total_time_seq += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_seq = total_time_seq / runs;

        // benchmark SIMD
        double total_time_simd = 0;
        for (int run = 0; run < runs; run++) {
            clock_t start = clock();
            matmul_simd_tiled(A, B, C_simd, size);
            total_time_simd += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_simd = total_time_simd / runs;

        // output
        double gflops_base = 2.0 * size * size * size / 1e9;
        printf("-----------------------------------------\n");
        printf("Matrix Size : %d x %d\n", size, size);
        printf("Sequential  : %.6f s (%.2f GFLOPS)\n", avg_seq, gflops_base / avg_seq);
        printf("SIMD + Tile : %.6f s (%.2f GFLOPS)\n", avg_simd, gflops_base / avg_simd);
        printf("Speedup     : %.2fx\n", avg_seq / avg_simd);

        free(A); 
        free(B); 
        free(C_seq); 
        free(C_simd);
    }
    printf("-----------------------------------------\n");
    return 0;
}