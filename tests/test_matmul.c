/*
Correctness tests (make test).
1. A 2x2 example checked by hand.
2. SIMD vs sequential on small and odd sizes, so the assembly kernel
   and the scalar leftover code both run.
*/
#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"
#include "utils.h"

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
Sizes 4 and up have full 4x4 blocks, so the assembly runs.
Odd sizes (5, 7, 67, ...) also run the scalar leftover code.
65 and 129 cross the 64 tile size.
*/
int test_sizes()
{
    int sizes[] = {1, 3, 4, 5, 7, 8, 11, 16, 63, 64, 65, 67, 128, 129};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        float *A = new_matrix(size);
        float *B = new_matrix(size);
        float *C_seq = new_matrix(size);
        float *C_simd = new_matrix(size);

        srand(67);
        fill_random(A, size * size);
        fill_random(B, size * size);

        matmul_seq(A, B, C_seq, size);
        matmul_simd_tiled(A, B, C_simd, size);
        int ok = verify_correctness(C_seq, C_simd, size);

        free(A);
        free(B);
        free(C_seq);
        free(C_simd);

        if (!ok) {
            printf("size %d failed\n", size);
            return 0;
        }
    }
    printf("Size tests passed (%d sizes)\n", num_sizes);
    return 1;
}

int main()
{
    if (!test_small() || !test_sizes())
        return 1;
    printf("All tests passed\n");
    return 0;
}
