/*
Benchmark: sequential vs SIMD + tiling across matrix sizes.
Correctness tests are in tests/test_matmul.c (make test).
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul.h"
#include "utils.h"

int main()
{
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