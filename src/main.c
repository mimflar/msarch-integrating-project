/*
Benchmark: sequential, tiled SIMD, and Apple Accelerate across matrix sizes.
Correctness tests are in tests/test_matmul.c (make test).
*/
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul.h"
#include "utils.h"

static void matmul_accelerate(float *A, float *B, float *C, int size)
{
    // beta = 0 tells SGEMM to overwrite C with A x B.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f, A, size, B, size, 0.0f, C, size);
}

static double average_cpu_time(void (*matmul)(float *, float *, float *, int),
                               float *A, float *B, float *C, int size, int runs)
{
    double total_time = 0;
    for (int run = 0; run < runs; run++) {
        clock_t start = clock();
        matmul(A, B, C, size);
        total_time += (double)(clock() - start) / CLOCKS_PER_SEC;
    }
    return total_time / runs;
}

int main()
{
    // sweep across multiple sizes (including non-multiples of 4 and TILE)
    int test_sizes[] = {64, 67, 128, 255, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int runs = 5; // timed runs, after one warm-up run

    puts("Average CPU time per multiplication");

    for (int s = 0; s < num_sizes; s++) {
        int size = test_sizes[s];

        float *A = new_matrix(size);
        float *B = new_matrix(size);
        float *C_seq = new_matrix(size);
        float *C_simd = new_matrix(size);
        float *C_accelerate = new_matrix(size);

        if (!A || !B || !C_seq || !C_simd || !C_accelerate) {
            fprintf(stderr, "Matrix allocation failed at size %d\n", size);
            free(A);
            free(B);
            free(C_seq);
            free(C_simd);
            free(C_accelerate);
            return 1;
        }

        srand(67);
        fill_random(A, size * size);
        fill_random(B, size * size);

        // warm-up runs
        matmul_seq(A, B, C_seq, size);
        matmul_simd_tiled(A, B, C_simd, size);
        matmul_accelerate(A, B, C_accelerate, size);

        int simd_correct = verify_correctness(C_seq, C_simd, size);
        int accelerate_correct = verify_correctness(C_seq, C_accelerate, size);
        if (!simd_correct || !accelerate_correct) {
            if (!simd_correct)
                fprintf(stderr, "SIMD correctness check failed at size %d\n", size);
            if (!accelerate_correct)
                fprintf(stderr, "Accelerate correctness check failed at size %d\n", size);
            free(A);
            free(B);
            free(C_seq);
            free(C_simd);
            free(C_accelerate);
            return 1;
        }

        double avg_seq = average_cpu_time(matmul_seq, A, B, C_seq, size, runs);
        double avg_simd = average_cpu_time(matmul_simd_tiled, A, B, C_simd, size, runs);
        double avg_accelerate = average_cpu_time(matmul_accelerate, A, B, C_accelerate, size, runs);

        // output
        double gflops_base = 2.0 * size * size * size / 1e9;
        printf("-----------------------------------------\n");
        printf("Matrix Size : %d x %d\n", size, size);
        printf("Sequential  : %.6f s (%.2f GFLOPS)\n", avg_seq, gflops_base / avg_seq);
        printf("SIMD + Tile : %.6f s (%.2f GFLOPS)\n", avg_simd, gflops_base / avg_simd);
        printf("Accelerate  : %.6f s (%.2f GFLOPS)\n", avg_accelerate, gflops_base / avg_accelerate);
        printf("Speedup     : %.2fx\n", avg_seq / avg_simd);

        free(A); 
        free(B); 
        free(C_seq); 
        free(C_simd);
        free(C_accelerate);
    }
    printf("-----------------------------------------\n");
    return 0;
}
