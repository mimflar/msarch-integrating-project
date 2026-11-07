#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
Small example that we can check by hand

[1 2]   [5 6]   [19 22]
[3 4] x [7 8] = [43 50]
*/
int test_small()
{
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float result[4] = {19, 22, 43, 50};
    float C[4];

    matmul_seq(A, B, C, 2);
    for (int i = 0; i < 4; i++) {
        if (C[i] != result[i]) {
            printf("test failed at C[%d]: got %f, want %f\n", i, C[i], result[i]);
            return 0;
        }
    }
    printf("Small test passed\n");
    return 1;
}

// Space for one size x size matrix.
float *new_matrix(int size)
{
    return malloc(sizeof(float) * size * size);
}

// Random values between -1 and 1.
void fill_random(float *matrix, int count)
{
    for (int i = 0; i < count; i++)
        matrix[i] = (float)((double)rand() / RAND_MAX * 2.0 - 1.0);
}

int main()
{
    int size = 512; // matrix size
    int runs = 5; // timed runs, after one warm-up run

    if (!test_small())
        return 1;

    float *A = new_matrix(size);
    float *B = new_matrix(size);
    float *C = new_matrix(size);

    srand(67);
    fill_random(A, size * size);
    fill_random(B, size * size);

    matmul_seq(A, B, C, size);

    double total_time = 0;
    for (int run = 0; run < runs; run++) {
        clock_t start = clock();
        matmul_seq(A, B, C, size);
        total_time += (double)(clock() - start) / CLOCKS_PER_SEC;
    }

    double average_time = total_time / runs;
    double gflops = 2.0 * size * size * size / average_time / 1e9;
    printf("sequential, size = %d: average %.6f s over %d runs, %.2f GFLOPS\n",
           size, average_time, runs, gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}
