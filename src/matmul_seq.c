#include "matmul.h"

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
