#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

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

int verify_correctness(float *C_seq, float *C_actual, int size)
{
    for (int i = 0; i < size * size; i++) {
        if (fabs(C_seq[i] - C_actual[i]) > 1e-3) {
            printf("Mismatch at index %d: Seq=%f, Actual=%f\n", i, C_seq[i], C_actual[i]);
            return 0;
        }
    }
    return 1;
}
