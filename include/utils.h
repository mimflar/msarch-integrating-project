#ifndef UTILS_H
#define UTILS_H

// Space for one size x size matrix.
float *new_matrix(int size);

// Random values between -1 and 1.
void fill_random(float *matrix, int count);

// Returns 1 if every element of C_simd is within 1e-3 of C_seq, else 0.
int verify_correctness(float *C_seq, float *C_simd, int size);

#endif
