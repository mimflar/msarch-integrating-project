#ifndef MATMUL_H
#define MATMUL_H

/*
All matrices are size x size, 32-bit floats, row-major.
Both functions overwrite C with A x B.
*/

// Sequential baseline, i-k-j order, no SIMD.
void matmul_seq(float *A, float *B, float *C, int size);

// Handwritten ARM64 NEON assembly + cache tiling + 4x4 register block.
void matmul_simd_tiled(float *A, float *B, float *C, int size);

#endif
