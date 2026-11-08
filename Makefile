# -fno-vectorize -fno-slp-vectorize stop the compiler from adding SIMD by itself,
# so the sequential version really is sequential.
CFLAGS = -O2 -Wall -fno-vectorize -fno-slp-vectorize

matmul: matmul.c
	clang $(CFLAGS) matmul.c -o matmul

clean:
	rm -f matmul