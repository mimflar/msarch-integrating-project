# -fno-vectorize -fno-slp-vectorize stop the compiler from adding SIMD by itself,
# so the sequential version really is sequential.
# -Iinclude lets the .c files find the headers in include/.
CC = clang
CFLAGS = -O2 -Wall -Iinclude -fno-vectorize -fno-slp-vectorize

SRC = src/matmul_seq.c src/matmul_simd.c src/utils.c
HEADERS = include/matmul.h include/utils.h

all: build/bench build/test_matmul

# Benchmark program
build/bench: $(SRC) src/main.c $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(SRC) src/main.c -o build/bench

# Test program
build/test_matmul: $(SRC) tests/test_matmul.c $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(SRC) tests/test_matmul.c -o build/test_matmul

run: build/bench
	./build/bench

test: build/test_matmul
	./build/test_matmul

clean:
	rm -rf build

.PHONY: all run test clean
