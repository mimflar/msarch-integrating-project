# -fno-vectorize -fno-slp-vectorize stop the compiler from adding SIMD by itself,
# so the sequential version really is sequential.
# -Iinclude lets the .c files find the headers in include/.
CC = clang
CFLAGS = -O2 -Wall -Iinclude -fno-vectorize -fno-slp-vectorize

SRC = src/matmul_seq.c src/matmul_simd.c src/utils.c
HEADERS = include/matmul.h include/utils.h

BUILD_DIR = build
ifdef TILE
BUILD_DIR = build/tile-$(TILE)
TILE_CFLAGS = -DMATMUL_TILE_SIZE=$(TILE)
endif

all: $(BUILD_DIR)/bench $(BUILD_DIR)/test_matmul

# Benchmark program
$(BUILD_DIR)/bench: $(SRC) src/main.c $(HEADERS) Makefile
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TILE_CFLAGS) -DACCELERATE_NEW_LAPACK $(SRC) src/main.c -o $@ -framework Accelerate

# Test program
$(BUILD_DIR)/test_matmul: $(SRC) tests/test_matmul.c $(HEADERS) Makefile
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TILE_CFLAGS) $(SRC) tests/test_matmul.c -o $@

run: $(BUILD_DIR)/bench
	./$(BUILD_DIR)/bench

test: $(BUILD_DIR)/test_matmul
	./$(BUILD_DIR)/test_matmul

clean:
	rm -rf build

.PHONY: all run test clean
