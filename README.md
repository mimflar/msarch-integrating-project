# SIMD-Accelerated Matrix Multiplication for Neural-Network Workloads

CSC612M Integrating Project · Group 7

We multiply two square matrices (C = A × B) in two ways and compare the speed:

1. **Sequential** – plain C loops, one multiply at a time.
2. **SIMD** – handwritten ARM64 NEON assembly, four multiplies at a time, with cache tiling and a 4×4 block of C kept in registers.

## What you need

- A Mac with Apple Silicon (M1 or newer)
- Xcode Command Line Tools. To install them, run:

  ```bash
  xcode-select --install
  ```

## How to run

From the project folder:

```bash
make        # build everything into build/
make test   # check that SIMD gives the same results as sequential
make run    # run the benchmark
```

Expected benchmark output (numbers vary a little between runs):

```
-----------------------------------------
Matrix Size : 64 x 64
Sequential  : 0.000184 s (2.85 GFLOPS)
SIMD + Tile : 0.000034 s (15.33 GFLOPS)
Speedup     : 5.38x
-----------------------------------------
Matrix Size : 67 x 67
...
```

To try other matrix sizes, change the `test_sizes[]` array in `src/main.c`, then run `make run` again.

## Project structure

```
include/
  matmul.h          declarations of the two multiply functions
  utils.h           declarations of the helper functions
src/
  matmul_seq.c      sequential version (i-k-j loops)
  matmul_simd.c     NEON assembly + cache tiling version
  utils.c           new_matrix, fill_random, verify_correctness
  main.c            benchmark
tests/
  test_matmul.c     correctness tests
Makefile            build commands
```

## Progress

| Task | Status |
|---|---|
| Sequential multiplication with a hand-checked test and timing | Done |
| ARM64 NEON assembly version | Done |
| Cache blocking and register reuse | Done |
| Check that SIMD output matches sequential output | Done |
| Benchmarks across matrix sizes | Done |
| Handle matrix sizes that are not multiples of 4 or the tile size | Done |
| Benchmark against Apple Accelerate reference | To do |
| Tune cache block size and test on massive matrices | To do |
