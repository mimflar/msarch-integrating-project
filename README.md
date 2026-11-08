# SIMD-Accelerated Matrix Multiplication for Neural-Network Workloads

CSC612M Integrating Project · Group 7

We multiply two square matrices (C = A × B) in two ways and compare the speed:

1. **Sequential** – plain C loops, one multiply at a time.
2. **SIMD** – handwritten ARM64 NEON assembly, four multiplies at a time.

## What you need

- A Mac with Apple Silicon (M1 or newer)
- Xcode Command Line Tools. To install them, run:

  ```bash
  xcode-select --install
  ```

## How to run

From the project folder:

```bash
make        # build the program
./matmul    # run it
```

Expected output (numbers vary a little between runs):

```
Small test passed
-----------------------------------------
Matrix Size : 64 x 64
Sequential  : 0.000229 s (2.29 GFLOPS)
SIMD + Tile : 0.000057 s (9.26 GFLOPS)
Speedup     : 4.04x
----------------------------------------- 
Matrix Size : 67 x 67
...
```

To try another matrix size, change the `test_sizes[]` array in `main` (in `matmul.c`), then run `make` and `./matmul` again.

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