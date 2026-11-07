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
sequential, size = 512: average 0.033277 s over 5 runs, 8.07 GFLOPS
```

To try another matrix size, change `int size = 512;` in `main` (in `matmul.c`), then run `make` and `./matmul` again.

## Progress

| Task | Status |
|---|---|
| Sequential multiplication with a hand-checked test and timing | Done |
| ARM64 NEON assembly version | To do |
| Cache blocking and register reuse | To do |
| Check that SIMD output matches sequential output | To do |
| Benchmarks across matrix sizes | To do |
