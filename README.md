# SIMD-Accelerated Matrix Multiplication for Neural-Network Workloads

CSC612M Integrating Project · Group 7

We multiply two square matrices (`C = A × B`) of single-precision floats and compare three implementations.

1. **Sequential.** Plain C loops in `i-k-j` order. The build disables compiler auto-vectorization for this baseline.
2. **SIMD + Tile.** Handwritten ARM64 NEON assembly updates four values per vector instruction. It processes the matrices in cache tiles, keeps a `4 × 4` block of `C` in registers during accumulation, and uses scalar C for leftover rows and columns.
3. **Apple Accelerate.** Apple's `cblas_sgemm` provides a library performance reference for the same multiplication.

## What you need

- A Mac with Apple Silicon (M1 or newer) and macOS 13.3 or newer
- Xcode Command Line Tools. To install them, run:

  ```bash
  xcode-select --install
  ```

## How to run

Run these commands from the project folder.

```bash
make        # build everything into build/
make test   # run the hand-checked case and compare NEON with sequential C
make run    # run the benchmark
```

The benchmark checks NEON and Accelerate against the sequential result before timing. It reports average CPU time over five runs after one warm-up, plus throughput in GFLOPS. The `Speedup` line compares NEON with sequential C. The timings below are from the recorded run with the default tile size.

```text
Average CPU time per multiplication
NEON tile size: 64
-----------------------------------------
Matrix Size : 1024 x 1024
Sequential  : 0.273318 s (7.86 GFLOPS)
SIMD + Tile : 0.060193 s (35.68 GFLOPS)
Accelerate  : 0.002258 s (950.97 GFLOPS)
Speedup     : 4.54x
-----------------------------------------
```

The benchmark uses matrix sizes `64`, `67`, `128`, `255`, `512`, and `1024`. To try others, edit `test_sizes[]` in [`src/main.c`](src/main.c) and run `make run` again.

## Reproduce the tile-size sweep

`make run` uses a default NEON tile size of `64 × 64`. Use `TILE` to select another positive multiple of four. Each override builds into its own directory under `build/`.

```bash
for tile in 32 64 128 256; do
  make run TILE="$tile" || break
done
```

Compare the `SIMD + Tile` CPU times for each matrix size. Use `make test TILE=32` to run the correctness tests at tile size `32`, or substitute another tile size.

In the recorded sweep, tile `32` had the lowest CPU time at the three largest matrix sizes. At `1024 × 1024` it took `55.003 ms`, an `8.6%` reduction from tile `64`. Tile `128` had the lowest printed times at the three smaller sizes, tying tile `256` at `64 × 64`. The sweep used one benchmark invocation per tile. See [`results.txt`](results.txt) for the full tile comparison and the Accelerate results at tile `64`.

## Project structure

```
include/
  matmul.h          declarations of the two multiply functions
  utils.h           declarations of the helper functions
src/
  matmul_seq.c      sequential version (i-k-j loops)
  matmul_simd.c     NEON assembly + cache tiling version
  utils.c           new_matrix, fill_random, verify_correctness
  main.c            benchmark with the Apple Accelerate reference
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
| Benchmark against Apple Accelerate reference | Done |
| Compare NEON tile sizes 32, 64, 128, and 256 | Done |
| Benchmark matrix sizes larger than 1024 × 1024 | To do |
