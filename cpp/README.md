# C++ multithreaded port

A C++20 port of the Douglas ADI time loop (European put), multithreaded with
`std::jthread` and `std::barrier`. It implements the contract in
[`docs/phase1_summary.md`, Section 4](../docs/phase1_summary.md#4-handoff-to-phase-2-what-c-and-cuda-implement).

## Build

Needs CMake 3.20+ and a C++20 compiler with `<barrier>` (Apple clang 17 works on macOS).

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

## Check against Python

```bash
cpp/build/adi_port fixtures/small out/small --threads 8
python -m adi_heston.fixtures check fixtures/small out/small
```

All seven outputs (the six kernels and the full time loop) match Python **bit for bit**, with relative
error 0, on the `tiny`, `small` and `medium` cases at every thread count from 1 to 10. Two things make
that possible:

- every kernel follows the Python reference operation by operation, and
- `-ffp-contract=off` stops the compiler from fusing multiply-adds, which NumPy does not do either.

Each output value is computed by exactly one thread, so results do not depend on the thread count.

## Benchmark

```bash
python -m adi_heston.fixtures generate --grid 800 400 50       # inputs only, into bench_inputs/
cpp/build/adi_port bench_inputs/grid_801x401_N50 --bench --threads 8 --repeats 5
python scripts/bench_cpu.py                                    # full sweep: CSV and figures in results/
```

`--bench` times only the time loop, measured between barriers, so file reading and thread start-up are
excluded. It prints `threads,rows,cols,steps,median_ms_per_step,min_ms_per_step`.

## Layout

| File | Contents |
|---|---|
| `src/io.*` | Read and write the raw float64 `.bin` files |
| `src/kernels.*` | The six kernels, each working on a given range of grid lines; `thomas_line` is the per-line solver |
| `src/solver.*` | Row and column blocks, the `std::jthread` team, and the Douglas loop with three barriers per step |
| `src/main.cpp` | Command line: check mode and benchmark mode |

## How the threads share one Douglas step

Each worker owns a fixed block of rows and a fixed block of columns. The threads are created once per
run, and a `std::barrier` (about 2 µs at 8 threads on an M1 Pro) is the only synchronization:

1. **By rows:** `A0U`, `A1U`, `A2U` and `R1 = U + dtau*A U - c*A1U`; the owner of row 0 writes the boundary value.
2. **By columns:** solve `(I - c*A1) Y1 = R1` along s, sweeping row by row across the block.
3. **By rows:** `R2 = Y1 - c*A2U`, then solve `(I - c*A2) U = R2` along each row, in place.

There is a barrier after each phase.
