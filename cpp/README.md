# C++ multithreaded port

A C++20 port of the ADI time loop, multithreaded with `std::jthread` and `std::barrier`. It covers all four
schemes (Douglas, Craig–Sneyd, Modified Craig–Sneyd, Hundsdorfer–Verwer) for European and American puts.
It implements the contract in
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

Every fixture case passes **bit for bit** (relative error 0) at every thread count from 1 to 16:

- the base cases `tiny`, `small` and `medium` (Douglas, European): the six kernels, one step and the full run;
- the seven variant cases `small_<scheme>` and `small_american_<scheme>`: one step and the full run of each
  scheme, plus the multiplier `lam` for American options. These folders contain a `settings.bin`, which tells
  the program which scheme to run and whether the option is American.

Two things make bit-for-bit agreement possible:

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
excluded. It prints `threads,rows,cols,steps,median_ms_per_step,min_ms_per_step`. It runs the folder's problem:
the scheme and option type in its `settings.bin` (`generate --grid M N STEPS --scheme NAME --american`), or the
Douglas scheme for a European put if there is none.

## Layout

| File | Contents |
|---|---|
| `src/io.*` | Read and write the raw float64 `.bin` files |
| `src/kernels.*` | The six kernels and the pointwise American update, each working on a given range of grid lines; `thomas_line` is the per-line solver |
| `src/solver.*` | Row and column blocks, the `std::jthread` team, and `AdiSolver`, the time loop for all four schemes |
| `src/main.cpp` | Command line: check mode and benchmark mode |

## How the threads share one time step

Each worker owns a fixed block of rows and a fixed block of columns. The threads are created once per
run, and a `std::barrier` (about 2 µs at 8 threads on an M1 Pro) is the only synchronization. There is a
barrier after each phase.

**Douglas, 3 phases:**

1. **By rows:** `A0U`, `A1U`, `A2U`, the predictor `Y0 = U + dtau*A U` (plus `dtau*lam` for an American
   option), and `R1 = Y0 - c*A1U`. The owner of row 0 writes the boundary value.
2. **By columns:** solve `(I - c*A1) Y1 = R1` along s, sweeping row by row across the block.
3. **By rows:** `R2 = Y1 - c*A2U`, then solve `(I - c*A2) U = R2` along each row, in place. For an American
   option, the pointwise update follows.

**Craig–Sneyd, Modified Craig–Sneyd, Hundsdorfer–Verwer, 6 phases.** Phase 3 writes `Y2` instead of `U`, and
three more phases follow:

4. **By rows:** A applied to `Y2` (only `A0` for Craig–Sneyd), the scheme's corrected predictor, and `R1`
   for the second pass.
5. **By columns:** the second s-solve.
6. **By rows:** the second v-solve, into `U`, then the American update if any.

The first two half-steps of every run (the Rannacher start-up) always use Douglas.
