# CUDA port

A CUDA port of the ADI time loop, written for the GPU from scratch. It runs all four schemes (Douglas, Craig–Sneyd,
Modified Craig–Sneyd, Hundsdorfer–Verwer) for European and American puts, and implements the same contract as the
C++ port ([`docs/phase1_summary.md`, Section 4](../docs/phase1_summary.md#4-handoff-to-phase-2-what-c-and-cuda-implement)).
Its exact build reproduces the Python reference byte for byte.

Author: Ganesh Arivoli <arivoli@wisc.edu>. Developed and measured on an RTX 3070 Ti (Ampere, sm_86) with CUDA 12.8.

## Build

Needs CMake 3.24+ and the CUDA 12 toolkit. CMake uses `/usr/local/cuda-12.8` when it exists; otherwise pass
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc`. The default architecture is sm_86 (`-DCMAKE_CUDA_ARCHITECTURES=...` to change).

```bash
cmake -S cuda -B cuda/build/default
cmake --build cuda/build/default -j
```

| Option (default OFF) | Effect |
|---|---|
| `ADI_EXACT` | No fused multiply-add, Python's operation order and divisions: every output is byte-identical to the fixtures |
| `ADI_RECOMPUTE_FACTORS` | Recompute the Thomas factors in every solve, as the C++ port does, instead of once at setup |
| `ADI_DEBUG_SYNC` | Synchronize and check for errors after every kernel launch |

Give each configuration its own folder under `cuda/build/`, for example
`cmake -S cuda -B cuda/build/exact -DADI_EXACT=ON`.

## Check against Python

```bash
python scripts/check_cuda.py --build cuda/build/default
```

It runs the port on every fixture case, generating `medium` and `large` if they are missing, each into a fresh folder.
Every output must exist (a missing one is a failure). It checks:

- the fixture outputs: within the fixture tolerances, or byte for byte for an `ADI_EXACT` build;
- the explicit stage's intermediates against NumPy, computed from the case's own files;
- the backward error of both tridiagonal solves;
- the number of factor sets (one per distinct `theta_adi * dtau`);
- for American cases: `U >= payoff`, `lam >= 0` and `lam * (U - payoff) = 0`.

All 50 fixture outputs of the 11 cases pass in every build configuration. One case by hand:

```bash
cuda/build/default/adi_cuda fixtures/small out/small
python -m adi_heston.fixtures check fixtures/small out/small
```

## Benchmark

```bash
python -m adi_heston.fixtures generate --grid 1600 800 50 --scheme hundsdorfer_verwer --american
cuda/build/default/adi_cuda bench_inputs/grid_1601x801_N50_hundsdorfer_verwer_american --bench --repeats 10
python scripts/bench_gpu.py      # full sweep against the C++ port: CSV and figures in results/
```

`--bench` runs the folder's problem (`settings.bin`): one warm-up and R timed runs of the whole schedule, timed with
CUDA events on the work stream, with `U` (and `lam`) reset on the device between runs. It prints
`gpu,rows,cols,steps,median_ms_per_step,min_ms_per_step,setup_ms,scheme,american`, per schedule row; `setup_ms`
(uploads and factorization) is paid once and not included in the per-row times.

`scripts/bench_gpu.py` sweeps 6 grids (101 x 51 to 3201 x 1601), the 4 schemes and both option types, and times the
C++ port on the same inputs (`adi_port --bench`, at 1, 8 and 16 threads). Pass `--cpu-baseline` a CSV from
`scripts/bench_cpu.py` to add NumPy to the figures.

## Layout

| File | Contents |
|---|---|
| `main.cu` | Command line: check mode and benchmark mode |
| `host/io.*` | Raw float64 `.bin` files; loading a fixture or benchmark folder, with load-time checks |
| `kernels/common.cuh` | `real` (double), error checking, `DeviceBuffer` (owning device memory), stream and event timer |
| `kernels/kernels.cuh` | Device copies of the operators, per-node stencils for A0, A1, A2, kernel declarations |
| `kernels/explicit.cu` | The explicit stage (K1), the Ikonen–Toivanen update, the probe kernel for single-kernel checks |
| `kernels/tridiag.cu` | Factorization and the Thomas solves along s and along v |
| `solver/schemes.cu` | The four schemes: kernel order per row, and the fused correction kernel of the corrected schemes |
| `solver/solver.*` | `AdiSolver`: one-time setup, the time loop, timing |

## How a run works on the GPU

1. **Load (host).** The folder's arrays, checked: every size matches the grid; A2's rows below the Dirichlet row are
   byte-identical (A2 does not depend on s), so A2 is kept as one row of coefficients per diagonal; every row of `U0`
   is constant, so the payoff is one value per s-node.
2. **Setup (once).** Every buffer is allocated once and the operators are uploaded. The tridiagonal systems are
   factored once per distinct value of `c = theta_adi * dtau`: 2D factors along s, two 1D sequences along v (the
   Dirichlet row and all other rows). The default build stores reciprocal pivots, so the solves never divide.
3. **Time loop.** The CPU only enqueues kernels on one stream: no copies, no waiting. Start-up rows use Douglas.
4. **Output.** One synchronization, then `U` (and `lam`) are copied back.

Kernels of one schedule row:

| Kernel | Threads | Work |
|---|---|---|
| explicit stage (K1) | one per node, 32 x 8 blocks along j | stencils on `U`, predictor `Y0 = U + dtau*A U (+ dtau*lam)`, right-hand side `Y0 - c*A1 U` |
| `solve_s` | one per column, 32-thread blocks | Thomas along s; a warp reads 32 neighbouring columns |
| `solve_v` | one per row, 32-thread blocks | Thomas along v, right-hand side `Y1 - c*A2 X` formed on the fly |
| correction (K4) | one per node | corrected schemes only: stencils on `Y2`, corrected predictor and the second right-hand side, in one pass |
| Ikonen–Toivanen | one per node | American only: `U = max(U - dtau*lam, g)`, `lam = max(0, lam + (g - U)/dtau)` |

Douglas runs K1, `solve_s`, `solve_v` (3 kernels per row); the corrected schemes add K4 and a second pair of solves
(6). American rows end with the update. Every kernel needs the whole result of the one before, so the parallelism is
inside each kernel. The node kernels keep the GPU full; the solves run one thread per grid line, which leaves most of
the GPU idle, so they take most of each row in this first version.

**Exactness.** The stencils, the explicit stage, the corrections and the Thomas sweeps follow the Python reference
operation by operation, including multiplying zero coefficients on the Dirichlet row (signed zeros) and NumPy's tie
rule for `max`. With `ADI_EXACT` (no fused multiply-add, divisions as in Python) the outputs are byte-identical to the
fixtures. The default build uses fused multiply-add and reciprocal pivots and agrees to about 1e-13.
