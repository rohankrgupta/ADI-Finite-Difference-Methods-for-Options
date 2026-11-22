# Phase 1: Python reference solver

**Last updated November 21, 2026.** This document records what the Python solver does, the design decisions behind it, and
how it was validated. Section 4 is the handoff to Phase 2: exactly what stays in Python and what the C++ (CPU threads)
and CUDA (GPU) versions implement.

The mathematical background (the PDE, the boundary conditions and why each piece is there) is in
[`formulation.pdf`](formulation.pdf). This document assumes it.

---

## 1. What exists

| Module (`python/adi_heston/`) | Purpose | Tests (`python/tests/`) |
|---|---|---|
| `params.py` | `HestonParams(r, q, kappa, theta, xi, rho)` and `OptionSpec(K, T, american)` with the put payoff | `test_params.py` |
| `grid.py` | Non-uniform grids: `stock_grid`, `variance_grid`, and `Grid` (with `Grid.build(...)` and `.mesh()`) | `test_grid.py` |
| `fd.py` | Finite-difference weights on non-uniform grids, and the mixed-derivative stencil | `test_fd.py` |
| `tridiag.py` | `solve_tridiagonal`: batched Thomas algorithm along any axis | `test_tridiag.py` |
| `operators.py` | The Heston operator split `A = A0 + A1 + A2` as coefficient arrays, with all boundaries | `test_operators.py` |
| `solver.py` | Douglas ADI time stepping, Rannacher start-up, and the price read-off | `test_solver.py` |
| `reference.py` | Independent prices: Black–Scholes (exact) and Heston (semi-analytic) | `test_reference.py` |
| `fixtures.py` | Writes the saved Python outputs for the C++ and CUDA ports, and checks a port against them (Section 4.5) | `test_fixtures.py` |

Setup is in the [README](../README.md). The whole suite is 137 tests and runs in about 9 seconds:

```bash
pytest
```

### What it can price now

**European and American puts under the Heston model**, with any of the four ADI schemes: `douglas`, `craig_sneyd`,
`modified_craig_sneyd` and `hundsdorfer_verwer`.

```python
from adi_heston import HestonParams, OptionSpec
from adi_heston.grid import Grid
from adi_heston.solver import solve

params = HestonParams(r=0.025, q=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
option = OptionSpec(K=100.0, T=1.0)
grid = Grid.build(K=100.0, S_max=800.0, V_max=5.0, m=100, n=50)   # 101 x 51 nodes

result = solve(params, option, grid, n_steps=100)
result.price(S0=100.0, v0=0.04)   # 6.4198 (semi-analytic reference: 6.4259)
result.U                          # the full solution at tau = T, shape (101, 51)
```

An American put is the same call with `american=True`. For example, on the published benchmark:

```python
from adi_heston.reference import AMERICAN_BENCHMARK_OPTION, AMERICAN_BENCHMARK_PARAMS

grid = Grid.build(K=10.0, S_max=80.0, V_max=5.0, m=100, n=50)
american = solve(AMERICAN_BENCHMARK_PARAMS, AMERICAN_BENCHMARK_OPTION, grid, n_steps=50, scheme="modified_craig_sneyd")
american.price(S0=10.0, v0=0.0625)   # 0.5192 (published: 0.5199 and 0.5202)
american.lam                         # the Ikonen-Toivanen multiplier; lam > 0 marks the exercise region
```

---

## 2. Validation results

Every row below is an automated test.

| Check | Result |
|---|---|
| **Black–Scholes reduction** (xi = 0, v0 = theta, S0 = K) against the exact formula | Error 1.9e-2, 4.8e-3, 1.2e-3, 3.0e-4 as the grid and step count double. That is 4.00x per doubling, so second order. |
| **Heston reference price** against a published benchmark (Fang & Oosterlee; Feller condition fails) | 5.7851554344 against 5.785155450, a difference of 1.6e-8 |
| **ADI against the Heston reference**, rho = 0 | Error 2.5e-2, 6.1e-3, 1.5e-3 on 51x26, 101x51, 201x101: **second order** |
| **ADI against the Heston reference**, rho = -0.9, 1000 steps | Error 2.9e-2, 6.9e-3, 1.7e-3: **second order in space** |
| **ADI on the Feller-failing benchmark** | Error 1.1e-2 at 101x51 and 3.0e-3 at 201x101: still about second order |
| **Order in time** (fixed grid, step count doubled) | **2.0** for rho = 0. **0.89, 0.95, 0.98** for rho = -0.9: Douglas is first order in time when the mixed term is present, as theory predicts. |
| **Rannacher start-up**, Gamma near K (T = 0.25, 10 steps) | Gamma error 0.44 without smoothing (larger than Gamma itself), 0.0012 with it |
| **Craig–Sneyd, Modified Craig–Sneyd, Hundsdorfer–Verwer** | Second order in time with rho = -0.9 (measured over the grid against a 640-step run), and second order in space against the Heston reference |
| **American puts** against the published benchmark (Ikonen & Toivanen; Zvan, Forsyth & Vetzal), 10 prices | Worst error against either source, all four schemes: 3.3e-3 at 51x26, 9.7e-4 at 101x51, 3.7e-4 at 201x101. The two sources differ from each other by up to 3e-4. |
| **American structure** | U >= payoff and U >= European at every node; lambda >= 0 and lambda*(U - payoff) = 0 exactly; the exercise region is one block per v-line, below K, moving to lower s as v rises; with r = 0, American equals European to 6e-13 |
| **Finite differences** on the sinh grid | Observed order about 2.0 for u', u'' and u_sv. On a deliberately jumpy grid, u'' drops to order 1.00, which confirms why the grids are smooth. |

---

## 3. Design decisions

### 3.1 Problem setup

- **Coordinates:** we solve in (s, v) directly, not in log-price. Log-price would send s = 0 to minus infinity and lose the
  exact boundary condition there.
- **Domain:** the box [0, S_max] x [0, V_max]. Most tests use S_max = 8K and V_max = 5.
- **Array layout:** `U[i, j] = u(s_i, v_j)`, shape `(m+1, n+1)`. Axis 0 is s and axis 1 is v. Everything is float64.
- **Names:** `theta` is the Heston long-run variance. The ADI weight is `theta_adi`. The papers call both theta, so keep
  them apart in code.

### 3.2 Grids (`grid.py`)

- **Smooth sinh stretch:** `s_i = K + c*sinh(xi_i)` and `v_j = d*sinh(eta_j)`, with evenly spaced helper variables.
  Neighbouring spacings change gradually, which keeps central second derivatives second order.
- **K is exactly a grid node**, and **S_max is hit exactly.** The helper step is very slightly different on each side of K
  (3% at m = 50); the difference goes to zero as the grid is refined.
- **Default clustering:** c = K/5 and d = V_max/500. These are inputs, so they can be varied in experiments.

### 3.3 Finite differences (`fd.py`)

- **Three-point weights** come from fitting a parabola through neighbouring nodes, so they are exact for quadratics.
  They are stored as `(N+1, 3)` arrays for the (left, centre, right) neighbours, with the end rows zero.
- **Mixed derivative u_sv** is the s-derivative of the v-derivative. Its weight on `U[i+k, j+l]` is
  `w_s[i, k+1] * w_v[j, l+1]`. On a non-uniform grid all 9 points of the 3x3 block are used; on an even grid only the
  4 corners would be.

### 3.4 Operators (`operators.py`)

```
A1 (s-direction):  1/2 s^2 v u_ss + (r - q) s u_s        - r/2 u
A2 (v-direction):  1/2 xi^2 v u_vv + kappa(theta - v) u_v - r/2 u
A0 (mixed):        rho xi s v u_sv
```

- **Storage:** stencil coefficient arrays, not sparse matrices. `a1` and `a2` have shape `(3, m+1, n+1)` and hold the
  (lower, diagonal, upper) coefficients along axis 0 and axis 1 respectively. `a0_coef` is `rho*xi*s*v` on the grid.
- **Coefficients do not depend on time.** They are built once per grid and parameter set.
- **The reaction term -r u is split equally** between A1 and A2.
- **Upwinding rule:** convection uses central differences wherever that keeps both off-diagonal coefficients
  non-negative (cell Peclet number at most 2). Elsewhere it uses a first-order upwind difference, taken from the side
  that information flows from. On the test case this happens in two places, and both are far from v0 = 0.04, where we
  read off the price:

  | Grid | v-direction upwinding starts at | s-direction upwinding |
  |---|---|---|
  | 51 x 26 | v of about 0.32 | only for v < 0.014 |
  | 101 x 51 | v of about 0.55 | only for v < 0.014 |
  | 201 x 101 | v of about 0.95 | only for v < 0.014 |

- **Why this matters for the solver:** with that rule, every A1 and A2 row has off-diagonals >= 0 and sums to -r/2.
  So `(I - c*A_j)` is strictly diagonally dominant for every c > 0, and **the Thomas algorithm is stable without
  pivoting.** The C++ and CUDA solvers can rely on this.

### 3.5 Boundaries, written into the arrays

Each operator handles only its own direction's edges, and A0 is zero on every edge.

| Edge | Condition | How it appears in the arrays |
|---|---|---|
| s = 0 (row 0) | Dirichlet: K e^{-r tau} (European) or K (American) | All coefficients in row 0 are zero in A0, A1 and A2. **Row 0 of U holds the boundary value.** |
| s = S_max (row m) | Neumann u_s = 0 | Mirror ghost node. A1: lower = s^2 v / h^2, diag = -lower - r/2, upper = 0. |
| v = 0 (column 0) | none: the PDE itself | A2: 2-point forward difference for kappa*theta*u_v. A1 has no diffusion there, so it upwinds automatically. |
| v = V_max (column n) | Neumann u_v = 0 | Mirror ghost node. A2: lower = xi^2 V_max / h^2, diag = -lower - r/2, upper = 0. |

The corners follow from these rules: the Dirichlet row wins at (0, 0) and (0, n).

### 3.6 Time stepping (`solver.py`)

**One Douglas step,** from U at the old time to the new time:

```
A0U, A1U, A2U = apply_A0(U), apply_A1(U), apply_A2(U)
Y0 = U + dtau * (A0U + A1U + A2U)              # (a) explicit predictor
c  = theta_adi * dtau
R1 = Y0 - c * A1U
R1[0, :] = boundary value at the new time      # the only boundary line
Y1 = solve (I - c*A1) Y1 = R1, along s         # (b) one tridiagonal system per v-line
U_new = solve (I - c*A2) U_new = Y1 - c * A2U  # (c) one tridiagonal system per s-line
```

- **The Dirichlet row needs no special code** beyond that one line. Its coefficients are zero, so stages (a) and (c)
  leave it unchanged, and stage (b) sets it.
- **Cost per step:** 3 stencil sweeps plus 2 batched tridiagonal solves.
- **Rannacher start-up:** the first full step is replaced by two half-steps of Douglas with `theta_adi = 1`. This damps the
  error from the payoff kink, and it does not lower the overall order.
- **Order in time:** first order when rho != 0 (the mixed term only gets an explicit Euler step), second order when rho = 0.

### 3.7 Price read-off

A bicubic spline (`scipy RectBivariateSpline`) through the final grid values, evaluated at (S0, v0). Its error is
O(h^4), so it stays well below the solver's O(h^2) error.

### 3.8 Reference prices (`reference.py`)

- **Black–Scholes:** exact put and Gamma formulas.
- **Heston:** the semi-analytic put `P = K e^{-rT}(1 - P2) - S0 e^{-qT}(1 - P1)`, where P1 and P2 are each one numerical
  integral of the characteristic function.
  - It uses the **"little Heston trap"** form (Albrecher et al., 2007), which avoids complex-log branch jumps at long maturities.
  - The put is computed directly rather than as call minus parity, which avoids cancellation for out-of-the-money puts.
  - For `xi = 0` it falls back to an exact Black–Scholes price with the time-averaged variance.
  - Accuracy is about 1e-9. It is a reference, not a fast pricer.

---

## 4. Handoff to Phase 2: what C++ and CUDA implement

### 4.1 The split

**Python does all the setup. The compiled code only runs the time loop.** The time loop is where all the run time goes,
so this keeps the performance comparison fair while keeping the compiled code small.

| Stays in Python | Done by the compiled code (C++ and CUDA) |
|---|---|
| Grid construction | the time loop |
| Parameters, payoff, initial values | the six kernels in Section 4.3 |
| Building the coefficient arrays (`build_heston_operators`) | |
| Rannacher start-up logic (it becomes entries in the step schedule) | |
| Price read-off (spline), Greeks, plots | |

The compiled code knows no finance: it never sees payoffs, grids or the Rannacher rule.

### 4.2 Inputs and outputs

| Array | Shape | Meaning |
|---|---|---|
| `a1` | (3, m+1, n+1) | A1 (lower, diag, upper) along axis 0 |
| `a2` | (3, m+1, n+1) | A2 (lower, diag, upper) along axis 1 |
| `a0_coef` | (m+1, n+1) | rho*xi*s*v, zero on all edges |
| `w_s` | (m+1, 3) | first-derivative weights in s (rows 0 and m are zero) |
| `w_v` | (n+1, 3) | first-derivative weights in v (rows 0 and n are zero) |
| `U0` | (m+1, n+1) | starting values (the payoff, with the boundary value in row 0) |
| schedule | list of (dtau, theta_adi, boundary_value) | one entry per step, in order |
| **output** `U` | (m+1, n+1) | values after the last step |

- **Memory layout:** row-major (C order), float64. Node (i, j) is at position `i*(n+1) + j`. **v-lines are contiguous; s-lines are strided.**
- **The schedule** for `n_steps = N` with the default start-up is two entries of `(T/(2N), 1.0, K e^{-r*k*T/(2N)})` for
  k = 1, 2, followed by N - 1 entries of `(T/N, 0.5, K e^{-r*k*T/N})` for k = 2..N. It is built by
  `solver.time_schedule(...)`, which `solve()` itself uses, so Python and the ports run exactly the same steps.

### 4.3 The six kernels

| Kernel | Definition |
|---|---|
| `apply_A1(U)` | `out[i,j] = a1[0,i,j]*U[i-1,j] + a1[1,i,j]*U[i,j] + a1[2,i,j]*U[i+1,j]` (out-of-range terms have zero coefficients; skip them) |
| `apply_A2(U)` | the same along j, with `a2` |
| `apply_A0(U)` | `out[i,j] = a0_coef[i,j] * sum over k,l in {-1,0,1} of w_s[i,k+1]*w_v[j,l+1]*U[i+k,j+l]` for interior nodes; 0 on all edges |
| `solve_A1(c, R)` | for each column j, solve the tridiagonal system with sub `-c*a1[0,:,j]`, main `1 - c*a1[1,:,j]`, super `-c*a1[2,:,j]`, right-hand side `R[:,j]` |
| `solve_A2(c, R)` | for each row i, the same along j with `a2` |
| `douglas_step(U, dtau, theta_adi, b)` | the composition in Section 3.6 |

The reference implementations are `HestonOperators.apply_A0/A1/A2`, `HestonOperators.solve_A1/A2` and
`solver.douglas_step`. When in doubt, the Python code defines the contract.

### 4.4 Performance notes for the ports

- **Every line is independent.** All s-lines (or all v-lines) can be solved at the same time: split them across CPU threads,
  or give each line its own GPU thread.
- **Memory access differs by direction:**
  - **C++:** solve the s-lines by sweeping row by row across all columns at once (as the NumPy version does). This stays cache-friendly.
  - **CUDA:** with one thread per line, `solve_A1` reads memory efficiently (neighbouring threads read neighbouring
    addresses), but `solve_A2` is strided. Speeding up `solve_A2` (for example by transposing, or by using a different
    tridiagonal algorithm) is the main GPU performance problem.
- **The tridiagonal matrices never change.** The start-up steps use `1.0 * dtau/2` and the main steps use `0.5 * dtau`,
  which give the same `c`. So the elimination factors can be computed once and reused for the whole run.
- **Python baseline:** on a 201 x 101 grid, one apply plus both solves takes about 1.5 ms in NumPy.

### 4.5 Checking a port against Python

Saved Python inputs and expected outputs live in [`fixtures/`](../fixtures). They are produced by
`python/adi_heston/fixtures.py`.

**File format.** Every array is a raw `.bin` file: float64, little-endian, row-major (C order), no header. It loads
with a single `fread` into a `double` buffer. Each case folder also has a `meta.json` with every shape and scalar.

| File | Contents |
|---|---|
| `a1.bin`, `a2.bin`, `a0_coef.bin`, `w_s.bin`, `w_v.bin` | operator inputs (Section 4.2) |
| `U0.bin`, `schedule.bin` | starting values, and the step schedule as rows of (dtau, theta_adi, boundary) |
| `kernel_input.bin` | one fixed test array fed to every kernel |
| `scalars.bin` | the four numbers a port needs from `meta.json`, so it never parses JSON: `[solve_c, dtau, theta_adi, boundary]`, the last three being the single Douglas step's arguments |
| `apply_A0.bin`, `apply_A1.bin`, `apply_A2.bin` | expected kernel outputs on `kernel_input` |
| `solve_A1.bin`, `solve_A2.bin` | expected solves with right-hand side `kernel_input` and `c = meta["solve_c"]` |
| `douglas_step.bin` | one step from `kernel_input`, with the arguments in `meta["douglas_step_args"]` |
| `U_final.bin` | the full run: start from `U0` and apply `douglas_step` for every row of `schedule.bin` |

**Cases.** All use the same test problem: r = 0.025, q = 0.01, kappa = 1.5, theta = 0.04, xi = 0.3, rho = -0.9,
K = 100, T = 1, S_max = 800, V_max = 5.

| Case | Grid | Steps | In the repo? |
|---|---|---|---|
| `tiny` | 6 x 5 | 4 | yes; small enough to print and debug by hand |
| `small` | 21 x 11 | 20 | yes; the main check |
| `medium` | 101 x 51 | 50 | no; generate with `python -m adi_heston.fixtures generate --case medium` |

**Checking a port.** The port writes its outputs into a folder, using the same file names and format. Then:

```bash
python -m adi_heston.fixtures check fixtures/small path/to/port_output
```

This prints PASS or FAIL per output, with its relative error (the largest difference divided by the largest expected
value). An output the port has not written yet is reported as "not provided", so kernels can be checked one at a time.
The exit code is 1 if anything fails.

**Tolerances** (double precision): **1e-12** for each kernel and for `douglas_step`, and **1e-10** for `U_final`.
The algorithm is the same, but compilers reorder arithmetic and use fused multiply-add instructions, so agreement is
close but not bit-for-bit. A float32 port would need its own, looser tolerances (about 1e-5).

**The C++ port** (`cpp/`, see [`cpp/README.md`](../cpp/README.md)) passes every check on `tiny`, `small` and `medium` bit for bit (relative error 0) at 1 to 10 threads. Large benchmark grids get inputs only: `python -m adi_heston.fixtures generate --grid M N STEPS` writes them to `bench_inputs/`.

**Evidence the files are enough.** A test (`test_fixtures.py`) implements the six kernels using only these files and
the definitions in Section 4.3, without importing the solver. It passes every check, with errors between 0 and 2e-15.

### 4.6 American options

Following Haentjens & in 't Hout (2015, eqs. 4.1–4.4), American pricing changes two things, for every scheme and also
for the two start-up half-steps:

- **Stage (a)** adds `dtau * lam` to the explicit predictor: `Y0 = U + dtau*(A0U + A1U + A2U) + dtau*lam`. All other
  stages are unchanged.
- **After each step**, a pointwise update (`solver.ikonen_toivanen_update`), with that step's own `dtau`:
  `U = max(U_bar - dtau*lam, g)` and `lam = max(0, lam + (g - U_bar)/dtau)`.

The extra inputs are the payoff array `g` (equal to `U0`) and the state array `lam`, which starts at zero. The update
needs no linear solve and parallelizes trivially. There are no American fixtures yet; they are worth adding once a port
reaches this step.

---

## 5. Not done yet

| Item | Notes |
|---|---|
| **Greeks and exercise boundary** | `SolveResult.U` and `SolveResult.lam` already hold what is needed |
| **Research experiments** | Strong correlation, Feller condition, exercise boundary, cost vs. accuracy |
| **Start-up steps for American options** | Haentjens & in 't Hout damp with two true backward-Euler half-steps (a full 2D solve), and only for Douglas and Craig–Sneyd. We use two Douglas half-steps with `theta_adi = 1` for every scheme, so the C++/CUDA ports need only ADI kernels. The exercise-boundary experiment will show whether this matters. |
| **American case where the Feller condition fails** | No published reference prices exist (Haentjens & in 't Hout note this), so only self-convergence can be checked |

## 6. Observations worth keeping for the report

- The upwinded region in v moves outward as the grid is refined (Section 3.4), so it stays away from where the price is read.
- With rho = -0.9 and the step count tied to the grid size, the ADI error falls faster than 4x per refinement (4.6x, 4.8x).
  Douglas's time error and the space error have opposite signs and partly cancel. This is a coincidence, not a property
  to rely on; with many time steps the space error alone falls at 4.2x.
- On the Feller-failing benchmark the solver keeps roughly second-order accuracy.
