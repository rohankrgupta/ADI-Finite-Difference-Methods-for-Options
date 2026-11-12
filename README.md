# ADI Finite-Difference Methods for American Options under the Heston Model

Math/CS 714 final project, UW–Madison, Fall 2026 · Rohan Gupta, Ganesh Arivoli · Instructor: Yue Sun

We price American put options under the Heston stochastic-volatility model by solving the pricing PDE
with ADI (alternating direction implicit) finite-difference schemes: Douglas, Craig–Sneyd,
Modified Craig–Sneyd and Hundsdorfer–Verwer. The early-exercise constraint is handled with
Ikonen–Toivanen operator splitting. We study when the schemes stay accurate and stable: under strong
correlation, when the variance can reach zero, and near the exercise boundary. We then speed up the
solver with C++ multithreading and CUDA, and compare their performance.

- Proposal: [`docs/proposal.pdf`](docs/proposal.pdf)
- Problem formulation (the PDE, boundary conditions and discretization): [`docs/formulation.pdf`](docs/formulation.pdf)
- Phase 1 summary, design decisions and the Phase 2 handoff: [`docs/phase1_summary.md`](docs/phase1_summary.md)

## Setup

Requires Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Code conventions

- **Layout.** The Python package is in `python/adi_heston/` and its tests are in `python/tests/`, one test file per module.
- **Arrays.** A 2D solution is an array `U` of shape `(m+1, n+1)` with `U[i, j] = u(s_i, v_j)`.
  Axis 0 is the stock price s and axis 1 is the variance v.
- **Operators.** These are stored as stencil coefficient arrays, not sparse matrices. Applying an operator is a stencil
  sweep, and solving is a batched tridiagonal (Thomas) solve along one axis. The C++ and CUDA versions use the same layout.
- **Precision.** Everything is float64.
- **Names.** `theta` is the Heston long-run variance and `theta_adi` is the ADI weight. Parameters are passed as
  `HestonParams` and `OptionSpec`.

## Status

Phase 1 (Python reference solver) is in progress. The commands to reproduce each figure will be added here as results land.
