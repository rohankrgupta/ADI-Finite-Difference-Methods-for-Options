"""Saved Python outputs that the C++ and CUDA ports are checked against.

Each test case is a folder of raw arrays plus one ``meta.json``:

* every ``.bin`` file is float64, little-endian, row-major (C order), with no header;
* ``meta.json`` holds the shapes and all scalars (parameters, the solve constant c,
  the arguments of the single Douglas step, and the tolerances);
* ``scalars.bin`` repeats the four numbers a port needs, so that it never has to parse
  JSON: [solve_c, dtau, theta_adi, boundary], the last three being the arguments of
  the single Douglas step.

The base cases (tiny, small, medium) use the Douglas scheme for a European put and also
check the six kernels one by one. The variant cases (small_<scheme> and
small_american_<scheme>) check one step and the full run of each scheme, European and
American. They add ``settings.bin`` = [scheme id, american (0 or 1), start-up half-steps],
with scheme ids from ``SCHEME_IDS``. American cases also add ``lam_input.bin``, the
multiplier fed to the one-step check. ``meta.json`` lists each case's outputs.

A port writes its outputs into its own folder, using the same file names and format.
``check`` then compares them with the Python outputs. Outputs the port has not
produced yet are reported as "not provided", not as failures.

Usage:
    python -m adi_heston.fixtures generate [--case tiny small medium small_craig_sneyd ...] [--out DIR]
    python -m adi_heston.fixtures generate --grid M N STEPS [--out DIR]   # inputs only, for benchmarks
    python -m adi_heston.fixtures check CASE_DIR PORT_OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .grid import Grid
from .operators import build_heston_operators, dirichlet_value
from .params import HestonParams, OptionSpec
from .solver import DEFAULT_THETA, SCHEMES, douglas_step, ikonen_toivanen_update, solve, time_schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "fixtures"
BENCH_DIR = REPO_ROOT / "bench_inputs"

# Fixed test problem: rho = -0.9 exercises the mixed term, and q != 0 exercises the full drift.
PARAMS = HestonParams(r=0.025, q=0.01, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
OPTION = OptionSpec(K=100.0, T=1.0)
S_MAX, V_MAX = 8.0 * OPTION.K, 5.0
RANNACHER_HALF_STEPS = 2
CASES = {"tiny": (5, 4, 4), "small": (20, 10, 20), "medium": (100, 50, 50)}  # (m, n, n_steps)

KERNEL_OUTPUTS = ["apply_A0", "apply_A1", "apply_A2", "solve_A1", "solve_A2", "douglas_step"]
OUTPUTS = KERNEL_OUTPUTS + ["U_final"]  # outputs of the base cases

# Variant cases: name -> (base case, scheme, american).
SCHEME_IDS = {"douglas": 0, "craig_sneyd": 1, "modified_craig_sneyd": 2, "hundsdorfer_verwer": 3}
VARIANTS = {f"small_{s}": ("small", s, False) for s in ("craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer")}
VARIANTS |= {f"small_american_{s}": ("small", s, True) for s in SCHEME_IDS}

TOLERANCES = {
    **{name: 1e-12 for name in KERNEL_OUTPUTS + ["step", "step_lam"]},
    "U_final": 1e-10,
    "lam_final": 1e-10,
}


def kernel_input(shape: tuple[int, int]) -> np.ndarray:
    """Return the fixed test array fed to every kernel. It uses formulas only (no random numbers), so it is reproducible anywhere."""
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return np.sin(0.37 * i + 0.11 * j) + 0.5 * np.cos(0.23 * i - 0.17 * j)


def lam_input(shape: tuple[int, int]) -> np.ndarray:
    """Return the fixed, non-negative multiplier fed to the one-step check of American cases."""
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return 0.5 * (1.0 + np.sin(0.29 * i + 0.07 * j))


def _write(path: Path, array: np.ndarray) -> list[int]:
    a = np.ascontiguousarray(array, dtype="<f8")
    a.tofile(path)
    return list(a.shape)


FORMAT = "float64, little-endian, row-major (C order), no header"


def _time_loop_inputs(m: int, n: int, n_steps: int, option: OptionSpec = OPTION, theta_adi: float = 0.5):
    """Return (grid, operators, input arrays of the time loop) for the test problem on an (m+1) x (n+1) grid."""
    grid = Grid.build(K=OPTION.K, S_max=S_MAX, V_max=V_MAX, m=m, n=n)
    ops = build_heston_operators(grid, PARAMS)
    S, _ = grid.mesh()
    inputs = {
        "a1": ops.a1,
        "a2": ops.a2,
        "a0_coef": ops.a0_coef,
        "w_s": ops.w_s,
        "w_v": ops.w_v,
        "U0": OPTION.payoff(S),
        "schedule": time_schedule(PARAMS, option, n_steps, theta_adi, rannacher_half_steps=RANNACHER_HALF_STEPS),
    }
    return grid, ops, inputs


def _problem_meta(m: int, n: int, n_steps: int) -> dict:
    return {
        "m": m,
        "n": n,
        "n_steps": n_steps,
        "rannacher_half_steps": RANNACHER_HALF_STEPS,
        "params": asdict(PARAMS),
        "option": asdict(OPTION),
        "S_max": S_MAX,
        "V_max": V_MAX,
    }


def generate(case: str, out_dir: Path = DEFAULT_DIR) -> Path:
    """Write all input and expected-output files for ``case`` into ``out_dir/case`` and return that folder."""
    if case in VARIANTS:
        return _generate_variant(case, out_dir)
    m, n, n_steps = CASES[case]
    folder = Path(out_dir) / case
    folder.mkdir(parents=True, exist_ok=True)

    grid, ops, inputs = _time_loop_inputs(m, n, n_steps)
    X = kernel_input(grid.shape)
    dtau = OPTION.T / n_steps
    solve_c = 0.5 * dtau  # the c every main step uses: theta_adi * dtau
    step_args = {"dtau": dtau, "theta_adi": 0.5, "boundary": dirichlet_value(PARAMS, OPTION, dtau)}

    arrays = {
        **inputs,
        "kernel_input": X,
        "scalars": np.array([solve_c, step_args["dtau"], step_args["theta_adi"], step_args["boundary"]]),
        # expected outputs
        "apply_A0": ops.apply_A0(X),
        "apply_A1": ops.apply_A1(X),
        "apply_A2": ops.apply_A2(X),
        "solve_A1": ops.solve_A1(solve_c, X),
        "solve_A2": ops.solve_A2(solve_c, X),
        "douglas_step": douglas_step(ops, X, step_args["dtau"], step_args["theta_adi"], step_args["boundary"]),
        "U_final": solve(PARAMS, OPTION, grid, n_steps, rannacher_half_steps=RANNACHER_HALF_STEPS).U,
    }
    shapes = {name: _write(folder / f"{name}.bin", a) for name, a in arrays.items()}
    meta = {
        "format": FORMAT,
        "case": case,
        **_problem_meta(m, n, n_steps),
        "solve_c": solve_c,
        "douglas_step_args": step_args,
        "shapes": shapes,
        "outputs": OUTPUTS,
        "tolerances": {name: TOLERANCES[name] for name in OUTPUTS},
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def _generate_variant(case: str, out_dir: Path) -> Path:
    """Write a variant case: one step and the full run of one scheme, European or American."""
    base, scheme, american = VARIANTS[case]
    m, n, n_steps = CASES[base]
    folder = Path(out_dir) / case
    folder.mkdir(parents=True, exist_ok=True)

    option = OptionSpec(K=OPTION.K, T=OPTION.T, american=american)
    theta = DEFAULT_THETA[scheme]
    grid, ops, inputs = _time_loop_inputs(m, n, n_steps, option, theta)
    X = kernel_input(grid.shape)
    dtau = OPTION.T / n_steps
    step_args = {"dtau": dtau, "theta_adi": theta, "boundary": dirichlet_value(PARAMS, option, dtau)}

    arrays = {
        **inputs,
        "kernel_input": X,
        "scalars": np.array([theta * dtau, dtau, theta, step_args["boundary"]]),
        "settings": np.array([SCHEME_IDS[scheme], float(american), RANNACHER_HALF_STEPS]),
    }
    result = solve(PARAMS, option, grid, n_steps, scheme=scheme, rannacher_half_steps=RANNACHER_HALF_STEPS)
    if american:
        lam0 = lam_input(grid.shape)
        U_bar = SCHEMES[scheme](ops, X, dtau, theta, step_args["boundary"], lam0)
        step_U, step_lam = ikonen_toivanen_update(U_bar, lam0, inputs["U0"], dtau)
        arrays |= {"lam_input": lam0, "step": step_U, "step_lam": step_lam, "U_final": result.U, "lam_final": result.lam}
        outputs = ["step", "step_lam", "U_final", "lam_final"]
    else:
        arrays |= {"step": SCHEMES[scheme](ops, X, dtau, theta, step_args["boundary"]), "U_final": result.U}
        outputs = ["step", "U_final"]
    shapes = {name: _write(folder / f"{name}.bin", a) for name, a in arrays.items()}
    meta = {
        "format": FORMAT,
        "case": case,
        "scheme": scheme,
        "american": american,
        **_problem_meta(m, n, n_steps),
        "step_args": step_args,
        "shapes": shapes,
        "outputs": outputs,
        "tolerances": {name: TOLERANCES[name] for name in outputs},
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def generate_inputs(m: int, n: int, n_steps: int, out_dir: Path = BENCH_DIR) -> Path:
    """Write only the inputs of the time loop for an (m+1) x (n+1) grid, for benchmarks.

    Uses the same test problem as the fixture cases but no expected outputs, which
    would be slow to compute for large grids. Returns the folder, named
    ``grid_<m+1>x<n+1>_N<n_steps>``.
    """
    folder = Path(out_dir) / f"grid_{m + 1}x{n + 1}_N{n_steps}"
    folder.mkdir(parents=True, exist_ok=True)
    _, _, inputs = _time_loop_inputs(m, n, n_steps)
    shapes = {name: _write(folder / f"{name}.bin", a) for name, a in inputs.items()}
    meta = {"format": FORMAT, **_problem_meta(m, n, n_steps), "shapes": shapes}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def load(folder: Path, name: str) -> np.ndarray:
    """Load ``name.bin`` from a case folder with the shape recorded in its meta.json."""
    meta = json.loads((Path(folder) / "meta.json").read_text())
    return np.fromfile(Path(folder) / f"{name}.bin", dtype="<f8").reshape(meta["shapes"][name])


def check(case_dir: Path, port_dir: Path) -> tuple[bool, list[tuple[str, str, float | None]]]:
    """Compare a port's outputs with the Python outputs.

    Returns (ok, rows), where each row is (name, status, relative error). The status is
    PASS, FAIL or "not provided". ok is False only if some output FAILs.
    """
    case_dir, port_dir = Path(case_dir), Path(port_dir)
    outputs = json.loads((case_dir / "meta.json").read_text()).get("outputs", OUTPUTS)
    rows = []
    for name in outputs:
        expected = load(case_dir, name)
        path = port_dir / f"{name}.bin"
        if not path.exists():
            rows.append((name, "not provided", None))
            continue
        got = np.fromfile(path, dtype="<f8")
        if got.size != expected.size:
            rows.append((name, f"FAIL (size {got.size}, expected {expected.size})", None))
            continue
        err = float(np.max(np.abs(got.reshape(expected.shape) - expected)) / np.max(np.abs(expected)))
        rows.append((name, "PASS" if err <= TOLERANCES[name] else "FAIL", err))
    ok = not any(status.startswith("FAIL") for _, status, _ in rows)
    return ok, rows


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m adi_heston.fixtures", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write the Python reference files")
    gen.add_argument("--case", nargs="+", choices=sorted(CASES) + sorted(VARIANTS), default=["tiny", "small", *VARIANTS])
    gen.add_argument("--grid", nargs=3, type=int, metavar=("M", "N", "STEPS"),
                     help="write inputs only for an (M+1) x (N+1) grid into bench_inputs/")
    gen.add_argument("--out", type=Path, default=None)
    chk = sub.add_parser("check", help="compare a port's outputs with the Python outputs")
    chk.add_argument("case_dir", type=Path)
    chk.add_argument("port_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.grid:
            print(f"wrote {generate_inputs(*args.grid, out_dir=args.out or BENCH_DIR)}")
            return 0
        for case in args.case:
            print(f"wrote {generate(case, args.out or DEFAULT_DIR)}")
        return 0

    ok, rows = check(args.case_dir, args.port_dir)
    for name, status, err in rows:
        err_text = f"{err:.2e}" if err is not None else "-"
        print(f"{name:14s} {status:14s} rel. error {err_text:>9s}   (tolerance {TOLERANCES[name]:.0e})")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
