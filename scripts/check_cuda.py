"""Run the CUDA port on the fixture cases and check every output it must produce.

For each case the port writes into a fresh folder, ``<build>/check/<case>``. Every output the
chosen stage requires must exist; unlike ``python -m adi_heston.fixtures check``, a missing output
is a failure. Fixture outputs are compared with the case's saved files; intermediate outputs (such
as the explicit stage's ``k1_R1``) are compared with values computed here in NumPy from the case's
own input files, through the Python reference functions.

A default build must agree within the fixture tolerances. A build configured with ``ADI_EXACT``
(read from its CMakeCache.txt) must be byte-identical.

Usage (from the repository root):
    uv run --no-project --with-requirements requirements.txt --with-editable . -- \\
        python scripts/check_cuda.py [--build cuda/build/default] [--stage N] [--cases tiny small ...]

Project: ADI finite-difference methods for American options under Heston
         Math/CS 714, UW–Madison, Fall 2026
Author:  Ganesh Arivoli <arivoli@wisc.edu>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from adi_heston import fixtures
from adi_heston.operators import HestonOperators
from adi_heston.solver import _predict

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CASES = ["tiny", "small", "medium", "large"]
ALL_CASES = BASE_CASES + sorted(fixtures.VARIANTS)
LATEST_STAGE = 7
# Schemes whose time loop the port runs (stage 6: Douglas; stage 7 adds the corrected schemes one by one).
LOOP_SCHEMES = {"douglas", "craig_sneyd"}
INTERMEDIATE_TOLERANCE = 1e-12  # the fixture tolerance for single kernels
RESIDUAL_TOLERANCE = 1e-12  # backward error of the tridiagonal solves
# Distinct values of c = theta_adi*dtau, so factor sets: the start-up c equals the main c when theta_adi = 1/2.
FACTOR_SETS = {"douglas": 1, "craig_sneyd": 1, "modified_craig_sneyd": 2, "hundsdorfer_verwer": 2}


def required_outputs(stage: int, case: str) -> tuple[list[str], list[str]]:
    """Return (fixture outputs, intermediate outputs) the port must write for `case` at `stage`."""
    fixture_outputs: list[str] = []
    intermediates: list[str] = []
    if stage >= 3:
        if case in BASE_CASES:
            fixture_outputs += ["apply_A0", "apply_A1", "apply_A2"]
        intermediates += ["k1_R1", "k1_A2U"]
    if stage >= 4:
        if case in BASE_CASES:
            fixture_outputs += ["solve_A1", "solve_A2"]
    if stage >= 5:
        if case in BASE_CASES:  # the Douglas time loop for a European put
            fixture_outputs += ["douglas_step", "U_final"]
    # Variant cases: the time loop, from stage 6 on, for every scheme the port implements.
    if case in fixtures.VARIANTS and stage >= 6:
        loop_schemes = LOOP_SCHEMES if stage >= 7 else {"douglas"}
        if fixtures.VARIANTS[case][1] in loop_schemes:
            fixture_outputs += json.loads((fixtures.DEFAULT_DIR / case / "meta.json").read_text())["outputs"]
    return fixture_outputs, intermediates


def american_properties(case_dir: Path, out_dir: Path, exact: bool) -> list[tuple[str, str, float | None]]:
    """U >= payoff, lam >= 0 and lam*(U - payoff) = 0 for the one-step and full-run outputs. Complementarity
    must be exactly 0 in an exact build; the default build multiplies by 1/dtau, so it may be off by rounding."""
    payoff = fixtures.load(case_dir, "U0")
    rows = []
    for U_name, lam_name in (("step", "step_lam"), ("U_final", "lam_final")):
        U_path, lam_path = out_dir / f"{U_name}.bin", out_dir / f"{lam_name}.bin"
        if not (U_path.exists() and lam_path.exists()):
            rows.append((f"american {U_name}", "MISSING", None))
            continue
        U = np.fromfile(U_path, dtype="<f8").reshape(payoff.shape)
        lam = np.fromfile(lam_path, dtype="<f8").reshape(payoff.shape)
        product = float(np.max(np.abs(lam * (U - payoff))))
        limit = 0.0 if exact else 1e-12 * float(np.max(np.abs(lam))) * float(np.max(np.abs(U)))
        problems = []
        if np.min(U - payoff) < 0:
            problems.append("U < payoff")
        if np.min(lam) < 0:
            problems.append("lam < 0")
        if product > limit:
            problems.append("lam*(U - payoff) != 0")
        rows.append((f"american {U_name}", "PASS" if not problems else "FAIL (" + ", ".join(problems) + ")", product))
    return rows


def operators_from_files(case_dir: Path, absolute: bool = False) -> HestonOperators:
    """The reference operators built from the case's own coefficient files (not rebuilt from parameters,
    which would bring in platform rounding differences). With absolute=True, every coefficient is |.|."""
    load = (lambda name: np.abs(fixtures.load(case_dir, name))) if absolute else (lambda name: fixtures.load(case_dir, name))
    return HestonOperators(grid=None, params=None, a1=load("a1"), a2=load("a2"), a0_coef=load("a0_coef"),
                           w_s=load("w_s"), w_v=load("w_v"), upwind_s=None, upwind_v=None)


def solve_residuals(case_dir: Path, out_dir: Path) -> list[tuple[str, str, float | None]]:
    """Backward error of the port's solve_A1/A2 on kernel_input: max|(I - cA)x - r| divided by the
    largest term summed, max(|x| + c|A||x|)."""
    ops, abs_ops = operators_from_files(case_dir), operators_from_files(case_dir, absolute=True)
    rhs = fixtures.load(case_dir, "kernel_input")
    c = float(np.fromfile(case_dir / "scalars.bin", dtype="<f8")[0])
    rows = []
    for name, apply, apply_abs in (("solve_A1", ops.apply_A1, abs_ops.apply_A1),
                                   ("solve_A2", ops.apply_A2, abs_ops.apply_A2)):
        path = out_dir / f"{name}.bin"
        if not path.exists():
            rows.append((f"{name} residual", "MISSING", None))
            continue
        x = np.fromfile(path, dtype="<f8").reshape(rhs.shape)
        residual = np.max(np.abs(x - c * apply(x) - rhs)) / np.max(np.abs(x) + c * apply_abs(np.abs(x)))
        rows.append((f"{name} residual", "PASS" if residual <= RESIDUAL_TOLERANCE else "FAIL", float(residual)))
    return rows


def factor_set_check(case: str, case_dir: Path, out_dir: Path) -> tuple[str, str, float | None]:
    """The port must build one factor set per distinct c = theta_adi*dtau of the schedule."""
    path = out_dir / "tridiag_sets.bin"
    if not path.exists():
        return ("tridiag_sets", "MISSING", None)
    schedule = fixtures.load(case_dir, "schedule")
    distinct = len({theta * dtau for dtau, theta, _ in schedule})
    scheme = fixtures.VARIANTS[case][1] if case in fixtures.VARIANTS else "douglas"
    got = int(np.fromfile(path, dtype="<f8")[0])
    ok = got == distinct == FACTOR_SETS[scheme]
    return ("tridiag_sets", "PASS" if ok else f"FAIL (got {got}, schedule has {distinct})", float(got))


def expected_intermediates(case_dir: Path) -> dict[str, np.ndarray]:
    """K1 on kernel_input with the one-step arguments, as solver._predict and _implicit_corrections compute it."""
    ops = operators_from_files(case_dir)
    X = fixtures.load(case_dir, "kernel_input")
    _, dtau, theta, boundary = np.fromfile(case_dir / "scalars.bin", dtype="<f8")
    lam = fixtures.load(case_dir, "lam_input") if (case_dir / "lam_input.bin").exists() else None
    A0U, A1U, A2U = ops.apply_A0(X), ops.apply_A1(X), ops.apply_A2(X)
    Y0 = _predict(X, dtau, A0U + A1U + A2U, lam)
    R1 = Y0 - (theta * dtau) * A1U
    R1[0] = boundary
    return {"k1_R1": R1, "k1_A2U": A2U}


def compare(got_path: Path, expected: np.ndarray, tolerance: float, exact: bool) -> tuple[str, float | None]:
    """Return (status, relative error) of one output file against the expected array."""
    if not got_path.exists():
        return "MISSING", None
    got = np.fromfile(got_path, dtype="<f8")
    expected = np.ascontiguousarray(expected, dtype="<f8").ravel()
    if got.size != expected.size:
        return f"FAIL (size {got.size}, expected {expected.size})", None
    scale = np.max(np.abs(expected))
    err = float(np.max(np.abs(got - expected)) / scale) if scale > 0 else float(np.max(np.abs(got)))
    if exact:
        return ("PASS" if got.tobytes() == expected.tobytes() else "FAIL (bytes differ)"), err
    return ("PASS" if err <= tolerance else "FAIL"), err


def check_case(binary: Path, build: Path, case: str, stage: int, exact: bool) -> bool:
    case_dir = fixtures.DEFAULT_DIR / case
    out_dir = build / "check" / case
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run = subprocess.run([str(binary), str(case_dir), str(out_dir)], capture_output=True, text=True)
    if run.returncode != 0:
        print(f"{case}: adi_cuda failed (exit {run.returncode}): {run.stderr.strip()}")
        return False

    fixture_outputs, intermediates = required_outputs(stage, case)
    rows = []
    for name in fixture_outputs:
        status, err = compare(out_dir / f"{name}.bin", fixtures.load(case_dir, name), fixtures.TOLERANCES[name], exact)
        rows.append((name, status, err))
    if intermediates:
        expected = expected_intermediates(case_dir)
        for name in intermediates:
            status, err = compare(out_dir / f"{name}.bin", expected[name], INTERMEDIATE_TOLERANCE, exact)
            rows.append((name, status, err))
    if stage >= 4:
        if case in BASE_CASES:
            rows += solve_residuals(case_dir, out_dir)
        rows.append(factor_set_check(case, case_dir, out_dir))
    if "step_lam" in fixture_outputs:
        rows += american_properties(case_dir, out_dir, exact)

    ok = all(status == "PASS" for _, status, _ in rows)
    print(f"{case}: {'ok' if ok else 'FAILED'}")
    for name, status, err in rows:
        value = "-" if err is None else (f"{int(err)}" if name == "tridiag_sets" else f"{err:.2e}")
        label = ("count" if name == "tridiag_sets" else "backward error" if "residual" in name
                 else "max|lam*(U-g)|" if name.startswith("american") else "rel. error")
        print(f"    {name:18s} {status:20s} {label} {value}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", type=Path, default=REPO_ROOT / "cuda" / "build" / "default")
    parser.add_argument("--stage", type=int, default=LATEST_STAGE, help="check the outputs required by this stage")
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=ALL_CASES)
    args = parser.parse_args()

    build = args.build.resolve()
    binary = build / "adi_cuda"
    if not binary.exists():
        raise SystemExit(f"{binary} not found; build it first")
    cache = (build / "CMakeCache.txt").read_text()
    exact = "ADI_EXACT:BOOL=ON" in cache
    for case in ("medium", "large"):
        if case in args.cases and not (fixtures.DEFAULT_DIR / case / "meta.json").exists():
            print(f"generating fixtures/{case}")
            fixtures.generate(case)

    print(f"build {build} ({'exact: byte-identical' if exact else 'default: fixture tolerances'}), stage {args.stage}")
    results = [check_case(binary, build, case, args.stage, exact) for case in args.cases]
    fixture_outputs = sum(len(required_outputs(args.stage, case)[0]) for case in args.cases)
    print(f"{fixture_outputs} fixture outputs required")
    print(f"{sum(results)}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
