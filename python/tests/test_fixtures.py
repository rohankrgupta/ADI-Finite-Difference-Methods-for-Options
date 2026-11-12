import json
import shutil

import numpy as np
import pytest

from adi_heston import fixtures
from adi_heston.fixtures import DEFAULT_DIR, OUTPUTS, check, generate, load
from adi_heston.tridiag import solve_tridiagonal


# ---- A "port" written only from the saved files and the contract in docs/phase1_summary.md, Section 4.3 ----
# It deliberately does not import HestonOperators or the solver. If it passes, then the files plus the
# written contract are enough for the C++ and CUDA versions to reproduce Python's results.


class ContractPort:
    def __init__(self, folder):
        self.a1, self.a2 = load(folder, "a1"), load(folder, "a2")
        self.a0, self.ws, self.wv = load(folder, "a0_coef"), load(folder, "w_s"), load(folder, "w_v")

    def apply_A1(self, U):
        lo, di, up = self.a1
        out = di * U
        out[1:] += lo[1:] * U[:-1]
        out[:-1] += up[:-1] * U[1:]
        return out

    def apply_A2(self, U):
        lo, di, up = self.a2
        out = di * U
        out[:, 1:] += lo[:, 1:] * U[:, :-1]
        out[:, :-1] += up[:, :-1] * U[:, 1:]
        return out

    def apply_A0(self, U):
        m1, n1 = U.shape
        out = np.zeros_like(U)
        for i in range(1, m1 - 1):
            for j in range(1, n1 - 1):
                block = U[i - 1 : i + 2, j - 1 : j + 2]
                out[i, j] = self.a0[i, j] * np.sum(np.outer(self.ws[i], self.wv[j]) * block)
        return out

    def solve_A1(self, c, R):
        lo, di, up = self.a1
        return solve_tridiagonal(-c * lo, 1.0 - c * di, -c * up, R, axis=0)

    def solve_A2(self, c, R):
        lo, di, up = self.a2
        return solve_tridiagonal(-c * lo, 1.0 - c * di, -c * up, R, axis=1)

    def douglas_step(self, U, dtau, theta_adi, boundary):
        A0U, A1U, A2U = self.apply_A0(U), self.apply_A1(U), self.apply_A2(U)
        c = theta_adi * dtau
        R1 = U + dtau * (A0U + A1U + A2U) - c * A1U
        R1[0, :] = boundary
        Y1 = self.solve_A1(c, R1)
        return self.solve_A2(c, Y1 - c * A2U)


def write_port_outputs(case_dir, port_dir):
    meta = json.loads((case_dir / "meta.json").read_text())
    port = ContractPort(case_dir)
    X = load(case_dir, "kernel_input")
    c = meta["solve_c"]
    args = meta["douglas_step_args"]
    U = load(case_dir, "U0")
    for dtau, theta_adi, boundary in load(case_dir, "schedule"):
        U = port.douglas_step(U, dtau, theta_adi, boundary)
    outputs = {
        "apply_A0": port.apply_A0(X),
        "apply_A1": port.apply_A1(X),
        "apply_A2": port.apply_A2(X),
        "solve_A1": port.solve_A1(c, X),
        "solve_A2": port.solve_A2(c, X),
        "douglas_step": port.douglas_step(X, args["dtau"], args["theta_adi"], args["boundary"]),
        "U_final": U,
    }
    port_dir.mkdir(parents=True, exist_ok=True)
    for name, array in outputs.items():
        np.ascontiguousarray(array, dtype="<f8").tofile(port_dir / f"{name}.bin")


@pytest.fixture
def small_case(tmp_path):
    return generate("small", tmp_path / "cases")


# ---- the checker ----------------------------------------------------------------


@pytest.mark.parametrize("case", ["tiny", "small"])
def test_contract_port_passes_every_check(tmp_path, case):
    case_dir = generate(case, tmp_path / "cases")
    write_port_outputs(case_dir, tmp_path / "port")
    ok, rows = check(case_dir, tmp_path / "port")
    assert ok
    assert [status for _, status, _ in rows] == ["PASS"] * len(OUTPUTS)


def test_corrupted_output_fails_only_that_kernel(tmp_path, small_case):
    port = tmp_path / "port"
    write_port_outputs(small_case, port)
    bad = np.fromfile(port / "apply_A1.bin", dtype="<f8")
    k = int(np.argmax(np.abs(bad)))
    bad[k] *= 1.0 + 1e-9
    bad.tofile(port / "apply_A1.bin")
    ok, rows = check(small_case, port)
    assert not ok
    assert {name: status for name, status, _ in rows} == {
        name: ("FAIL" if name == "apply_A1" else "PASS") for name in OUTPUTS
    }


def test_missing_outputs_are_not_provided_rather_than_failures(tmp_path, small_case):
    port = tmp_path / "port"
    port.mkdir()
    shutil.copy(small_case / "apply_A1.bin", port / "apply_A1.bin")
    ok, rows = check(small_case, port)
    assert ok
    status = {name: s for name, s, _ in rows}
    assert status["apply_A1"] == "PASS" and status["solve_A2"] == "not provided"


def test_wrong_size_output_fails(tmp_path, small_case):
    port = tmp_path / "port"
    port.mkdir()
    np.zeros(7).tofile(port / "U_final.bin")
    ok, rows = check(small_case, port)
    assert not ok and dict((n, s) for n, s, _ in rows)["U_final"].startswith("FAIL")


def test_command_line_check_exit_codes(tmp_path, small_case, capsys):
    port = tmp_path / "port"
    write_port_outputs(small_case, port)
    assert fixtures._main(["check", str(small_case), str(port)]) == 0
    np.zeros(7).tofile(port / "U_final.bin")
    assert fixtures._main(["check", str(small_case), str(port)]) == 1
    assert "FAILED" in capsys.readouterr().out


# ---- the committed files ----------------------------------------------------------


@pytest.mark.parametrize("case", ["tiny", "small"])
def test_committed_fixtures_are_up_to_date(tmp_path, case):
    """If a Python kernel changes, this fails until the fixtures are regenerated."""
    fresh = generate(case, tmp_path)
    committed = DEFAULT_DIR / case
    assert json.loads((fresh / "meta.json").read_text()) == json.loads((committed / "meta.json").read_text())
    for path in sorted(fresh.glob("*.bin")):
        new = np.fromfile(path, dtype="<f8")
        old = np.fromfile(committed / path.name, dtype="<f8")
        np.testing.assert_allclose(old, new, rtol=1e-14, atol=1e-14 * np.max(np.abs(new)), err_msg=path.name)


def test_schedule_matches_rannacher_start_up(small_case):
    meta = json.loads((small_case / "meta.json").read_text())
    sched = load(small_case, "schedule")
    dtau = meta["option"]["T"] / meta["n_steps"]
    assert sched.shape == (meta["n_steps"] + 1, 3)  # 2 half-steps replace 1 full step
    np.testing.assert_allclose(sched[:2, :2], [[dtau / 2, 1.0], [dtau / 2, 1.0]])
    np.testing.assert_allclose(sched[2:, :2], np.tile([dtau, 0.5], (meta["n_steps"] - 1, 1)))
    assert sched[:, 0].sum() == pytest.approx(meta["option"]["T"])
    assert np.allclose(sched[:, 0] * sched[:, 1], dtau / 2)  # the same c in every step
