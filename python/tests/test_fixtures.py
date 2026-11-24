import json
import shutil

import numpy as np
import pytest

from adi_heston import fixtures
from adi_heston.fixtures import DEFAULT_DIR, OUTPUTS, check, generate, generate_inputs, load
from adi_heston.params import OptionSpec
from adi_heston.solver import DEFAULT_THETA, time_schedule
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


@pytest.mark.parametrize("case", ["tiny", "small", *sorted(fixtures.VARIANTS)])
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


def test_scalars_file_repeats_the_meta_values(small_case):
    meta = json.loads((small_case / "meta.json").read_text())
    args = meta["douglas_step_args"]
    np.testing.assert_array_equal(
        np.fromfile(small_case / "scalars.bin", dtype="<f8"),
        [meta["solve_c"], args["dtau"], args["theta_adi"], args["boundary"]],
    )


def test_inputs_only_generator_writes_the_time_loop_inputs(tmp_path):
    from adi_heston.fixtures import generate_inputs

    folder = generate_inputs(20, 10, 20, tmp_path)
    assert folder.name == "grid_21x11_N20"
    reference = generate("small", tmp_path / "cases")  # same problem and size as the small case
    for name in ("a1", "a2", "a0_coef", "w_s", "w_v", "U0", "schedule"):
        np.testing.assert_array_equal(load(folder, name), load(reference, name))
    assert not (folder / "U_final.bin").exists()


def test_inputs_generator_default_problem_is_douglas_european(tmp_path):
    folder = generate_inputs(20, 10, 20, tmp_path)
    np.testing.assert_array_equal(load(folder, "settings"), [0.0, 0.0, fixtures.RANNACHER_HALF_STEPS])
    meta = json.loads((folder / "meta.json").read_text())
    assert (meta["scheme"], meta["american"], meta["theta_adi"]) == ("douglas", False, 0.5)


@pytest.mark.parametrize("american", [False, True])
@pytest.mark.parametrize("scheme", sorted(fixtures.SCHEME_IDS))
def test_inputs_generator_writes_the_schemes_schedule(tmp_path, scheme, american):
    folder = generate_inputs(20, 10, 20, tmp_path, scheme=scheme, american=american)
    suffix = ("" if scheme == "douglas" else f"_{scheme}") + ("_american" if american else "")
    assert folder.name == f"grid_21x11_N20{suffix}"

    schedule = load(folder, "schedule")  # rows of (dtau, theta_adi, boundary)
    half = fixtures.RANNACHER_HALF_STEPS
    assert (schedule[:half, 1] == 1.0).all() and (schedule[half:, 1] == DEFAULT_THETA[scheme]).all()
    if american:
        assert (schedule[:, 2] == fixtures.OPTION.K).all()  # s = 0: immediate exercise is worth K
    else:
        assert (schedule[:, 2] < fixtures.OPTION.K).all()  # K exp(-r tau)
    option = OptionSpec(K=fixtures.OPTION.K, T=fixtures.OPTION.T, american=american)
    np.testing.assert_array_equal(schedule, time_schedule(fixtures.PARAMS, option, 20, DEFAULT_THETA[scheme], half))
    np.testing.assert_array_equal(load(folder, "settings"), [fixtures.SCHEME_IDS[scheme], float(american), half])


def test_inputs_generator_matches_the_variant_case(tmp_path):
    """Benchmark inputs for a scheme are exactly the inputs of the variant case that the fixtures verify."""
    folder = generate_inputs(20, 10, 20, tmp_path, scheme="hundsdorfer_verwer", american=True)
    reference = generate("small_american_hundsdorfer_verwer", tmp_path / "cases")
    for name in ("a1", "a2", "a0_coef", "w_s", "w_v", "U0", "schedule", "settings"):
        np.testing.assert_array_equal(load(folder, name), load(reference, name), err_msg=name)


def test_command_line_grid_with_scheme(tmp_path, capsys):
    argv = ["generate", "--grid", "20", "10", "20", "--scheme", "craig_sneyd", "--american", "--out", str(tmp_path)]
    assert fixtures._main(argv) == 0
    assert (tmp_path / "grid_21x11_N20_craig_sneyd_american" / "settings.bin").exists()
    with pytest.raises(SystemExit):  # --scheme and --american need --grid
        fixtures._main(["generate", "--scheme", "craig_sneyd", "--out", str(tmp_path)])


def test_variant_case_settings_and_outputs(tmp_path):
    folder = generate("small_american_hundsdorfer_verwer", tmp_path)
    meta = json.loads((folder / "meta.json").read_text())
    np.testing.assert_array_equal(np.fromfile(folder / "settings.bin", dtype="<f8"), [3.0, 1.0, 2.0])
    assert meta["outputs"] == ["step", "step_lam", "U_final", "lam_final"]
    lam = load(folder, "lam_final")
    assert (lam >= 0.0).all() and (lam > 0.0).any()  # an exercise region exists
    assert (load(folder, "U_final") >= load(folder, "U0")).all()  # never below the payoff


def test_check_uses_each_cases_own_output_list(tmp_path):
    folder = generate("small_craig_sneyd", tmp_path / "cases")
    port = tmp_path / "port"
    port.mkdir()
    shutil.copy(folder / "step.bin", port / "step.bin")
    ok, rows = check(folder, port)
    assert ok
    assert [(name, status) for name, status, _ in rows] == [("step", "PASS"), ("U_final", "not provided")]
