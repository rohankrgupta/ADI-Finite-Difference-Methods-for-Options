"""Benchmark the C++ multithreaded port against the NumPy reference.

Times only the Douglas time loop (no file reading, setup or thread start-up) on the
fixture test problem for several grid sizes, then writes a CSV and two figures:

    results/cpu_benchmark.csv
    results/cpu_speedup.pdf         speed-up of C++ over C++ on 1 thread, vs threads
    results/cpu_time_per_step.pdf   time per step vs grid size: NumPy, C++ 1 thread, C++ 8 threads

Usage (from the repository root, after building cpp/ as in cpp/README.md):
    python scripts/bench_cpu.py
    python scripts/bench_cpu.py --grids 100x50,200x100 --threads 1,2,4 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

from adi_heston.fixtures import OPTION, PARAMS, RANNACHER_HALF_STEPS, S_MAX, V_MAX, generate_inputs
from adi_heston.grid import Grid
from adi_heston.operators import build_heston_operators
from adi_heston.solver import douglas_step, time_schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = REPO_ROOT / "cpp" / "build" / "adi_port"
FIELDS = ["machine", "rows", "cols", "steps", "implementation", "threads", "median_ms_per_step", "min_ms_per_step"]


def machine_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def time_numpy(m: int, n: int, steps: int, repeats: int) -> tuple[float, float]:
    """Return (median, min) milliseconds per step of the NumPy Douglas loop, after one warm-up run."""
    grid = Grid.build(K=OPTION.K, S_max=S_MAX, V_max=V_MAX, m=m, n=n)
    ops = build_heston_operators(grid, PARAMS)
    S, _ = grid.mesh()
    U0 = OPTION.payoff(S)
    schedule = time_schedule(PARAMS, OPTION, steps, rannacher_half_steps=RANNACHER_HALF_STEPS)
    per_step = []
    for run in range(repeats + 1):
        U = U0.copy()
        start = time.perf_counter()
        for dtau, theta, boundary in schedule:
            U = douglas_step(ops, U, dtau, theta, boundary)
        if run > 0:
            per_step.append(1e3 * (time.perf_counter() - start) / len(schedule))
    return statistics.median(per_step), min(per_step)


def time_cpp(inputs: Path, threads: int, repeats: int, invocations: int) -> tuple[float, float]:
    """Return (median, min) milliseconds per step of `adi_port --bench` over separate runs.

    Each run of the program starts its own threads, and macOS does not always place all
    of them on performance cores, so one run can be an outlier. The median over several
    runs is reported.
    """
    medians, mins = [], []
    for _ in range(invocations):
        out = subprocess.run([str(PORT), str(inputs), "--bench", "--threads", str(threads), "--repeats", str(repeats)],
                             capture_output=True, text=True, check=True).stdout.strip()
        fields = out.split(",")
        medians.append(float(fields[4]))
        mins.append(float(fields[5]))
    return statistics.median(medians), min(mins)


def run(grids: list[tuple[int, int]], steps: int, thread_counts: list[int], repeats: int, invocations: int,
        numpy_repeats: int, out_dir: Path) -> list[dict]:
    machine = machine_name()
    rows = []
    schedule_rows = len(time_schedule(PARAMS, OPTION, steps, rannacher_half_steps=RANNACHER_HALF_STEPS))
    for m, n in grids:
        inputs = generate_inputs(m, n, steps)
        median, best = time_numpy(m, n, steps, numpy_repeats)
        rows.append(dict(machine=machine, rows=m + 1, cols=n + 1, steps=schedule_rows,
                         implementation="numpy", threads=1, median_ms_per_step=median, min_ms_per_step=best))
        print(f"{m + 1}x{n + 1}: numpy {median:.3f} ms/step")
        for t in thread_counts:
            median, best = time_cpp(inputs, t, repeats, invocations)
            rows.append(dict(machine=machine, rows=m + 1, cols=n + 1, steps=schedule_rows,
                             implementation="cpp", threads=t, median_ms_per_step=median, min_ms_per_step=best))
            print(f"{m + 1}x{n + 1}: cpp {t:2d} threads {median:.3f} ms/step")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cpu_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---- figures ------------------------------------------------------------------------

INK, INK_2, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
GRID_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # one blue, light -> dark by grid size
IMPL_COLORS = {"numpy": "#2a78d6", "cpp1": "#eb6834", "cpp8": "#1baf7a"}  # categorical slots 1-3


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted({(r["rows"], r["cols"]) for r in rows})
    threads = sorted({r["threads"] for r in rows if r["implementation"] == "cpp"})

    def ms(impl, t, g):
        return next(r["median_ms_per_step"] for r in rows
                    if r["implementation"] == impl and r["threads"] == t and (r["rows"], r["cols"]) == g)

    # Speed-up over C++ on 1 thread.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(threads, threads, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), label="ideal")
    for color, g in zip(GRID_RAMP[-len(grids):], grids):
        speedup = [ms("cpp", 1, g) / ms("cpp", t, g) for t in threads]
        ax.plot(threads, speedup, color=color, linewidth=1.6, marker="o", markersize=4, label=f"{g[0]}x{g[1]}")
    if max(threads) > 8:
        ax.axvline(8, color=RULE, linewidth=1)
        ax.text(7.85, max(threads) * 0.97, "8 performance cores", color=INK_2, fontsize=7, ha="right", va="top")
    ax.set_xlabel("threads", color=INK, fontsize=9)
    ax.set_ylabel("speed-up over 1 thread", color=INK, fontsize=9)
    ax.set_xticks(threads)
    ax.set_ylim(bottom=0)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, title="grid", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cpu_speedup.pdf")
    plt.close(fig)

    # Time per step against grid size.
    nodes = [g[0] * g[1] for g in grids]
    top = 8 if 8 in threads else max(threads)
    series = [("numpy", "NumPy", [ms("numpy", 1, g) for g in grids], "s"),
              ("cpp1", "C++, 1 thread", [ms("cpp", 1, g) for g in grids], "o"),
              ("cpp8", f"C++, {top} threads", [ms("cpp", top, g) for g in grids], "^")]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for key, label, values, marker in series:
        ax.plot(nodes, values, color=IMPL_COLORS[key], linewidth=1.6, marker=marker, markersize=4, label=label)
        ax.annotate(label, (nodes[-1], values[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid nodes (m+1)(n+1)", color=INK, fontsize=9)
    ax.set_ylabel("time per step (ms)", color=INK, fontsize=9)
    ax.set_xlim(right=nodes[-1] * 6)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "cpu_time_per_step.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grids", default="100x50,200x100,400x200,800x400,1600x800", help="comma-separated MxN")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threads", default="1,2,4,6,8,10")
    parser.add_argument("--repeats", type=int, default=5, help="timed loops per C++ run, after one warm-up")
    parser.add_argument("--invocations", type=int, default=3, help="separate C++ runs per setting; median reported")
    parser.add_argument("--numpy-repeats", type=int, default=3, help="timed NumPy runs after one warm-up")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    if not PORT.exists():
        raise SystemExit(f"{PORT} not found; build it first (see cpp/README.md)")
    grids = [tuple(int(x) for x in g.split("x")) for g in args.grids.split(",")]
    thread_counts = [int(t) for t in args.threads.split(",")]
    rows = run(grids, args.steps, thread_counts, args.repeats, args.invocations, args.numpy_repeats, args.out)
    plot(rows, args.out)
    print(f"wrote {args.out / 'cpu_benchmark.csv'}, cpu_speedup.pdf, cpu_time_per_step.pdf")


if __name__ == "__main__":
    main()
