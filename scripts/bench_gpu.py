"""Benchmark the CUDA port against the C++ port over grids, schemes and option types.

For every grid, scheme (Douglas, Craig–Sneyd, Modified Craig–Sneyd, Hundsdorfer–Verwer) and option
type (European, American) this writes the time-loop inputs with ``generate_inputs``, times the GPU
(``adi_cuda --bench``) and the C++ port (``adi_port --bench`` at a few thread counts), then deletes the
inputs again (one 3201 x 1601 folder is about 330 MB). Both ports time only the time loop; the GPU's
one-time setup (uploads and factorization) is reported separately. Times are per schedule row.

Writes into --out (default results/, gitignored):
    gpu_benchmark.csv      one row per grid, scheme, option type, implementation and thread count
    gpu_state.txt          GPU clocks, temperature and power before and after the sweep
    gpu_time_per_row.pdf   Douglas, European: time per row against grid size for NumPy (if a
                           bench_cpu.py CSV is given), C++ on 1 thread, the best C++ and the GPU
    gpu_speedup.pdf        GPU speed-up over the best C++ time, per scheme, European and American

Usage (from the repository root, after building cuda/ and cpp/ as their READMEs describe):
    uv run --no-project --with-requirements requirements.txt --with-editable . -- python scripts/bench_gpu.py
    ... python scripts/bench_gpu.py --grids 100x50,400x200 --schemes douglas --cpu-threads 1,8 \\
            --cpu-baseline results/cpu_benchmark.csv

Project: ADI finite-difference methods for American options under Heston
         Math/CS 714, UW–Madison, Fall 2026
Author:  Ganesh Arivoli <arivoli@wisc.edu>
"""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import statistics
import subprocess
from pathlib import Path

from adi_heston.fixtures import BENCH_DIR, SCHEME_IDS, generate_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_PORT = REPO_ROOT / "cuda" / "build" / "default" / "adi_cuda"
CPU_PORT = REPO_ROOT / "cpp" / "build" / "adi_port"
SCHEMES = list(SCHEME_IDS)  # douglas, craig_sneyd, modified_craig_sneyd, hundsdorfer_verwer
LABELS = {"douglas": "Douglas", "craig_sneyd": "Craig–Sneyd", "modified_craig_sneyd": "Modified Craig–Sneyd",
          "hundsdorfer_verwer": "Hundsdorfer–Verwer"}
FIELDS = ["cpu", "gpu", "rows", "cols", "steps", "scheme", "american", "implementation", "threads",
          "median_ms_per_row", "min_ms_per_row", "setup_ms"]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:  # macOS
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def nvidia_smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True,
                          text=True, check=True).stdout.strip()


def time_gpu(inputs: Path, repeats: int, invocations: int) -> tuple[float, float, float, int]:
    """(median, min, setup, schedule rows) over separate runs of adi_cuda --bench: median of medians, min of mins."""
    medians, mins, setups = [], [], []
    for _ in range(invocations):
        out = subprocess.run([str(GPU_PORT), str(inputs), "--bench", "--repeats", str(repeats)],
                             capture_output=True, text=True, check=True).stdout.strip()
        fields = out.split(",")  # gpu,rows,cols,steps,median,min,setup_ms,scheme,american
        medians.append(float(fields[4]))
        mins.append(float(fields[5]))
        setups.append(float(fields[6]))
    return statistics.median(medians), min(mins), statistics.median(setups), int(fields[3])


def time_cpu(inputs: Path, threads: int, repeats: int, invocations: int) -> tuple[float, float]:
    """(median, min) over separate runs of adi_port --bench."""
    medians, mins = [], []
    for _ in range(invocations):
        out = subprocess.run([str(CPU_PORT), str(inputs), "--bench", "--threads", str(threads), "--repeats",
                              str(repeats)], capture_output=True, text=True, check=True).stdout.strip()
        fields = out.split(",")  # threads,rows,cols,steps,median,min
        medians.append(float(fields[4]))
        mins.append(float(fields[5]))
    return statistics.median(medians), min(mins)


def run(args: argparse.Namespace) -> list[dict]:
    cpu, gpu = cpu_name(), nvidia_smi("name")
    scratch = BENCH_DIR / "bench_gpu_tmp"
    rows = []
    for m, n in args.grids:
        for scheme in args.schemes:
            for american in (False, True):
                inputs = generate_inputs(m, n, args.steps, BENCH_DIR if args.keep_inputs else scratch, scheme, american)
                median, best, setup, schedule_rows = time_gpu(inputs, args.repeats, args.invocations)
                base = dict(cpu=cpu, gpu=gpu, rows=m + 1, cols=n + 1, steps=schedule_rows, scheme=scheme,
                            american=int(american))
                rows.append(dict(base, implementation="gpu", threads="", median_ms_per_row=median,
                                 min_ms_per_row=best, setup_ms=setup))
                line = f"{m + 1}x{n + 1} {scheme:20s} {'american' if american else 'european':8s} gpu {median:9.4f}"
                for t in args.cpu_threads:
                    median, best = time_cpu(inputs, t, args.cpu_repeats, args.cpu_invocations)
                    rows.append(dict(base, implementation="cpp", threads=t, median_ms_per_row=median,
                                     min_ms_per_row=best, setup_ms=""))
                    line += f" | cpp {t:2d}t {median:9.4f}"
                print(line + " ms/row", flush=True)
                if not args.keep_inputs:
                    shutil.rmtree(inputs)
    shutil.rmtree(scratch, ignore_errors=True)
    return rows


# ---- figures ------------------------------------------------------------------------------------

INK, INK_2, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
# Categorical slots 1-4 in fixed order (validated: CVD and normal-vision separation pass; slots 3-4 are
# below 3:1 contrast on white, so every series also gets its own marker and a direct label).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["s", "o", "^", "D"]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def _series(ax, x, y, slot: int, label: str) -> tuple[float, float, str]:
    """Plot one series; return its last point and label for _direct_labels."""
    ax.plot(x, y, color=SLOTS[slot], linewidth=1.6, marker=MARKERS[slot], markersize=5, label=label)
    return x[-1], y[-1], label


def _direct_labels(ax, ends: list[tuple[float, float, str]], min_gap: float = 9.0) -> None:
    """Label each series at its last point, pushing labels that share an x apart by at least min_gap points.
    Call after the axis limits are final."""
    to_points = ax.transData + ax.figure.dpi_scale_trans.inverted()  # data -> inches
    by_x: dict[float, list] = {}
    for x, y, label in ends:
        by_x.setdefault(x, []).append((to_points.transform((x, y))[1] * 72.0, x, y, label))
    for group in by_x.values():
        group.sort()
        placed = []
        for y_pt, x, y, label in group:
            target = max(y_pt, placed[-1] + min_gap) if placed else y_pt
            placed.append(target)
            ax.annotate(label, (x, y), xytext=(6, target - y_pt), textcoords="offset points", va="center",
                        fontsize=7, color=INK_2)


def plot(rows: list[dict], numpy_rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted({(r["rows"], r["cols"]) for r in rows})
    nodes = [g[0] * g[1] for g in grids]

    def pick(impl, scheme, american, g, threads=None):
        return [r["median_ms_per_row"] for r in rows if r["implementation"] == impl and r["scheme"] == scheme
                and r["american"] == american and (r["rows"], r["cols"]) == g
                and (threads is None or r["threads"] == threads)]

    def best_cpp(scheme, american, g):
        return min(pick("cpp", scheme, american, g))

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    def legend_above(fig, ax, ncol):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=7, labelcolor=INK_2)

    # Douglas, European: time per row against grid size.
    if "douglas" in {r["scheme"] for r in rows}:
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        ends = []
        if numpy_rows:
            numpy_by_grid = {(int(r["rows"]), int(r["cols"])): float(r["median_ms_per_step"]) for r in numpy_rows}
            pts = [(g[0] * g[1], numpy_by_grid[g]) for g in grids if g in numpy_by_grid]
            if pts:
                ends.append(_series(ax, [p[0] for p in pts], [p[1] for p in pts], 0, "NumPy"))
        one_thread = [pick("cpp", "douglas", 0, g, 1) for g in grids]
        if all(one_thread):
            ends.append(_series(ax, nodes, [v[0] for v in one_thread], 1, "C++, 1 thread"))
        ends.append(_series(ax, nodes, [best_cpp("douglas", 0, g) for g in grids], 2, "C++, best"))
        ends.append(_series(ax, nodes, [pick("gpu", "douglas", 0, g)[0] for g in grids], 3, "GPU"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid nodes (m+1)(n+1)", color=INK, fontsize=9)
        ax.set_ylabel("time per row (ms)", color=INK, fontsize=9)
        ax.set_xlim(right=nodes[-1] * 8)
        _style(ax)
        legend_above(fig, ax, len(ends))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _direct_labels(ax, ends)
        fig.savefig(out_dir / "gpu_time_per_row.pdf")
        plt.close(fig)

    # GPU speed-up over the best C++ time, per scheme; European and American side by side, one shared axis.
    schemes = [s for s in SCHEMES if s in {r["scheme"] for r in rows}]
    speedups = {(s, a): [best_cpp(s, a, g) / pick("gpu", s, a, g)[0] for g in grids] for s in schemes for a in (0, 1)}
    low = min(min(min(v) for v in speedups.values()) / 1.3, 0.8)  # keep the 1x line in view
    high = max(max(v) for v in speedups.values()) * 1.3
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if low <= t <= high]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
    for ax, american, title in ((axes[0], 0, "European put"), (axes[1], 1, "American put")):
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ends = [_series(ax, nodes, speedups[(s, american)], SCHEMES.index(s), LABELS[s]) for s in schemes]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_xlabel("grid nodes (m+1)(n+1)", color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlim(right=nodes[-1] * 30)
        ax.text(nodes[0], 1.0, "GPU = best C++ ", color=INK_2, fontsize=7, ha="left", va="bottom")
        _style(ax)
        ax.ends = ends
    axes[0].set_ylabel("GPU speed-up over the best C++", color=INK, fontsize=9)
    legend_above(fig, axes[0], len(schemes))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ax in axes:
        _direct_labels(ax, ax.ends)
    fig.savefig(out_dir / "gpu_speedup.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grids", default="100x50,200x100,400x200,800x400,1600x800,3200x1600",
                        help="comma-separated MxN")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--repeats", type=int, default=10, help="timed GPU runs per invocation, after a warm-up")
    parser.add_argument("--invocations", type=int, default=3, help="separate GPU runs per problem; median reported")
    parser.add_argument("--cpu-threads", default="1,8,16", help="C++ thread counts; empty to skip the CPU")
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--cpu-invocations", type=int, default=1)
    parser.add_argument("--cpu-baseline", type=Path, default=None,
                        help="cpu_benchmark.csv from scripts/bench_cpu.py, for the NumPy times")
    parser.add_argument("--keep-inputs", action="store_true", help="keep the generated inputs in bench_inputs/")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    args.grids = [tuple(int(x) for x in g.split("x")) for g in args.grids.split(",")]
    args.schemes = [s for s in args.schemes.split(",") if s]
    unknown = set(args.schemes) - set(SCHEMES)
    if unknown:
        raise SystemExit(f"unknown schemes {sorted(unknown)}; choose from {SCHEMES}")
    args.cpu_threads = [int(t) for t in args.cpu_threads.split(",") if t]
    for port in [GPU_PORT] + ([CPU_PORT] if args.cpu_threads else []):
        if not port.exists():
            raise SystemExit(f"{port} not found; build it first")

    args.out.mkdir(parents=True, exist_ok=True)
    state = "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate"
    before = nvidia_smi(state)
    rows = run(args)
    after = nvidia_smi(state)
    (args.out / "gpu_state.txt").write_text(f"{state}\nbefore: {before}\nafter:  {after}\n")
    with open(args.out / "gpu_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    numpy_rows = []
    if args.cpu_baseline:
        with open(args.cpu_baseline) as f:
            numpy_rows = [r for r in csv.DictReader(f) if r["implementation"] == "numpy"]
    if args.cpu_threads:
        plot(rows, numpy_rows, args.out)
        print(f"wrote {args.out / 'gpu_benchmark.csv'}, gpu_state.txt, gpu_time_per_row.pdf, gpu_speedup.pdf")
    else:
        print(f"wrote {args.out / 'gpu_benchmark.csv'} and gpu_state.txt (no CPU times, so no figures)")


if __name__ == "__main__":
    main()
