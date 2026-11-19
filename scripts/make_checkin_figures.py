"""Make the figures for the check-in report.

Writes two PDFs into report/figures/:

    space_convergence.pdf  price error against grid size (Craig-Sneyd, N = m steps)
    time_convergence.pdf   time error against step count for the four ADI schemes (rho = -0.9)

Run from the repository root (takes about 20 seconds):

    python scripts/make_checkin_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adi_heston import HestonParams, OptionSpec
from adi_heston.grid import Grid
from adi_heston.reference import heston_put
from adi_heston.solver import SCHEMES, solve

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"

K, T = 100.0, 1.0
OPTION = OptionSpec(K, T)
S0, V0 = 100.0, 0.04

# Categorical colours in fixed order (checked for colour-blind separation). Two of them are
# light against white, so every series also gets its own marker and a legend entry.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"

SCHEME_LABELS = {
    "douglas": "Douglas",
    "craig_sneyd": "Craig–Sneyd",
    "modified_craig_sneyd": "Modified Craig–Sneyd",
    "hundsdorfer_verwer": "Hundsdorfer–Verwer",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "savefig.bbox": "tight",
    }
)


def heston(rho: float) -> HestonParams:
    """The test case used throughout the check-in report."""
    return HestonParams(r=0.025, q=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=rho)


def guide(ax, x, y_start, slope, label, where="below"):
    """Draw a dashed reference line of the given log-log slope starting at (x[0], y_start)."""
    x = np.asarray(x, dtype=float)
    y = y_start * (x / x[0]) ** slope
    ax.plot(x, y, ls="--", lw=0.8, color=MUTED, zorder=1)
    va = "top" if where == "below" else "bottom"
    ax.annotate(label, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8, va=va)


def space_convergence() -> None:
    """Price error at (S0, v0) against the semi-analytic price as the grid is refined.

    Craig-Sneyd is second order in time, so with N = m steps its time error stays small and the
    plot shows the space error.
    """
    ms = [50, 100, 200, 400]
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    largest_first_error = 0.0
    for k, rho in enumerate((0.0, -0.9)):
        params = heston(rho)
        ref = heston_put(params, OPTION, S0, V0)
        errors = []
        for m in ms:
            grid = Grid.build(K, 8 * K, 5.0, m, m // 2)
            errors.append(abs(solve(params, OPTION, grid, m, scheme="craig_sneyd").price(S0, V0) - ref))
        print(f"space  rho={rho:+.1f}", " ".join(f"{e:.2e}" for e in errors))
        ax.loglog(ms, errors, color=COLORS[k], marker=MARKERS[k], label=f"ρ = {rho:g}".replace("-", "−"), zorder=3)
        largest_first_error = max(largest_first_error, errors[0])
    guide(ax, ms, 2.0 * largest_first_error, -2, "slope 2", where="above")

    ax.set_xlabel("intervals in $s$ ($m$; $v$ has $m/2$)")
    ax.set_ylabel("absolute price error")
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.minorticks_off()
    ax.grid(True, which="major")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "space_convergence.pdf")
    plt.close(fig)


def time_convergence() -> None:
    """Largest difference over a fixed 61 x 31 grid from a run with 2560 steps, rho = -0.9."""
    params, grid = heston(-0.9), Grid.build(K, 8 * K, 5.0, 60, 30)
    Ns = [10, 20, 40, 80, 160]
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    for k, scheme in enumerate(SCHEMES):
        U_ref = solve(params, OPTION, grid, 2560, scheme=scheme).U
        errors = [np.abs(solve(params, OPTION, grid, N, scheme=scheme).U - U_ref).max() for N in Ns]
        print(f"time   {scheme:22s}", " ".join(f"{e:.2e}" for e in errors))
        # Craig-Sneyd and modified Craig-Sneyd almost coincide: draw CS with larger markers underneath.
        size = 8 if scheme == "craig_sneyd" else 5
        ax.loglog(Ns, errors, color=COLORS[k], marker=MARKERS[k], markersize=size, label=SCHEME_LABELS[scheme],
                  zorder=2 if scheme == "craig_sneyd" else 3)
    guide(ax, Ns, 2.0e-1, -1, "slope 1", where="above")
    guide(ax, Ns, 4.0e-2, -2, "slope 2")

    ax.set_xlabel("time steps $N$")
    ax.set_ylabel("time error (max over grid)")
    ax.set_xticks(Ns, [str(N) for N in Ns])
    ax.minorticks_off()
    ax.grid(True, which="major")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "time_convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    space_convergence()
    time_convergence()
    print(f"wrote figures to {OUT}")
