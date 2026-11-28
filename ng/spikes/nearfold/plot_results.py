"""Draw the two A0.5 result figures from the near-fold CSVs.

Usage (from the repo root):
    uv run ng/spikes/nearfold/plot_results.py --linear LINEAR_CSV [LINEAR_CSV ...] \
        --eig EIG_CSV --out-dir docs/results/figures
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "jacobi": ("tab:red", "o"),
    "gs": ("tab:green", "s"),
    "bordered": ("tab:blue", "D"),
}


def read(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_iterations(rows: list[dict[str, str]], output: Path) -> None:
    """Krylov iterations against distance to the fold, one line per solver and N."""
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for solver, (color, marker) in STYLE.items():
        for n, line in (("128", "-"), ("256", "--")):
            sel = [r for r in rows if r["solver"] == solver and r["N"] == n and r["iterations"]]
            if not sel:
                continue
            sel.sort(key=lambda r: float(r["dist"]))
            dist = [float(r["dist"]) for r in sel]
            its = [int(r["iterations"]) for r in sel]
            ok = [r["converged"] == "true" for r in sel]
            ax.plot(dist, its, line, color=color, label=f"{solver}, N={n}")
            ax.scatter(
                [d for d, o in zip(dist, ok, strict=True) if o],
                [i for i, o in zip(its, ok, strict=True) if o],
                color=color,
                marker=marker,
            )
            ax.scatter(
                [d for d, o in zip(dist, ok, strict=True) if not o],
                [i for i, o in zip(its, ok, strict=True) if not o],
                color=color,
                marker="x",
                s=60,
            )
    ax.axvline(0.0, color="k", lw=0.8, ls=":", label="fold estimate")
    ax.set_yscale("log")
    ax.set_xlabel(r"distance to the fold, $\psi_0^* - \psi_0$")
    ax.set_ylabel("FGMRES iterations to 1e-10 (x = did not converge)")
    ax.set_title("Linear solves along the l=1 branch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.savefig(output, dpi=100)
    print(f"wrote {output}")


def plot_eigenvalues(rows: list[dict[str, str]], output: Path) -> None:
    """Smallest eigenvalue magnitude of the fixed-omega and fixed-phi Jacobians against psi0."""
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    psi0 = [float(r["psi0"]) for r in rows]
    for key, label, color in (
        ("lambda_fixed_omega", "fixed-omega Jacobian", "tab:red"),
        ("lambda_fixed_phi", "fixed-phi Jacobian", "tab:blue"),
    ):
        values = [abs(float(r[key])) for r in rows]
        ax.plot(psi0, values, "o-", color=color, label=label)
    ax.set_xlabel(r"$\psi_0$")
    ax.set_ylabel(r"$|\lambda_{\min}|$ (inverse iteration)")
    ax.set_title("Smallest Jacobian eigenvalue, N=128")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output, dpi=100)
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--linear", type=Path, nargs="+", required=True)
    parser.add_argument("--eig", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    linear = [row for path in args.linear for row in read(path)]
    plot_iterations(linear, args.out_dir / "a05_iterations_vs_distance.png")
    plot_eigenvalues(read(args.eig), args.out_dir / "a05_smallest_eigenvalue.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
