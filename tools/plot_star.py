"""Plot a boson star solution, and compare two solutions of the same problem.

Reads either format:
  * the text format of tools/export_nearfold_states.py and the Rust l1_star example (*.txt);
  * the C solver's solution.h5 (through tools/rotboson_io.py).

Usage:
    uv run tools/plot_star.py fields SOLUTION --output PNG
    uv run tools/plot_star.py compare SOLUTION_A SOLUTION_B --output PNG [--labels A B]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rotboson_io import read_hdf5  # noqa: E402

FIELDS = ["log_alpha_f", "beta_f", "log_h_f", "log_a_f", "psi_f", "lambda_f"]
TITLES = {
    "log_alpha_f": r"$\log\alpha$",
    "beta_f": r"$\beta$",
    "log_h_f": r"$\log h$",
    "log_a_f": r"$\log a$",
    "psi_f": r"$\psi$",
    "lambda_f": r"$\lambda$",
}


@dataclass
class Solution:
    """One solution: grid sizes and steps, l, omega, and the six fields as (nr_total, nz_total) arrays."""

    nr_total: int
    nz_total: int
    ghost: int
    dr: float
    dz: float
    l: int  # noqa: E741
    omega: float
    fields: dict[str, np.ndarray]

    def r(self) -> np.ndarray:
        """Cell-centered rho coordinates of the interior points."""
        return (np.arange(self.ghost, self.nr_total - self.ghost) - self.ghost + 0.5) * self.dr

    def z(self) -> np.ndarray:
        """Cell-centered z coordinates of the interior points."""
        return (np.arange(self.ghost, self.nz_total - self.ghost) - self.ghost + 0.5) * self.dz

    def interior(self, name: str) -> np.ndarray:
        """The field without its ghost cells, indexed [i_rho, j_z]."""
        g = self.ghost
        return self.fields[name][g : self.nr_total - g, g : self.nz_total - g]

    def phi(self) -> np.ndarray:
        """The scalar field phi = rho^l * psi on the interior."""
        return self.r()[:, None] ** self.l * self.interior("psi_f")


def read_text(path: Path) -> Solution:
    """Read the near-fold export text format."""
    header: dict[str, str] = {}
    values: dict[str, list[float]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "field":
            current = parts[1]
            values[current] = []
        elif current is None:
            header[parts[0]] = parts[1]
        else:
            values[current].append(float(parts[0]))
    nr, nz = int(header["nr_total"]), int(header["nz_total"])
    return Solution(
        nr_total=nr,
        nz_total=nz,
        ghost=int(header["ghost"]),
        dr=float(header["dr"]),
        dz=float(header["dz"]),
        l=int(header["l"]),
        omega=float(header["omega"]),
        fields={name: np.array(values[name]).reshape(nr, nz) for name in FIELDS},
    )


def read_h5(path: Path) -> Solution:
    """Read a C solution.h5."""
    datasets, attrs = read_hdf5(path)
    a = cast("dict[str, Any]", attrs)
    nr, nz = int(a["NrTotal"]), int(a["NzTotal"])
    return Solution(
        nr_total=nr,
        nz_total=nz,
        ghost=int(a["ghost"]),
        dr=float(a["dr"]),
        dz=float(a["dz"]),
        l=int(a["l"]),
        omega=float(np.asarray(datasets["w_f.asc"]).reshape(-1)[0]),
        fields={name: np.asarray(datasets[name + ".asc"]).reshape(nr, nz) for name in FIELDS},
    )


def read_solution(path: Path) -> Solution:
    """Read a solution in either format, chosen by the file suffix."""
    return read_h5(path) if path.suffix == ".h5" else read_text(path)


def panel(
    fig: Any, ax: Any, sol: Solution, data: np.ndarray, title: str, cmap: str = "viridis"
) -> None:
    """Draw one (rho, z) color map."""
    mesh = ax.pcolormesh(sol.r(), sol.z(), data.T, shading="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$z$")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, shrink=0.8)


def plot_fields(sol: Solution, output: Path) -> None:
    """Six fields plus phi = rho^l psi, on a 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    flat = list(axes.flat)
    for ax, name in zip(flat, FIELDS, strict=False):
        panel(fig, ax, sol, sol.interior(name), TITLES[name])
    panel(fig, flat[6], sol, sol.phi(), rf"$\phi = \rho^{sol.l}\,\psi$", cmap="magma")
    flat[7].axis("off")
    flat[7].text(
        0.0,
        0.5,
        f"l = {sol.l}\nomega = {sol.omega:.12f}\nN = {sol.nr_total - 2 * sol.ghost}\ndr = {sol.dr}",
        fontsize=14,
        family="monospace",
    )
    fig.savefig(output, dpi=100)
    print(f"wrote {output}")


def plot_compare(a: Solution, b: Solution, labels: list[str], output: Path) -> int:
    """log10 |a - b| for the six fields; print the max difference per field. Return 1 on a grid mismatch."""
    if (a.nr_total, a.nz_total, a.ghost, a.dr, a.dz) != (
        b.nr_total,
        b.nz_total,
        b.ghost,
        b.dr,
        b.dz,
    ):
        print("the two solutions are on different grids")
        return 1
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    print(
        f"omega {labels[0]} {a.omega!r}  {labels[1]} {b.omega!r}  rel diff {abs(a.omega - b.omega) / abs(b.omega):.3e}"
    )
    for ax, name in zip(axes.flat, FIELDS, strict=True):
        diff = np.abs(a.interior(name) - b.interior(name))
        scale = np.abs(b.interior(name)).max()
        print(f"{name:12s} max|diff| {diff.max():.3e}  max|field| {scale:.3e}")
        panel(fig, ax, a, np.log10(diff + 1e-300), f"log10 |diff| {TITLES[name]}", cmap="inferno")
    fig.suptitle(f"{labels[0]} vs {labels[1]}")
    fig.savefig(output, dpi=100)
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_fields = sub.add_parser("fields", help="plot one solution")
    p_fields.add_argument("solution", type=Path)
    p_fields.add_argument("--output", type=Path, required=True)
    p_cmp = sub.add_parser("compare", help="plot the difference of two solutions")
    p_cmp.add_argument("solution_a", type=Path)
    p_cmp.add_argument("solution_b", type=Path)
    p_cmp.add_argument("--output", type=Path, required=True)
    p_cmp.add_argument("--labels", nargs=2, default=["A", "B"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fields":
        plot_fields(read_solution(args.solution), args.output)
        return 0
    return plot_compare(
        read_solution(args.solution_a), read_solution(args.solution_b), args.labels, args.output
    )


if __name__ == "__main__":
    sys.exit(main())
