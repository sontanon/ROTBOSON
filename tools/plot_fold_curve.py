"""Plot the fold curve of a continuation campaign: omega against psi0 (full and zoomed), and mass against omega.

Marks the fold estimate from summary.json and, optionally, the states listed in a
near-fold manifest.csv (from tools/export_nearfold_states.py).

Usage:
    uv run tools/plot_fold_curve.py --campaign out/campaigns/runbook-l1 \
        --manifest out/nearfold/l1/manifest.csv --output docs/results/figures/a05_l1_fold_curve.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

COLORS = {128: "tab:blue", 256: "tab:orange"}
ZOOM_PSI0 = 0.30  # the zoom panel shows psi0 from here on


def load_steps(campaign: Path) -> list[dict[str, Any]]:
    """Converged steps of the campaign (exit_code 0), in step order."""
    state = json.loads((campaign / "state.json").read_text())
    return [s for s in state["steps"] if s["exit_code"] == 0]


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Rows of a near-fold manifest.csv."""
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps = load_steps(args.campaign)
    summary = json.loads((args.campaign / "summary.json").read_text())
    by_step = {s["i"]: s for s in steps}

    fig, (ax_w, ax_z, ax_m) = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    fold_psi0, fold_omega = summary["psi0"], summary["omega"]
    for n in sorted({s["N"] for s in steps}):
        sub = [s for s in steps if s["N"] == n]
        color = COLORS.get(n, "tab:gray")
        ax_w.plot(
            [s["psi0"] for s in sub],
            [s["omega"] for s in sub],
            ".-",
            color=color,
            ms=3,
            label=f"N={n}",
        )
        ax_m.plot(
            [s["omega"] for s in sub],
            [s["M_Komar"] for s in sub],
            ".-",
            color=color,
            ms=3,
            label=f"N={n}",
        )
        near = [s for s in sub if s["psi0"] >= ZOOM_PSI0]
        ax_z.plot(
            [s["psi0"] for s in near],
            [s["omega"] - fold_omega for s in near],
            ".-",
            color=color,
            ms=4,
            label=f"N={n}",
        )

    ax_z.axhline(0.0, color="k", lw=0.8, ls=":")
    ax_z.plot([fold_psi0], [0.0], "k*", ms=14, label="fold estimate")
    ax_w.axhline(fold_omega, color="k", lw=0.8, ls=":")
    ax_w.plot(
        [fold_psi0],
        [fold_omega],
        "k*",
        ms=14,
        label=f"fold estimate: psi0={fold_psi0:.5f}, omega={fold_omega:.6f}",
    )
    ax_m.axvline(fold_omega, color="k", lw=0.8, ls=":", label=f"omega_min={fold_omega:.6f}")

    if args.manifest:
        rows = load_manifest(args.manifest)
        marked = [by_step[int(r["step"])] for r in rows if int(r["step"]) in by_step]
        ax_w.plot(
            [s["psi0"] for s in marked],
            [s["omega"] for s in marked],
            "o",
            mfc="none",
            mec="red",
            ms=10,
            label=f"near-fold test states ({len(marked)})",
        )
        zoomed = [s for s in marked if s["psi0"] >= ZOOM_PSI0]
        ax_z.plot(
            [s["psi0"] for s in zoomed],
            [s["omega"] - fold_omega for s in zoomed],
            "o",
            mfc="none",
            mec="red",
            ms=10,
            label="near-fold test states",
        )
        ax_m.plot(
            [s["omega"] for s in marked],
            [s["M_Komar"] for s in marked],
            "o",
            mfc="none",
            mec="red",
            ms=10,
            label="near-fold test states",
        )
        print(f"marked {len(marked)} of {len(rows)} manifest states")

    ax_w.set_xlabel(r"$\psi_0$ (central scalar field)")
    ax_w.set_ylabel(r"$\omega$")
    ax_w.set_title("Frequency along the branch")
    ax_w.legend(fontsize=9)
    ax_w.grid(alpha=0.3)
    ax_z.set_xlabel(r"$\psi_0$")
    ax_z.set_ylabel(r"$\omega - \omega_{\rm fold}$")
    ax_z.set_title(f"Zoom: psi0 >= {ZOOM_PSI0}")
    ax_z.legend(fontsize=9)
    ax_z.grid(alpha=0.3)
    ax_m.set_xlabel(r"$\omega$")
    ax_m.set_ylabel(r"$M$ (Komar)")
    ax_m.set_title("Mass against frequency")
    ax_m.legend(fontsize=9)
    ax_m.grid(alpha=0.3)
    fig.suptitle(f"Fold curve: {args.campaign.name} ({len(steps)} converged steps)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
