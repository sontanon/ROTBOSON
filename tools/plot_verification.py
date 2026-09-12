#!/usr/bin/env python3
"""Generate the M2 paper-verification figures (l = 1..4) from campaign data.

Reads only campaign ``state.json`` step records and
``data/paper/table_ix1.csv`` — no HDF5 field data — and writes PNGs to
``docs/figures/``. Scope: l=1..4 (the verification claim); the l=5/6
campaigns are descoped (see docs/milestone-report-2026-09.md §7).

Critical-point estimation (sample-based, matching the milestone report's
headline Table IX.1 comparison):
- Samples per l: converged steps of ``sweep-l<l>-final`` (full branch) plus
  all ``fold-l<l>-*`` fold campaigns, EXCLUDING the dr=0.02/N=320 chain
  steps — those shrink the domain to 6.4 and drag ω below the paper at
  M/R ≳ 0.4 (report §4.1), so they are off-branch for comparison purposes.
- **ω_min** = minimum ω over the samples (refined fold probes dominate
  where they exist). All fold chains approach the fold one-sided, so this
  slightly *over*estimates ω_min (conservative); l=3/4 folds are
  un-bracketed (chains stopped before turning).
- **M_max / J_max** = maximum M_Komar/J_Komar over the samples. M peaks
  mid-branch for l=1..3; for l=4 the peak sits at the sampled edge (the
  resolution-limited fold end), so the sample maximum is a lower bound.

Usage:
    uv run tools/plot_verification.py [--outdir docs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CAMPAIGNS = REPO / "out" / "campaigns"
PAPER_CSV = REPO / "data" / "paper" / "table_ix1.csv"

Step = dict[str, Any]

L_RANGE = (1, 2, 3, 4)
BRANCH_CAMPAIGN = "sweep-l{l}-final"
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}


def load_paper() -> dict[int, dict[str, float]]:
    """Parse data/paper/table_ix1.csv — header row then l, M_max, J_max, w_min."""
    rows: dict[int, dict[str, float]] = {}
    with open(PAPER_CSV, newline="") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            l, m_max, j_max, w_min = line.split(",")
            if not l.strip().isdigit():
                continue  # header row
            rows[int(l)] = {
                "M_max": float(m_max),
                "J_max": float(j_max),
                "w_min": float(w_min),
            }
    return rows


def load_steps(campaign: str, grid: tuple[float, int] | None = None) -> list[Step]:
    """Converged steps of one campaign, optionally filtered to one (dr, N)."""
    state = CAMPAIGNS / campaign / "state.json"
    if not state.exists():
        return []
    raw = json.loads(state.read_text())
    steps = [s for s in raw.get("steps", []) if s.get("exit_code") == 0]
    if grid is not None:
        steps = [
            s
            for s in steps
            if None not in (s.get("dr"), s.get("N")) and (float(s["dr"]), int(s["N"])) == grid
        ]
    return [s for s in steps if None not in (s.get("omega"), s.get("psi0"))]


def branch_samples(l: int) -> list[Step]:
    return load_steps(BRANCH_CAMPAIGN.format(l=l))


def fold_samples(l: int) -> list[Step]:
    """Fold-region samples from all fold-l{l}-* campaigns."""
    out: list[Step] = []
    for d in sorted(CAMPAIGNS.glob(f"fold-l{l}-*")):
        if d.is_dir():
            out.extend(load_steps(d.name))
    return out


def refined_samples(l: int) -> list[Step]:
    """Branch + fold-campaign samples, excluding the harmful dr=0.02 chains."""
    out: list[Step] = list(branch_samples(l))
    for d in sorted(CAMPAIGNS.glob(f"fold-l{l}-*")):
        if d.is_dir():
            out.extend(load_steps(d.name))
    return [s for s in out if (float(s.get("dr") or 0), int(s.get("N") or 0)) != (0.02, 320)]


def critical_points(l: int) -> dict[str, float | None]:
    """Our critical points for one l, sample-based (report-consistent)."""
    pts = refined_samples(l)
    if not pts:
        return {"w_min": None, "M_max": None, "J_max": None}
    return {
        "w_min": min(s["omega"] for s in pts),
        "M_max": max(s["M_Komar"] for s in pts),
        "J_max": max(s["J_Komar"] for s in pts),
    }


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def fig_branch_curves(
    samples_by_l: dict[int, list[Step]],
    paper: dict[int, dict[str, float]],
    mode: str,
    fname: str,
) -> None:
    """Figures 1–3: branch curves.

    mode = "m" | "j": x = ω, y = M_Komar / J_Komar. The paper's M_max/J_max
    is drawn as a short horizontal dashed segment spanning each branch's ω
    range (Table IX.1 gives the critical M/J but not their ω — M/J peaks are
    distinct critical points from the ω_min fold, so they must not be drawn
    as a single point in this plane). mode = "psi0": x = ψ₀ (log), y = ω,
    fold-grid min-ω sample marked.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for l in L_RANGE:
        pts = samples_by_l[l]
        if mode == "psi0":
            ax.plot(
                [s["psi0"] for s in pts],
                [s["omega"] for s in pts],
                color=COLORS[l],
                lw=1.8,
                label=f"l = {l}",
            )
            cp = critical_points(l)
            if cp["w_min"] is not None:
                best = min(fold_samples(l), key=lambda s: s["omega"])
                ax.plot(
                    [best["psi0"]],
                    [best["omega"]],
                    marker="v",
                    color=COLORS[l],
                    ls="none",
                    ms=9,
                    mec="k",
                    mew=0.5,
                    label="ω_min (ours, fold grid)" if l == 1 else None,
                )
        else:
            ykey = "M_Komar" if mode == "m" else "J_Komar"
            xs = [s["omega"] for s in pts]
            ax.plot(xs, [s[ykey] for s in pts], color=COLORS[l], lw=1.8, label=f"l = {l}")
            ref = paper[l]["M_max" if mode == "m" else "J_max"]
            ax.plot(
                [min(xs), max(xs)],
                [ref, ref],
                color=COLORS[l],
                ls="--",
                lw=1.0,
                alpha=0.65,
                label=("paper Table IX.1 " + ("M_max" if mode == "m" else "J_max"))
                if l == 1
                else None,
            )
    if mode == "psi0":
        ax.set_xscale("log")
        ax.set_xlabel(r"ψ₀ (origin-referenced, log scale)")
    else:
        ax.set_xlabel(r"ω")
    ax.set_ylabel(
        r"$M_{\mathrm{Komar}}$"
        if mode == "m"
        else r"$J_{\mathrm{Komar}}$"
        if mode == "j"
        else r"$\omega$"
    )
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def fig_error_bars(
    crit: dict[int, dict[str, float | None]],
    paper: dict[int, dict[str, float]],
    fname: str,
) -> None:
    """Figure 4: % deviation from paper Table IX.1 for ω_min / M_max / J_max."""
    metrics = [
        ("w_min", "ω_min", "w_min"),
        ("M_max", "M_max", "M_max"),
        ("J_max", "J_max", "J_max"),
    ]
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, (key, label, pkey) in enumerate(metrics):
        xs = [l + (k - 1) * width for l in L_RANGE]
        ys = []
        for l in L_RANGE:
            ours = crit[l][key]
            if ours is None:
                ys.append(float("nan"))
            else:
                ref = paper[l][pkey]
                ys.append(100.0 * (ours - ref) / ref)
        ax.bar(xs, ys, width=width, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(list(L_RANGE))
    ax.set_xticklabels([f"l = {l}" for l in L_RANGE])
    ax.set_ylabel("deviation from paper (%)")
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def fig_domain_study(fname: str) -> None:
    """Figure 5: the §4.1 resolution-vs-domain conclusions (report-consistent).

    Δω of the same physical solution when the grid changes, relative to the
    campaign grid, as concluded in-session (values quoted from the report;
    the study's fixedPhi node radius scales with dr, so cross-grid deltas
    are quoted from the report's analysis rather than re-derived here):
    resolution ×2 moves ω by −1.56%; domain ×2 is negligible (−9e-6); the
    chained dr=0.02/domain-6.4 regrids at high l are actively harmful
    (−2.1% l=3, −4.3% l=4).
    """
    bars = [
        ("resolution ×2\ndr÷2 @ dom 16.4", -1.56),
        ("domain ×2\ndr fixed", -9.0e-4),
        ("chained regrid\n(l=3 fold)", -2.1),
        ("chained regrid\n(l=4 fold)", -4.3),
    ]
    labels = [b[0] for b in bars]
    deltas = [b[1] for b in bars]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    patches = ax.bar(labels, deltas, color=["tab:blue", "tab:orange", "tab:red", "tab:red"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("Δω relative to the campaign grid (%)")
    ax.set_ylim(-5.2, 0.8)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    for bar, d in zip(patches, deltas, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d - 0.12 if d < -0.05 else d + 0.06,
            f"{d:+.2f}%" if abs(d) >= 5e-3 else "≈ 0",
            ha="center",
            va="top" if d < -0.05 else "bottom",
            fontsize=9,
        )
    ax.text(
        0.99,
        0.97,
        "memory ceiling: N=400 (980k unknowns) OOM-killed at 8 GB · chained-regrid bars: dr=0.02, domain 6.4",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=REPO / "docs" / "figures")
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    paper = load_paper()
    samples_by_l = {l: refined_samples(l) for l in L_RANGE}
    crit = {l: critical_points(l) for l in L_RANGE}

    fig_branch_curves(samples_by_l, paper, "m", str(outdir / "branches_m_omega.png"))
    fig_branch_curves(samples_by_l, paper, "psi0", str(outdir / "omega_vs_psi0.png"))
    fig_branch_curves(samples_by_l, paper, "j", str(outdir / "branches_j_omega.png"))
    fig_error_bars(crit, paper, str(outdir / "table_ix1_error.png"))
    fig_domain_study(str(outdir / "domain_resolution_study.png"))

    # Console table: the numbers behind figure 4, visible in the run log.
    print("l   ω_min      M_max      J_max      (paper: w_min / M_max / J_max)")
    for l in L_RANGE:
        cp, pk = crit[l], paper[l]
        print(
            f"{l}   "
            f"{_fmt(cp['w_min']):<10} "
            f"{_fmt(cp['M_max']):<10} "
            f"{_fmt(cp['J_max']):<10} "
            f"({pk['w_min']} / {pk['M_max']} / {pk['J_max']})"
        )
    print(f"\nFigures written to {outdir}")


if __name__ == "__main__":
    main()
