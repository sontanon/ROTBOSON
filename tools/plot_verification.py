#!/usr/bin/env python3
"""Generate the M2 paper-verification figures (l = 1..4) from campaign data.

Usage:
    uv run tools/plot_verification.py --campaigns runbook-l1,runbook-l2,...
    uv run tools/plot_verification.py --campaigns runbook-l1 --outdir docs/figures

``--campaigns`` is a manifest: the explicit list of campaign names (under
``out/campaigns/``) feeding the figures. This is deliberate — exploratory
campaigns must never pollute the verification figures (the 2026-09 figures
were polluted by mixed-grid fold-campaign samples). Branch segments are
plotted per grid (dr, N) with distinct line styles, so a refinement's ω
kink is visible rather than hidden.

Critical-point estimation (sample-based, matching the milestone report):
per l, over the manifest campaigns' converged steps (excluding the harmful
dr=0.02/N=320 chain samples — off-branch, 2026-09 §4.1):
- ω_min = the minimum sampled ω;
- M_max / J_max = the maximum sampled M_Komar / J_Komar (for l=4 the M peak
  sits at the sampled edge — the resolution-limited fold end — so the
  sample maximum is a lower bound).

Reads only campaign ``state.json`` files and ``data/paper/table_ix1.csv``;
no HDF5 field data. Writes PNGs to ``docs/figures/``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CAMPAIGNS = REPO / "out" / "campaigns"
PAPER_CSV = REPO / "data" / "paper" / "table_ix1.csv"

Step = dict[str, Any]

L_RANGE = (1, 2, 3, 4)
COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
# Line style per grid resolution: the segment's grid is visible per branch.
N_STYLES = {
    128: "solid",
    192: (0, (5, 2)),
    256: (0, (1, 1)),
}


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
    state = (CAMPAIGNS / campaign) / "state.json"
    if not state.exists():
        state = Path(campaign) / "state.json"
    if not state.exists():
        raise SystemExit(f"campaign not found: {campaign} (no state.json)")
    raw = json.loads(state.read_text())
    steps = [s for s in raw.get("steps", []) if s.get("exit_code") == 0]
    if grid is not None:
        steps = [
            s
            for s in steps
            if None not in (s.get("dr"), s.get("N")) and (float(s["dr"]), int(s["N"])) == grid
        ]
    return [s for s in steps if None not in (s.get("omega"), s.get("psi0"))]


def _l_of_step(step: Step) -> int:
    """l from the step's solution dirname (l=X,w=...)."""
    name = step.get("sol_dir") or ""
    m = re.search(r"l=(\d+),", name)
    if not m:
        raise SystemExit(f"cannot parse l from sol_dir: {name}")
    return int(m.group(1))


def manifest_samples(names: list[str]) -> dict[int, list[Step]]:
    """Converged steps grouped per l, over exactly the manifest's campaigns."""
    out: dict[int, list[Step]] = {}
    for name in names:
        for step in load_steps(name):
            out.setdefault(_l_of_step(step), []).append(step)
    for l in out:
        out[l].sort(key=lambda s: s["psi0"])
    return out


def drop_off_branch(samples: dict[int, list[Step]]) -> dict[int, list[Step]]:
    """Exclude the harmful dr=0.02/N=320 chain samples (2026-09 §4.1)."""
    return {
        l: [s for s in pts if (float(s.get("dr") or 0), int(s.get("N") or 0)) != (0.02, 320)]
        for l, pts in samples.items()
    }


def critical_points(samples: list[Step]) -> dict[str, float | None]:
    """Sample-based critical points for one branch (report-consistent)."""
    if not samples:
        return {"w_min": None, "M_max": None, "J_max": None}
    return {
        "w_min": min(s["omega"] for s in samples),
        "M_max": max(s["M_Komar"] for s in samples),
        "J_max": max(s["J_Komar"] for s in samples),
    }


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def _segments(samples: list[Step]) -> list[tuple[tuple[float, int], list[Step]]]:
    """Split a branch into contiguous same-grid segments (in ψ₀ order)."""
    segs: list[tuple[tuple[float, int], list[Step]]] = []
    for s in samples:
        key = (float(s["dr"]), int(s["N"]))
        if segs and segs[-1][0] == key:
            segs[-1][1].append(s)
        else:
            segs.append((key, [s]))
    return segs


def fig_branch_curves(
    samples_by_l: dict[int, list[Step]],
    paper: dict[int, dict[str, float]],
    mode: str,
    fname: str,
) -> None:
    """Figures 1–3: branch curves, segments styled per grid (dr, N).

    mode = "m" | "j": x = ω, y = M_Komar / J_Komar, paper critical values as
    short horizontal dashed references (Table IX.1 gives the critical M/J
    but not their ω — they are distinct critical points from ω_min and must
    not be drawn as single points in this plane). mode = "psi0": x = ψ₀
    (log), y = ω, the fold's min-ω sample marked.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for l in L_RANGE:
        pts = samples_by_l.get(l) or []
        if not pts:
            continue
        segs = _segments(pts)
        for seg_i, ((_dr, n), seg) in enumerate(segs):
            style = N_STYLES.get(n, "solid")
            first = seg_i == 0
            if mode == "psi0":
                ax.plot(
                    [s["psi0"] for s in seg],
                    [s["omega"] for s in seg],
                    color=COLORS[l],
                    lw=1.8,
                    ls=style,
                    label=f"l = {l}" if first else None,
                )
            else:
                ykey = "M_Komar" if mode == "m" else "J_Komar"
                ax.plot(
                    [s["omega"] for s in seg],
                    [s[ykey] for s in seg],
                    color=COLORS[l],
                    lw=1.8,
                    ls=style,
                    label=f"l = {l}" if first else None,
                )
                ref = paper[l]["M_max" if mode == "m" else "J_max"]
                ax.plot(
                    [min(s["omega"] for s in pts), max(s["omega"] for s in pts)],
                    [ref, ref],
                    color=COLORS[l],
                    ls=":",
                    lw=1.0,
                    alpha=0.6,
                    label=("paper Table IX.1 " + ("M_max" if mode == "m" else "J_max"))
                    if l == 1 and first
                    else None,
                )
        if mode == "psi0":
            cp = critical_points(pts)
            if cp["w_min"] is not None:
                best = min(pts, key=lambda s: s["omega"])
                ax.plot(
                    [best["psi0"]],
                    [best["omega"]],
                    marker="v",
                    color=COLORS[l],
                    ls="none",
                    ms=9,
                    mec="k",
                    mew=0.5,
                    label="ω_min (ours, sampled)" if l == 1 else None,
                )
    handles, labels = ax.get_legend_handles_labels()
    used_n = {int(s.get("N") or 0) for pts in samples_by_l.values() for s in pts}
    for n, style in sorted(N_STYLES.items()):
        if n in used_n:
            handles.append(Line2D([], [], color="0.4", lw=1.5, ls=style))
            labels.append(f"N={n} segment")
    ax.set_xscale("log") if mode == "psi0" else None
    ax.set_xlabel(r"ψ₀ (origin-referenced, log scale)" if mode == "psi0" else r"ω")
    ax.set_ylabel(
        r"$M_{\mathrm{Komar}}$"
        if mode == "m"
        else r"$J_{\mathrm{Komar}}$"
        if mode == "j"
        else r"$\omega$"
    )
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(handles=handles, labels=labels)
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
        ("resolution ×2\nndr÷2 @ dom 16.4", -1.56),
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
    ap.add_argument(
        "--campaigns",
        required=True,
        help="comma-separated campaign names (manifest): exactly the campaigns "
        "feeding the figures, e.g. runbook-l1,runbook-l2,runbook-l3,runbook-l4",
    )
    ap.add_argument("--outdir", type=Path, default=REPO / "docs" / "figures")
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.campaigns.split(",") if n.strip()]

    paper = load_paper()
    samples = drop_off_branch(manifest_samples(names))
    missing = [l for l in L_RANGE if not samples.get(l)]
    if missing:
        raise SystemExit(f"no samples for l={missing} — check the manifest")
    crit = {l: critical_points(samples.get(l) or []) for l in L_RANGE}

    fig_branch_curves(samples, paper, "m", str(outdir / "branches_m_omega.png"))
    fig_branch_curves(samples, paper, "psi0", str(outdir / "omega_vs_psi0.png"))
    fig_branch_curves(samples, paper, "j", str(outdir / "branches_j_omega.png"))
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
