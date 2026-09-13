# Runbook: rotating boson-star catalogue regeneration (l = 1..4)

One document, top to bottom: from the committed starter seed to the four
branches and the verification figures. One campaign per branch; every
command is deterministic; the validated numbers are recorded inline. This
is the M2 deliverable — a first-time user should be able to reproduce the
catalogue on modest hardware (8 GB RAM, 4 cores) without tuning anything.

Everything runs from the repo root unless a step says otherwise.

## 0. Prerequisites

```bash
# build (oneMKL) — see README.md for the full prerequisite list
source /opt/intel/oneapi/setvars.sh
cmake --preset release && cmake --build --preset release -j
uv sync --dev
```

Memory budget: the finest grid used is N=256 (≈405k unknowns, ~1.4–2.3 GB
during factorization). N=512 (1.6M) does **not** fit in 8 GB — every spec
below caps the refinement ladder accordingly.

## a. Starter seed (l=1, ω=0.95)

```bash
cd out && ../build/release/ROTBOSON ../configs/l1_seed.toml && cd ..
```

Produces `out/l=1,w=9.50000E-01,dr=1.25000E-01,N=0128` (~1 min): the
validated from-scratch ansatz (ω fixed at 0.95, ψ0=0.01, σ=4, rExt=12).
Verified on-branch: `M_Komar1 = M_Komar2`, ψ(0)=2.8388e-3.

Known properties (documented, do not "fix" them):
- **Never retune the ansatz.** Lower-ω from-scratch seeding failed 5/5
  attempts (trivial attractor, spurious fat solution, PARDISO −4) — the
  ansatz is ω-tuned as well as l-tuned.
- The seed's support fills the domain (r99 ≈ 95% of 16.5 at ω=0.95). This
  is harmless for a seed: the continuation moves denser immediately and
  every regrid re-solves with its own boundary treatment.

## b. Regrid the seed (the grid/resolution tool)

```bash
uv run tools/sweep_driver.py configs/l1_regrid_n256.toml
```

Re-solves the seed's branch point on N=256/dr=0.0625 **keeping the
domain**, via the driver's cross-grid interpolation — ψ₀ lands exactly and
the campaign stops after one solve (`done:psi0_target`). Verified: ω 0.95 →
0.950315 (+0.033%, the resolution refinement), M_Komar 0.78931 → 0.78694.
This spec is the template for "same solution, different grid": edit
`[grid]`, recompute `psi0_target` (the formula is in the file), run.

## c. The l=1 branch: down-campaign to the fold

```bash
uv run tools/sweep_driver.py configs/runbook_l1_down.toml
uv run tools/sweep_driver.py configs/runbook_l1_down.toml --summarize
```

What it does: steps ψ₀ upward in 3% increments from the starter (ω falls
0.95 → the fold). Near the fold the field's half-width drops below 8 cells
(`hwl_min`) → **one refinement fires, N 128 → 256, domain kept** — the
vertex is measured at the finer resolution. `stop_at_turning_point` stops
the campaign 1–2 steps past the vertex; `--summarize` localizes ω_min
(degree-4 fit over the vertex-region samples, paper §IX).

Recorded expectations (l=1, achieved 2026-09-13):
- the refinement trigger fires around ψ₀ ≈ 0.31–0.35 (the half-width dips
  8 → 7 cells), a few steps before the vertex (ψ₀ ≈ 0.39–0.46);
- ω_min = 0.6460482 (paper: 0.64561; **+0.068%** — the dr=0.0625
  discretization on the kept domain; the fold is the most
  resolution-sensitive critical point);
- the ψ0_target=0.6 is a safety net — the detector fires first;
- wall-time ≈ 40–70 min (the coarse stretch at N=128, the fold region at
  N=256).

The figure shows a small ω kink where the branch changes grid
(+0.23% at dr=0.125 → the fine grid): branch segments are colored per N so
the kink is visible, not hidden.

## d. Cross-l seeds (l = 2..4) — no campaign needed

The branch-point amplitude at ω≈0.9 scales nearly **×0.073–0.079 per l**
(measured: 0.0738 (l=1→2), 0.0726 (2→3), 0.0790 (3→4)). Cross-l jumps
happen **at ω≈0.9**, where that ratio is measured — jumping from a
high-ω donor with the ω=0.9 ratio fails (the amplitude ratio is
ω-dependent; the l=3 attempt from a ω=0.97 donor diverged 3/3).

The ladder — one solve per arrow, every link live-validated:

| arrow | spec | result |
|---|---|---|
| l=1 starter → l=2 seed | `configs/l2_seed_from_l1.toml` (ψ₀ ×0.0738) | l=2 @ ω=0.9696 (M=0.7749) |
| l=2 @ 0.9696 → l=2 @ 0.9 (waypoint) | `configs/l2_waypoint_w09.toml` | l=2 @ ω=0.9000 |
| l=2 @ 0.9 → l=3 seed | `configs/l3_seed_at_w09.toml` (ψ₀ ×0.0726) | l=3 @ ω=0.8998 (M=1.7939) |
| l=3 @ 0.9 → l=4 seed (ψ₀ ×0.0790) | `configs/l4_seed_at_w09.toml` | l=4 @ ω=0.8999 (M=2.0338) |

The seed-generation specs stand down the adaptive triggers
(`max_widenings=0`, `support_fraction=0.98`, `boundary_fraction=0.99`):
they manufacture seeds, they do not measure the dilute end — without this
the ω=0.95 seed's support fraction (0.9506) wastes a coarsening probe and
can trip the boundary guard.

The l=2 seed lands at ω=0.9696, *above* 0.95 — the l=2 branch therefore
needs no increasing-ω leg either; its down-campaign starts from that seed
and covers 0.97 → fold in one run.

## e. The l=2..4 down-campaigns

Same shape as (c); the donors are the ladder's seed solutions:

```bash
for l in 2 3 4; do
  uv run tools/sweep_driver.py configs/runbook_l${l}_down.toml
  uv run tools/sweep_driver.py configs/runbook_l${l}_down.toml --summarize
done
```

ψ₀ targets (past-fold safety nets, raised from the 2026-09 values — on the
kept-domain convention the l=3/l=4 folds sit further out than the old
labels suggested): l=2 → 1.5, l=3 → 4.0, l=4 → 2.5. Expected wall-time
≈ 1–1.5 h each.

Achieved critical points (2026-09-13, all campaigns stopped on
`stopped:turning_point`):

| l | ω_min (ours) | paper | Δ |
|---|---|---|---|
| 1 | 0.6460482 | 0.64561 | +0.068% |
| 2 | 0.5212434 | 0.51657 | +0.90% |
| 3 | 0.4533087 | 0.44339 | +2.24% |
| 4 | 0.4186076 | 0.40756 | +2.71% |

The mass/angular-momentum peaks (M_max/J_max) are substantially more
accurate (l=1,2 within 0.05%; l=3 −0.55%/−1.0%; l=4 −1.2%/−2.3%) — they
live mid-branch, away from the fold. The fold ω_min is the most
resolution-sensitive critical point: these values are the dr=0.0625
discrete solutions (verified fit-robust and domain-independent for l=2).
A fold-polish stage (a fractional-N regrid 256→384 at dr≈0.0417, ≈30–45
min per l) is a documented optional extension for when tighter fold
accuracy matters — deliberately not part of this runbook's flow.

**Up-legs (ω rising toward the dilute end) are optional extensions**, not
needed for the verification (the seeds already sit at ω≥0.9, and Table
IX.1's critical points live at the fold end). The widening machinery is
demonstrated in `configs/l1_coarsen_demo.toml`; a future up-leg spec is
`direction = "down"` with `omega_target` — the driver widens when the
support fills the domain (`support_fraction`) and stops cleanly at the
`dr_max` floor or the `max_widenings` budget (`stopped:domain_budget`).

## f. Figures

```bash
uv run tools/plot_verification.py --campaigns runbook-l1,runbook-l2,runbook-l3,runbook-l4
```

Writes `out/figures/`: M_Komar(ω) and J_Komar(ω) branch curves with the
paper's Table IX.1 critical points, ω vs ψ₀ with the folds marked, the
Table IX.1 deviation bars, and the grid study. Branch segments are colored
per grid (N), so a refinement kink is visible per branch. The tool reads
only campaign `state.json` files — give it the runbook's campaign names
(a manifest, not a glob), so exploratory campaigns can never pollute the
verification figures.

## Legitimate stop reasons (not errors)

- `stopped:turning_point` — the expected end of every down-campaign: the
  vertex was crossed; `--summarize` localizes ω_min.
- `done:psi0_target` — the safety net fired (the detector missed; still a
  usable branch, and `--summarize` still applies if the fold was crossed).
- `stopped:boundary` — a down-leg's tail grazing the boundary with widening
  exhausted: the campaign ends at the best achievable grid. Fine for
  seed-generation specs; for a measurement campaign it means the branch's
  dilute end is short.
- `stopped:domain_budget` — the coarseness floor (`dr_max`) or the widening
  budget (`max_widenings`) spent: same meaning.
- Anything `failed:*` is a bug or an out-of-spec parameter — not expected
  from this runbook.

## Verification

Compare each branch's localized critical points against
`data/paper/table_ix1.csv` (the paper's Table IX.1). Achieved (2026-09-13):
the mass/angular-momentum peaks within 0.05% (l=1,2) and ~2.3% (l=4);
the fold frequencies +0.07% (l=1) to +2.7% (l=4) — the fold is the most
resolution-sensitive critical point at the runbook's grids; the causes are
diagnosed (resolution, not physics). `plot_verification.py` prints the
comparison table and draws the deviation bars.
