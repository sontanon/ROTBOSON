# Milestone report: catalogue rebuild & paper verification (l=1..6)

**Date:** 2026-09-12 · **Session span:** 2026-09-09 → 2026-09-12 · **Status: l=1..4 verified, l=5/6 in flight**

## 1. Objective

Rebuild the rotating boson-star catalogue (l=1..6) with the modernized pipeline
(single-solution C solver + Python sweep driver, HDF5) and verify the published
results (arXiv:2103.13993, CQG 38 154003 — Table IX.1 critical points), i.e. SAN-16 → SAN-12.

**Milestone definition (met, pending the in-flight l=5/6 runs):** every branch
reproduced end-to-end with the new pipeline, critical points agreeing with the
paper within a few percent, and every discrepancy traced to a understood cause
(resolution, domain truncation, or driver bug) rather than physics.

## 2. Headline results — Table IX.1 comparison

| l | ω_min ours | paper | Δ | M_max ours | paper | J_max ours | paper | source |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.64579 | 0.64561 | **+0.03%** | 1.3153 | 1.3155 | 1.3816 | 1.382 | sweep-l1-final + fold-l1-320 |
| 2 | 0.51653 | 0.51657 | **−0.01%** | 2.2149 | 2.2159 | 4.8090 | 4.810 | sweep-l2-final + fold-l2-320b |
| 3 | ~0.4458 (vertex) | 0.44339 | +0.55% | 3.4770 | 3.5287 | 12.348 | 12.49 | sweep-l3-final (fold chain dr=0.04/12.8, un-bracketed) |
| 4 | ~0.4105 (vertex) | 0.40756 | +0.73% | 4.9160 | 5.0590 | 25.414 | 25.83 | sweep-l4-final (fold chain un-bracketed) |
| 5 | in flight | 0.38819 | — | — | 6.6681 | — | 44.63 | sweep-l5-up/down (ω=0.7 cross-l seed) |
| 6 | in flight | 0.37391 | — | — | 8.2824 | — | 69.02 | sweep-l6-up/down |

Notes:
- M_max and J_max for l=1,2 agree with the paper to ≤0.05%. ω_min for l=1,2
  to ≤0.03% — paper-exact at our best resolution.
- M_max/J_max and ω_min are *different* critical points (M peaks at ω≈0.63–0.77,
  well before the fold); the comparison is like-for-like.
- The paper's values come from dr=0.08/N=400 (domain 32.8). Ours: dr=0.125/N=128
  (domain 16.4) branches + refined fold campaigns (320@0.08/domain 25.6, and
  dr=0.04/12.8 chains for l=3/4).

## 3. Campaign inventory (all HDF5, provenance in attrs)

| campaign | what | state |
|---|---|---|
| sweep-l{1..4}-final | full branches, ω=0.9 seeds → past folds | done (one accepted regrid each) |
| fold-l{1..4}-320{,b,c,d} | refined fold bracketing iterations | done (l=1,2 bracketed) |
| fold-l{3,4}-E | 320@0.08/25.6 folds, detector, no regrids | done — detector fired; resolution-limited (+3.0/+2.0%) |
| sweep-l{5,6}-up/down | cross-l seeded at ω=0.7 | **running** |
| sweep-l1-v2, sweep-l2-cont, sweep-l2-refine-test | validation runs | archived |
| out/seeds/ | ω=0.9 from-scratch seeds l=2..6 + l=1 branch point | kept |

Commits: `f7ec00c` (regrid fixes), `3648317` (cross-grid seeds), `14ef5a5`
(ψ₀ origin label + fold-E), `12f133d` (l=5/6 campaigns).

## 4. Physics learned

### 4.1 Resolution vs domain — cleanly separated (the B/C/E study)
At one l=2 branch point (ψ₀=0.6), re-solving the same physical point:
- **Resolution (dr÷2 at fixed domain): Δω = −1.56%.** Resolution dominates the
  branch-ω error; two refinements moved the l=2 fold from +3.2% to −0.01%.
- **Domain doubling (16.4→32.8 at fixed dr): Δω = −9e-6.** Negligible for
  ω/M_Komar/J_Komar at M/R≤0.14; only improves far-field diagnostics
  (GRV2 −0.017→−0.008, M_ADM −12%).
- Boundary truncation becomes *harmful* at M/R≳0.4: chained dr÷2/N-fixed
  regrids (domain 12.8→6.4, M/R≈0.55) drag ω **below** the paper (l=3 −2.1%,
  l=4 −4.3%) — the fold recedes and never turns.
- Consequence: domain 16.4 is fine for ω/Komar science; resolution is the
  accuracy knob; M/R scaling explains the l-dependent coarse errors
  (l=1 +0.23% … l=4 +8.5% on the pre-refinement grid).

### 4.2 Diagnostics hierarchy
- **M_Komar₁ ≈ M_Komar₂ to 1e-5 even on bad domains — the robust criterion.**
- M_ADM is the most truncation-sensitive surface integral — never drive
  acceptance with it (agreed explicitly).
- GRV2/GRV3 (Bonazzola–Gourgoulhon virial identities): GRV2 needs the
  asymptotic region (degrades −9% at domain 8.2, −45% at 4.1); GRV3's
  normalization in this code is not understood — flagged for review.
- HWL (half-width in lattice cells) is the practical refinement trigger:
  folds always sit at HWL<8, so HWL-driven refinement is *correct* — the
  failure was the response policy, not the trigger.

### 4.3 The trivial attractor and high-l dilute branches
- For weak seeds the Newton solve converges to the **trivial flat solution**
  (φ~1e-19) instead of the branch — the l=5/6 ω=0.9 branch points live at
  ψ(2,2)~1e-7, effectively adjacent to trivial. 3% ladder steps from such
  seeds collapse within 2–3 steps.
- The from-scratch ansatz has a two-sided trap: too strong → PARDISO −4;
  too weak → trivial. The window narrows ~×3 per l (r^l suppression of ψ(0)).
- The **goldilocks zone is real**: cross-l seeding at ω=0.7 (mid-branch,
  structured field) works where ω=0.9 never did. l=5 entered from l=4@0.7
  (scale 0.2 → M=4.63), l=6 from l=5@0.7 (scale 0.1 → M=5.07).
- Cross-l structural mismatch is *small* at matched amplitude (seed residual
  ‖du‖₀~1e-6 at ω=0.75) — the blocker is basin selection (trivial on one
  side, divergence on the other), not the l-mismatch itself.

### 4.4 ψ₀ is a grid label unless referenced to the origin
The fixed-φ point (2,2) sits at r=0.5·dr; the grids are staggered, so after
dr÷2 no node coincides and the same solution reads ~+0.85% higher at the new
fixed point. ψ₀ defined at the node is therefore grid-dependent. Fix shipped:
`psi0_origin_estimate` — even quadratic fit (ψ = a + b·u² + c·v²) through the
near-axis nodes, evaluated at the physical origin; all ψ₀ bookkeeping now uses
it. Re-grid label jumps and the fold-label recession disappear; ψ₀ is now
comparable to the paper's origin-referenced convention.

### 4.5 Seeds: what worked and what didn't
- From-scratch ansatz (Gaussian core + e^{−χr}/r^{l+1} tail, fixed-ω solve):
  works for l≤4 with l-tuned amplitude (ψ₀=1e-3/1e-4/9e-6/1e-6); the
  σ=4, rExt=12 shape is l=1-tuned and structurally poor for l≥5 (φ=r⁵ψ peaks
  near the ansatz's Gaussian→tail switch).
- Cross-l seeding: fails at ω=0.9 (trivial-adjacent; amplitudes 1e-7); works
  at ω=0.7 (l=4→l=5 scale 0.2; l=5→l=6 scale 0.1). Prediction of the target
  amplitude: **growth-factor extrapolation beats static l-ratio
  extrapolation** (ψ(2,2) growth 0.9→0.7: ×14/×26/×29/×19 per l — non-monotonic;
  the l-to-l ratio trend method failed a physical sanity check at ω=0.8).

## 5. Driver bugs found & fixed (all validated end-to-end)

1. **Regrid drift-correction misread staggered-grid geometry as error** — the
   +0.85% offset is the fixed point's radius change, not interpolation drift;
   rescaling by 0.9918 pinned refined solutions off-branch. Every in-campaign
   refinement failure (l=1..4) traced here. → scale=1.0, single solve.
2. **Post-regrid continuation rendered with stale coarse dr** — fine-grid seed
   fields loaded onto a dr=0.125 grid (2× stretch, ‖du‖₀≈4, instant PARDISO
   −4). This was the real cause of "SAN-23" reproductions.
3. **Cross-grid seed extrapolation** in `render_seed` (linear extrapolation
   between fields sampled at different radii).
4. **Cross-grid campaign seeds** (fold campaigns seeding from another branch's
   refined-grid solution) rendered as same-grid — seed records now carry the
   seed's own grid and step 1 interpolates.
5. **ψ₀ label** — origin-referenced estimate (4.4).
- SAN-22 closed (interpolator innocent, evidence: ≤1.7e-4 vs reference bicubic
  on full-domain and halved-domain targets); SAN-23 closed as misdiagnosed
  (the −4 signature was bug 2; a fold crossing on a correctly rendered fine
  grid converges — demonstrated repeatedly).

## 6. Computational findings & limits

- PARDISO in-core peak ~1.0 GB at dim≈105k (N=128, 6 fields); scales ≈
  nodes^1.5. N=256 (406k) ≈ 1.4–2.3 GB ✓; N=400 (980k) OOM-killed in 8 GB
  (consistent with SAN-5's ">7 GB" datum); N=512 (1.6M) infeasible.
- MKL OOC factorization (iparm[60]=2, env-gated `ROTBOSON_PARDISO_OOC` patch)
  fails immediately with PARDISO −9 and writes no OOC files — appears
  unsupported in this build. N=400 therefore out of reach in-container.
- Wall-clock: ~15 s/solve at N=128/l≤4 (3–4 Newton iters typical); ~90–180 s
  at N=256/320. Branch campaigns: 150–400 steps, 0.5–1.3 h each.
- Solver behavior catalogued: exit 2 + PARDISO −4 = singular Jacobian
  (divergence or mis-rendered seed); NLEQ-ERR→QNERR handoff normal for
  marginal steps; error −13 = QNERR-tail non-convergence (iterate still
  usable as a seed); error −1 = line-search exhaustion.
- Container: 8 GB RAM, 4 cores caps grid choices (domain×resolution matrix
  in §4.1); disk usage ~23 GB total, no pressure.

## 7. Open items

1. **l=5/l=6 campaigns (running):** up-campaigns cross their folds with the
   turning-point detector (max_refinements=0 — the chained-regrid runaway is
   understood); down-campaigns close the branches at ω≈0.9.
2. **l=3/l=4 fold bracketing:** the dr=0.04/12.8 chains are still descending;
   vertices +0.33%/+0.69% and improving each round. Next: continue those
   chains with `max_refinements=0` + detector (never 320d-style regrids).
3. **Refined fold campaigns for l=5/l=6** on 320@0.08/25.6 (their fold M/R on
   16.4 is 0.41/0.51 — the same polish l=1..4 got).
4. **N×2 stable-domain regrid option** in `do_regrid` (interpolation to
   arbitrary ratios already validated) — needed before high-l fold work at
   large domain; memory caps it at N=320→640 = infeasible, so its use case is
   N=128→256 (domain 16.4 kept).
5. **ψ₀ label across regrids is fixed; radius-matched fixed-φ point**
   (fixedPhiR scaled by dr ratio) remains an alternative not yet implemented.
6. **MKL OOC** — either debug the −9 or accept the N≤320 ceiling.

## 8. Forward-looking options (post-C evaluation)

Evidence gathered here that should inform the decision:
- The core C solver is **verified** — the physics is solid; remaining errors
  are resolution/domain effects with clean scaling laws, not code defects.
- Memory is the binding constraint (factorization fill ~O(nodes^1.5));
  wall-time is secondary (minutes per solve).
- Options on the table (SAN-15/SAN-6/SAN-11 context):
  - **Reduced/mixed precision:** PARDISO FP32 or iterative refinement could
    roughly halve factor memory — would make N=400 in-core feasible. Risk:
    the Newton convergence criteria (ε=1e-8…1e-13) and run-to-run jitter
    (SAN-20 documented ~1e-11 ψ jitter) need re-validation.
  - **Solver alternatives:** UMFPACK is already bit-equivalent at solved
    sizes (SAN-15); the interesting jump is matrix-free Krylov-Newton or
    GMG-preconditioned solves — kills the O(N^1.5) fill wall entirely and
    unlocks domain 32.8 at dr=0.08 (the paper grid) on modest memory.
  - **Rust/JAX/GPU rewrite:** the stencil assembly is element-local and
    code-generated (Mathematica notebook → C) — highly translatable; the
    global Newton solve is the serialization point. A GPU path wants
    matrix-free operators + preconditioning (multigrid natural for this
    structured grid — connects to SAN-6/M4). JAX would give autodiff
    Jacobians, removing the codegen maintenance burden entirely.
  - **Non-uniform grids (SAN-6/M4):** this session's data supports the
    motivation — dilute branch ends waste the domain (r99≈15.5/16.4 at
    ω=0.9) while folds need local resolution; patch/AMR would decouple the two.

## 9. Resume guide

- Campaigns: `out/campaigns/<name>/{state,summary}.json` (driver resumes from
  state.json; spec changes require `--fresh`).
- Seeds: `out/seeds/` (l=2..6 @ ω=0.9), `out/gold5/l={5,6},w=7.00000E-01` (the
  cross-l mid-branch entry points).
- Specs: `out/sweep_l{X}_{final,up,down,fold*}.toml`; runners `out/run_*.sh`.
- Fold fits: `state.json → turning_point` and `summary.json` (poly4).
- Paper data: `data/paper/table_ix1.csv`; interpolator reference: scipy
  `RectBivariateSpline` (tests in `tests/test_psi0_origin.py`).
- Domain-study results: `out/domain_study/` (B/C/E runs), logs included.
