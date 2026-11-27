# SAN-67 — the N1 backend decision: equilibration experiment + field-blocked Newton–Krylov

**Branch:** `n1/backend-blocknk` · **Deliverable:** `rbs_linalg::BlockFieldKrylov`
(`ng/rbs-linalg/src/block_field.rs`, 13 unit tests) + this report
**Measurement harness:** `ng/spikes/faer-lu/` modes `equil` / `blocknk`
(path deps into the workspace members; the spike stays standalone)
**Date:** 2026-11-19 · **Box:** the same `pi-dev` container as every C
reference number (`docs/profiling.md`): i7-6700HQ, cgroup CPU quota 4,
memory cap 12 GiB. All runs sequential, 4 threads pinned
(`faer::set_global_parallelism(Par::rayon(4))` — the 8-hyperthread
default is the documented 2.5× oversubscription trap).
**Systems measured:** the REAL first-Newton Jacobian of the l=1 smoke
spec — rbs-elliptic assembly at the ansatz, the same system the N=64
parity gate builds — on the production grid ladder (domain fixed at 16:
dr = 16/N → 0.25 / 0.125 / 0.0625 at N = 64 / 128 / 256, matching
`configs/l1_smoke_ci.toml` → `l1_seed.toml` → `probe_l1_umfp_n256.toml`).

---

## 1. Verdict up front

**Block-field Newton–Krylov is the N1 backend.** The decision rule was
"block-NK within ~2× of PARDISO at N=256 under 2–3 GB → N1 backend; MG
targets N ≥ 400." The measurement clears it by an order of magnitude
on both axes, not marginally:

| N=256, 4 threads | C reference (PARDISO) | block-NK (this spike) |
|---|---|---|
| cold solve (factor + 1 solve) | 92 s | **7.7–7.9 s** (5.3 s factor + 2.4 s Krylov, 12 iters) |
| campaign step (stale preconditioner) | ~69 s (CGS phase) | **3.0–3.3 s** (16–17 iters) |
| peak RSS | 4.95 GB | **1.88 GiB** (fp64) |
| full cold-start Newton solve | ~17.8 s factor + ~12 × 69 s ≈ **14 min** | 5.3 s factor + 8 Krylov solves ≈ **29 s** |

Additional gates the rule did not ask for:

* **Oracle agreement.** The full Newton loop driven by the block backend
  with blocks factored **once at the ansatz** (stale for every step)
  converges in 8 steps at N=64 and reproduces the pinned C-oracle
  fixture to worst field-relative **2.3e-11** / worst absolute
  **4.4e-15** — inside the parity gate's own bars (1e-10 / 5e-14).
* **N-independent iteration counts.** 12 Krylov iterations fresh /
  16–18 stale, and 8 Newton steps, at *every* size N=64/128/256: the
  block preconditioner converges on physics, not on problem size —
  the property the multigrid escalation was meant to buy.
* **fp64 projections clear N=400 and N=512.** Block-NK projects to
  ~4.6 GiB at N=400 *without* the fp32+IR recipe the C needs
  (8.97 GB), and ~7.8 GiB at N=512 — the size where the C's sparse LU
  is OOM-killed at 12 GiB (§6).

**faer full-matrix sparse LU stays where it was:** the parity-gate
oracle backend at N ≤ 128, never a scaling path. **Row equilibration
is a measured no-op on the real systems** (§3): keep it on inside the
block backend as free insurance, but it is not load-bearing.

---

## 2. Step 1 — the equilibration experiment (the SAN-54 §5.1 rescue, tested)

The hypothesis from `REPORT.md` §5: faer's unscaled partial pivoting
loses ~1 decade of solve accuracy per 3 grid points on this matrix
class because the C's Robin/decay boundary rows are scaled by
`dr·dz/rr²` and shrink under refinement; row equilibration might widen
the accuracy window at the parity sizes.

**Result: the collapse does not reproduce on the real assembly, and
equilibration changes nothing (≤ ±20%, both directions).** The SAN-54
residual blow-up (2.1e-13 → ~1e3 over N=16→64, measured on the spike's
*representative* interior values) is a **value-model artifact**: on the
real ansatz Jacobian — same pattern, same boundary formulas, real
SymPy-derived values — faer's raw LU stays honest at every size and
every grid spacing tested, including the exact grid (N=64, domain 8)
where the generator collapsed to ~1e3.

### 2.1 The real-system table (faer sparse LU, supernodal AUTO, 4 threads)

| N | dr | rows | nnz | symbolic (s) | numeric (s) | solve (s) | raw residual | row-equil. | 2-sided | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.0 | 2,401 | 62,131 | 0.025 | 0.046 | 0.001 | **1.24e-13** | 1.08e-13 | 1.05e-13 | 32 MiB |
| 32 | 0.5 | 7,777 | 199,773 | 0.068 | 0.21 | 0.005 | **7.78e-13** | 5.64e-13 | 5.80e-13 | 115 MiB |
| 64 | 0.25 | 27,745 | 873,619 | 0.21 | 1.24 | 0.023 | **6.43e-12** | 7.53e-12 | 6.09e-12 | 493 MiB |
| 128 | 0.125 | 104,545 | 3,417,875 | 1.04 | 8.32 | 0.113 | **7.10e-11** | 6.71e-11 | 6.83e-11 | 2.59 GiB |

The N=64 raw residual reproduces the parity gate's own check
(6.4e-12) exactly. The trend is a mild ~×6–8 per refinement doubling —
the boundary-row mechanism is real (the min row-max halves exactly
with dr: 1.56e-2 at dr=0.125) but on real values it costs 2.5 decades
total from N=16 to N=128, not 15.

### 2.2 The collapse-control and the spacing control

| system | dr | raw residual | row-equil. |
|---|---|---|---|
| N=64, **domain 8** (the REPORT.md collapse grid) | 0.125 | **8.90e-11** | 8.51e-11 |
| N=128, domain 8 (= the N=256 production spacing) | 0.0625 | **1.02e-9** | 1.34e-9 |

Three facts fall out:

1. **The real assembly is 13 orders of magnitude healthier than the
   generator model** at the same grid (8.9e-11 vs ~1e3 at N=64/domain 8).
   At the real ansatz the metric fields are flat, so the generator's
   O(1) cross-field coupling magnitudes never materialize; the
   conditioned matrix faer actually sees is far closer to a scaled
   Laplacian.
2. **The degradation tracks dr, not N** (N=64@dr=0.125 ≈ N=128@dr=0.125
   ≈ 7–9e-11; N=128@dr=0.0625 ≈ 1e-9): a property of the boundary-row
   scaling class, exactly as REPORT.md §5 diagnosed — but with a
   coefficient that keeps every tested spacing inside a valid-solve
   regime for direct LU at the sizes where direct LU fits at all.
3. **Equilibration is cosmetic**: row-max or row+col scaling moves the
   residual by less than ±20% at every point measured, and at N=128 it
   *slows* the factorization by ~20–30% (8.3 → 10.0–11.0 s) for
   nothing. faer's COLAMD ordering under equilibration is not better,
   and pivoting on the real values was never in trouble.

**Answer to the Step-1 question ("does equilibration widen faer's
accuracy window enough to matter at parity sizes N=64/128?"): NO —
there is no window problem to widen.** faer's raw LU solves the real
parity systems to 6.4e-12 (N=64) and 7.1e-11 (N=128), both far below
the 1e-10 parity-gate bar, and equilibration's ±20% is noise. The
conditional-go clause of REPORT.md §5.1 is therefore not triggered.
The faer NO-GO for *scaling* stands unchanged on the memory/fill
argument (full-matrix LU at N=128 already costs 2.59 GiB / 8.3 s for a
0.11 s solve; N=256 projects ~22 GiB — REPORT.md §3.5).

---

## 3. Step 2 — the deliverable: `BlockFieldKrylov`

`ng/rbs-linalg/src/block_field.rs` (~1000 lines with tests/docs), no
new dependencies (faer's own `dyn_stack` re-export covers the scratch
management). Design, as documented in the module:

* **Matrix contract:** square, `nfields · dim + 1` rows (6 field
  blocks of size dim + the ω slot). `factor()` extracts each field's
  diagonal block from the field-major CSR, row-equilibrates it
  (default on), and factors: **LDLT** (faer simplicial, natural order)
  when the block is symmetric, **sparse LU** (faer, COLAMD, AUTO
  supernodal threshold) otherwise. Every block of every real Jacobian
  measured here is nonsymmetric (first-derivative advection terms) —
  verified at factor time — so LU is the live path; the LDLT branch is
  exercised by unit tests and falls back to LU on a zero pivot.
* **`solve_into` runs restarted FGMRES by hand** (~150 lines,
  documented in-module): right-preconditioned with the block-diagonal
  inverse, modified Gram–Schmidt against ≤ 30 Krylov vectors per
  cycle, Givens least squares, *flexible storage* of the preconditioned
  basis `Z` so the update never re-applies M⁻¹ and the residual
  estimate is the true residual. Restart on a fresh true-residual
  recomputation. The eigenvalue slot is preconditioned by identity; the
  Krylov loop cleans up the same-node cross-field couplings and the ω
  row/column.
* **`solve_stale(a_current, b, x)` is the campaign entry** — the C
  line's stale-LU CGS analogue: the operator is the caller's *current*
  Jacobian, the preconditioner is the block factorization kept from the
  last `factor()`. FGMRES minimizes the true residual of the current
  system, so stale blocks cost iterations, not accuracy. This is a
  deliberate extension beyond the `LinearSolver` trait shape (the trait
  alone can only express fresh-blocks); `solve_into` keeps the trait
  contract for the Newton loop and unit tests.
* **Strict mode** (default) raises `SolveError::NotConverged` on a
  missed tolerance; `last_stats()` reports iterations / cycles / true
  residual / converged for campaign logs. Lenient mode returns
  best-effort. The residual check is wired into every solve by
  construction — the SAN-54 "silent garbage" failure mode cannot
  recur through this backend silently.
* **Parallelism:** block factors/solves honor faer's process-global
  setting; the driver pins it once
  (`faer::set_global_parallelism(Par::rayon(4))`). The Krylov matvec
  is sequential (memory-bound; the block solves dominate the wall —
  measured 12–17 block-solve rounds per solve).

Known scope limits, stated honestly: the LDLT branch uses faer's
*simplicial* path with the natural ordering (the symmetric case never
occurs on real systems; if it ever does at scale, switch it to the
supernodal LDLT before relying on it). Rank-deficient blocks do not
error at factor time (faer's low-level LU tolerates them); a genuinely
broken preconditioner surfaces as `NotConverged` — pinned by a unit
test.

---

## 4. Step 3 — the measured table (real Newton systems, 4 threads)

Configuration: tol 1e-8 relative, restart 30, max cycles 40,
equilibrated blocks. "Fresh" = `factor(J_k)` + Krylov (cold solve);
"stale" = Krylov against a *drifted* operator (blocks kept) — measured
on real damped Newton steps, not synthetic perturbations. RSS = VmHWM.

### 4.1 Per-solve costs

| N | rows | block dim | factor (s) | fresh solve (s / iters) | stale step (s / iters) | peak RSS |
|---|---|---|---|---|---|---|
| 64 | 27,745 | 4,624 | 0.24 | 0.091–0.097 / **12** | 0.11–0.14 / **16–18** | 119.7 MiB |
| 128 | 104,545 | 17,424 | 1.06–1.08 | 0.44–0.55 / **12** | 0.52–0.85 / **16–18** | 470.8 MiB |
| 256 | 405,601 | 67,600 | 5.33–5.40 | 2.39–2.47 / **12** | 2.98–3.32 / **16–17** | **1.88 GiB** |

Per-solve true residuals: 7.1e-9 fresh, 1–9e-9 stale — at tolerance at
every step, independently recomputed by the harness
(`*_residual_rel_independent`) and matching the backend's own
accounting exactly. At tol 1e-6 (the C's CGS stopping level) the fresh
solve takes 11 iterations — the tolerance is not a lever worth
campaign-tuning.

**Iteration counts are N-independent** (12 fresh / 16–18 stale at every
size): the block preconditioner captures the physics coupling; the
Krylov loop's work is a fixed ~15 block-solve rounds regardless of
grid size. This is the scalability property the C's fill-growing LU
cannot have, and it is why the wall column grows only ~×3.2 per size
doubling (roughly the block-solve fill growth) rather than ×8+.

### 4.2 End-to-end Newton solves (stale blocks factored once at the ansatz)

| N | Newton steps to ‖du‖<1e-8‖u‖ | factor + 8 linear solves | total wall | final ‖du‖/‖u‖ |
|---|---|---|---|---|
| 64 | 8 | 0.24 + 0.99 | **1.22 s** | 2.3e-14 |
| 128 | 8 | 1.08 + 5.04 | **6.1 s** | 6.3e-14 |
| 256 | 8 | 5.33 + 23.5 | **28.8 s** | 2.5e-14 |

N=64 vs the pinned oracle fixture: worst field-relative 2.3e-11,
worst absolute 4.4e-15 (per-field: log_alpha 9.6e-15, beta 5.4e-15,
log_h 9.1e-15, log_a 4.8e-14, psi 9.4e-15, lambda 2.3e-11 — the lambda
rel sits on near-zero axis points, absolute 5.2e-16). The block
backend therefore passes the parity gate's own accuracy bars while
running the stale-block policy the campaign wants.

### 4.3 Against the C reference table (`docs/profiling.md`)

| quantity | C (PARDISO, 4 thr) | block-NK | ratio |
|---|---|---|---|
| N=128 peak RSS | 1.33 GB | 0.46 GiB | 2.9× less |
| N=256 cold solve | 92 s | 7.7–7.9 s | **12× faster** |
| N=256 campaign step (stale) | ~69 s | 3.0–3.3 s | **~22× faster** |
| N=256 peak RSS | 4.95 GB | 1.88 GiB | 2.6× less |
| N=256 full cold Newton solve | ~14 min | 28.8 s | **~29× faster** |
| fill/iters growth | LU nnz ~rows^1.20 | Krylov iters **constant**, factor RSS ~dim^1.1 | different scaling class |

The decision rule ("within ~2× of PARDISO at N=256 under 2–3 GB") is
satisfied with margin: ~12× faster cold, ~22× faster per campaign
step, 1.88 GiB < 3 GB.

### 4.4 Block equilibration arm

`--no-equil` at N=64: identical iterations (12/16/18/16) and residuals
(7.154e-9 fresh) — cosmetic on blocks too, consistent with §2. The
default stays ON (one row-max pass per block, measured cost within
noise) as insurance for spacings/states not tested here; it is not
load-bearing for any number above.

---

## 5. What the block path does NOT yet prove (honest caveats)

* **Fold-adjacent behavior.** All measurements are the ansatz plus 3–7
  real Newton steps of drift (λ backtracking exercised, first step
  damped 0.5). The C line's stale-LU CGS exists for *near-fold*
  Jacobians, where refactorization blows up (56 s/factor class); the
  block analogue near folds is untested. The API is ready for it
  (`solve_stale`), and the guard is structural (`NotConverged` +
  refactor-on-stall is the driver policy to try). This is the first
  thing the SAN-65 campaign driver should exercise.
* **The l=5/l=6 badly-scaled seeds** (the C's PARDISO −4 class) are
  untested — a different ill-conditioning regime than grid scaling.
* **N ≥ 400 is projection, not measurement** (§6).
* The stale solves here reuse blocks across *Newton steps of one
  solve*; campaign reuse across *continuation steps* (w0 stepping) has
  longer drift paths — measure before trusting beyond ~10 steps or
  refactor on iteration-count growth (e.g. when a stale step needs
  >2× the fresh iteration count).

---

## 6. Projections to N=400 / N=512 (fp64, against the 12 GiB cap)

Measured laws (from §4.1): 6-block factor-phase RSS delta
73.4 → 311.9 → 1377.5 MiB at dim 4,624 → 17,424 → 67,600 →
**delta ≈ dim^1.09–1.10** (the guard's projection was within 3% of the
measured N=256 delta); factor wall ≈ dim^1.13–1.19; stale-step wall
≈ dim^1.1–1.3. Assembly+operator+arena baseline ≈ 455.7 MiB at N=256,
∝ nnz.

| projected, fp64 | N=400 (dim 163,216) | N=512 (dim 266,256) |
|---|---|---|
| factor RSS delta | ~3.5 GiB | ~6.0 GiB |
| **peak RSS** | **~4.6 GiB** | **~7.8 GiB** |
| factor wall | ~15 s | ~28 s |
| campaign step | ~8–9 s | ~15–17 s |
| C reference | 8.97 GB **fp32+IR**, ~220 s/step (8 thr, likely ~2.5× oversubscribed) | **OOM-killed at 12 GiB** |

Consequences, if the projections hold (verify with the guard protocol
before committing a campaign — the exponent may steepen at supernodal
panel boundaries):

* **N=400 fits in fp64 with ~7 GiB headroom** — no fp32+IR recipe
  needed, removing a whole accuracy-machinery dependency from the NG
  campaign path.
* **N=512 becomes feasible-in-principle** (~7.8 GiB < 12 GiB) — the
  size where the C's global LU OOMs. The binding constraint moves from
  "memory wall" to "wall time grows like dim^1.1–1.3", which is the
  multigrid problem statement, at one grid size later than the C's.

---

## 7. The recommendation

1. **N1 backend: `BlockFieldKrylov`** (block-diagonal faer LU/LDLT
   preconditioner + hand FGMRES), wired behind `LinearSolver` for
   fresh-blocks and `solve_stale` for campaign steps. The measured N=256
   numbers (~12× faster than PARDISO cold, ~22× per stale step, 1.88 GiB,
   oracle-accurate end-to-end) clear the decision rule with an order of
   magnitude of margin.
2. **faer sparse LU keeps the parity-gate role only** (N ≤ 128 seed
   solves, unit tests). Its measured accuracy on real systems
   (6.4e-12 at N=64, 7.1e-11 at N=128, 1.0e-9 at dr=0.0625) is honest
   — the SAN-54 collapse is reclassified as a value-model artifact —
   but its memory/fill scaling at N ≥ 256 remains disqualifying, and
   the block path now beats it at every size anyway (N=64: 0.24 s
   factor vs 1.24 s; the parity gate keeps it as an *independent*
   cross-check backend, which is exactly what G2/G3 want).
3. **Row equilibration: keep as per-block default inside the backend,
   drop as a full-matrix rescue project.** Measured cosmetic (±20%) at
   every tested size/spacing; the full-matrix experiment that would
   have justified it is closed by §2.
4. **Multigrid is NOT pulled earlier by memory — its schedule pressure
   relaxes, its justification shifts.** Block-NK projects to clear
   N=400/512 inside the cap, so MG is no longer needed for memory
   survival at the paper sizes. What still mandates MG on the roadmap:
   (a) the grid-ladder rungs 2–3 (stretched-radial, compactified) need
   a scalable per-field inverter for the *same* elliptic blocks MG
   targets — block-LU's dim^1.1 fill growth is the wrong exponent for
   the ladder's endgame; (b) per-step wall beyond N=256 grows
   superlinearly; (c) the fold-robustness regime (§5) is where a
   MG-cycle preconditioner (linear-time, refactor-free) may beat stale
   block-LU. MG targets the ladder rungs and the fold regime, not a
   memory rescue at N=400.

---

## 8. API / numeric notes for SAN-65 (the campaign driver)

* **Drive campaigns through `solve_stale(&current_j, b, x)`.** The
  C-mirroring policy: `factor()` at the first step of a continuation
  leg (or after a regrid), `solve_stale` per Newton step; refactor
  on `NotConverged` or when a step's iteration count exceeds ~2× the
  fresh count (the drift budget measured here: ≤7 steps, 16–18
  iterations, no growth).
* **`KrylovStats` + `NotConverged`** are the campaign log surface.
  Strict mode is the default; the residual check is built into every
  solve (the SAN-54 lesson is structural here, not advisory).
* **Memory accounting:** the backend owns an operator copy (nnz-sized)
  plus the Krylov arena ((2m+1)·n ≈ 198 MiB at N=256, m=30). The
  measured N=256 peak 1.88 GiB includes: assembly+pattern+residual
  455.7 MiB (driver-side) + operator clone + arena + 6 block factors
  1377.5 MiB. The driver should not keep a second Jacobian copy alive
  around solves.
* **Pin faer's parallelism once** at driver start
  (`faer::set_global_parallelism(Par::rayon(4))`) — the default rayon
  pool sizes to `nproc` (8) and oversubscribes the 4-CPU quota.
* **Tolerance:** 1e-8 relative reproduces the oracle inside the parity
  bars; 1e-6 saves one iteration. Newton exit is the driver's test, as
  in the C.
* **The Newton count is the loop's, not the backend's:** 8 steps here
  (the slice's backtracking) vs the C's 12 (NLEQ-ERR). SAN-52's QNERR
  port remains the outer-loop wall-time lever.
* The l=1 6-field contract (`nfields = 6`) is the default config; other
  field counts work (unit-tested at 2) if the evolve side ever wants
  the same machinery.

---

## 9. Reproduction

```bash
export CARGO_HOME=$HOME/.cargo
cd ng/spikes/faer-lu
cargo build --release

# Step 1 — equilibration (each invocation one sequential run)
./target/release/faer-lu-spike --mode equil --n 16   --variant none  --threads 4
./target/release/faer-lu-spike --mode equil --n 64   --variant row   --threads 4
./target/release/faer-lu-spike --mode equil --n 128  --variant 2sided --threads 4
./target/release/faer-lu-spike --mode equil --n 64   --domain 8 --variant none --threads 4   # collapse control

# Step 3 — block-NK per-size measurements (guarded at 256)
./target/release/faer-lu-spike --mode blocknk --n 64  --stale-steps 3 --threads 4
./target/release/faer-lu-spike --mode blocknk --n 128 --stale-steps 3 --threads 4
./target/release/faer-lu-spike --mode blocknk --n 256 --stale-steps 3 --threads 4 \
    --budget-gb 8 --delta64-mib 73.4 --delta128-mib 311.9

# End-to-end: full Newton to convergence, stale blocks, oracle fixture check
./target/release/faer-lu-spike --mode blocknk --n 64 --newton --stale-steps 20 --fixture --threads 4
./target/release/faer-lu-spike --mode blocknk --n 256 --newton --stale-steps 20 --threads 4 \
    --budget-gb 8 --delta64-mib 73.4 --delta128-mib 311.9

# The backend's own suite (workspace)
cd ../../ && cargo test -p rbs-linalg
```

The N=256 guard arguments are the measured N=64/N=128 factor-phase RSS
deltas (73.4 / 311.9 MiB); the guard projects dim^exp from them, aborts
over `--budget-gb`, and sets an RLIMIT_AS net before the numeric phase.
Its N=256 projection (1368 MiB) was within 3% of the measured delta
(1377.5 MiB).
