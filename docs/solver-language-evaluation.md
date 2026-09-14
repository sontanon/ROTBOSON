# Solver & language evaluation — SAN-15 (M3)

**Status: decision-ready.** Evidence: `docs/profiling.md` and its living
record `data/summaries/profile-baseline.md` (all measurements 2026-09-13/14).

This memo answers SAN-15's question — *is PARDISO worth its dependency, and
is a Rust/Python rewrite worth doing?* — with measured numbers, per
component, ending in an explicit go/no-go. Santiago decides; this is the
input artifact, not the decision.

---

## 1. The question, restated

PLAN.md's Phase 6 (Rust migration) is deferred until it earns its place.
The profiling study (SAN-11) was commissioned precisely to decide whether
any of the following modernizations pay for themselves:

1. Drop PARDISO/MKL for another sparse solver.
2. Move hot components to Rust or a Python-native hybrid.
3. Adopt a production recipe change (precision/algorithm) that improves
   throughput without a rewrite.

The default answer is deliberately allowed to be *"do nothing, C is fine."*

---

## 2. Where wall time actually goes (measured)

Production-representative N=256 single solve at the correct 4-thread
setting, 92 s wall:

| phase | share | note |
|---|---|---|
| PARDISO solve (phase 33) | 73% | LU-preconditioned CGS loop, not a plain solve |
| PARDISO factor (phase 22) | 21% | numeric factorization |
| PARDISO analysis (phase 11) | 4% | reorder + symbolic factor |
| Jacobian/residual assembly | ~1% | our code; negligible at every size |
| norms + misc | ~1% | negligible |

At N=400 (paper grid) the split shifts toward the factor being dominant
(74%), because LU fill grows ~rows^1.20. In every case, **≥95% of wall time
is inside the vendor factorization/solve library.** Our own code is a few
percent.

> **Threading caveat (see `docs/profiling.md` §4):** the box is 4 physical
> cores capped at 4 CPUs; nothing sets `OMP_NUM_THREADS`, so runs default to
> `nproc=8` and are **2.5× slower than at 4 threads** (228 s vs 92 s). Pin
> `OMP_NUM_THREADS=4` — a free ~2.5× on every run.

---

## 3. Solver options — per component

### 3.1 PARDISO (MKL) — **GO / keep**
- Pays for exactly the hard part: factorization of a badly-scaled 6-field
  Jacobian at ~60 GFLOPS, threaded across 8 cores.
- `iparm[3]=61` CGS-solve mode is load-bearing near folds (A/B tested:
  direct refactorization of near-singular Jacobians blows up 56 s/factor
  vs ~20 s/step under stale-LU CGS). Kept as campaign default.
- **MKL dependency is justified by the profile.** Not the bottleneck to fix.

### 3.2 UMFPACK / SuiteSparse — **NO-GO (killed by head-to-head)**
- Bit-equivalent at smoke sizes (established pre-M3), and remains the
  MKL-free fallback (`umfpack` preset).
- Head-to-head at production size (N=256, 405k rows): PARDISO best 92 s
  (4 thr; the 216 s run was the 8-thread oversubscription artifact); UMFPACK
  did **not** reach its first solve within 330 s (>3.5× even vs PARDISO's
  best) — killed and ruled out. Consistent with the historical small-grid
  experience (weaker ordering at 2D mesh fill, single-threaded factor).
- Not worth the dependency swap; evidence closes the "drop MKL" question.

### 3.3 MUMPS-BLR (compressed factors) — **DEPRIORITIZED**
- fp32 + double-residual IR already halves factor memory and reaches the
  paper's N=400 in-core at 8.97 GB (≈3 GB headroom at the 12 GiB cap).
- BLR only earns its cost if N>512 in-core or concurrent large-grid runs
  become requirements. Neither is current. Revisit only on that trigger.

### 3.4 GPU / hybrid memory (cuDSS) — **NOT NOW**
- Conceptually the path to N≥512 without a new algorithm, but requires an
  NVIDIA host (not present) and is a backend project of its own. Out of
  scope for this decision; M4 is a better investment.

### 3.5 Iterative solvers / multigrid — **deferred to M4**
- M4 (multigrid / non-uniform grids) attacks the resolution-vs-domain
  tension that uniform grids turn into a memory wall — the *real* lever for
  l>4 resolution and extended boundaries, which fp32 only postpones.
- Not a solver-swap decision for M3; sequenced to M4 (exploratory).

---

## 4. Language / runtime — per component

### 4.1 C (status quo) — **GO / keep**
- Assembly is 1–2% of wall at every measured size; the solve is a vendor
  library call. A Rust/Python rewrite of hot components has **no measured
  headroom** — it would optimize the 1% and leave the 95% untouched.
- Maintenance/longevity value alone does not justify a rewrite while the
  numerics are the bottleneck and C+MKL is at the floor.

### 4.2 Rust port of hot components — **NO-GO (now); revisit after M4**
- Would target faer or SuiteSparse; but the hot loop is MKL PARDISO, which
  a Rust port would still call (via FFI). Net: all the cost, none of the
  speed.
- The only scenario that reopens this is a rewrite that also changes the
  algorithm (M4 grid strategy) or drops PARDISO — neither is earned yet.

### 4.3 Python-with-native-cores hybrid — **NO-GO**
- The orchestration (sweep driver) is already Python; moving the solver to
  Python would keep the PARDISO call but add a marshalling layer. No win.

---

## 5. The one change worth adopting — **GO**

**fp32 + double-residual IR as the production recipe for N≥256:**
`ROTBOSON_PRECISION=32 ROTBOSON_PARDISO_DIRECT=1` (IR default 2).

- Halves factor memory (N=400 peak 8.97 GB vs ~double's projected 14+ GB).
- Full l=1 campaign reproduces the double-direct fold vertex ω_min to
  ~1e-16; 0 retries/stalls across 169 steps.
- Unlocks the paper's **N=400 in-core at 12 GiB** — the operational prize.
- Newton's own double-precision convergence test acts as the outer
  corrector; scalar fields land on-branch (~1e-13 rel vs double).

Adopt as the documented default for high-resolution campaigns. Keep
double-CGS for low-N smoke/CI where it is fine.

---

## 6. Recommendation summary (go/no-go)

| component | decision | expected gain | effort | risk |
|---|---|---|---|---|
| PARDISO/MKL | **GO** (keep) | — (already the floor) | none | none |
| UMFPACK swap | **NO-GO** | − (killed, >1.5× slower) | — | — |
| MUMPS-BLR | **DEFER** | fp32 covers N=400 | med | trigger-gated |
| GPU/cuDSS | **NO-GO (now)** | N≥512 only | high | no GPU host |
| Rust rewrite | **NO-GO (now)** | ~0 (optimizes the 1%) | high | revisit post-M4 |
| Python hybrid | **NO-GO** | ~0 | med | — |
| **fp32+IR recipe** | **GO** | N=400 in-core, −34% mem | low | adopted |
| **Pin `OMP_NUM_THREADS=4`** | **GO** | ~2.5× wall on this box | trivial | add to driver/runbook |
| **M4 multigrid** | **PRIORITIZE** | real resolution/domain lever | — | M4 track |

**Bottom line:** keep C + PARDISO, adopt fp32+IR for N≥256, and put the
effort into M4 (non-uniform grids) rather than any solver/language
migration. M3's migration question is answered: *do not migrate.*

---

## 7. Open for Santiago's decision

- Record the go/no-go verdict on SAN-15.
- Reconsider CGS-vs-direct campaign default for N≥256 (open follow-up;
  iparm[3]=31 tolerance check pending).
- Scope the N=400 full-validation campaign (l=1..6) as the operational
  payoff; note l=5/6 were descoped in M2 and need re-scoping.
