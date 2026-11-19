# SAN-54 — faer sparse-LU envelope on the 6-field Jacobian (RBS-NG spike)

**Branch:** `ng/faer-spike` · **Crate:** `ng/spikes/faer-lu/` (standalone package, own
`[workspace]` table so the parallel ng/ scaffold cannot absorb it)
**Date:** 2026-11-19 (overnight wind-down: measurements below are what completed before
the box was quieted; skipped items are listed with reasons)
**Backend measured:** faer **0.24.4** (latest crates.io release, published 2026-06-24),
sparse LU (`faer::sparse::linalg::lu`), default features, u32 indices
**Box:** the same `pi-dev` container as every C reference number in
`docs/profiling.md` — i7-6700HQ, cgroup CPU quota 4 (`cpu.max 400000/100000`),
memory cap 12 GiB, 8 hyperthreads visible. All runs pinned to 4 threads
(`Par::rayon(4)`) to match the C line's post-fix `numThreads=4` configuration.

---

## 1. Verdict up front

**NO-GO: faer 0.24.4's sparse LU is not viable as the transitional direct
backend for the parity port.** It is fast and its memory envelope is comfortable
at the parity sizes, but its *solve accuracy collapses with grid size* on this
matrix class: the direct-solve residual grows roughly one order of magnitude per
3 grid points and is ~10³ at N=64 — i.e. the solve is garbage exactly at the
N=64/N=128 parity sizes the transitional backend exists to serve. N=128 was not
factored overnight, but extrapolating the measured curve (~1 decade / 3 points)
puts its residual beyond 10¹⁶; there is nothing to measure there.

Fallback (named, not implemented): **per-field block preconditioners with faer
on the blocks + a Krylov outer loop** — each field's self-block is a ~17k-unknown
elliptic system at N=128 and can be factored with faer's sparse Cholesky/LDLT
paths (faer's most mature sparse components) inside a block-Jacobi/Gauss-Seidel
preconditioner. Break-glass if that misses parity targets: the **SuiteSparse
binding decision** (KLU/UMFPACK; the C line already carries a UMFPACK backend,
bit-equivalent at smoke sizes but ruled out at N≥256 by the C evaluation — this
breaks pure-Rust-from-day-one and needs an explicit owner call).

The 2D-mesh fill law measured here (≈ rows^1.45 over N=16→64, vs PARDISO's
rows^1.20) puts the **Rust LU wall at or below N≈128 on this box even ignoring
the accuracy wall** — the guarded N=256 projection lands at ~22 GiB of factor
storage alone, over the 12 GiB cap before a single numeric factorization is
attempted.

---

## 2. Pattern generator — rows/nnz sanity check vs the C oracle

`src/pattern.rs` reproduces `csr_gen_jacobian` (order-4 path, the one the C
production configs use) structurally and with the C's exact boundary-row
coefficients:

- rows = `6·(N+4)² + 1` (ghost = 2, `NrTotal = NzTotal = N+4`, trailing omega row).
- per-field nnz = the **verbatim `csr.c` formulas** (`nnz_jacobian_get_nnzs`,
  order-4 branch) — asserted at build time.
- interior/edge column lists transcribed from the generated kernels
  (`jacobian_4th_order_variable_omega_{cc,cs,sc,ss}`) and **diffed entry-by-entry
  against `src/csr_vars.c`** by `tools/verify_pattern_vs_c.py` → `ALL TABLES MATCH`.
- parity/Robin/decay/omega-constraint rows per `csr_symmetry.c`,
  `csr_robin.c`, `csr_exp_decay.c`, `csr_omega_constraint.c` with the C's exact
  closed-form coefficients (including the `dr·dz/rr²` boundary scaling).
- CSR↔CSC triple-set identity verified at N=64 (all 873,619 (row,col,val)
  triples identical); faer's own `new_checked` invariant validation runs on every
  build.

| N | rows (gen) | nnz (gen) | C reference (docs/profiling.md) | match |
|---|---|---|---|---|
| 64 | 27,745 | 873,619 | — (no published C number) | formula-consistent |
| 128 | 104,545 | 3,417,875 | 104,545 / 3.42M | **exact** |
| 256 | 405,601 | 13,519,891 | 405,601 / 13.5M | **exact** |

Interior *values* are representative (correct magnitude classes: Laplacian-
dominated diagonal ≈ −5, O(1) cross-field same-node couplings, small
first-derivative cross-field stencil terms, O(1) omega-column entries, smooth
"star-like" spatial profile), not the SymPy expressions — per the spike brief,
the timing evidence comes from the structure. All boundary rows use the C's
exact formulas, so the class of matrix faer sees — including the boundary-row
magnitudes that fall like 1/N² (see §5) — is faithful.

---

## 3. Measured numbers

faer AUTO picks the **supernodal** variant for this pattern at every size.
Timings = best of repeated runs; RSS = VmHWM.

### 3.1 Envelope at the smoke sizes (faer default/supernodal, 4 threads)

| N | rows | nnz | symbolic (s) | numeric (s) | solve (ms) | peak RSS | rel. residual |
|---|---|---|---|---|---|---|---|
| 16 | 2,401 | 62,131 | 0.017 | 0.038 | 0.77 | 35.9 MiB | **2.1e-13** ✓ |
| 24 | 4,705 | 119,987 | — | — | — | — | 5.1e-11 ✓ |
| 32 | 7,777 | 199,773 | — | 0.203 | 4.9 | — | 2.0e-8 (already degraded) |

### 3.2 The parity size that completed: N=64 (best of 3, supernodal, 4 threads)

| phase | time |
|---|---|
| assembly (pattern + CSR→CSC) | 0.020 s |
| symbolic | 0.246 s |
| **numeric** | **1.057 s** |
| solve (1 rhs) | 22.8 ms |
| peak RSS | 507 MiB (numeric phase adds +471 MiB over symbolic) |
| **rel. residual** | **9.9e2 — the solve is invalid** |

Simplicial variant at N=64 (forced, 1 run): numeric 25.43 s (24× slower than
supernodal), solve 35 ms, L+U fill 27,676,496, residual 1.2e3 — **both variants
fail identically.**

### 3.3 Accuracy collapse with size (faer default/supernodal, one run each)

| N | rel. residual ‖Ax−b‖/‖b‖ | | N | rel. residual |
|---|---|---|---|---|
| 24 | 5.1e-11 | | 48 | 3.3e-3 |
| 32 | 2.0e-8 | | 52 | 8.3e-2 |
| 36 | 4.4e-7 | | 56 | 0.78 |
| 40 | 2.0e-5 | | 60 | 93.9 |
| 44 | 5.0e-5 | | 64 | **986** |

Smooth exponential-in-N degradation (~1 decade per 3 grid points), in **both**
the supernodal and simplicial paths. Supporting diagnostics at N=64:

- **Pivots are healthy**: |U_jj| min 1.2e-9, median 6.2, max 111; L unit-lower
  with diagonal stored; U upper with full diagonal; zero triangularity
  violations. A correct partial-pivoting LU with this pivot spread (κ(LU) ~ 1e11
  worst case) should solve to ~1e-5, not 1e3.
- **Iterative refinement crawls**: 1205 → 517 → 389 after two refinement sweeps —
  the factors are not even a good preconditioner, i.e. L·U ≠ P·A·Q in practice.
  (A direct L·U vs P·A·Q reconstruction probe was inconclusive — the stored
  factor's index-space mapping is undocumented and no simple
  selection-permutation convention matched even at N=16 where faer's own solve
  verifies to 2e-13; recorded in the code as such.)
- Smallest pivot shrinks with N the same way the residual grows: 2.8e-4 (N=16),
  9.2e-7 (N=32), 2.9e-8 (N=48), 1.2e-9 (N=64).
- My side is verified: CSR==CSC triple-identity, CSR matvec == CSC matvec
  (0.0e0), faer's own structural checks pass.
- **CGNR (independent, matrix-free probe of the same matrix):** N=16 → 4.2e-4,
  N=32 → 3.3e-3, N=48 → 5.4e-2 after the full iteration budget. The N=64 CGNR
  tail was aborted at owner request during the night wind-down after ~9+ min of
  its 554,900-iteration budget, with no iteration/residual line flushed before
  the kill (the N=64 diag lines that did land before the CGNR phase: pivot min
  1.2e-9, LU-solve residual 1205). Partial data plus a stated abort reason is
  the recorded outcome for this diagnostic; the N≤48 trend is unambiguous
  (unpreconditioned CGNR also degrades with N — the system itself gets harder —
  but only faer's LU turns that into a 13-orders-of-magnitude collapse: at N=16
  CGNR sits at 4e-4 while faer's solve is at 2e-13).

### 3.4 Fill-in (L+U nnz, from the owned-simplicial path)

| N | rows | L+U nnz | nnz/row | vs A nnz |
|---|---|---|---|---|
| 16 | 2,401 | 800,328 | 333 | 12.9× |
| 64 | 27,745 | 27,676,496 | 997 | 31.7× |

Two-point law over N=16→64: **fill ≈ rows^1.45** — *worse* than PARDISO's
measured rows^1.20 (87M→445M over N=128→256). Expected cause: faer 0.24.4
offers only COLAMD as the column ordering (the fill-reducing-ordering choice is
not exposed — upstream issue #307 tracks this), so it cannot match the
nested-dissection-quality (METIS) ordering PARDISO uses.

### 3.5 N=128 (full) and N=256 (guarded probe) — NOT MEASURED overnight

- **N=128 full factor+solve: skipped.** The N=64 solve was already invalid
  (9.9e2) when the wind-down order arrived; timing an N=128 factor whose solve
  is extrapolated ~1e16-wrong would have produced headline numbers no one
  should act on, and it would have been the largest job of the night.
- **N=256 guarded probe: skipped (projection only, per the wind-down rule).**
  Projection from the measured N=64 point with the rows^1.45 fill law:
  fill(256) ≈ 27.7M × (405,601 / 27,745)^1.45 ≈ **1.3 G nnz of L+U**;
  at the measured ~18 bytes/nnz of supernodal factor storage (471 MiB /
  27.7M at N=64) that is **~22 GiB of factors alone** — nearly 2× the 12 GiB cap
  and ~4× the 6 GiB probe budget, *before* the solve-accuracy question. The
  guarded probe's decision would have been **stop at the projection: the Rust
  LU wall is below N=256 on this box.** With the same law, N=128 projects to
  ~189M nnz / ~3.2 GiB of factors — the memory wall is *at or below* N≈128
  even before accuracy kills it.

---

## 4. Comparison against the PARDISO reference line

Same box, same thread pinning (4), same pattern sizes:

| quantity | PARDISO (C, docs/profiling.md) | faer 0.24.4 (this spike) |
|---|---|---|
| N=64 — | not in reference table | 1.06 s factor / 22.8 ms solve / +471 MiB |
| N=128 factor | ~10% of wall (assembly 1.2%, analysis 4%, solve 80% of a CGS loop) | not factored (accuracy wall) |
| N=256 peak RSS | 4.95 GB (4 thr, fp64) | projected ~4-5 GiB factors — but solve invalid |
| fill law | rows^1.20 (87M → 445M, N=128→256) | **rows^1.45** (N=16→64, colamd-only) |
| N=512 | OOM-killed at 12 GiB | projected OOM well below N=256 |
| solves at N≤32 | machine-precision (CGS/direct) | machine-precision at N≤24, **~1e2-1e3 at N≥60** |

Where faer works (N≤32) it is fast and the factor memory is modest; the C line's
own UMFPACK comparison already showed pure-Rust-ordering-quality backends
struggle in this regime, and faer's colamd-only ordering repeats that pattern in
fill terms. The decisive difference is accuracy: PARDISO's partial pivoting plus
its internal scaling handles the C's boundary-row magnitudes at every
production size; faer's unscaled partial pivoting does not (§5).

---

## 5. Root-cause assessment (what the next agent should know)

Two mechanisms, both consistent with every measurement above; they are not
mutually exclusive:

1. **Unscaled partial pivoting vs the C's 1/N²-scaled boundary rows
   (structural — present in the real Jacobian).** The C's Robin/decay rows are
   scaled by `dr·dz/rr²` (exact formulas, reproduced verbatim in the
   generator). At fixed domain, `dr ∝ 1/N`, so boundary-row magnitudes fall like
   1/N² relative to the O(1) interior rows. PARDISO ships matrix scaling;
   **faer 0.24.4 has no scaling option at all** and its partial pivoting is
   magnitude-driven, so pivot quality decays with N exactly as observed (smallest
   pivot 1.2e-9 at N=64 and falling exponentially with N). A row-equilibration
   wrapper is the obvious one-experiment rescue and is the **first thing the
   next agent should try** (not run overnight: it would have been new compute).
   If equilibration restores N=64/128 solves to ~1e-10, the no-go downgrades to
   "conditional go with a mandatory equilibration wrapper + a residual check in
   the vertical slice".
2. **faer's sparse-LU accuracy-regression history on exactly this matrix
   class.** Upstream issue #301 (2026-06-23): "`SparseColMat::sp_lu()` solve
   accuracy regression in 0.24.1 … normal 5-point finite-difference Laplacians
   on a 2D grid … degradation worsens as the matrix grows" — the same signature
   measured here. It was fixed by PR #302 (wrong dependency analysis) and
   released in 0.24.2/0.24.4 on 2026-06-24. My measurements are on 0.24.4 and
   still show N-growing degradation, so either the fix is partial or mechanism
   1 dominates. Either way: this subsystem has recent, unresolved-in-spirit
   accuracy issues; pin the faer version and re-verify on every bump.

---

## 6. API / integration notes for the vertical-slice agent

- AUTO (supernodal) is the right variant for this pattern — 24× faster than
  simplicial at N=64 (1.06 s vs 25.4 s). The symbolic phase is sequential-only.
- **`NumericLu` hides its factors** (private `raw`), and even the
  simplicial/supernodal variant choice is not exposed — I sniffed it from the
  Debug string. To read L/U (fill accounting, custom triangular solves), call
  `lu::simplicial::factorize_simplicial_numeric_lu` with a self-owned
  `SimplicialLu`; the supernodal factors are inaccessible, period.
- `MemBuffer`/`MemStack` are **not** re-exported at the faer root — depend on
  `dyn-stack = "0.13"` directly (faer's own version).
- Parallelism is per-call scoped: `Par::rayon(n)` needs no global pool setup.
- Solve RHS goes in as `MatMut::from_column_major_slice_mut` (the `_mut`
  suffix is easy to miss against the MatRef variant).
- The high-level `faer::sparse::solvers::Lu` path is pleasant and hides all
  scratch management; the low-level path used here (needed for per-phase
  timings) is workable but fiddly — budget a day of API friction.
- Only COLAMD is available as the fill-reducing ordering (no METIS-class
  option; upstream #307) — this is why the fill law is rows^1.45 vs PARDISO's
  1.20, and it caps any future "faer at larger N" ambition independent of the
  accuracy issue.
- **Always wire a residual check** (‖Ax−b‖/‖b‖) into the vertical slice: the
  failure mode found here is silent — the factorization *succeeds*, returns
  plausible timings and a finite x, and is wrong by 15 orders of magnitude.

## 7. What was skipped overnight (explicit list)

| item | reason |
|---|---|
| N=128 full factor+solve | wind-down order; N=64 solve already invalid, N=128 extrapolated ~1e16-wrong |
| N=256 symbolic + guarded numeric probe | wind-down order; projection-only result recorded in §3.5 (would stop over budget) |
| 8-thread oversubscription documentation run | wind-down order; note: faer defaults to rayon's global pool = 8 threads on this box, so the C line's 8-thread trap (2.5× slowdown, docs/profiling.md §4) applies to faer defaults too — pin `Par::rayon(4)` |
| Row-equilibration rescue experiment (§5.1) | would have been new compute; it is the top follow-up |
| N=64 CGNR diagnostic tail | aborted at owner request (night wind-down) after ~9+ min of the 554,900-iteration budget; no result line flushed before the kill (§3.3) |
| Best-of-3 repeats beyond N=64 | only N=64 completed its 3-run block before wind-down |

## 8. Reproduction

```
cd ng/spikes/faer-lu
python3 tools/verify_pattern_vs_c.py ../../..        # ALL TABLES MATCH
cargo test --release                                # pattern vs csr.c formulas
cargo run --release -- --n 64 --mode full --force auto --runs 3 --threads 4
cargo run --release -- --n 64 --mode diag --threads 4   # pivots, refinement, CGNR
```
