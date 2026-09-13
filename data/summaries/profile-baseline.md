# Profiling baseline — master l=1 seed (M3)

Branch: `m3/profiling-baseline`. Config: `configs/l1_seed.toml` (fixed-ω=0.95,
N=128, order 4, `solverType=1` nleq_err, `useLowRank=0`). MKL PARDISO
(`pardiso_64`, mtype 11, METIS reorder, matching on), 4 threads.
Enabled by `ROTBOSON_PROFILE=1` (see `src/profile.h`).

Run 2026-09-13, exit 0, converged in 8 Newton iterations.

## Matrix

| quantity | value |
|---|---|
| rows (6 vars + ω constraint) | 104,545 |
| input nnz (CSR) | 3,417,875 |
| **LU factors nnz** (iparm[17]) | **87,016,119** (25× fill-in) |
| factorization work (iparm[18]) | 91,574 MFLOP (≈60 GFLOPS on 4 threads) |
| PARDISO peak internal (iparm 15+16) | ≈1.02 GB |
| process peak RSS (VmHWM) | **1.33 GB** |

## Phase split (57.96 s wall)

| phase | calls | total s | ms/call | share |
|---|---|---|---|---|
| pardiso solve (phase 33) | 17 | 46.70 | 2747 | **80.5%** |
| pardiso factor (phase 22) | 4 | 5.93 | 1482 | 10.2% |
| pardiso analysis (phase 11) | 1 | 2.45 | 2453 | 4.2% |
| jacobian assembly | 4 | 0.70 | 174 | 1.2% |
| rhs residual | 14 | 0.24 | 17 | 0.4% |
| norms | 48 | 0.65 | 14 | 1.1% |

4 factorizations for 8 outer Newton iterations; 17 total linear solves
(the nleq_err trial loop re-solves with a frozen factorization via
`solver_repeated_solve`).

## Findings

1. **The solve phase dominates (80%), and it is not a triangular solve.**
   After the first analysis, `solver_solve` sets `iparm[3]=61`, which makes
   every phase-33 call an LU-preconditioned CGS iteration loop (stopping
   criterion 1e-6) instead of a direct back-substitution. ~2.7 s per solve
   vs ~1.5 s for a full numerical factorization. iparm[19] diagnostics show
   CGS excursions (errors 1, 2, 4, 9 across the 4 primary solves).
   MKL's own documented iparm[19]=5 hint applies: *"factorization is too
   fast for this matrix — it is better to use iparm[3]=0"*.
   → Candidate quick win: `iparm[3]=0` + explicit `iparm[7]` iterative
   refinement. Must re-verify Newton convergence behavior before adopting
   (the CGS setting presumably earns its keep somewhere — possibly at folds).

2. **Memory is LU fill-in, and it is huge relative to the matrix.**
   87M factor nnz ≈ 700 MB for a 3.4M-nnz input — this is the N=512
   blocker (2D direct-solver fill growth). Process peak 1.33 GB at N=128;
   ~4× unknowns at N=256 → ~4 GB+; N=512 → ~12-16 GB, i.e. "does not fit
   in 8 GB" as the runbook says. Any memory win (BLR/MUMPS, fp32 factors,
   OOC — `ROTBOSON_PARDISO_OOC` hook already exists) acts directly on this.

3. **Assembly is a non-issue** (1.2%); language rewrites would buy nothing
   on the current algorithm. OpenMP thread sweeps and hotspot-level
   profiling remain worthwhile only for the factor/solve phases.

## Harness

`src/profile.h/.c`: `ROTBOSON_PROFILE=1` enables per-phase timers
(omp_get_wtime) + VmRSS/VmHWM sampling, registered via atexit so every exit
path prints the summary. Instrumented chokepoints (cover all solverType
paths): `csr_gen_jacobian`, `rhs`, `norm2_all_variables`,
`solver_solve` (analysis/factor/solve split), `solver_repeated_solve`.
PARDISO factor diagnostics (iparm[17]/[18] reporting enabled via -1) print
after each numerical factorization; CGS diagnostics (iparm[19]) print after
each solve.

Raw log: `out/profile_seed.log` (not tracked).

## Next experiments (proposed order)

1. `iparm[3]=0`, `iparm[7]=2` (default IR): expect solve phase 2747 ms →
   tens of ms; Newton wall −70%+. Validate identical solutions (M_Komar
   match, branch agreement) vs this baseline.
2. N=256 run (`l1_regrid_n256.toml`) for the fill-growth exponent and
   RSS curve — the number that sizes every memory option.
3. fp32 factors (iparm[27-1]=1, requires matrix in single precision) +
   double-precision IR: ~40% factor memory saving; accuracy recovery via
   refinement to be demonstrated.
4. MUMPS-BLR evaluation against the 87M-nnz LU (target: factor nnz
   reduction ≥2× at comparable wall time).
