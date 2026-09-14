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

## N=256 scaling run (2026-09-13)

`sweep_driver` + `configs/l1_regrid_n256.toml` (domain-keeping regrid,
405,601 rows, 13.5M nnz, 5 Newton iterations, wall 130.3 s):

| metric | N=128 | N=256 | ratio | exponent |
|---|---|---|---|---|
| rows | 104,545 | 405,601 | 3.88× | — |
| LU nnz | 87.0M | 445.3M | 5.11× | **1.20** |
| PARDISO peak internal | 1.02 GB | 4.73 GB | 4.63× | 1.11 |
| process peak RSS | 1.33 GB | 5.07 GB | 3.81× | 1.07 |
| factor ms/call | 1,482 | 14,115 | 9.5× | ~1.6 (memory-bound) |
| solve ms/call (CGS) | 2,747 | 4,785 | 1.74× | ~0.4 |
| wall total | 58 s | 130 s | 2.25× | — |

At N=256 the phase split shifts: factor (43.3%) ≈ solve (44.1%) — at
larger N the **factorization becomes the dominant cost and the dominant
memory item**. Extrapolating LU nnz ~ rows^1.20:

| N | rows | LU nnz est. | factor values | est. peak RSS |
|---|---|---|---|---|
| 384 | 903k | ~1.2G | ~9.3 GB | ~11–13 GB (does NOT fit 8 GB) |
| 512 | 1.60M | ~2.3G | ~18 GB | ~19–22 GB |

Note: N=256 at 5.07 GB already runs against the 8 GB container limit
(system-wide peak ~7.1 GB). fp32 factors (÷2 on values) make N=384
borderline-feasible (~6 GB) but not N=512 (~10 GB). Reaching the paper's
N=400 and beyond in doubles on this box requires compression (BLR/MUMPS),
OOC factors, hybrid/GPU memory — or M4 non-uniform grids to cut unknowns
at equal accuracy.

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

## Stage 1 fp32 (2026-09-13) — PASSED

`ROTBOSON_PRECISION=32` (pardiso_solve.c float conversion, iparm[28]=1 set
in solver_start; Newton/assembly/output stay double). Master seed,
CGS default:

| metric | double | fp32 | Δ |
|---|---|---|---|
| PARDISO internal peak (iparm 15+16) | 1.02 GB | 0.65 GB | −36% |
| factor+workspace (iparm[16]) | 746 MB | 381 MB | **−49% (predicted −50%)** |
| process peak RSS | 1.356 GB | 0.979 GB | −28% |
| wall (CGS default) | 58 s | 55 s | −5% |
| M_Komar1 vs double | — | 7.89310e-01 | **1.3e-5 rel** |

fp32 + direct (ROTBOSON_PARDISO_DIRECT=1), bonus probe:

| metric | double baseline | fp32 direct |
|---|---|---|
| wall | 58 s | **28.7 s (−50%)** |
| solve ms/call | 2747 | 389 |
| M_Komar1/M_Komar2 | 7.89309e-01 | 7.89309e-01 (match at 6 digits) |
| Newton | 8 iters | converged (7) |

Verdict: memory win is exactly as predicted; robustness OK in both modes;
Newton's own double-precision convergence test acts as an outer corrector,
so fp32+direct already lands on-branch. Residual accuracy (~1e-5 mass
under CGS-fp32) is exactly what Stage 2 (double-residual IR) targets.
Next: Stage 2 IR, then N=256 and N=384 probes (fp32 factor memory at
N=384 ≈ 4.7 GB → peak ~7-8 GB: the go/no-go number).

## Stage 2 fp32: double-residual IR (2026-09-14) — PASSED

`ROTBOSON_FP32_IR=0..4` (default 2), active only with fp32 + direct
(`ROTBOSON_PRECISION=32 ROTBOSON_PARDISO_DIRECT=1`). Mechanism: after each
fp32 solve, r = f - A·du in double (OpenMP CSR matvec), correction solve
reusing the same fp32 factors (extra phase-33), accept only on strict
residual drop. Implemented in pardiso_solve.c; PROF_IR bucket added.

Gates (references: fresh double seed run; banked `runbook-l1-direct-ref`):

| gate | case | result |
|---|---|---|
| G1 | seed, fp32+direct+IR(2) | scalars rel ~1e-13, fields ~1e-15 vs double; wall 40.5 s (vs 58 double) PASS |
| G2 | fold step (post-regrid, N=256) | converges 98 s (vs the ~50-min fp32-CGS stall); see fold-noise note PASS |
| G2b | healthy N=256 regrid | scalars ~1e-12, fields ~1e-14 vs double PASS |
| G3 | IR ladder, seed | IR=1 → ‖r‖~1e-7; IR=2 → ~1e-10; IR=3 → ~3e-13 (wall 34/40.5/43 s). Contraction ~1e-4/step; IR=2 is the right default |
| G4 | CTest 4/4; double path untouched | PASS |

Fold-noise note (G2): at the near-singular fold step, fp32+direct+IR and a
fresh DOUBLE-direct run differ from the banked double reference by
*identical* amounts (worst field 1e-4, ω rel 6e-8): the fold's κ amplifies
run-to-run solver noise identically for any method. fp32+IR adds zero error
over double-direct there; a strict 1e-10 agreement at fold steps is
unreachable by any solver, and the documented catalogue fold sensitivity
(M2: vertex ±1e-4 across legit grids) dominates anyway.

Incidents found while testing (both pre-existing, not Stage-2):
1. **Missing initial-data file segfaults** the binary during allocation
   instead of erroring (initial-read path; worth hardening later).
2. **Final-dir rename collision is silent**: if the final solution dir
   already exists, the write lands in the placeholder dir
   (`l=1,w=X.XXXXXE-01...`) with no warning — campaign output hygiene
   relies on externally removing stale dirs.

## Stage 3: full fp32 campaign + N=384 probe (2026-09-14) — GO

Production recipe: `ROTBOSON_PRECISION=32 ROTBOSON_PARDISO_DIRECT=1`
(IR default 2). Campaign: fresh `runbook_l1_down.toml`, zero retries/warnings
across all 169 steps incl. the fold regrid (the fp32-CGS attempt had stalled).

| metric | M2 CGS (double) | direct (double) | **fp32+direct+IR** |
|---|---|---|---|
| total binary wall | 2778 s | 3825 s | **3233 s** |
| fold steps (166–168) | ~20 s ea | 138 s ea | **~100 s ea** |
| peak RSS, fold step | (ref: 5.07 GB @ N=256) | 5.14 GB | **3.38 GB (−34%)** |
| retries / stalls | 1 retry (step 65) | 0 | **0** |
| fold vertex ω_min | 0.6460482 (poly2) | 0.6455295 (poly3) | **0.64552950 (poly3)** — agrees to 1e-16 with the direct-double campaign |

### N=384 feasibility probe (configs/probe_l1_n384.toml)

Master seed, fixed ω=0.95, 6·388²+1 = 903k unknowns, fp32+direct+IR:

- **Converged** (12 NLEQ iterations, 4 factors, 17 solves, on-branch
  invariant M_Komar1 = M_Komar2 = 0.789975 preserved), exit 0.
- Wall **556 s**; factor 99.6 s/call (near-singular ~3× healthy cost);
  IR adds 4.6 s × ~2 × 17 = 78 s; analysis 27 s.
- LU fill **1.14 G-nnz** (matches the rows^1.20 extrapolation).
- Peak RSS **7.79 GB vs the container's exact 8 GiB cgroup cap — fits
  with ~0.2 GB headroom** (razor-thin; swap/other processes could kill it).

### M3 decision input (cap raised 8→12 GiB, 2026-09-14)

With the container cap at 12 GiB (host 16 GB, other containers live):

| probe | unknowns | LU nnz | wall | peak RSS | verdict |
|---|---|---|---|---|---|
| N=384 (earlier) | 903k | 1.14G | 556 s | 7.79 GB | fits 8 GiB |
| **N=400 (paper grid)** | 970k | 1.28G | 583 s | **8.97 GB** | **fits 12 GiB (≈3 GB headroom)** |
| N=512 | 1.60M | — | — | **OOM-killed** | does NOT fit 12 GiB (`oom_kill 1`, cgroup memory.events) |

Both N=400 and N=384 converged (12 NLEQ iters, on-branch invariant
M_Komar1=M_Komar2 preserved). N=512 dies during the first PARDISO
factorization. So the fp32 frontier at 12 GiB is **N≈400–450**; the paper's
N=400 is reachable in-core, N=512 is not without BLR/OOC/GPU/16-GiB.

### M3 decision input

fp32+direct+IR reaches the paper's N=400 at 12 GiB. Reaching N=512 needs
either a bigger cap (~16 GiB; host has 16 but shares with other containers)
or memory compression (BLR/MUMPS) / OOC / GPU hybrid. M4 non-uniform grids
remain the algorithmic answer to the resolution-vs-domain tension.

## Next experiments (proposed order)

1. ~~iparm[3]=0, iparm[7]=2~~ — DONE 2026-09-13, mixed verdict:
   - Master seed (cold start, 17 solves / 4 factors): 58 s → 42 s (−28%),
     **bit-identical solution** (M_Komar/J/ω equal to display precision).
   - Full l=1 runbook campaign (169 steps, regrid at 165, turning point
     stop): 2778 s (M2 ref) → 3825 s (+38%) — **direct mode is a net loss
     for continuation**. Near-fold Newton matrices are near-singular:
     direct factorization costs 56 s/call at N=256 fold steps (vs 14 s at
     the healthy N=256 regrid), while CGS mode reuses the stale LU cheaply
     (ref fold steps ≈ 20 s total).
   - Correctness of the new-mode campaign confirmed: ω equal to ~1e-15
     relative at steps 20/60/100/140/160/164; psi_f/log_alpha_f/beta_f
     equal to ~1e-14; fold vertex estimate 0.6455295 (poly3, 5 points) vs
     paper 0.64561 → −0.012% (M2/C-GS estimate: +0.068%, poly2).
   - Resolution: keep iparm[3]=61 (CGS) as the campaign default — it is a
     deliberate, load-bearing algorithm choice. New env switch
     ROTBOSON_PARDISO_DIRECT=1 opts into direct+IR solves (worth it for
     cold-start seeds and standalone N=256 regrids, −28% to −40%).
   - Open follow-up: iparm[3]=31 (CGS, tolerance 1e-3) — cheaper CGS
     iterations; must check branch accuracy vs the 1e-6 baseline.
2. N=256 run — DONE: see "N=256 scaling run" above.
3. ~~fp32 + IR + full campaign + N=384 probe~~ — DONE (see Stage 3
   section). Production recipe: `ROTBOSON_PRECISION=32
   ROTBOSON_PARDISO_DIRECT=1`.
4. MUMPS-BLR: **deprioritized by Stage 3** — fp32 alone covers N=384 at
   8 GiB and projected N=512 at a 12–16 GiB cap (host has 16 GB; container
   cap is the only blocker). Revisit BLR/MUMPS only if N>512 in-core, or
   side-by-side large-grid concurrency, becomes a requirement.
5. Ask the user to raise the container memory cap (~12–16 GiB), then run
   the N=512 probe + an l=4/5 high-resolution validation at fp32.
6. Housekeeping tickets: SAN-32 (robustness: segfault + rename collision);
   optional CGS-vs-direct default reconsideration for N≥256 campaigns.
