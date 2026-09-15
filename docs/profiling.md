# Profiling report — ROTBOSON solver (SAN-11, M3)

**Status:** acceptance-ready. Living record with every raw number:
`data/summaries/profile-baseline.md`. All runs on the `pi-dev` container
(i7-6700HQ, 4 physical cores / 8 hyperthreads, cgroup CPU quota 4,
memory cap 12 GiB).

## 1. Scope & method

Systematic measurement of where wall time goes at production sizes, how the
solver scales with threads, and whether the PARDISO dependency / C language
pay for themselves — the decision input for the solver & language
evaluation (SAN-15).

Instrumentation: lightweight in-tree profiler (`src/profile.c`), enabled by
`ROTBOSON_PROFILE=1` — per-phase `omp_get_wtime` timers + VmRSS/VmHWM
sampling, reported at every exit path via `atexit`. Phase buckets live at
shared chokepoints (`csr_gen_jacobian`, `rhs`, `norm2_all_variables`,
`solver_solve`, `solver_repeated_solve`) so all solverType paths are
covered. PARDISO factor diagnostics (iparm[17]/[18]) and per-solve CGS
diagnostics (iparm[19]) are enabled. Results reproducible via the commands
recorded in `data/summaries/profile-baseline.md`.

Representative workloads:

| label | config | rows | nnz | notes |
|---|---|---|---|---|
| N=128 smoke | `l1_seed.toml` | 104,545 | 3.42M | fixed-ω cold start |
| N=256 production | `probe_l1_umfp_n256.toml` | 405,601 | 13.5M | fixed-ω cold start |
| N=400 paper | `probe_l1_n400.toml` | 970,240 | — | fp32+direct+IR |

## 2. Phase breakdown (where time goes)

| size | assembly | analysis | factor | solve | peak RSS |
|---|---|---|---|---|---|
| N=128 (4 thr) | 1.2% | 4% | 10% | **80%** | 1.33 GB |
| N=256 (4 thr) | 1% | 4% | 21% | **73%** | 4.95 GB |
| N=400 fp32 (8 thr) | 0.8% | 4% | **74%** | 7% | 8.97 GB |

Assembly is 1–2% at every size — there is **no headroom in our own code**;
≥95% of wall time is inside the vendor factorization/solve library. At
N≥256 factorization becomes the dominant cost and the dominant memory item.

## 3. Surprising finding #1: "solve" is a CGS loop, not a solve

After the first analysis, `solver_solve` sets `iparm[3]=61`, turning every
phase-33 call into an **LU-preconditioned CGS iteration loop** (stopping
1e-6) instead of a plain back-substitution — ~80% of wall at N=128. This is
a deliberate, load-bearing choice: A/B testing showed direct refactorization
of near-singular (fold) Jacobians blows up (56 s/factor vs ~20 s/step under
stale-LU CGS), so CGS mode is the right campaign default. `ROTBOSON_PARDISO_DIRECT=1`
opts into direct+IR solves, worth it for cold-start seeds and standalone
regrids (−28% to −40%).

## 4. Surprising finding #2 (this report): 8 threads is 2.5× slower than 4

OpenMP scaling sweep, N=256 single solve, sequential runs:

| threads | wall | assembly | analysis | factor | solve | peak RSS |
|---|---|---|---|---|---|---|
| 1 | 138 s | 2.66 | 8.77 | 35.99 | 89.79 | 4.87 GB |
| 2 | 100 s | 1.31 | 5.46 | 22.91 | 70.49 | 4.89 |
| **4** | **92 s** | **0.71** | **4.19** | **17.80** | **68.94** | 4.95 |
| 8 | **228 s** | 2.52 | 11.98 | 46.00 | 165.54 | 5.10 |

**8 threads is 2.5× slower than 4, and every phase degrades** (assembly
3.5×, analysis 2.9×, factor 2.6×, solve 2.4× worse). Verified cause:

- cgroup v2 `cpu.max = 400000/100000` → **container CPU quota = 4**.
- Host is 4 physical cores / 8 hyperthreads (`cpu cores: 4`, `siblings: 8`);
  `nproc=8` counts hyperthreads, not the usable cap.
- Nothing in the repo sets `OMP_NUM_THREADS` (zero hits), so runs silently
  default to `nproc=8` → oversubscribe the 4-CPU quota → thread thrash.
- The observed "~67% load on all 8 cores" is the oversubscription signature.

**Action:** pin `OMP_NUM_THREADS=4` (or `MKL_DYNAMIC=FALSE`) for solver runs
— a ~2.5× free speedup on this box. The RUNBOOK's "4 cores" premise is
right but unenforced. Re-verify the N=400 probe: if the 583 s measurement
used the 8-thread default, it too was oversubscribed (~2.5× slower than at
4 threads). This is the OpenMP scaling answer — not a marginal gain but a
silent 2.5× penalty currently baked into every production run.

## 5. Memory wall — the real scaling limit

LU fill grows ~rows^1.20 (measured 87M → 445M nnz, N=128 → N=256); PARDISO
factor memory is the binding constraint, not CPU.

| probe | unknowns | LU nnz | wall | peak RSS | verdict |
|---|---|---|---|---|---|
| N=384 | 903k | 1.14G | 556 s | 7.79 GB | fits 8 GiB |
| **N=400 (paper)** | 970k | 1.28G | 583 s | **8.97 GB** | **fits 12 GiB (≈3 GB headroom)** |
| N=512 | 1.60M | — | — | OOM-killed | does NOT fit 12 GiB |

## 6. fp32 + double-residual IR — the production lever

`ROTBOSON_PRECISION=32` + `ROTBOSON_PARDISO_DIRECT=1` (IR default 2) halves
factor memory (N=400 peak 8.97 GB vs projected 14+ GB double) and unlocks
the paper's N=400 in-core at the 12 GiB cap. Full l=1 campaign: 0
retries/stalls across 169 steps; fold vertex ω_min reproduces double-direct
to ~1e-16; scalars land on-branch (~1e-13 rel). Newton's own double
convergence test acts as the outer corrector.

## 7. UMFPACK head-to-head — not a path

N=256 identical config, both backends. PARDISO best (4 thr) 92 s; UMFPACK
**did not reach its first solve within 330 s** (>3.5× even vs PARDISO's
best) — killed per the 1.5× rule and ruled out. Consistent with the
historical small-grid experience (weaker ordering at 2D mesh fill,
single-threaded factor). The UMFPACK backend remains only as the MKL-free
fallback (bit-equivalent at smoke sizes). **PARDISO stays.**

## 8. Top hotspots (symbol-level)

Not run separately: the hot phases are inside MKL PARDISO (a closed vendor
library, not symbol-profileable), and our own code is 1–2% of wall. A
symbol-level hotspot pass would re-confirm "assembly negligible, rest is a
library call" — assessed as no headroom, so deliberately skipped.

## 9. Recommendations

| # | recommendation | expected gain | status |
|---|---|---|---|
| 1 | Pin `OMP_NUM_THREADS=4` (now `[solver] numThreads`, default 4) | ~2.5× wall speedup on this box | **implemented** |
| 2 | Adopt fp32+IR as production recipe for N≥256 | N=400 in-core, −34% memory | **adopted** |
| 3 | Keep PARDISO/MKL (dependency justified) | — | **decision** |
| 4 | UMFPACK swap: no-go (killed) | — | **closed** |
| 5 | Re-verify N=400 at 4 threads | likely ~2.5× faster than 583 s | **follow-up** |
| 6 | Scope full N=400 validation (l=1..6); note l=5/6 descoped | closes paper validation | **follow-up** |

Full decision framing and go/no-go per component: `docs/solver-language-evaluation.md`.
