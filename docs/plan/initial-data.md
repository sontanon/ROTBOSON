# Plan: Initial Data (Linear project "Initial Data", team NR)

Agreed 2026-11-27. Decision record: `docs/adr/0002-initial-data-scope.md`.

## Goal

A pure-Rust solver for rotating boson star initial data that we are proud to ship on `master`:

- **v1.0** reproduces the paper's fold points (the minimum frequency ω_min on each branch) for l = 1..6, **each with an error bar**.
- **v1.1** improves the numerics (compactified radial coordinate, 6th/8th-order stencils) so the same answers need smaller grids.

Its output file format is the **interface** a future evolution code will read (defined in A1).

## Principles

1. **Physics and numerics results are the gates, not code metrics.** Every milestone ends with a result you can show: a table, a figure, a converged fold.
2. **The C code is the oracle.** It is frozen. Rust results are compared against it until Rust surpasses it.
3. **Precision is configurable per component.** Residuals and convergence tests are always fp64 for fold work; preconditioner factors may be fp32; an all-fp32 mode exists for exploration.
4. **No Python compatibility after A2.** The Rust driver replaces `tools/sweep_driver.py`.
5. **Small, exact issues.** Most issues go to a small local model. See `docs/plan/issue-template.md`.

## Milestones

| Milestone | Goal | Exit check | Demo (what you can look at) |
|---|---|---|---|
| **A0 · Consolidate** | One clean, green `master`. Settle the open Rust PRs, build the C oracle on this machine, remove out-of-scope code, format the workspace. | CI green (build, test, clippy, codegen check); C `ctest` passes locally; no stale PRs. | An l=1 star solved from scratch in Rust (`cargo run --release -p rbs-elliptic --example l1_star`), opened in ParaView, and a figure of the six fields plus φ = ρ^l ψ next to the C oracle's solution of the same problem, with their difference. `docs/results/A0-demo.md`. |
| **A0.5 · Near-fold solver spike** | Does the BlockFieldKrylov solver still work near the fold? Compare three preconditioners. | Table: Krylov iterations against distance to the fold (N=128, 256); one Newton solve converging at the closest state; verdict. | `docs/results/A0.5-near-fold.md` with three figures: the l=1 fold curve (ω against ψ0, and mass against ω) with the test states marked; Krylov iterations against distance to the fold, one line per preconditioner; the smallest Jacobian eigenvalue against ψ0, fixed-ω against fixed-φ (the fixed-ω one goes to zero at the fold). |
| **A1 · Single-solve CLI** | `rbs solve config.toml` writes `solution.h5`, with the C contract. Precision and stencil order designed in. Initial-data format v1 written down. | Matches the C oracle at N=64 and N=128. | `rbs solve` and `rbs plot` on any config; fp64 against fp32-factor runs side by side. |
| **A2 · Continuation to the fold** | Rust driver runs real solves; the Python driver retires. | l=1 down-campaign at N=256 reaches the turning point, matching C. | The l=1 fold curve drawn by the Rust driver, overlaid on the C campaign. |
| **A3 · Scaling and benchmark** | Rayon over blocks and assembly; measure. | Time and memory table, N=128..512, C (fp64, fp32+IR) against Rust (fp64, fp32 factors), at the initial guess and near the fold. | Time and memory against N, log-log, C against Rust. |
| **A4 · Reproduction with error bars → v1.0** | ω_min for l=1..6 with a resolution study (Richardson, check order p≈4) and a domain-size study. | Table of ω_min ± error against paper Table IX.1; figures; tag v1.0. | Fold curves for l=1..6 with error bars; convergence plots. |
| **A5 · Numerics beyond the paper → v1.1** | Compactification (after a ρz-vs-rθ spike), then order 6/8, via the mapped-grid abstraction. | Smaller error bars at smaller N than A4; tag v1.1; C and Python retire to a tag. | Error against N for orders 4, 6, 8, with and without compactification. |

Specs: `docs/specs/<milestone>-*.md`. A0 and A0.5 are written. The others are written when their milestone is next.

## Parked (not in this project)

- **Multigrid** (per-block, as a drop-in for block LU). Only if we need N > 512.
- **Time evolution, mesh refinement.** A separate future project that consumes the A1 file format.
- **Spectral methods** (as in Kadath/LORENE). Revisit if A4 shows finite differences struggle at the fold.

## Decision log

| Date | Decision |
|---|---|
| 2026-11-27 | Rust, not C, for all new work. The speed win measured so far comes from the algorithm (block-preconditioned Krylov against global sparse LU), not the language. |
| 2026-11-27 | The ω constraint is **not** eliminated exactly (Keller bordering is unstable at the fold, where the fixed-ω Jacobian is singular). Instead the bordered structure may be used inside the preconditioner. Tested in A0.5. |
| 2026-11-27 | Python byte-compatibility of the driver's JSON is dropped once the Rust driver works (A2). |
| 2026-11-27 | Release split: v1.0 = faithful reproduction with error bars, v1.1 = better numerics. |
| 2026-11-27 | Linear: new team NR; old projects ROTBOSON and RBS-NG closed. |
