# ADR 0002: Re-scope the Rust work to an initial-data release

**Date:** 2026-11-27
**Status:** Accepted
**Amends:** ADR 0001 (`0001-pivot-to-rbs-ng.md`)

## Context

ADR 0001 set up one program, RBS-NG, that would port the C solver, then build a grid ladder and multigrid, then add time evolution, all in this repository. By 2026-11-27:

- the Rust port solves the l=1 star at N=64 and matches the C oracle;
- a field-blocked Newton–Krylov backend (BlockFieldKrylov) measured 12–20× faster than PARDISO at N=256 with 2.6× less memory, with iteration counts that do not grow with N;
- nothing on the grid ladder, multigrid, or evolution existed beyond placeholders;
- the plan's milestones were too large to hand to agents, and gated on parity rather than on results.

## Decision

1. **This repository ships an initial-data solver**, the Linear project "Initial Data" (team NR), with milestones A0–A5 in `docs/plan/initial-data.md`. v1.0 reproduces the paper's l=1..6 folds with error bars; v1.1 adds compactification and higher-order stencils.
2. **Time evolution and mesh refinement leave this repository's scope.** They will be a separate project, which reads this solver's output through a versioned file format defined in milestone A1. The `rbs-evolve` placeholder crate is removed.
3. **BlockFieldKrylov is the backend.** ADR 0001's "geometric multigrid + Krylov is the production path" is withdrawn. Multigrid is parked and comes back only if grids beyond N=512 are needed, as a per-block replacement for block LU. faer sparse LU stays only as a cross-check for small parity tests.
4. **The fold is the critical test of the backend.** Milestone A0.5 measures the solver near the ω turning point before further work builds on it.
5. **Python byte-compatibility is dropped** once the Rust driver runs real campaigns (A2).

Kept from ADR 0001: pure Rust; the C code frozen as the oracle; the SymPy source of truth with generated, checked Rust kernels; HDF5 and VTR output from day one.

## Consequences

- `ng/docs/architecture.md` is rewritten to this scope (an A0 issue).
- The RBS-NG and ROTBOSON Linear projects are closed. Their history stays in team SAN.
