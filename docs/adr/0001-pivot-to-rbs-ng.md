# ADR 0001: Pivot to RBS-NG — a pure-Rust next generation of ROTBOSON

**Date:** 2026-11-19
**Status:** Accepted
**Supersedes:** the SAN-15 verdict *"Rust port of hot components — NO-GO (now)"*
(`docs/solver-language-evaluation.md` §4.2), reopened under that memo's own
pre-declared trigger.

## Context

ROTBOSON (the C codebase, `src/`) solves rotating boson star initial data:
axisymmetry, quasi-isotropic coordinates, six coupled nonlinear elliptic PDEs
plus a scalar-field frequency eigenvalue ω, global Newton with MKL PARDISO
sparse-direct solves. It is validated to ~1e-13 against its own golden
solutions and reproduces the reference paper's critical points
(`RUNBOOK.md`).

The M3 profiling study (`docs/profiling.md`, living record
`data/summaries/profile-baseline.md`) measured the structural limits of that
architecture, and SAN-15 (§4.2) recorded the no-go with an explicit reopener:

> The only scenario that reopens this is a rewrite that also changes the
> algorithm (M4 grid strategy) or drops PARDISO — neither is earned yet.

Both halves of that reopener are now earned. The measured evidence:

- **≥95% of wall time is inside the vendor factorization/solve library** at
  every measured size; our own assembly is 1–2%. A language swap alone has no
  headroom — that verdict stands.
- **LU fill grows ~rows^1.20** (87M → 445M nnz from N=128 → N=256). Sparse
  direct is superlinear in both time and memory; no vendor library changes
  that exponent.
- **N=512 OOMs at the 12 GiB cap**; the paper grid N=400 fits only through
  fp32+IR (8.97 GB peak, ~3 GB headroom). The memory wall is the scaling
  limit, not CPU.
- **UMFPACK (MKL-free direct) is not a path**: >3.5× slower at N=256 —
  killed head-to-head. Dropping MKL within the sparse-direct paradigm
  trades a wall for a slower wall.
- **A dilute l=4 campaign dies on domain size**: the down-leg widens until
  `stopped:boundary` by step 6 — the seed's support fills the box while the
  grid stays uniform. Resolution-vs-domain is the physics blocker for l>4.
- **l=5/l=6 seeds fail with PARDISO −4 at N=128**: the badly-scaled 6-field
  Jacobian defeats direct pivoting itself, before any scaling argument
  even starts.

The conclusion: the *algorithm* (uniform grid + global sparse direct) is the
binding constraint. The rewrite changes the grid strategy (M4: multigrid /
non-uniform grids) and drops PARDISO — the exact scenario SAN-15 said would
reopen the question.

## Decision

1. **RBS-NG is pure Rust from day one.** No MKL, no FFI solver cores. The
   transitional direct backend is `faer` sparse LU, used **only at parity
   sizes (N ≤ 128)** to port and validate the elliptic solver against the
   oracle.
2. **The production solver path is geometric multigrid + Krylov**
   (FGMRES-class outer iteration). **Never rely on global sparse LU
   scaling**: direct solves exist for parity gates and unit tests only, and
   every design choice that bakes in a direct solve must be labeled as
   transitional. The intermediate step is per-field block preconditioners
   exploiting the 6-field Jacobian structure.
3. **The C codebase is frozen as the golden oracle.** No new physics enters
   C. Every Rust stage is gated by parity against the oracle's 1e-13-validated
   solutions (gate ladder in `ng/docs/architecture.md`).
4. **Evolution is staged**: 1D spherical full-GR first
   (Einstein–Klein–Gordon; formulation study BSSN vs CCZ4 vs
   generalized-harmonic in 1D), stationary-hold validation against the
   elliptic initial data, then axisymmetry; 3D/AMR is gated behind those.
5. **The symbolic chain is retargeted, not rewritten**: `tools/sympy_system.py`
   remains the single source of truth; a Rust code generator
   (`tools/generate_rust_kernels.py`, SAN-56) emits the residual + Jacobian
   kernels checked in under `ng/`, with a byte-identical `--check` CI gate
   exactly like the C kernels. The C kernels stay as the structural
   cross-reference.
6. **Visualization is first-class**: VTR/VTH (ParaView-native) export
   alongside HDF5 from day one, mirroring the C HDF5 writer's
   self-describing contract so `tools/rotboson_io.py` reads Rust output
   unchanged.

## Consequences

- The `ng/` Cargo workspace (SAN-53) is the new development surface:
  `rbs-core` (grids/fields/stencils), `rbs-elliptic` (residual/Jacobian
  assembly, Newton, ω constraint), `rbs-linalg` (backend trait; faer
  transitional + MG-Krylov placeholder), `rbs-io` (HDF5 + VTR, trait-based),
  `rbs-driver` (campaign port target), `rbs-evolve`, `rbs-viz`.
- Direct-solve parity gates (N=64, N=128 seed solves) are the first
  milestones; the grid ladder then buys the dilute-l domain reach that the
  C campaign could not (stretched radial coordinate → compactification /
  multiblock → block-structured refinement).
- SAN-15's remaining valid verdicts stand unchanged: keep C + PARDISO for
  the frozen oracle, fp32+IR recipe for high-N C campaigns, UMFPACK stays
  the C fallback backend.

## References

- `docs/solver-language-evaluation.md` — SAN-15 memo (source of the
  superseded verdict and its reopener).
- `docs/profiling.md` — M3 profiling report (the phase-shares, rows^1.20
  fill growth, N=512 OOM, fp32+IR recipe).
- Both land on `master` with the M3 profiling merge; on this branch they
  live on `m3/profiling-baseline`.
- `RUNBOOK.md` — fold table, `stopped:boundary` semantics, campaign
  validation records.
- `ng/docs/architecture.md` — the RBS-NG architecture this ADR mandates.
