# RBS-NG architecture

**RBS-NG** is the pure-Rust next generation of ROTBOSON (ADR 0001,
`docs/adr/0001-pivot-to-rbs-ng.md`). The C codebase is the frozen golden
oracle: every stage of the Rust rewrite is gated by parity against its
1e-13-validated solutions. This document is the working architecture the
`ng/` workspace implements.

The problem being solved (unchanged): axisymmetric rotating boson star
initial data — six coupled nonlinear elliptic PDEs plus the scalar-field
frequency eigenvalue ω in quasi-isotropic coordinates, solved as one
global Newton system.

## 1. Grid ladder

The ladder buys, one rung at a time, the domain reach that uniform grids
turn into a memory wall (the l=4 down-campaign died at `stopped:boundary`
six steps in; N=512 OOMs at 12 GiB; LU fill grows ~rows^1.20):

1. **Uniform (r, z)** — the parity rung. Identical to the C oracle's grid
   (`NrTotal`/`NzTotal`/`ghost`, cell centers at `r = dr(i + 0.5 - ghost)`)
   so the direct parity gates at N=64/128 are meaningful. No new physics,
   no new numerics — just the port.
2. **Stretched radial coordinate** — the first domain lever. A smooth
   radial map `r = r(ξ)` concentrates resolution near the star and spares
   it in the dilute tail; the finite differences move to Fornberg
   non-uniform weights on the mapped grid (the C code already validates
   Fornberg stencils — `tests/fornberg.c`, and the same machinery
   generates the kernel subtype layout). Gate: reproduce the uniform-grid
   solution on the mapped grid to discretization error at fixed dr.
3. **Compactified outer domain / multiblock** — the rung that removes the
   boundary entirely: Robin conditions applied at radial infinity on a
   compactified block, matched to the interior block. This is what the
   dilute l>4 branches actually need (their support fills any finite box).
4. **Block-structured refinement** — resolution where the star is, coupled
   through the multigrid hierarchy that the solver strategy (§2) already
   requires; AMR-like behavior falls out of block refinement plus the
   continuation driver's triggers rather than a separate machinery.

Each rung is additive: the previous rung's gates keep running, and the
oracle C solves stay the reference at every rung the oracle can still
reach.

## 2. Solver strategy

The measured profile motivates everything here: in the C code, ≥95% of
wall time is vendor sparse-direct factorization/solve, and that cost grows
superlinearly. RBS-NG never builds its production path on that exponent.

- **Transitional: faer sparse LU** (`rbs-linalg::FaerSparseLu`). Direct
  solves exist in RBS-NG for exactly two purposes: the oracle parity gates
  at N ≤ 128 (seed solves) and unit tests of the Newton machinery. They
  are parity and scaffolding — **never a scaling path**. Anything that
  bakes in a global direct solve must be labeled transitional.
- **Intermediate: per-field block preconditioners.** The Jacobian's
  6-field structure (each field's operator plus couplings) admits block
  factorizations that are cheap to invert field-by-field and capture most
  of the physics coupling; this is the bridge between the LU scaffolding
  and multigrid, and the analysis target for where multigrid smoothers
  must be strengthened.
- **Production: geometric multigrid + FGMRES/Krylov**
  (`rbs-linalg::MgKrylov`, currently an honest placeholder implementing
  the same `LinearSolver` trait). The hierarchy is geometric over the
  grid ladder (§1), so rung 4's block refinement and the MG coarsening are
  the same machinery; the outer Krylov loop handles the eigenvalue row
  and the near-fold ill-conditioning the C campaign handles with
  stale-factor CGS reuse.

**Explicit principle: never rely on global sparse LU scaling.** The
`LinearSolver` trait makes backends swappable precisely so the Newton loop
can be developed against faer at parity sizes and moved to MG-Krylov
without touching assembly or the driver.

## 3. Codegen plan

The symbolic chain is the single source of truth and it already exists:
`tools/sympy_system.py` re-derives the six residuals and the 6×31
Jacobian in SymPy and regenerates the checked-in C kernels
(`tools/generate_kernels.py` → `src/rhs_vars.c`, `src/csr_vars.c`) with a
byte-identical CI gate (`uv run tools/generate_kernels.py --check`).

RBS-NG retargets the emitter, not the mathematics
(`tools/generate_rust_kernels.py`):

- it emits the same system as Rust into
  `ng/rbs-elliptic/src/generated/kernels.rs` — the interior residual
  (physical-derivative form, mirroring `rhs_vars.c`), the Robin/decay
  boundary rows (mirroring `rhs_bdry`), and the step-scaled subtype
  Jacobian (mirroring the `jacobian_submatrix_*` assignments of
  `csr_vars.c`);
- a `--check` gate regenerates and asserts byte-identical output, exactly
  like the C gate;
- the checked-in C kernels remain the **structural cross-reference**: the
  Rust kernels are the same equations term-for-term, and any kernel edit
  flows through `tools/sympy_system.py` so both languages regenerate
  together;
- numerical parity is pinned by a checked-in fixture
  (`ng/rbs-elliptic/tests/fixtures/kernel_parity.txt`): seeded random
  field states with SymPy-evaluated reference values, asserted by
  `cargo test` (a few ULP of evaluation-order noise).

The C generator is untouched — the oracle's chain stays frozen.

## 4. Validation gates

Every gate is parity against the frozen C oracle, and every gate number
below is an already-measured C result that the Rust solve must reproduce:

| gate | what | target |
|---|---|---|
| G1 | kernel parity vs SymPy | exact to evaluation-order noise (checked-in fixture, `cargo test`) |
| G2 | **N=64 seed solve** (l=1) | oracle solution fields to solver tolerance — the SAN-62 vertical slice |
| G3 | **N=128 seed solve** (l=1) | same, at the size where faer direct is still parity-honest |
| G4 | **N=256 l=1 down-campaign fold** | fold vertex ω_min = 0.6460482 (paper 0.64561, the +0.068% dr=0.0625 C result — `RUNBOOK.md` §c) |
| G5 | **stretch N=400 fold** | ω_min = 0.6456114 on the stretched radial rung — the first gate that must beat the uniform box |

Gates are cumulative: nothing above G2 may proceed while a lower gate
fails, and the fold gates double as the accuracy contract for the grid
ladder rungs. The stretch gate (G5) is where RBS-NG first exceeds the
oracle's reach rather than matching it.

## 5. Evolution path

Staged per ADR 0001 (3D/AMR gated at the far end):

1. **1D spherical full-GR Einstein–Klein–Gordon** — the formulation study
   first (`rbs-evolve::Formulation`): BSSN vs CCZ4 vs generalized-harmonic
   in 1D, evaluated on constraint preservation, collapse robustness, and
   how naturally each fits the multigrid + Krylov solver stack (the same
   backend the elliptic side already needs).
2. **Stationary-hold validation** — evolve the elliptic initial data and
   require the stationary solution to hold: drift bounded by solver
   tolerance over many dynamical times. This is the evolution track's
   parity gate, and it needs no C reference — the oracle for it *is* the
   elliptic solve.
3. **Collapse thresholds** — the scalar-field amplitude ladder in 1D:
   where does the stationary branch end in time? The thresholds are the
   physics deliverable of the evolution track.
4. **Axisymmetric evolution** — the (r, z) system evolved with the chosen
   formulation; the grid ladder and solver stack carry over. 3D/AMR stays
   gated behind all of the above.

## 6. Visualization

First-class from day one (ADR 0001 decision 6):

- **VTR/VTH export** (`rbs-io::VtrWriter`, real): ParaView-native
  rectilinear-grid XML for every field snapshot, plus VTH series
  collection for campaign trajectories (the `rbs-viz` pipeline consumes
  these).
- **HDF5 export** (`rbs-io::Hdf5Writer`, feature `hdf5`): mirrors the C
  writer's self-describing contract — `<field>.asc` datasets with
  `(nr, nz)` C-order layout, `r`/`z` meshgrids, and the full root
  attribute set — so `tools/rotboson_io.py` reads Rust output
  **unchanged**; the format contract is pinned on the C side by the
  fixture roundtrip test and on the Rust side by the SAN-55 parity
  fixture (`ng/rbs-io/tests/fixtures/l1_smoke_n64_rust.h5`).
- **Animation pipeline** (`rbs-viz`, placeholder): field movies along
  continuation campaigns, fold-approach visualizations, and
  collapse-threshold sequences — the campaign driver writes VTRs as it
  steps, so the animation is a post-process, never a solver-time cost.

## The workspace map

| crate | role | status |
|---|---|---|
| `rbs-core` | grids, fields, stencils, index maps (C-convention ports) | real, tested |
| `rbs-elliptic` | residual/Jacobian assembly, Newton, ω constraint | kernels generated (SAN-56); assembly/Newton are the SAN-62 port target |
| `rbs-linalg` | `LinearSolver` trait + faer LU + MG-Krylov placeholder | real, tested |
| `rbs-io` | HDF5 + VTR writers behind `FieldWriter` | real, parity-pinned (HDF5 C-contract mirror, feature-gated; VTR VTK-order-correct) |
| `rbs-driver` | continuation campaigns | parameter skeleton; port target |
| `rbs-evolve` | 1D-first evolution | formulation roster only |
| `rbs-viz` | animation/streaming pipeline | placeholder |

See each crate's `README.md` for its role and parity target, and
`ng/README.md` for build, test, and codegen instructions.
