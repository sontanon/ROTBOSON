# Architecture of the Rust initial-data solver

The `ng/` workspace is a pure-Rust solver for **rotating boson star initial data**. The scope is set by ADR 0002 (`docs/adr/0002-initial-data-scope.md`). The milestones are in `docs/plan/initial-data.md`. The C code in `src/` is frozen and serves as the oracle: Rust results are compared against it until Rust surpasses it.

This document describes what exists today and where the planned pieces will go. Planned items are marked **(planned)**.

## 1. The problem

Axisymmetric, stationary, rotating boson stars in quasi-isotropic coordinates (ρ, z), with azimuthal number l. There are six unknown fields on the grid:

| Index | Field | Meaning |
|---|---|---|
| 0 | `log_alpha` | log of the lapse α |
| 1 | `beta` | the shift (frame dragging) |
| 2 | `log_h` | log of the metric function h |
| 3 | `log_a` | log of the metric function a |
| 4 | `psi` | the regularized scalar field; the physical field is φ = ρ^l ψ |
| 5 | `lambda` | the regularization field |

There is also the scalar-field frequency ω, which is an eigenvalue. The solver works with ξ, where ω = ½ m (1 + tanh ξ), so that ω stays in (0, m). The equations come from `tools/sympy_system.py`.

All unknowns form one vector: the six fields field-major (field g occupies `[g·dim, (g+1)·dim)`, dim = NrTotal·NzTotal, ghost cells included), then ξ in the last slot. Newton solves

```
J = [ A    b  ]     A = ∂F/∂u (the fixed-ω Jacobian),  b = ∂F/∂ξ
    [ cᵀ   c_w ]     last row: the constraint row
```

The constraint row selects the mode:

- **Fixed-ω** (`ProblemParams::fixed_phi = None`): ξ is given.
- **Fixed-φ** (`fixed_phi = Some((i, j))`): ψ at one grid node is given, and ω is solved for. Continuation near the fold uses this mode, because there the fixed-ω A is singular and the fixed-φ J is not.

`docs/notes/bordered-omega-system.md` derives all of this and gives the numerical evidence.

## 2. Crates

| Crate | Role |
|---|---|
| `rbs-core` | The uniform (ρ, z) grid with ghost cells, the field-major `FieldSet`, and the finite-difference stencils (orders 2, 4, 6), with the C code's index and coordinate conventions. |
| `rbs-elliptic` | Residual and Jacobian assembly from the generated kernels, the initial guess, the ω map, and the Newton drivers. |
| `rbs-linalg` | The `LinearSolver` trait, the CSR matrix type, and the backends: `BlockFieldKrylov` (production) and `FaerSparseLu` (cross-check). |
| `rbs-io` | Output: VTR for ParaView, and HDF5 (feature `hdf5`) in the C code's format, readable by `tools/rotboson_io.py`. |
| `rbs-driver` | The solver-independent half of the continuation driver, ported from `tools/sweep_driver.py`: the campaign spec, `state.json`, step decisions and the fold fit. Driving real solves is A2 **(planned)**. |
| `rbs-viz` | Placeholder for a visualization pipeline. Today plots come from `tools/plot_star.py` and `tools/plot_fold_curve.py`. |

Spikes live under `ng/spikes/`. Each is a standalone crate outside the workspace:

- `faer-lu` decided the backend;
- `nearfold` tested it near the fold (A0.5).

## 3. The solver stack

```
Newton (rbs-elliptic::newton)
   │  NLEQ-ERR (+QNERR), NLEQ-RES (+QNRES), or plain damped Newton, ported from src/
   ▼
LinearSolver trait (rbs-linalg)
   ├── BlockFieldKrylov: per-field sparse LU blocks (faer) as preconditioner + FGMRES
   └── FaerSparseLu:     full sparse LU, for parity tests at N ≤ 128 only
```

**BlockFieldKrylov** row-equilibrates each field's diagonal block of the Jacobian and factors it with faer (sparse LU, or LDLT when the block is symmetric). These factors precondition a restarted FGMRES, which handles the couplings between fields and the ξ row. `solve_stale` reuses the factors across Newton steps and continuation steps. The `Preconditioner` option selects how the blocks are combined:

| Variant | What | Use |
|---|---|---|
| `BorderedBlockJacobi` | The pinned field's block (ψ) is factored together with the ξ column and the constraint row, as one (dim+1) block. | **Fixed-φ solves: always.** The A0.5 verdict. |
| `BlockJacobi` | Plain block diagonal; identity on ξ. | Currently the default. It fails at converged states, because the ψ block is singular at every solution. |
| `BlockGaussSeidel` | Forward substitution over the lower block couplings. | Fails for the same reason. Kept for comparison. |

Measured at N=256 on 4 threads, at the initial guess (`ng/spikes/faer-lu/REPORT-BACKEND.md`):

| N=256, 4 threads | C (PARDISO) | BlockFieldKrylov |
|---|---|---|
| cold solve (factor + 1 solve) | 92 s | 7.7–7.9 s (5.3 s factor + 2.4 s Krylov, 12 iterations) |
| campaign step (stale preconditioner) | ~69 s | 3.0–3.3 s (16–17 iterations) |
| peak memory | 4.95 GB | 1.88 GiB |
| full cold-start Newton solve | ≈ 14 min | ≈ 29 s |

Near the fold (A0.5, `docs/results/A0.5-near-fold.md`), with the bordered preconditioner: 21–25 iterations at N=128 and 33–35 at N=256, flat through the fold. A Newton solve at the state closest to the fold converges and matches C (ω to 6e-12 at N=256).

**Rule:** the production path never depends on a global sparse LU. faer LU is a cross-check for small problems.

## 4. Code generation

`tools/sympy_system.py` is the single source of truth for the equations. Two generators read it:

- `tools/generate_kernels.py` writes the C kernels in `src/`. This generator is frozen together with the C code.
- `tools/generate_rust_kernels.py` writes `ng/rbs-elliptic/src/generated/kernels.rs`: the interior residual, the boundary rows and the step-scaled Jacobian, term for term the same as the C.

Both generators have a `--check` mode, which CI runs: it regenerates the file and fails unless the result is byte-identical. The fixture `ng/rbs-elliptic/tests/fixtures/kernel_parity.txt` holds SymPy-evaluated reference values, so `cargo test` checks the kernels numerically without Python. Never edit `generated/` by hand: change the SymPy source and regenerate.

## 5. Validation

The tests that exist today:

| Test | What it pins |
|---|---|
| `rbs-elliptic` `kernel_parity` (unit) | Generated kernels against SymPy values, to evaluation-order noise. |
| `rbs-elliptic/tests/l1_smoke_n64_parity.rs` | The l=1, N=64 fixed-ω solve against the C oracle's fields. |
| `rbs-elliptic/tests/fixedphi_jacobian_parity.rs` | The full fixed-φ CSR Jacobian, including the ω column, against a dump from the C code. |
| `rbs-elliptic/tests/fixedphi_n64_parity.rs` | The fixed-φ N=64 state: ω, the pin and the fields. |
| `rbs-elliptic/tests/newton_machinery_parity.rs`, `newton_fault_injection.rs` | The Newton drivers' iteration histories and their failure paths. |
| `rbs-linalg` unit tests | Every preconditioner in both modes, stale blocks, and a singular field block. |
| `rbs-io/tests/hdf5_parity.rs`, `vtr_selfcheck.rs` | The HDF5 file contract and the VTR layout. |
| `rbs-driver/tests/*` | The ported Python driver tests (decisions, `state.json` round trip, summarize). |

Results by milestone (in `docs/results/`):

- **A0:** an l=1 star solved from scratch in Rust agrees with C to 1.6e-12 in the fields and 8e-13 in ω (`A0-demo.md`).
- **A0.5:** the near-fold behavior described above (`A0.5-near-fold.md`).
- **Later milestones** each add a result: A1 C parity at N=64 and N=128 through the CLI; A2 the l=1 fold at N=256 against the C campaign; A4 ω_min for l=1..6 with error bars.

## 6. Planned (A1 and later)

None of these exist yet.

- **A1: the `rbs solve` CLI.** It reads a TOML config and writes `solution.h5`, with a versioned initial-data format v1 that a future evolution code will read.
- **A1: a precision parameter.** fp64 residuals always; fp32 optional for the preconditioner factors.
- **A1: stencil order as a parameter.** A grid/stencil abstraction that admits mapped (non-uniform) coordinates. It will live in `rbs-core` and be used by `rbs-elliptic`'s assembly.
- **A1: make `BorderedBlockJacobi` the default** for fixed-φ. Add ψ deflation for fixed-ω.
- **A2:** the solver-facing half of `rbs-driver`, which replaces `tools/sweep_driver.py`.
- **A3:** rayon parallelism over blocks and assembly.
- **A5:** compactification, and stencil orders 6 and 8, on the mapped-grid abstraction.

## Parked

These are not part of this repository's plan (ADR 0002):

- multigrid (only if grids beyond N=512 are needed, as a per-block replacement for block LU);
- time evolution and mesh refinement (a separate future project that reads the A1 file format);
- spectral methods.

## History

This workspace began as "RBS-NG" under ADR 0001 (`docs/adr/0001-pivot-to-rbs-ng.md`), tracked in the old Linear team SAN. That plan had a grid ladder, multigrid as the production path, and an evolution track. ADR 0002 replaced it on 2026-11-27. The backend measurements above come from SAN-67. The spike reports under `ng/spikes/faer-lu/` are records of that period and keep their original wording.
