# rbs-linalg

**Role.** The linear solvers that Newton calls, behind one trait, `LinearSolver`. The matrix type `CsrMatrix` mirrors the C code's 3-array CSR (`src/tools.h`), 0-based. Conversion to faer's CSC happens inside the backends.

| Backend | Use |
|---|---|
| `BlockFieldKrylov` | **Production.** Each field's diagonal block is row-equilibrated and factored with faer (sparse LU, or LDLT when the block is symmetric). The blocks precondition a restarted FGMRES. `solve_stale` reuses old factors. `Preconditioner` selects block Jacobi, bordered block Jacobi or block Gauss–Seidel. For fixed-φ solves use `BorderedBlockJacobi`: the ψ block is singular at every converged state, and only the bordered variant handles that (`../docs/notes/bordered-omega-system.md`). |
| `FaerSparseLu` | Full sparse LU. A cross-check for parity tests at N ≤ 128. Never a scaling path. |
| `MgKrylov` | Placeholder. Multigrid is parked (ADR 0002). |

Measurements: `../spikes/faer-lu/REPORT-BACKEND.md` (at the initial guess) and `../../docs/results/A0.5-near-fold.md` (near the fold).

**Tests** (`cargo test -p rbs-linalg`):
- every preconditioner solves small bordered systems in both constraint modes, with fresh and with stale blocks;
- the bordered preconditioner solves a system whose field block is exactly singular, where block Jacobi does not;
- block equilibration, strict and lenient convergence modes, and error paths;
- CSR validation and the matrix-vector product;
- the faer LU solve, factor reuse, and singular or invalid input.
