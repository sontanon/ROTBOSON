# rbs-linalg

Sparse linear solver backends behind one trait.

**Role:** the [`LinearSolver`] contract the Newton loop codes against, plus
the checked-in backends: `BlockFieldKrylov` (field-blocked Newton–Krylov,
the SAN-67 N1 backend), `FaerSparseLu` (faer sparse LU — the parity-gate
cross-check, N ≤ 128 only) and the MG-Krylov placeholder for the far
production path. The matrix contract type [`CsrMatrix`] mirrors the C
oracle's `csr_matrix` (`src/tools.h`: ia/ja/a, 0-based here).

**Parity target:** PARDISO at N <= 128 — same matrix in, same solution out
to LU working precision. Per ADR 0001 the full-matrix direct backend is
*transitional*: it exists for the parity gates and unit tests, never for
production scaling. The N1 campaign path is the field-blocked Krylov
backend (block preconditioners + restarted FGMRES, `block_field` module);
the far production path is geometric multigrid + FGMRES (`mg_krylov`
module). Measured evidence and the backend decision:
`ng/spikes/faer-lu/REPORT-BACKEND.md`.

**Dependency choice:** we ship our own minimal CSR type rather than `sprs`
(or similar) so the port contract stays dependency-free and structurally
identical to the C 3-array layout; converting to faer's CSC happens inside
the faer backends only.
