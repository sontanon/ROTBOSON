# rbs-linalg

Sparse linear solver backends behind one trait.

**Role:** the [`LinearSolver`] contract the Newton loop codes against, plus
the checked-in backends: `FaerSparseLu` (faer sparse LU) and the
MG-Krylov placeholder for the production path. The matrix contract type
[`CsrMatrix`] mirrors the C oracle's `csr_matrix` (`src/tools.h`: ia/ja/a,
0-based here).

**Parity target:** PARDISO at N <= 128 — same matrix in, same solution out
toLU working precision. Per ADR 0001 this backend is *transitional*: it
exists for the parity gates and unit tests, never for production scaling.
The production path is geometric multigrid + FGMRES (`mg_krylov` module).

**Dependency choice:** we ship our own minimal CSR type rather than `sprs`
(or similar) so the port contract stays dependency-free and structurally
identical to the C 3-array layout; converting to faer's CSC happens inside
the faer backend only.
