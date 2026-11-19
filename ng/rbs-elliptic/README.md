# rbs-elliptic

The elliptic initial-data stage: residual/Jacobian assembly, the global
Newton loop, and the omega constraint.

**Role:** evaluates the generated kernels over the grid
([`rbs_elliptic::generated`]) into the global residual vector and the CSR
Jacobian, and drives Newton with the frequency eigenvalue folded in.

**Parity target:** the C oracle's elliptic solve — same residual vector
(`src/rhs.c` calling `src/rhs_vars.c` + `rhs_bdry`), same Jacobian
(`src/csr_gen_jacobian` calling `src/csr_vars.c`), same Newton semantics
(`src/newton.c`, `src/qnerr.c`), same omega constraint
(`src/omega_calc.c`, `src/csr_omega_constraint.c`).

**Gates** (`ng/docs/architecture.md` §4): oracle parity at N=64 and N=128
seed solves first (SAN-62: l=1, N=64, pure Rust vs the oracle solution).
