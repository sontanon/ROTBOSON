# rbs-elliptic

The elliptic initial-data stage: residual/Jacobian assembly, the global
Newton loop, and the omega constraint.

**Role:** evaluates the generated kernels over the grid
([`rbs_elliptic::generated`]) into the global residual vector and the CSR
Jacobian, and drives Newton with the frequency eigenvalue folded in.

**Parity target:** the C oracle's elliptic solve — same residual vector
(`src/rhs.c` calling `src/rhs_vars.c` + `rhs_bdry`), same Jacobian
(`src/csr_gen_jacobian` calling `src/csr_vars.c`), same Newton machinery
(SAN-52: `src/nleq_err.c` + `src/qnerr.c`, `src/nleq_res.c` +
`src/qnres.c`, and `src/newton.c` behind the `NewtonConfig::kind`
switch), same omega constraint
(`src/omega_calc.c`, `src/csr_omega_constraint.c`).

**Gates** (`ng/docs/architecture.md` §4): oracle parity at N=64 and N=128
seed solves first (SAN-62: l=1, N=64, pure Rust vs the oracle solution).

fixedPhi mode (SAN-58): the omega row pins psi at the fixedPhi grid node
(`ProblemParams::fixed_phi`, raw grid coordinates) and the frequency
becomes the free eigenvalue. Pinned against the C oracle by
`tests/fixedphi_jacobian_parity.rs` (the full CSR vs the C's own
`csr_gen_jacobian` dump — the first gate covering the omega COLUMN, which
the fixedOmega seed solve never exercises) and
`tests/fixedphi_n64_parity.rs` (the N=64 plumbing gate at the oracle's
converged star: omega carried to 1e-15, the pin bit-exact, fields to
1e-14; plus the eigenvalue-walk characterization — the from-scratch walk
the C closes in ~9 NLEQ-ERR steps is blocked on the slice Newton's
residual-RMS line search, a SAN-52 dependency documented in the test).
