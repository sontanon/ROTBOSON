// Sparse direct solver backend interface.
//
// The nonlinear solver only depends on these five entry points (plus the
// low-rank setup hook). Two backends implement them:
//   - PARDISO (Intel oneMKL): pardiso_start.c / pardiso_solve.c /
//     pardiso_stop.c / low_rank.c
//   - UMFPACK (SuiteSparse):   umfpack_backend.c
//
// The low-rank update is a PARDISO-only feature; the UMFPACK backend treats
// solver_solve_low_rank as solver_solve and solver_diff_gen as a no-op.
#ifndef ROTBOSON_SOLVER_H
#define ROTBOSON_SOLVER_H

// Assumes tools.h is included first (defines MKL_INT and csr_matrix), matching
// the codebase convention for headers that reference these types.

// Initialize the backend for a square system of dimension n.
void solver_start(const MKL_INT n);

// Release backend memory.
void solver_stop(void);

// Full solve: analyze (once) + factor + solve. Solves A u = f.
void solver_solve(double *u, csr_matrix *A, double *f);

// Solve using a low-rank factorization update (PARDISO only; UMFPACK falls
// back to solver_solve).
void solver_solve_low_rank(double *u, csr_matrix *A, double *f);

// Repeated solve reusing the existing factorization (must follow solver_solve).
void solver_repeated_solve(double *u, csr_matrix *A, double *f);

// Prepare low-rank update bookkeeping (PARDISO only; no-op for UMFPACK).
void solver_diff_gen(void);

#endif /* ROTBOSON_SOLVER_H */
