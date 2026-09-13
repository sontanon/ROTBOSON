// PARDISO global parameters header.
#define PARDISO_MAIN_FILE
#include "pardiso_param.h"
#include <string.h>

#undef DEBUG

// Initialize PARDISO parameters and memory.
#ifdef FORTRAN
extern "C" void solver_start_(const MKL_INT *p_matrix_dim)
{
    MKL_INT matrix_dim = *p_matrix_dim;
#else
void solver_start(const MKL_INT matrix_dim)
{
#endif
    // Real unsymmetric matrix.
    mtype = 11;
    // One RHS.
    nrhs = 1;
    // Matrix dimension.
    n = matrix_dim;
    // Auxiliary integer.
    MKL_INT k = 0;

    // Set everything to zero beforehand.
    for (k = 0; k < 64; k++)
    {
        iparm[k] = 0;
        pt[k] = 0;
    }

#ifdef VERBOSE
    printf("PARDISO: Intializing parameters.\n");
#endif

    // Problem fine-tune parameters. Used for perm_use = precond_use = 0.
    // Index minus 1 is done to reference to FORTRAN form in manual.
    iparm[1 - 1] = 1; // Do not use default parameters.
    iparm[2 - 1] = 3; // Parallel fill-in reordering from METIS.
    iparm[4 - 1] = 0; // No iterative-direct algorithm.
    iparm[5 - 1] = 0; // No user fill-in reducing permutation.
    iparm[6 - 1] = 0; // Do not write solution into RHS.
    iparm[7 - 1] = 0; // Not in use.
    iparm[8 - 1] = 2; // Max numbers of iterative refinement steps (direct solve).
    iparm[9 - 1] = 0; // Not in use.
    iparm[10 - 1] = 13; // Perturb the pivot elements with 1E-13.
    iparm[11 - 1] = 1; // Use nonsymmetric permutation and scaling MPS.
    iparm[12 - 1] = 0; // No conjugate transposed/transpose solve.
    iparm[13 - 1] =
        1; // Maximum weighted matching algorithm is switched-on (default for non-symmetric).
    iparm[14 - 1] = 0; // Output: Number of perturbed pivots.
    iparm[15 - 1] = 0; // Not in use.
    iparm[16 - 1] = 0; // Not in use.
    iparm[17 - 1] = 0; // Not in use.
    iparm[18 - 1] = -1; // Output: Number of nonzeros in the factor LU (report if < 0).
    iparm[19 - 1] = -1; // Output: Mflops for LU factorization (report if < 0).
    iparm[20 - 1] = 0; // Output: Numbers of CG Iterations.
    iparm[24 - 1] = 1; // Parallel Numerical Factorization.
    iparm[25 - 1] = 0; // Parallel Forward/Backward Solve.
    iparm[27 - 1] = 0; // Matrix check.
    iparm[60 - 1] = getenv("ROTBOSON_PARDISO_OOC") != NULL ? 2 : 0; // OOC: 2 = keep factors on disk (MKL_PARDISO_OOC_* env vars).
    // Single-precision mode (Stage 1 of the M3 fp32 plan): ROTBOSON_PRECISION=32
    // sets iparm[28]=1. The value is stored in the solver handle and can only
    // be set during phase 11, so it is fixed here, before any solver call.
    // With this mode the arrays a, x, b handed to pardiso_64 must be float
    // (pardiso_solve.c does the conversion; Newton-side stays double).
    if (getenv("ROTBOSON_PRECISION") != NULL &&
        strcmp(getenv("ROTBOSON_PRECISION"), "32") == 0)
    {
        iparm[28 - 1] = 1;
        printf("PARDISO: SINGLE PRECISION MODE (ROTBOSON_PRECISION=32).\n");
    }
    maxfct = 1; // Maximum number of numerical factorizations.
    mnum = 1; // Which factorization to use.
    msglvl = MESSAGE_LEVEL; // Print statistical information in file.
    error = 0; // Initialize error flag.

    // Allocate permutation vector.
    perm = (MKL_INT *)malloc(n * sizeof(MKL_INT));

    // Set Low Rank array pointing towards NULL.
    diff = NULL;

    // Setup matrix-vector multiplication type.
    // Non-transposed, i.e. y = A*x.
    uplo[0] = 'N';

#ifdef VERBOSE
    printf("PARDISO: Setup solver memory and parameters.\n");
#endif

#ifdef DEBUG
    printf("\nPARDISO IPARM and DPARM parameters:\n");
    for (k = 0; k < 64; k++)
    {
        printf("iparm[%lld] = %lld\n", k + 1, iparm[k]);
    }
#endif

    return;
}
