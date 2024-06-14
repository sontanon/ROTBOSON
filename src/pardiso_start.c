// PARDISO global parameters header.
#define PARDISO_MAIN_FILE
#include "pardiso_param.h"

#undef DEBUG

// Initialize PARDISO parameters and memory.
#ifdef FORTRAN
extern "C" void pardiso_start_(const MKL_INT *p_matrix_dim)
{
	MKL_INT matrix_dim = *p_matrix_dim;
#else
void pardiso_start(const MKL_INT matrix_dim)
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
	iparm[1 - 1] = 1;	// Do not use default parameters.
	iparm[2 - 1] = 3;	// Parallel fill-in reordering from METIS.
	iparm[4 - 1] = 0;	// No iterative-direct algorithm.
	iparm[5 - 1] = 0;	// No user fill-in reducing permutation.
	iparm[6 - 1] = 0;	// Do not write solution into RHS.
	iparm[7 - 1] = 0;	// Not in use.
	iparm[8 - 1] = 0;	// Max numbers of iterative refinement steps.
	iparm[9 - 1] = 0;	// Not in use.
	iparm[10 - 1] = 13;	// Perturb the pivot elements with 1E-13.
	iparm[11 - 1] = 1;	// Use nonsymmetric permutation and scaling MPS.
	iparm[12 - 1] = 0;	// No conjugate transposed/transpose solve.
	iparm[13 - 1] = 1;      // Maximum weighted matching algorithm is switched-on (default for non-symmetric).
	iparm[14 - 1] = 0;	// Output: Number of perturbed pivots.
	iparm[15 - 1] = 0;	// Not in use.
	iparm[16 - 1] = 0;	// Not in use.
	iparm[17 - 1] = 0;	// Not in use.
	iparm[18 - 1] = 0;	// No Output: Number of nonzeros in the factor LU.
	iparm[19 - 1] = 0;	// No Output: Mflops for LU factorization.
	iparm[20 - 1] = 0;      // Output: Numbers of CG Iterations.
	iparm[24 - 1] = 1;	// Parallel Numerical Factorization.
	iparm[25 - 1] = 0;	// Parallel Forward/Backward Solve.
	iparm[27 - 1] = 0; 	// Matrix check.
	iparm[60 - 1] = 0;	// OOC.
	maxfct = 1;		// Maximum number of numerical factorizations.
	mnum = 1;		// Which factorization to use.
	msglvl = MESSAGE_LEVEL;	// Print statistical information in file.
	error = 0;		// Initialize error flag.

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
