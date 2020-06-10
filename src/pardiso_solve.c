#include "tools.h"
#include "pardiso_param.h"
#include "pardiso.h"

// Define for matrix, vector checks.
#undef DEBUG

void pardiso_simple_solve(
		double *u,			// Solution array.
		csr_matrix *A, 			// Matrix system to solve: Au = f.
		double *f			// RHS array.
		)
{
	// Debugging and recheck procedures.
#ifdef DEBUG
	// Check matrix for errors.
	printf("PARDISO: Will check matrix for errors...\n");
	iparm[27 - 1] = 1;

	sparse_struct handle;

	handle.n = n;
	handle.csr_ia = A->ia;
	handle.csr_ja = A->ja;
	handle.indexing = MKL_ONE_BASED;
	handle.matrix_format = MKL_CSR;
	handle.message_level = MKL_PRINT;
	handle.print_style = MKL_C_STYLE;

	sparse_matrix_checker_init(&handle);

	int mcheck_err;
	printf("Running MKL Matrix Check...\n");
	mcheck_err = sparse_matrix_checker(&handle);
	printf("MKL Matrix Check %d.\n", mcheck_err);
	if (mcheck_err == MKL_SPARSE_CHECKER_SUCCESS)
	{
		printf("MKL Matrix Check Successfull.\n");
	}
	else
	{
		printf("MKL Matrix Critical Error!\n");
	}
#endif

	// Reordering and symbolic factorization.
	if (!(A->analysis_phase))
	{
		phase = 11;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, perm, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);

		if (error != 0)
		{
			printf("ERROR during symbolic factorization: %lld.\n", error);
			exit(1);
		}

#ifdef VERBOSE
	printf("PARDISO: Reordering completed.\n");
	printf("PARDISO: Number of nonzeros in factors = %lld.\n", iparm[18 - 1]);
	printf("PARDISO: Number of factorization MFLOPS = %lld.\n", iparm[19 - 1]);
#endif

		// Analysis phase has been completed.
		A->analysis_phase = 1;

		// Enable CG preconditioning.
		iparm[3] = 61;
	}

	// Numerical factorization.
	phase = 22;
	pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
		&n, A->a, A->ia, A->ja, perm, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error);

	if (error != 0)
	{
		printf("ERROR during numerical factorization: %lld.\n", error);
		exit(2);
	}

#ifdef VERBOSE
	printf("PARDISO: Factorization completed.\n");
#endif

	// Back substitution and iterative refinement.
	phase = 33;
	pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
		&n, A->a, A->ia, A->ja, perm, &nrhs,
		iparm, &msglvl, f, u, &error);

	if (error != 0)
	{
		printf("ERROR during solution: %lld,\n", error);
		printf("iparm(30) = %lld\n", iparm[30 - 1]);
		exit(3);
	}


	// Return.
	return;
}

void pardiso_solve_low_rank(
		double *u,			// Solution array.
		csr_matrix *A, 			// Matrix system to solve: Au = f.
		double *f			// RHS array.
		)
{
	// Debugging and recheck procedures.
#ifdef DEBUG
	// Check matrix for errors.
	printf("PARDISO: Will check matrix for errors...\n");
	iparm[27 - 1] = 1;
#endif

	// If using low-rank, calls are different.
	// Notice in particular that diff is used instead of perm array.
	if (A->analysis_phase)
	{
#ifdef VERBOSE
		printf("PARDISO: Using Low Rank update to skip analysis phase.\n");
		printf("PARDISO: ndiff = %lld.\n", diff[0]);
#endif
		// Numerical factorization.
		phase = 22;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, diff, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);

		if (error != 0)
		{
			printf("ERROR during numerical factorization: %lld.\n", error);
			exit(2);
		}

#ifdef VERBOSE
		printf("PARDISO: Factorization completed.\n");
#endif

		// Back substitution and iterative refinement.
		phase = 33;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, diff, &nrhs,
			iparm, &msglvl, f, u, &error);

		if (error != 0)
		{
			printf("ERROR during solution: %lld,\n", error);
			exit(3);
		}

	}
	// Complete phases if not using low-rank.
	else
	{
		// Modify parameters according to Low-Rank update.
		iparm[24 - 1] = 10;
		// No permutation.
		iparm[5 - 1] = 0;
		// No CGS.
		iparm[4 - 1] = 0;
		// Additional values.
		iparm[ 6 - 1] = 0;
		iparm[12 - 1] = 0;
		iparm[28 - 1] = 0;
		iparm[31 - 1] = 0;
		iparm[36 - 1] = 0;
		iparm[37 - 1] = 0;
		iparm[56 - 1] = 0;
		// Ensure we are using IC.
		iparm[60 - 1] = 0;

		// Reordering and symbolic factorization.
		phase = 11;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, perm, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);

		if (error != 0)
		{
			printf("ERROR during symbolic factorization: %lld.\n", error);
			exit(1);
		}

#ifdef VERBOSE
		printf("PARDISO: Reordering completed.\n");
		printf("PARDISO: Number of nonzeros in factors = %lld.\n", iparm[18 - 1]);
		printf("PARDISO: Number of factorization MFLOPS = %lld.\n", iparm[19 - 1]);
#endif

		// Now that we have done the analysis phase, set the matrix handle.
		A->analysis_phase = 1;

		// Numerical factorization.
		phase = 22;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, perm, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);

		if (error != 0)
		{
			printf("ERROR during numerical factorization: %lld.\n", error);
			exit(2);
		}

#ifdef VERBOSE
		printf("PARDISO: Factorization completed.\n");
#endif

		// Back substitution and iterative refinement.
		phase = 33;
		pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
			&n, A->a, A->ia, A->ja, perm, &nrhs,
			iparm, &msglvl, f, u, &error);

		if (error != 0)
		{
			printf("ERROR during solution: %lld,\n", error);
			exit(3);
		}

		// Turn Low-Rank for future updates.
		iparm[39 - 1] = 1;
	}

	// Return.
	return;
}

void pardiso_repeated_solve(
		double *u,			// Solution array.
		csr_matrix *A, 		// Matrix system to solve: Au = f.
		double *f			// RHS array.
		)
{
	// Debugging and recheck procedures.
#ifdef DEBUG
	// Check matrix for errors.
	printf("PARDISO: Will check matrix for errors...\n");
	iparm[27 - 1] = 1;
#endif

	/*
	// Numerical factorization.
	phase = 22;
	pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
		&n, A->a, A->ia, A->ja, perm, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error);

	if (error != 0)
	{
		printf("ERROR during numerical factorization: %lld.\n", error);
		exit(2);
	}

#ifdef VERBOSE
	printf("PARDISO: Factorization completed.\n");
#endif
	*/

	// Back substitution and iterative refinement.
	phase = 33;
	pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
		&n, A->a, A->ia, A->ja, perm, &nrhs,
		iparm, &msglvl, f, u, &error);

	if (error != 0)
	{
		printf("ERROR during solution: %lld,\n", error);
		exit(3);
	}

	// Return.
	return;
}

