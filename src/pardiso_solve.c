#include "tools.h"
#include "pardiso_param.h"
#include "pardiso.h"

void pardiso_simple_solve(	double *u	,	// Solution array.
				csr_matrix *A	, 	// Matrix system to solve: Au = f.
				double *f	)	// RHS array.
{
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
			for (MKL_INT k = 0; k < 64; ++k)
				printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
			exit(1);
		}

		printf("PARDISO MEMORY DIAGNOSTICS\n");
		printf("iparm[14] : % 9lld kB : Peak memory on symbolic factorization.\n", iparm[14]);
		printf("iparm[15] : % 9lld kB : Permament memory on symbolic factorization.\n", iparm[15]);
		printf("iparm[16] : % 9lld kB : Peak memory on numerical factorization and solution.\n", iparm[16]);
		printf("iparm[62] : % 9lld kB : Minimum OOC memory for numerical factorization and solution.\n", iparm[62]);
		printf("Total peak memory consumption for IC  : %09lld kB : MAX(iparm[14], iparm[15] + iparm[16]).\n", MAX(iparm[14], iparm[15] + iparm[16]));
		printf("Total peak memory consumption for OOC : %09lld kB : MAX(iparm[14], iparm[15] + iparm[62]).\n", MAX(iparm[14], iparm[15] + iparm[62]));
		printf("\n");

		printf("PARDISO FACTOR DIAGNOSTICS\n");
		printf("iparm[17] : % 9lld : Number of nonzeros in factors.\n", iparm[18 - 1]);
		printf("iparm[18] : % 9lld : Number of factorization MFLOPS.\n", iparm[19 - 1]);
		printf("\n");

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
		for (MKL_INT k = 0; k < 64; ++k)
			printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
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
		for (MKL_INT k = 0; k < 64; ++k)
			printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
		exit(3);
	}

	// Return.
	return;
}

void pardiso_solve_low_rank(	double *u	,	// Solution array.
				csr_matrix *A	, 	// Matrix system to solve: Au = f.
				double *f	)	// RHS array.
{
	// If using low-rank, calls are different.
	// Notice in particular that diff is used instead of perm array.
	if (!(A->analysis_phase))
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
			for (MKL_INT k = 0; k < 64; ++k)
				printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
			exit(1);
		}

		printf("PARDISO MEMORY DIAGNOSTICS\n");
		printf("iparm[14] : % 9lld kB : Peak memory on symbolic factorization.\n", iparm[14]);
		printf("iparm[15] : % 9lld kB : Permament memory on symbolic factorization.\n", iparm[15]);
		printf("iparm[16] : % 9lld kB : Peak memory on numerical factorization and solution.\n", iparm[16]);
		printf("iparm[62] : % 9lld kB : Minimum OOC memory for numerical factorization and solution.\n", iparm[62]);
		printf("Total peak memory consumption for IC  : %09lld kB : MAX(iparm[14], iparm[15] + iparm[16]).\n", MAX(iparm[14], iparm[15] + iparm[16]));
		printf("Total peak memory consumption for OOC : %09lld kB : MAX(iparm[14], iparm[15] + iparm[62]).\n", MAX(iparm[14], iparm[15] + iparm[62]));
		printf("\n");

		printf("PARDISO FACTOR DIAGNOSTICS\n");
		printf("iparm[17] : % 9lld : Number of nonzeros in factors.\n", iparm[18 - 1]);
		printf("iparm[18] : % 9lld : Number of factorization MFLOPS.\n", iparm[19 - 1]);
		printf("\n");

		// Now that we have done the analysis phase, set the matrix handle.
		A->analysis_phase = 1;
	}

	// Numerical factorization.
	phase = 22;
	pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
		&n, A->a, A->ia, A->ja, diff, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error);

	if (error != 0)
	{
		printf("ERROR during numerical factorization: %lld.\n", error);
		for (MKL_INT k = 0; k < 64; ++k)
			printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
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
		for (MKL_INT k = 0; k < 64; ++k)
			printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
		exit(3);
	}

	// Turn Low-Rank for future updates.
	iparm[39 - 1] = 1;

	// Return.
	return;
}

void pardiso_repeated_solve(	double *u	,	// Solution array.
				csr_matrix *A	, 	// Matrix system to solve: Au = f.
				double *f	)	// RHS array.
{
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
		for (MKL_INT k = 0; k < 64; ++k)
			printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
		exit(3);
	}

	// Return.
	return;
}

