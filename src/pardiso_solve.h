void pardiso_simple_solve(	double *u,		// Solution array.
				csr_matrix *A, 		// Matrix system to solve: Au = f.
				double *f);		// RHS array.

void pardiso_solve_low_rank(	double *u,	// Solution array.
			csr_matrix *A, 		// Matrix system to solve: Au = f.
			double *f);		// RHS array.

void pardiso_repeated_solve(	double *u,		// Solution array.
				csr_matrix *A, 		// Matrix system to solve: Au = f.
				double *f);		// RHS array.
