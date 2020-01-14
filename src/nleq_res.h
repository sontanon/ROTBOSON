MKL_INT nleq_res(	      MKL_INT 	*err_code,		// OUTPUT: Pointer to integer containing error code.
		      double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
		      					//            First entry contains initial guess.
		      double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
		      					//            First entry contains initial RHS.
		      double	*lambda,		// IN-OUTPUT: Pointer to array of damping factors.
		      					//            First entry contains initial damping factor.
		      double	**du,			// OUTPUT: Pointer to array of updates.
		      double	*norm_f,		// OUTPUT: Pointer to array of RHS norms.
		      double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
		      double	*mu,			// OUTPUT: Pointer to array of mu's.
		      double	*lambda_prime,		// OUTPUT: Pointer to array of lambda_prime's.
		      double	*mu_prime,		// OUTPUT: Pointer to array of mu_primes's.
		csr_matrix 	*J,			// INPUT: Pointer to jacobian matrix type.
		const double 	epsilon,		// INPUT: Exit tolerance.
		const MKL_INT	max_newton_iterations,	// INPUT: Maximum number of Newton iterations.	
		const double 	lambda_min,		// INPUT: Minimum damping factor.
		const MKL_INT	qnres,			// INPUT: Boolean to indicate whether to use QNERR.
		      void	(*RHS_CALC)(double *, const double *),		// INPUT: RHS calculation subroutine.
		      void	(*JACOBIAN_CALC)(csr_matrix, const double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		      double	(*NORM)(const double *)	,			// INPUT: Norm calculation subroutine.
		      double	(*DOT)(const double *, const double *)	,	// INPUT: Dot product calculation subroutine.
		      void 	(*LINEAR_SOLVE)(double *, csr_matrix *, double *)	// INPUT: Linear solver subroutine.
	);
