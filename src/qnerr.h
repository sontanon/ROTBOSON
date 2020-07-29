MKL_INT nleq_err_qnerr(	MKL_INT 	*err_code,		// OUTPUT: Pointer to integer containing error code.
		double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
						//            First entry contains initial guess.
		double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
						//            First entry contains initial RHS.
		double 	**du,			// OUTPUT: Pointer to array of updates.
		double	**du_bar,		// OUTPUT: Pointer to array of updates (bar).
		double	*norm_du,		// OUTPUT: Pointer to array of update norms.
		double	*norm_du_bar,		// OUTPUT: Pointer to array of update (bar) norms.
		double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
	      	double	*alpha,			// OUTPUT: Pointer to array of alpha's.
	csr_matrix	*J,			// INPUT: Pointer to Jacobian matrix type.
	const 	double	epsilon,		// INPUT: Exit tolerance.
	const	MKL_INT	max_newton_iterations,	// INPUT: Maximum number of Newton iterations.
		void	(*RHS_CALC)(double *, double *),		// INPUT: RHS calculation subroutine.
	   	void	(*JACOBIAN_CALC)(csr_matrix, double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		double	(*NORM)(double *)	,			// INPUT: Norm calculation subroutine.
		double	(*DOT)(double *, double *)	,	// INPUT: Dot product calculation subroutine.
		void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),
		void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)	
	 );
