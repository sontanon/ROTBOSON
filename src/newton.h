MKL_INT newton(	      
		      MKL_INT 	*err_code,		// OUTPUT: Pointer to integer containing error code.
		      double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
		      					//            First entry contains initial guess.
		      double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
		      					//            First entry contains initial RHS.
		      double	*lambda,		// IN-OUTPUT: Pointer to array of damping factors.
		      					//            First entry contains initial damping factor.
		      double	**du,			// OUTPUT: Pointer to array of updates.
		      double 	*norm_du,		// OUTPUT: Pointer to array of update norms.
		      double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
		csr_matrix 	*J,			// INPUT: Pointer to jacobian matrix type.
		const double 	epsilon,		// INPUT: Exit tolerance.
		const MKL_INT	max_newton_iterations,	// INPUT: Maximum number of Newton iterations.	
		      void	(*RHS_CALC)(double *, double *),				// INPUT: RHS calculation subroutine.
		      void	(*JACOBIAN_CALC)(csr_matrix, double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		      double	(*NORM)(double *),					// INPUT: Norm calculation subroutine.
		      void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *)		// INPUT: Linear solver subroutine.
	);