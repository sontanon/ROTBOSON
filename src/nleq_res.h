#include "context.h"

MKL_INT nleq_res(
		      rb_context 	*ctx,			// INPUT: Runtime context.
		      MKL_INT 	*err_code,		// OUTPUT: Pointer to integer containing error code.
		      double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
		      					//            First entry contains initial guess.
		      double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
		      					//            First entry contains initial RHS.
		      double	*lambda,		// IN-OUTPUT: Pointer to array of damping factors.
		      					//            First entry contains initial damping factor.
		      double	**du,			// OUTPUT: Pointer to array of updates.
		      double	*norm_f,		// IN-OUTPUT: Pointer to array of RHS norms.
		      					//            First entry contains initial RHS norm.
		      double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
		      double	*mu,			// OUTPUT: Pointer to array of mu's.
		      double	*lambda_prime,		// OUTPUT: Pointer to array of lambda_prime's.
		      double	*mu_prime,		// OUTPUT: Pointer to array of mu_primes's.
		csr_matrix 	*J,			// INPUT: Pointer to jacobian matrix type.
		const double 	epsilon,		// INPUT: Exit tolerance.
		const MKL_INT	max_newton_iterations,	// INPUT: Maximum number of Newton iterations.
		const MKL_INT	max_trial_A_iterations,	// INPUT: Maximum number of trial A iterations.
		const MKL_INT	max_trial_B_iterations,	// INPUT: Maximum number of trial A iterations.
		const double 	lambda_min,		// INPUT: Minimum damping factor.
		const MKL_INT	qnres,			// INPUT: Boolean to indicate whether to use QNERR.
		      rb_rhs_fn	RHS_CALC,		// INPUT: RHS calculation subroutine.
		      rb_jacobian_fn	JACOBIAN_CALC,		// INPUT: Jacobian calculation subroutine.
		      rb_norm_fn	NORM,			// INPUT: Norm calculation subroutine.
		      rb_dot_fn	DOT,			// INPUT: Dot product calculation subroutine.
		      rb_linear_solve_fn LINEAR_SOLVE_1,	// INPUT: Linear solver subroutine.
		      rb_linear_solve_fn LINEAR_SOLVE_2		// INPUT: Linear solver subroutine.
	);
