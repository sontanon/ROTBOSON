#include "tools.h"

// Error codes.
#define ERROR_CODE_SUCCESS 				  0
#define ERROR_CODE_QNERR_THETA_INCREASE_EXIT 		- 2
#define ERROR_CODE_EXCEEDED_MAX_ITERATIONS 		- 3

#define NORMALIZED

#define MIN_ITERATIONS 0

MKL_INT nleq_err_qnerr(	
	        MKL_INT	*err_code,		// OUTPUT: Pointer to integer containing error code.
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
	      	void	(*RHS_CALC)(double *, double *),				// INPUT: RHS calculation subroutine.
	      	void	(*JACOBIAN_CALC)(csr_matrix, double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
	      	double	(*NORM)(double *)	,					// INPUT: Norm calculation subroutine.
	      	double	(*DOT)(double *, double *)	,			// INPUT: Dot product calculation subroutine.
	      	void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),		// INPUT: Linear solver subroutine.
	      	void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)		// INPUT: Linear solver subroutine.
	 )
{
	/* Iteration counters. */
	MKL_INT l = 0, i = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	/* Auxiliary double. */
	double alpha_bar = 0.0;

	/* Solution norm for relative error. */
	double norm_u = 1.0;

	/* Calculate Jacobian J(u^0). */
	JACOBIAN_CALC(*J, u[l], 0);

	// Solve  linear system J(u^0) du^0 = -f(u^0).
	LINEAR_SOLVE_1(du[l], J, f[l]);

	// Calculate ||du^0||.
	norm_du[l] = NORM(du[l]);

	// 1. Step l.
	for (l = 0; l < max_newton_iterations; ++l)
	{
		/* Print table header every 50 iterations. */
		if (l % 50 == 0)
		{
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** | QNERR ITER | ||du[l+1]||  | alpha_bar    | Theta[l]     | alpha[l+1]   | STATUS      |\n"
		"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");
		}

		// New iterate u^{l+1} = u^l + du^l.
		ARRAY_SUM(u[l + 1], 1.0, u[l], 1.0, du[l]);

		// Evaluation f(u^{l+1}).
		RHS_CALC(f[l + 1], u[l + 1]);

		// Linear system solve J(u^0) du_bar^{l+1} = -f(u^{l+1}).
		LINEAR_SOLVE_2(du_bar[l + 1], J, f[l + 1]);

		// 2. If l > 0. For i = 1, ... , l:
		if (l > 0)
		{
			for (i = 1; i < l + 1; ++i)
			{
				// alpha_bar = (du_bar^{l+1} . du^{i-1}) / ||du^{i-1}||^2.
				alpha_bar = DOT(du_bar[l + 1], du[i - 1]) / (norm_du[i - 1] * norm_du[i - 1]);
				// du_bar^{l+1} += alpha_bar * du^i.
				ARRAY_SUM(du_bar[l + 1], 1.0, du_bar[l + 1], alpha_bar, du[i]);
			}
		}

		// 3. Compute.
		// alpha_{l+1} = (du_bar^{l+1} . du^l) / ||du^l||^2.
		alpha[l + 1] = DOT(du_bar[l + 1], du[l]) / (norm_du[l] * norm_du[l]);

		norm_du_bar[l + 1] = NORM(du_bar[l + 1]);

		// Theta_l = ||du_bar^{l+1}|| / ||du^l||.
		Theta[l] = norm_du_bar[l + 1] / norm_du[l];

		// If Theta_l > 1/2: stop, no convergence.
		if (Theta[l] > 0.5 && (l + 1) > MIN_ITERATIONS)
		{

			/* Print message. */
	printf(	"***** | %-10lld |              | % 11.5E | % -9.5E | % 11.5E | %-11s |\n", l, alpha_bar, Theta[l], alpha[l + 1], "EXIT QNERR");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n");

			/* Error code -2: Theta increases beyond 0.5. */
			*err_code = ERROR_CODE_QNERR_THETA_INCREASE_EXIT;

			/* Return negative index: last update at u[l + 1]. */
			return -(l + 1);
		}

		// 4. Compute.
		// du^{l+1} = du_bar^{l+1} / (1 - alpha_{l+1}).
		ARRAY_SUM(du[l + 1], 1.0 / (1.0 - alpha[l + 1]), du_bar[l + 1], 0.0, du[l + 1]);
		norm_du[l + 1] = NORM(du[l + 1]);
		norm_u = NORM(u[l + 1]);

		// Convergence test: If ||du^{l+1}|| < epsilon: stop. Solution found u* = u^{l+1} + du^{l+1}.
#ifdef NORMALIZED
		if (norm_du[l + 1] < epsilon * norm_u)
#else
		if (norm_du[l + 1] < epsilon)
#endif
		{
			/* Update solution */
			ARRAY_SUM(u[l + 1], 1.0, u[l + 1], 1.0, du[l + 1]);
			
			/* Print message */
	printf(	"***** | %-10lld | % -9.5E | % 11.5E | % -9.5E | % 11.5E | %-11s |\n", l, norm_du[l + 1] / norm_u, alpha_bar, Theta[l], alpha[l + 1], "CONVERGED C");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n");

			/* No error code. */
			*err_code = ERROR_CODE_SUCCESS;
	
			/* Return positive index where solution is stored. */
			return l + 1;
		}
		/* Print message before continuing. */
	printf(	"***** | %-10lld | % -9.5E | % 11.5E | % -9.5E | % 11.5E | %-11s |\n", l, norm_du[l + 1] / norm_u, alpha_bar, Theta[l], alpha[l + 1], "ACCEPT");
	}

	/* If we reach this point we did not converge after the maximum iterations. */
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n");

	/* Error code -3: Reached maximum number of iterations. */
	*err_code = ERROR_CODE_EXCEEDED_MAX_ITERATIONS;

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}
