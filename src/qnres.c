// Include headers.
#include "tools.h"
#include "regularization_coupling.h"

// Error codes.
#define ERROR_CODE_SUCCESS 				  0
#define ERROR_CODE_QNRES_THETA_INCREASE_EXIT 		- 2
#define ERROR_CODE_EXCEEDED_MAX_ITERATIONS 		- 3
#define ERROR_CODE_QNRES_ILL_CONDITIONED		- 4

// Maximum expansion.
#define THETA_MAX 0.5

// Maximum preconditioner.
#define KAPPA_MAX 1.0E+6

// Set a required residual error sufficiently above the machine precision.
// Guess an initial iterate u^0. Evaluate F(u^0) and ||F(u^0)||.
MKL_INT nleq_res_qnres(
	        MKL_INT	*err_code,		// OUTPUT: Pointer to integer containing error code.
		double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
						//            First entry contains initial guess.
		double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
						//            First entry contains initial RHS.
		double 	**du,			// OUTPUT: Pointer to array of updates.
		double	*norm_f,		// OUTPUT: Pointer to array of RHS norms.
		double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
	      	double	*gamma,			// OUTPUT: Pointer to array of gamma's.
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
	// Remeber:
	// Intial RHS is stored in f[0].
	// Inital ||f[0]|| is also stored already in norm_f[0].

	// Iteration counters.
	MKL_INT l = 0, i = 0;

	// Get matrix dimension.
	MKL_INT dim = J->nrows;

	// Auxiliary memory blocks.
	double *v = (double *)SAFE_MALLOC(sizeof(double) * dim);
	double *w = (double *)SAFE_MALLOC(sizeof(double) * dim);

	// Auxiliary doubles.
	double beta = 0.0;
	double z = 0.0;

	// Preconditioner monitor.
	double kappa = 1.0;

	// Unique Jacobian.
	// Calculate Jacobian J(u^0).
	JACOBIAN_CALC(*J, u[l], 0);

	// First linear solve.
	// Solve linear system J(u^0) du^0 = -f(u^0).
	LINEAR_SOLVE_1(du[l], J, f[l]);

#ifdef REGULARIZATION_COUPLING
	coupled_du(du[l], u[l], solver_NrTotal, solver_NzTotal, solver_ghost, solver_dr, REG_MU);
#endif

	// Step l.
	for (l = 0; l < max_newton_iterations; ++l)
	{
		/* Print table header every 50 iterations. */
		if (l % 50 == 0)
		{
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n"
		"***** | QNRES ITER | ||df[l+1]||  | gamma[l]    | Theta[l]     | kappa       | STATUS      |\n"
		"***** |------------|--------------|-------------|--------------|-------------|-------------|\n");
		}

		// New iterate u^{l+1} = u^l + du^l.
		ARRAY_SUM(u[l + 1], 1.0, u[l], 1.0, du[l]);

		// Evaluation f(u^{l+1}).
		RHS_CALC(f[l + 1], u[l + 1]);

		// RHS norm.
		norm_f[l + 1] = NORM(f[l + 1]);

		// Theta.
		Theta[l] = norm_f[l + 1] / norm_f[l];

		// Preconditioner monitor kappa.
		kappa /= (1.0 - 2.0 * Theta[l]);

		// Calculate w = dF[l+1] = F[l+1] - F[l].
		ARRAY_SUM(w, 1.0, f[l + 1], -1.0, f[l]);

		// gamma[l] = <dF[l+1], dF[l+1]> = ||dF[l+1]||**2.
		gamma[l] = DOT(w, w);

		// Convergence test: If ||F(u^{l+1})|| < epsilon: stop. Solution found u* = u^{l+1}.
		if (norm_f[l + 1] < epsilon)
		{
			/* Print message */
	printf(	"***** | %-10lld | %11.5E  |% -11.5E | %9.5E  |% -11.5E | %-11s |\n", l, norm_f[l + 1], gamma[l], Theta[l], kappa, "CONVERGED C");
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n");

			/* No error code. */
			*err_code = ERROR_CODE_SUCCESS;

			/* Clean-up.*/
			SAFE_FREE(v);
			SAFE_FREE(w);
	
			/* Return positive index where solution is stored. */
			return l + 1;
		}

		// Sanity tests.
		// First expansion.
		// If Theta_l > 1/4: stop, no convergence.
		if (Theta[l] > THETA_MAX)
		{
			/* Print message */
	printf(	"***** | %-10lld | %11.5E  |% -11.5E | %9.5E  |% -11.5E | %-11s |\n", l, norm_f[l + 1], gamma[l], Theta[l], kappa, "EXIT QNRES");
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n");

			/* Error code -2: Theta increases beyond 0.25. */
			*err_code = ERROR_CODE_QNRES_THETA_INCREASE_EXIT;

			/* Clean-up. */
			SAFE_FREE(v);
			SAFE_FREE(w);

			/* Return negative index: last update at u[l + 1]. */
			return -(l + 1);
		}

		// Second, preconditioner.
		if (kappa > KAPPA_MAX)
		{
			/* Print message */
	printf(	"***** | %-10lld | %11.5E  |% -11.5E | %9.5E  |% -11.5E | %-11s |\n", l, norm_f[l + 1], gamma[l], Theta[l], kappa, "EXIT QNRES");
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n");

			/* Error code -2: Theta increases beyond 0.25. */
			*err_code = ERROR_CODE_QNRES_ILL_CONDITIONED;

			/* Clean-up. */
			SAFE_FREE(v);
			SAFE_FREE(w);

			/* Return negative index: last update at u[l + 1]. */
			return -(l + 1);
		}

		// If sanity tests are passed, we can calculate RHS.
		// v = (1 - <w,F[l+1]>/gamma[l]) * F[l+1].
		z = DOT(w, f[l + 1]);
		ARRAY_SUM(v, (1.0 - z / gamma[l]), f[l + 1], 0.0, v);

		// Recursive update.
		for (i = l - 1; i >= 0; --i)
		{
			// beta = <dF[i+1], v> / gammma[i].
			beta = (DOT(f[i + 1], v) - DOT(f[i], v)) / gamma[i];
			// v = v - beta * F[i+1].
			ARRAY_SUM(v, 1.0, v, -beta, f[i + 1]);
		}

		// Linear solve.
		// Notice that we are actually solving J(u^0) du^{l+1} = -v.
		// The minus signs work out in the end and there is no need to change v.
		LINEAR_SOLVE_2(du[l + 1], J, v);

#ifdef REGULARIZATION_COUPLING
		coupled_du(du[l + 1], u[l + 1], solver_NrTotal, solver_NzTotal, solver_ghost, solver_dr, REG_MU);
#endif

		// Print message before continuing.
	printf(	"***** | %-10lld | %11.5E  |% -11.5E | %9.5E  |% -11.5E | %-11s |\n", l, norm_f[l + 1], gamma[l], Theta[l], kappa, "ACCEPT");
		continue;
	}

	/* If we reach this point we did not converge after the maximum iterations. */
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n");

	/* Error code -3: Reached maximum number of iterations. */
	*err_code = ERROR_CODE_EXCEEDED_MAX_ITERATIONS;

	/* Clean-up.*/
	SAFE_FREE(v);
	SAFE_FREE(w);

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}