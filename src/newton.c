// Include headers.
#include "tools.h"

// Debug print Jacobian CSR matrix.
#define DEBUG_PRINT 0

// Error codes.
#define ERROR_CODE_SUCCESS 				  0
#define ERROR_CODE_EXCEEDED_MAX_ITERATIONS 		- 3

#define NORMALIZED

#define THETA_MAX 1.0E+06

// Set a required error accuracy epsilon sufficiently above the machine precision.
// Guess an initial iterate u^0. Evaluate F(u^0).
// Set a damping factor either lambda_0 = 1 or lambda_0 << 1.
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
	)
{
	// Print initial message.
	printf(	"***** \n"
		"***** WELCOME TO CLASSIC ERROR-BASED NEWTON SOLVER.\n"
		"***** \n");

	/* Iteration counter. */
	MKL_INT k = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	printf(	"***** INPUT PARAMETERS ARE:\n"
		"***** \n"
		"*****  ------------- --------------- ------------- ------------- --------- ----------- \n"
		"***** | epsilon     | maxNewtonIter | lambda[0]   | lambdaMin   | dim     | nnz       |\n"
		"***** |-------------|---------------|-------------|-------------|---------|-----------|\n"
		"***** | %-11.5E | %-13lld | %-11.5E | %-11.5E | %-7lld | %-9lld |\n"
		"*****  ------------- --------------- ------------- ------------- --------- ----------- \n"
		"***** \n", epsilon, max_newton_iterations, lambda[0], lambda[0], dim, J->nnz);

	// For iteration index k = 0, 1, ... , max_newton_iterations - 1 do:
	for (k = 0; k < max_newton_iterations; ++k)
	{
		// 1. Step k: Evaluate Jacobian matrix J(u^k). Solve linear system for du^k.
		//            J(u^k) du^k = -f(u^k).

		/* Now calculate Jacobian matrix J(u^k) into matrix. */
		JACOBIAN_CALC(*J, u[k], DEBUG_PRINT);

		/* Solve linear system. */
		LINEAR_SOLVE_1(du[k], J, f[k]);

		/* Calculate ||du^k||. */
#ifdef NORMALIZED
		norm_du[k] = NORM(du[k]) / NORM(u[k]);
#else
		norm_du[k] = NORM(du[k]);
#endif

		/* Print table header every 50 iterations. */
		if (k % 50 == 0)
		{
	printf(	"*****  ------------ -------------- -------------- -------------- ------------- \n"
		"***** | NCER ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | STATUS      |\n"
		"***** |------------|--------------|--------------|--------------|-------------|\n");
		}

		// Convergence test: If ||du^k|| < epsilon: stop. Solution found u* = u^k + du^k.
		if (norm_du[k] < epsilon)
		{
			/* Update solution and store into u[k + 1] */
			ARRAY_SUM(u[k + 1], 1.0, u[k], 1.0, du[k]);

			/* Print message. */
	printf(	"***** | %-10lld | %-12.5E |              |              | %-11s |\n", k, norm_du[k], "CONVERGED A");
	printf(	"*****  ------------ -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NCER-ERR Algorithm converged successfully after %lld iterations. Converged on ||du||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

			/* No error code. */
			*err_code = ERROR_CODE_SUCCESS;

			/* Return positive index where solution is stored. */
			return k + 1;
		}

		// 2. Compute new iterate. 
		ARRAY_SUM(u[k + 1], 1.0, u[k], lambda[0], du[k]);
		RHS_CALC(f[k + 1], u[k + 1]);

		// 3. Compute the monitoring quantities
		//    Theta_k    = ||f^{k + 1}|| / ||f^k||
		Theta[k]    = NORM(f[k + 1]) / NORM(f[k]);

		// Check for THETA_MAX violation.
		if (Theta[k] > THETA_MAX)
		{
		}
		// Else: let lambda_prime_k = min(1, mu_prime_k).
		else
		{
		}

		/* Make replacement. */
		lambda[k + 1] = lambda[k];

		/* Print message. Iterate accepted because Theta is not too big; we have not reached safe region; and lambda_prime is not too big. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11s |\n", k, norm_du[k], lambda[k], Theta[k],  "ACCEPT");
	}

	/* If we reach this point we did not converge after the maximum iterations. */
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NCER-ERR Algorithm failed to converge after %lld maximum number of iterations.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", max_newton_iterations);

	/* Error code -3: Reached maximum number of iterations. */
	*err_code = ERROR_CODE_EXCEEDED_MAX_ITERATIONS;

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}
