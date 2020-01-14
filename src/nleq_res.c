#include "tools.h"

// Set a required error accuracy epsilon sufficiently above the machine precision.
// Guess an initial iterate u^0. Evaluate F(u^0).
// Set a damping factor either lambda_0 = 1 or lambda_0 << 1.
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
	)
{
	printf(	"***** \n"
		"***** WELCOME TO GLOBAL RESIDUAL-BASED NEWTON SOLVER NLEQ-RES.\n"
		"***** \n"
		"***** Original algorithm based on (2011) Peter Deuflhard, \"Newton Methods for Nonlinear Problems\".\n"
		"***** This C program was written by Santiago Ontanon Sanchez, ICN UNAM, Mexico City September 2019.\n"
		"***** \n");
	
	/* Iteration counter. */
	MKL_INT k = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	/* Auxiliary memory block the size of solution vector. */
	double *aux = (double *)SAFE_MALLOC(sizeof(double) * dim);

	/* Auxiliary doubles. */
	double norm_f_minus_one_minus_lambda_f;

	/* QNERR parameters. */
	//MKL_INT qnres_code; MKL_INT qnres_stop;

	/* Prediction start. */
	MKL_INT prediction_start = 0;

	printf(	"***** INPUT PARAMETERS ARE:\n"
		"***** \n"
		"*****  ------------- --------------- ------------- ------------- ------- ----------- \n"
		"***** | epsilon     | maxNewtonIter | lambda[0]   | lambdaMin   | dim   | nnz       |\n"
		"***** |-------------|---------------|-------------|-------------|-------|-----------|\n"
		"***** | %-11.5E | %-13lld | %-11.5E | %-11.5E | %-5lld | %-9lld |\n"
		"*****  ------------- --------------- ------------- ------------- ------- ----------- \n"
		"***** \n", epsilon, max_newton_iterations, lambda[0], lambda_min, dim, J->nnz);

	/* Calculate initial RHS norm. */
	norm_f[0] = NORM(f[0]);

	// For iteration index k = 0, 1, ... , max_newton_iterations - 1 do:
	for (k = 0; k < max_newton_iterations;++k)
	{
		/* Print table header every 50 iterations. */
		if (k % 50 == 0)
		{
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n"
		"***** | NLEQ ITER  | ||f[k]||     | lambda[k]   | Theta[k]     | lambda'[k]  | STATUS      |\n"
		"***** |------------|--------------|-------------|--------------|-------------|-------------|\n");
		}

		// 1. Step k. 
		// Convergence test. If ||f(u^k)|| < epsilon: stop. Solution found u* = u^k.
		if (norm_f[k] < epsilon)
		{
			/* Print message. */
	printf(	"***** | %-10lld | %-12.5E |             |              |             | %-11s |\n", k, norm_f[k], "CONVERGED A");
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n"
		"***** \n"
		"***** NLEQ-RES Algorithm converged successfully after %lld iterations. Converged on ||f||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

			/* Clear auxiliary memory block. */
			SAFE_FREE(aux);

			/* No error code. */
			*err_code = 0;

			/* Return positive index where solution is stored. */
			return k;
		}

		// Else: Evaluate Jacobian matrix J(u^k). Solve linear system J(u^k) du^k = -f(u^k).

		/* Now calculate Jacobian matrix J(u^k) into matrix. */
		JACOBIAN_CALC(*J, u[k], 0);

		/* Solve linear system. */
		LINEAR_SOLVE(du[k], J, f[k]);

		// For k > 0: compute a prediction value for the damping factor.
		if (k > prediction_start)
		{
			mu[k] = norm_f[k - 1] * mu_prime[k - 1] / norm_f[k];

			/* Expected damping factor. */
			lambda[k] = MIN(1.0, mu[k]);
		}

		// Regularity test: If lambda_k < lambda_min: stop. Convergence failure.
REGULARITY_TEST:if (lambda[k] < lambda_min)
		{
			/* Print message */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E |              |             | %-11s |\n", k, norm_f[k], lambda[k], "ERROR -1");
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n"
		"***** \n"
		"***** NLEQ-RES Algorithm failed after %lld iterations.\n"
		"***** Reason for failure is that damping factor %11.5E is smaller that minimum specified %11.5E.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1, lambda[k], lambda_min);

			/* Clear auxiliary memory block. */
			SAFE_FREE(aux);

			/* Error code -1: failed regularity test. */
			*err_code = -1;

			/* Return negative index: last update at u[k]. */
			return -k;
		}

		// 2. Else: compute the trial iterate u^{k+1} = u^k + lambda_k du^k and evaluate
		//          f(u^{k+1}) and its norm.
TRIAL_ITERATE:	ARRAY_SUM(u[k + 1], 1.0, u[k], lambda[k], du[k]);
		RHS_CALC(f[k + 1], u[k + 1]);
		norm_f[k + 1] = NORM(f[k + 1]);

		// 3. Compute the monitoring quantities.
		//    Theta_k    = ||f(u^{k+1})|| / ||f(u^k)||.
		//    mu_prime_k = (1/2) * ||f(u^k)|| * lambda_k^2 / ||f(u^{k+1}) - (1 - lambda_k)f(u^k)||.

		/* Auxiliary memory block. */
		ARRAY_SUM(aux, 1.0, f[k + 1], (lambda[k] - 1.0), f[k]);
		norm_f_minus_one_minus_lambda_f = NORM(aux);

		Theta[k]    = norm_f[k + 1] / norm_f[k];
		mu_prime[k] = 0.5 * norm_f[k] * lambda[k] * lambda[k] / norm_f_minus_one_minus_lambda_f;

		// If Theta_k > 1 - lambda_k / 4: 
		// then replace lambda_k by lambda_prime_k = min(mu_prime_k, lambda_k / 2). Go to regularity test.
		if (Theta[k] > 1.0 - 0.25 * lambda[k]) /* If not restricted (Theta[k]  >= 1.0) */
		{
			lambda_prime[k] = MIN(0.5 * lambda[k], mu_prime[k]);

			/* Print message: iterate is rejected because Theta[k] > 1.0 - 0.25 * lambda[k]. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "REJECT A");

			lambda[k] = lambda_prime[k];

			/* Check new lambda against regularity test. */
			goto REGULARITY_TEST;
		}
		// Else: let lambda_prime_k = min(1, mu_prime_k).
		else
		{
			lambda_prime[k] = MIN(1.0, mu_prime[k]);
		}

		// If lambda_prime_k = lambda_k = 1:
		if (lambda_prime[k] == lambda[k] && lambda[k] == 1.0 && Theta[k] < 0.5)
		{
			if (qnres)
			{
				/* NOT IMPLEMENTED FOR NOW! */
			}
			else
			{
				/* Print message. Iterate accepted. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

				/* Goto 1. */
				continue;
			}
		}
		// Else: If lambda_prime_k > 4 * lambda_k: replace lambda_k by lambda_prime_k and goto 2.
		else if (lambda_prime[k] > 4.0 * lambda[k])
		{
			/* Print message. Iterate rejected because lambda_prime is bigger than lambda by at leaste a factor of 4. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "REJECT B");

			lambda[k] = lambda_prime[k];

			/* No need for regularity test. Go directly to try new iterate. */
			goto TRIAL_ITERATE;
		}
		// Else: accept u^{k+1} as new iterate and goto 1. with k -> k + 1.
		else
		{
			/* Print message. Iterate accepted because Theta is not too big; we have not reached safe region; and lambda_prime is not too big. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

			/* This continue is really unnecessary and does the goto 1 instruction. */
			continue;
		}
	}

	/* If we reach this point we did not converge after the maximum iterations. */
	printf(	"*****  ------------ -------------- ------------- -------------- ------------- ------------- \n"
		"***** \n"
		"***** NLEQ-RES Algorithm failed to converge after %lld maximum number of iterations.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", max_newton_iterations);

	/* Clear auxiliary memory block. */
	SAFE_FREE(aux);

	/* Error code -3: Reached maximum number of iterations. */
	*err_code = -3;

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}
