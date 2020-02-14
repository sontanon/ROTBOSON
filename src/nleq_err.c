#include "tools.h"
#include "qnerr.h"

#define DEBUG_PRINT 0

// Set a required error accuracy epsilon sufficiently above the machine precision.
// Guess an initial iterate u^0. Evaluate F(u^0).
// Set a damping factor either lambda_0 = 1 or lambda_0 << 1.
MKL_INT nleq_err(	      
			MKL_INT 	*err_code,		// OUTPUT: Pointer to integer containing error code.
		      double 	**u,			// IN-OUTPUT: Pointer to array of solution vectors.
		      					//            First entry contains initial guess.
		      double 	**f,			// IN-OUTPUT: Pointer to array of RHS's.
		      					//            First entry contains initial RHS.
		      double	*lambda,		// IN-OUTPUT: Pointer to array of damping factors.
		      					//            First entry contains initial damping factor.
		      double	**du,			// OUTPUT: Pointer to array of updates.
		      double	**du_bar,		// OUTPUT: Pointer to array of updates (bar).
		      double 	*norm_du,		// OUTPUT: Pointer to array of update norms.
		      double	*norm_du_bar,		// OUTPUT: Pointer to array of update (bar) norms.
		      double	*Theta,			// OUTPUT: Pointer to array of monitoring quantity.
		      double	*mu,			// OUTPUT: Pointer to array of mu's.
		      double	*lambda_prime,		// OUTPUT: Pointer to array of lambda_prime's.
		      double	*mu_prime,		// OUTPUT: Pointer to array of mu_primes's.
		csr_matrix 	*J,			// INPUT: Pointer to jacobian matrix type.
		const double 	epsilon,		// INPUT: Exit tolerance.
		const MKL_INT	max_newton_iterations,	// INPUT: Maximum number of Newton iterations.	
		const double 	lambda_min,		// INPUT: Minimum damping factor.
		const MKL_INT	qnerr,			// INPUT: Boolean to indicate whether to use QNERR.
		      void	(*RHS_CALC)(double *, const double *),		// INPUT: RHS calculation subroutine.
		      void	(*JACOBIAN_CALC)(csr_matrix, const double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		      double	(*NORM)(const double *)	,			// INPUT: Norm calculation subroutine.
		      double	(*DOT)(const double *, const double *)	,	// INPUT: Dot product calculation subroutine.
		      void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),	// INPUT: Linear solver subroutine.
		      void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)	// INPUT: Linear solver subroutine.
	)
{
	printf(	"***** \n"
		"***** WELCOME TO GLOBAL ERROR-BASED NEWTON SOLVER NLEQ-ERR.\n"
		"***** \n"
		"***** Original algorithm based on (2011) Peter Deuflhard, \"Newton Methods for Nonlinear Problems\".\n"
		"***** This C program was written by Santiago Ontanon Sanchez, ICN UNAM, Mexico City September 2019.\n"
		"***** \n");

	/* Iteration counter. */
	MKL_INT k = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;
	//MKL_INT subdim = (dim - 1) / 5;
	//MKL_INT g_num = 0;

	/* Auxiliary memory block the size of solution vector. */
	double *aux = (double *)SAFE_MALLOC(sizeof(double) * dim);

	/* Auxiliary doubles. */
	double norm_du_bar_minus_du;
	double norm_du_bar_minus_one_minus_lambda_du;

	/* QNERR parameters. */
	MKL_INT qnerr_code; MKL_INT qnerr_stop;

	/* Prediction start. */
	MKL_INT prediction_start = 0;

	printf(	"***** INPUT PARAMETERS ARE:\n"
		"***** \n"
		"*****  ------------- --------------- ------------- ------------- --------- ----------- \n"
		"***** | epsilon     | maxNewtonIter | lambda[0]   | lambdaMin   | dim     | nnz       |\n"
		"***** |-------------|---------------|-------------|-------------|---------|-----------|\n"
		"***** | %-11.5E | %-13lld | %-11.5E | %-11.5E | %-7lld | %-9lld |\n"
		"*****  ------------- --------------- ------------- ------------- --------- ----------- \n"
		"***** \n", epsilon, max_newton_iterations, lambda[0], lambda_min, dim, J->nnz);

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
		//norm_du[k] = NORM(du[k] + g_num * subdim, subdim);
		norm_du[k] = NORM(du[k]);

		/* Print table header every 50 iterations. */
		if (k % 50 == 0)
		{
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** | NLEQ ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | lambda'[k]   | STATUS      |\n"
		"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");
		}

		// Convergence test: If ||du^k|| < epsilon: stop. Solution found u* = u^k + du^k.
		if (norm_du[k] < epsilon)
		{
			/* Update solution and store into u[k + 1] */
			ARRAY_SUM(u[k + 1], 1.0, u[k], 1.0, du[k]);

			/* Print message. */
	printf(	"***** | %-10lld | %-12.5E |              |              |             | %-11s  |\n", k, norm_du[k], "CONVERGED A");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on ||du||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

			/* Clear auxiliary memory block. */
			SAFE_FREE(aux);

			/* No error code. */
			*err_code = 0;

			/* Return positive index where solution is stored. */
			return k + 1;
		}

		// For k > 0: compute a prediction value for the damping factor.
		if (k > prediction_start)
		{
			/* Auxiliary memory block calculation */
			ARRAY_SUM(aux, 1.0, du_bar[k], -1.0, du[k]);
			//norm_du_bar_minus_du = NORM(aux + g_num * subdim, subdim);
			norm_du_bar_minus_du = NORM(aux);

			mu[k] = (norm_du[k - 1] * norm_du_bar[k] * lambda[k - 1]) / (norm_du_bar_minus_du * norm_du[k]);

			/* Expected damping factor. */
			lambda[k] = MIN(1.0, mu[k]);
		}

		// Regularity test: If lambda_k < lambda_min: stop. Convergence failure.
REGULARITY_TEST:if (lambda[k] < lambda_min)
		{
			/* Print message */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  |              |              | %-11s |\n", k, norm_du[k], lambda[k], "ERROR -1");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm failed after %lld iterations.\n"
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

		// 2. Else: compute the trial iterate u^{k + 1} = u^k + lambda_k du^k and evaluate
		//          f(u^{k + 1}). Solve the linear system ('old' Jacobian, 'new' RHS):
		//          J(u^k) du_bar^{k + 1} = -f(u^{k + 1}).
TRIAL_ITERATE:	ARRAY_SUM(u[k + 1], 1.0, u[k], lambda[k], du[k]);
		RHS_CALC(f[k + 1], u[k + 1]);
		LINEAR_SOLVE_2(du_bar[k + 1], J, f[k + 1]);
		//norm_du_bar[k + 1] = NORM(du_bar[k + 1] + g_num * subdim, subdim);
		norm_du_bar[k + 1] = NORM(du_bar[k + 1]);

		// 3. Compute the monitoring quantities
		//    Theta_k    = ||du_bar^{k + 1}|| / ||du^k||
		//    mu_prime_k = (1/2) * ||du^k|| lambda_k^2 / ||du_bar^{k + 1} - (1 - lambda_k) du^k||.

		/* Auxiliary memory block. */
		ARRAY_SUM(aux, 1.0, du_bar[k + 1], (lambda[k] - 1.0), du[k]);
		//norm_du_bar_minus_one_minus_lambda_du = NORM(aux + g_num * subdim, subdim);
		norm_du_bar_minus_one_minus_lambda_du = NORM(aux);

		Theta[k]    = norm_du_bar[k + 1] / norm_du[k];
		mu_prime[k] = 0.5 * norm_du[k] * lambda[k] * lambda[k] / norm_du_bar_minus_one_minus_lambda_du;

		// If Theta_k > 1 - lambda_k / 4: 
		// then replace lambda_k by lambda_prime_k = min(mu_prime_k, lambda_k / 2). Go to regularity test.
		if (Theta[k] > 1.0 - 0.25 * lambda[k]) /* If not restricted (Theta[k]  >= 1.0) */
		{
			lambda_prime[k] = MIN(0.5 * lambda[k], mu_prime[k]);

			/* Print message: iterate is rejected because Theta[k] > 1.0 - 0.25 * lambda[k]. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k], lambda[k], Theta[k], lambda_prime[k], "REJECT A");

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
		if (lambda_prime[k] == lambda[k] && lambda[k] == 1.0)
		{
			// Convergence test: If ||du_bar^{k+1}|| < epsilon: stop. Solution found u* = u^{k+1} + du_bar^{k+1}.
			if (norm_du_bar[k + 1] < epsilon)
			{
				/* Update solution */
				ARRAY_SUM(u[k + 1], 1.0, u[k + 1], 1.0, du_bar[k + 1]);

				/* Print message for convergence. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du_bar[k + 1], lambda[k], Theta[k], lambda_prime[k], "CONVERGED B");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on ||du_bar||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

				/* Clear auxiliary memory block. */
				SAFE_FREE(aux);

				/* No error code. */
				*err_code = 0;

				/* Return positive index where solution is stored. */
				return k + 1;
			}

			// If Theta_k < 1/2: Switch to QNERR with iterate u[k+1], f[k+1] as input.
			if (Theta[k] < 0.5)
			{
				if (qnerr)
				{
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du_bar[k + 1], lambda[k], Theta[k], lambda_prime[k], "ENTER QNERR");

					qnerr_stop = nleq_err_qnerr(&qnerr_code, u + k + 1, f + k + 1, 
							du + k + 1, du_bar + k + 1,
							norm_du + k + 1, norm_du_bar + k + 1, Theta + k + 1, mu + k + 1,
							J, epsilon, max_newton_iterations - (k + 1), 
							RHS_CALC, JACOBIAN_CALC, NORM, DOT, LINEAR_SOLVE_1, LINEAR_SOLVE_2);

					/* Check for convergence. */
					if (qnerr_code == 0)
					{

	printf(	"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on QNERR.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1 + qnerr_stop);

						/* Clear auxiliary memory block. */
						SAFE_FREE(aux);

						/* No error code. */
						*err_code = 0;

						/* Return positive index where solution is stored. */
						return k + 1 + qnerr_stop;
					}
					/* Check if we exited by increase in Theta and still have iterations to go. */
					else if (qnerr_code == -2)
					{
	printf(	"***** | NLEQ ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | lambda'[k]   | STATUS      |\n"
		 	"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");

						/* Prepare to restart NLEQ-ERR. */
						lambda[k - qnerr_stop + 1] = 1.0;

						/* Set k to last update. */
						k -= qnerr_stop;

						/* Increase prediction start. */
						prediction_start = k + 1;

						/* Goto 1. */
						continue;
					}
					/* Otherwise, we might have spent all iterations. */
					else if (qnerr_code == -3)
					{
	printf(	"***** \n"
		"***** NLEQ-ERR Algorithm failed to converge after %lld maximum number of iterations. Failure on QNERR.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", max_newton_iterations);
					}

					/* Clear auxiliary memory block. */
					SAFE_FREE(aux);

					/* No error code. */
					*err_code = qnerr_code;

					/* Return negative index where solution is stored. */
					return -(k + 1 - qnerr_stop);
				}
				/* Global algorithm for no QNERR. */
				else
				{
					/* Print message. Iterate accepted. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k], lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

					/* Goto 1. */
					continue;
				}
			}
		}
		// Else: If lambda_prime_k > 4 * lambda_k: replace lambda_k by lambda_prime_k and goto 2.
		else if (lambda_prime[k] > 4.0 * lambda[k])
		{
			/* Print message. Iterate rejected because lambda_prime is bigger than lambda by at leaste a factor of 4. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k], lambda[k], Theta[k], lambda_prime[k], "REJECT B");

			lambda[k] = lambda_prime[k];

			/* No need for regularity test. Go directly to try new iterate. */
			goto TRIAL_ITERATE;
		}
		// Else: accept u^{k+1} as new iterate and goto 1. with k -> k + 1.
		else
		{
			/* Print message. Iterate accepted because Theta is not too big; we have not reached safe region; and lambda_prime is not too big. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k], lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

			/* This continue is really unnecessary and does the goto 1 instruction. */
			continue;
		}
	}

	/* If we reach this point we did not converge after the maximum iterations. */
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm failed to converge after %lld maximum number of iterations.\n"
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
