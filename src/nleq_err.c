// Include headers.
#include "tools.h"
#include "qnerr.h"

// Debug print Jacobian CSR matrix.
#define DEBUG_PRINT 0

// Error codes.
#define ERROR_CODE_SUCCESS 				  0
#define ERROR_CODE_FAILED_REGULARITY_TEST 		- 1
#define ERROR_CODE_QNERR_THETA_INCREASE_EXIT 		- 2
#define ERROR_CODE_EXCEEDED_MAX_ITERATIONS 		- 3
#define ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_A 	-11
#define ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_B 	-13

#define NORMALIZED

#define LOCAL_LAMBDA 	1.00000E+00
#define THETA_MAX	1.00000E+00
#define RESTRICTED	1.00000E+00

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
		const MKL_INT	max_trial_A_iterations,	// INPUT: Maximum number of trial A iterations.
		const MKL_INT	max_trial_B_iterations,	// INPUT: Maximum number of trial B iterations.
		const double 	lambda_min,		// INPUT: Minimum damping factor.
		const MKL_INT	qnerr,			// INPUT: Boolean to indicate whether to use QNERR.
		      void	(*RHS_CALC)(double *, double *),				// INPUT: RHS calculation subroutine.
		      void	(*JACOBIAN_CALC)(csr_matrix, double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		      double	(*NORM)(double *),					// INPUT: Norm calculation subroutine.
		      double	(*DOT)(double *, double *),				// INPUT: Dot product calculation subroutine.
		      void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),		// INPUT: Linear solver subroutine.
		      void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)		// INPUT: Linear solver subroutine.
	)
{
	// Print initial message.
	printf(	"***** \n"
		"***** WELCOME TO GLOBAL ERROR-BASED NEWTON SOLVER NLEQ-ERR.\n"
		"***** \n"
		"***** Original algorithm based on (2011) Peter Deuflhard, \"Newton Methods for Nonlinear Problems\".\n"
		"***** This C program was written by Santiago Ontanon Sanchez, ICN UNAM, Mexico City September 2019.\n"
		"***** \n");

	/* Iteration counter. */
	MKL_INT k = 0;

	/* Trial iteration counters. */
	MKL_INT l_A = 0;
	MKL_INT l_B = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	/* Auxiliary memory block the size of solution vector. */
	double *aux = (double *)SAFE_MALLOC(sizeof(double) * dim);

	/* Auxiliary doubles. */
	double norm_du_bar_minus_du;
	double norm_du_bar_minus_one_minus_lambda_du;

	/* QNERR parameters. */
	MKL_INT qnerr_code; 
	MKL_INT qnerr_stop;

	/* Prediction start. */
	MKL_INT prediction_start = 0;

	/* Solution norm for normalization or relative error. */
	double norm_u = 1.0;

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
		norm_du[k] = NORM(du[k]);

#ifdef NORMALIZED
		/* Calculate solution norm. */
		norm_u = NORM(u[k]);
#endif

		/* Print table header every 50 iterations. */
		if (k % 50 == 0)
		{
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** | NLEQ ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | lambda'[k]   | STATUS      |\n"
		"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");
		}

		// Convergence test: If ||du^k|| < epsilon: stop. Solution found u* = u^k + du^k.
#ifdef NORMALIZED
		if (norm_du[k] < epsilon * norm_u)
#else
		if (norm_du[k] < epsilon)
#endif
		{
			/* Update solution and store into u[k + 1] */
			ARRAY_SUM(u[k + 1], 1.0, u[k], 1.0, du[k]);

			/* Print message. */
	printf(	"***** | %-10lld | %-12.5E |              |              |              | %-11s |\n", k, norm_du[k] / norm_u, "CONVERGED A");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on ||du||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

			/* Clear auxiliary memory block. */
			SAFE_FREE(aux);

			/* No error code. */
			*err_code = ERROR_CODE_SUCCESS;

			/* Return positive index where solution is stored. */
			return k + 1;
		}

		// For k > 0: compute a prediction value for the damping factor.
		if (k > prediction_start)
		{
			/* Auxiliary memory block calculation */
			ARRAY_SUM(aux, 1.0, du_bar[k], -1.0, du[k]);
			norm_du_bar_minus_du = NORM(aux);

			mu[k] = (norm_du[k - 1] * norm_du_bar[k] * lambda[k - 1]) / (norm_du_bar_minus_du * norm_du[k]);

			/* Expected damping factor. */
			if (Theta[k - 1] > 0.5)
			{
				//lambda[k] = MIN(2.00 * lambda[k - 1], MIN(LOCAL_LAMBDA, mu[k]));
				lambda[k] = MIN(LOCAL_LAMBDA, mu[k]);
			}
			else
			{
				//lambda[k] = MIN(1.25 * lambda[k - 1], MIN(LOCAL_LAMBDA, mu[k]));
				lambda[k] = MIN(LOCAL_LAMBDA, mu[k]);
			}
		}

		// Regularity test: If lambda_k < lambda_min: stop. Convergence failure.
REGULARITY_TEST:if (lambda[k] < lambda_min)
		{
			/* Print message */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  |              |              | %-11s |\n", k, norm_du[k] / norm_u, lambda[k], "ERROR -1");
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
			*err_code = ERROR_CODE_FAILED_REGULARITY_TEST;

			/* Return negative index: last update at u[k]. */
			return -k;
		}

		// 2. Else: compute the trial iterate u^{k + 1} = u^k + lambda_k du^k and evaluate
		//          f(u^{k + 1}). Solve the linear system ('old' Jacobian, 'new' RHS):
		//          J(u^k) du_bar^{k + 1} = -f(u^{k + 1}).
TRIAL_ITERATE:	ARRAY_SUM(u[k + 1], 1.0, u[k], lambda[k], du[k]);
		RHS_CALC(f[k + 1], u[k + 1]);
		LINEAR_SOLVE_2(du_bar[k + 1], J, f[k + 1]);

		norm_du_bar[k + 1] = NORM(du_bar[k + 1]);

#ifdef NORMALIZED
		norm_u = NORM(u[k + 1]);
#endif

		// 3. Compute the monitoring quantities
		//    Theta_k    = ||du_bar^{k + 1}|| / ||du^k||
		//    mu_prime_k = (1/2) * ||du^k|| lambda_k^2 / ||du_bar^{k + 1} - (1 - lambda_k) du^k||.

		/* Auxiliary memory block. */
		ARRAY_SUM(aux, 1.0, du_bar[k + 1], (lambda[k] - 1.0), du[k]);
		norm_du_bar_minus_one_minus_lambda_du = NORM(aux);

		Theta[k]    = norm_du_bar[k + 1] / norm_du[k];
		mu_prime[k] = 0.5 * norm_du[k] * lambda[k] * lambda[k] / norm_du_bar_minus_one_minus_lambda_du;

		// If Theta_k > 1 - lambda_k / 4: 
		// then replace lambda_k by lambda_prime_k = min(mu_prime_k, lambda_k / 2). Go to regularity test.
		//if (Theta[k] > 1.0 - 0.25 * lambda[k]) /* If not restricted (Theta[k]  >= 1.0) */
		if (Theta[k] > THETA_MAX - RESTRICTED * 0.25 * lambda[k])
		{
			lambda_prime[k] = MIN(0.5 * lambda[k], mu_prime[k]);

			/* Print message: iterate is rejected because Theta[k] > 1.0 - 0.25 * lambda[k]. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k] / norm_u, lambda[k], Theta[k], lambda_prime[k], "REJECT A");

			/* Replace lambda_k by lambda_prime_k. */
			lambda[k] = lambda_prime[k];

			/* Increase trial A counter. */
			l_A++;

			/* Check if we can still do iterations. */
			if (l_A < max_trial_A_iterations)
			{
				/* Check new lambda against regularity test. */
				goto REGULARITY_TEST;
			}
			/* Else we have exceeded max_trail_A_iterations, which might indicate that we are stuck in a loop. */
			/* Therefore, exit to be safe. */
			else
			{
				/* Print message */
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm failed after %lld iterations.\n"
		"***** Reason for failure is that A type lambda_k replacement by lambda_prime_k has undergone a maximum of %lld iterations.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1, max_trial_A_iterations);

				/* Clear auxiliary memory block. */
				SAFE_FREE(aux);

				/* Error code -11: exceeded maximum trial iterations A. */
				*err_code = ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_A;

				/* Return negative index: last update at u[k + 1]. */
				return -(k + 1);
				
			}
		}
		// Else: let lambda_prime_k = min(1, mu_prime_k).
		else
		{
			/* Make replacement. */
			lambda_prime[k] = MIN(LOCAL_LAMBDA, mu_prime[k]);

			/* Reset l_A trial counter since we have found a proper lambda factor that makes Theta[k] < 1 - lambda[k] / 4. */
			l_A = 0;
		}

		// Test if we are inside "local" region.
		// If lambda_prime_k = lambda_k = 1:
		if (lambda_prime[k] == lambda[k] && lambda[k] == LOCAL_LAMBDA)
		{
			// Convergence test: If ||du_bar^{k+1}|| < epsilon: stop. Solution found u* = u^{k+1} + du_bar^{k+1}.
#ifdef NORMALIZED
			if (norm_du_bar[k + 1] < epsilon * norm_u)
#else
			if (norm_du_bar[k + 1] < epsilon)
#endif
			{
				/* Update solution */
				ARRAY_SUM(u[k + 1], 1.0, u[k + 1], 1.0, du_bar[k + 1]);

				/* Print message for convergence. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du_bar[k + 1] / norm_u, lambda[k], Theta[k], lambda_prime[k], "CONVERGED B");
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on ||du_bar||.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1);

				/* Clear auxiliary memory block. */
				SAFE_FREE(aux);

				/* No error code. */
				*err_code = ERROR_CODE_SUCCESS;

				/* Return positive index where solution is stored. */
				return k + 1;
			}

			// If Theta_k < 1/2: Switch to QNERR with iterate u[k+1], f[k+1] as input.
			if (Theta[k] < 0.5)
			{
				if (qnerr)
				{
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du_bar[k + 1] / norm_u, lambda[k], Theta[k], lambda_prime[k], "ENTER QNERR");

					/* Call QNERR with initial iterates u[k+1], f[k+1]. J(u[k+1]) will be calculated inside QNERR. */
					/* Theta will remain the monitoring quantity, whereas mu is the alpha parameter inside QNERR. */
					/* At this point, we have done k + 1 iterations, so QNERR is called with that number less iterations. */
					/* qnerr_stop is the returned integer: it will be positive if QNERR succeeded, but negative otherwise. */
					qnerr_stop = nleq_err_qnerr(&qnerr_code, u + k + 1, f + k + 1, 
							du + k + 1, du_bar + k + 1,
							norm_du + k + 1, norm_du_bar + k + 1, Theta + k + 1, mu + k + 1,
							J, epsilon, max_newton_iterations - (k + 1), 
							RHS_CALC, JACOBIAN_CALC, NORM, DOT, LINEAR_SOLVE_1, LINEAR_SOLVE_2);

					/* Check for convergence. */
					if (qnerr_code == ERROR_CODE_SUCCESS)
					{

	printf(	"***** \n"
		"***** NLEQ-ERR Algorithm converged successfully after %lld iterations. Converged on QNERR.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1 + qnerr_stop);

						/* Clear auxiliary memory block. */
						SAFE_FREE(aux);

						/* No error code. */
						*err_code = ERROR_CODE_SUCCESS;

						/* Return positive index where solution is stored. */
						return k + 1 + qnerr_stop;
					}
					/* Check if we exited by increase in Theta and still have iterations to go. */
					else if (qnerr_code == ERROR_CODE_QNERR_THETA_INCREASE_EXIT)
					{
	printf(	"***** | NLEQ ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | lambda'[k]   | STATUS      |\n"
	 	"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");

						/* Prepare to restart NLEQ-ERR. */
		 				/* In this case, the last update is stored in u[k + 1 - qnerr_stop] since qnerr_stop is negative. */
						/* Therefore, this will function as the initial iteration for NLEQ. We need, however, an initial */
						/* lambda damping factor which will be set to 1.0. */
						lambda[k + (-qnerr_stop) + 1] = LOCAL_LAMBDA;

						/* Set k to one place before last update. */
						/* This k is not where the last solution is stored, but, rather, one place before. */
						/* This is due to the fact that the "continue" statement below will increase k via */
						/* the ++k statement at the for loop above. */
						k += (-qnerr_stop);

						/* Increase prediction start. */
						/* This is because we do not have a lambda[k - 1] and have to start a prediction again. */
						prediction_start = k + 1;

						/* Goto 1. */
						/* Notice that if we have reached max_newton_iterations, we will immediately go to the */
						/* appropriate error code. */
						continue;
					}
					/* Check if QNERR spent all remaining iterations. */
					else if (qnerr_code == ERROR_CODE_EXCEEDED_MAX_ITERATIONS)
					{
	printf(	"***** \n"
		"***** NLEQ-ERR Algorithm failed to converge after %lld maximum number of iterations. Failure on QNERR.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", max_newton_iterations);

						/* Clear auxiliary memory block. */
						SAFE_FREE(aux);

						/* No error code. */
						*err_code = qnerr_code;

						/* Return negative index where solution is stored. */
						return -(k + 1 + (-qnerr_stop));
					}
				}
				/* Global algorithm for no QNERR. */
				else
				{
					/* Print message. Iterate accepted. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k] / norm_u, lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

					/* Goto 1. */
					continue;
				}
			}
		}
		// Else: If lambda_prime_k > 4 * lambda_k: replace lambda_k by lambda_prime_k and goto 2.
		else if (lambda_prime[k] > 4.0 * lambda[k])
		{
			/* Print message. Iterate rejected because lambda_prime is bigger than lambda by at leaste a factor of 4. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k] / norm_u, lambda[k], Theta[k], lambda_prime[k], "REJECT B");

			lambda[k] = lambda_prime[k];

			/* Increase trial B counter. */
			l_B++;

			/* Check if we can still do iterations. */
			if (l_B < max_trial_B_iterations)
			{
				/* No need for regularity test. Go directly to try new iterate. */
				goto TRIAL_ITERATE;
			}
			/* Else we have exceeded max_trail_B_iterations, which might indicate that we are stuck in a loop. */
			/* Therefore, exit to be safe. */
			else
			{
				/* Print message */
	printf(	"*****  ------------ -------------- -------------- -------------- -------------- ------------- \n"
		"***** \n"
		"***** NLEQ-ERR Algorithm failed after %lld iterations.\n"
		"***** Reason for failure is that B type lambda_k replacement by lambda_prime_k has undergone a maximum of %lld iterations.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1, max_trial_B_iterations);

				/* Clear auxiliary memory block. */
				SAFE_FREE(aux);

				/* Error code -11: exceeded maximum trial iterations B. */
				*err_code = ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_B;

				/* Return negative index: last update at u[k + 1]. */
				return -(k + 1);
			}
		}
		// Else: accept u^{k+1} as new iterate and goto 1. with k -> k + 1.
		else
		{
			/* Print message. Iterate accepted because Theta is not too big; we have not reached safe region; and lambda_prime is not too big. */
	printf(	"***** | %-10lld | %-12.5E | %-11.5E  | %-12.5E | %-11.5E  | %-11s |\n", k, norm_du[k] / norm_u, lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

			/* Reset l_B counter. */
			l_B = 0;

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
	*err_code = ERROR_CODE_EXCEEDED_MAX_ITERATIONS;

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}
