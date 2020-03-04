// Include headers.
#include "tools.h"
#include "qnres.h"

// Debug print Jacobian CSR matrix.
#define DEBUG_PRINT 0

// Error codes.
#define ERROR_CODE_SUCCESS 				  0
#define ERROR_CODE_FAILED_REGULARITY_TEST 		- 1
#define ERROR_CODE_QNRES_THETA_INCREASE_EXIT 		- 2
#define ERROR_CODE_EXCEEDED_MAX_ITERATIONS 		- 3
#define ERROR_CODE_QNRES_ILL_CONDITIONED		- 4
#define ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_A 	-11
#define ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_B 	-13

// Maximum expansion.
#define THETA_MAX 0.25

// Set a required error accuracy epsilon sufficiently above the machine precision.
// Guess an initial iterate u^0. Evaluate F(u^0) and ||F(u^0)||.
// Set a damping factor either lambda_0 = 1 or lambda_0 << 1.
MKL_INT nleq_res(	      
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
		      void	(*RHS_CALC)(double *, const double *),				// INPUT: RHS calculation subroutine.
		      void	(*JACOBIAN_CALC)(csr_matrix, const double *, const MKL_INT),	// INPUT: Jacobian calculation subroutine.
		      double	(*NORM)(const double *),					// INPUT: Norm calculation subroutine.
		      double	(*DOT)(const double *, const double *),				// INPUT: Dot product calculation subroutine.
		      void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),		// INPUT: Linear solver subroutine.
		      void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)		// INPUT: Linear solver subroutine.
	)
{
	// Print initial message.
	printf(	"***** \n"
		"***** WELCOME TO GLOBAL RESIDUAL-BASED NEWTON SOLVER NLEQ-RES.\n"
		"***** \n"
		"***** Original algorithm based on (2011) Peter Deuflhard, \"Newton Methods for Nonlinear Problems\".\n"
		"***** This C program was written by Santiago Ontanon Sanchez, ICN UNAM, Mexico City September 2019.\n"
		"***** \n");
	
	/* Iteration counter. */
	MKL_INT k = 0;

	// Trial iteration counters.
	MKL_INT l_A = 0;
	MKL_INT l_B = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	/* Auxiliary memory block the size of solution vector. */
	double *aux = (double *)SAFE_MALLOC(sizeof(double) * dim);

	/* Auxiliary doubles. */
	double norm_f_minus_one_minus_lambda_f;

	/* QNRES parameters. */
	MKL_INT qnres_code; 
	MKL_INT qnres_stop;

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
			*err_code = ERROR_CODE_SUCCESS;

			/* Return positive index where solution is stored. */
			return k;
		}

		// Else: Evaluate Jacobian matrix J(u^k). Solve linear system J(u^k) du^k = -f(u^k).

		/* Now calculate Jacobian matrix J(u^k) into matrix. */
		JACOBIAN_CALC(*J, u[k], 0);

		/* Solve linear system. */
		LINEAR_SOLVE_1(du[k], J, f[k]);

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
			*err_code = ERROR_CODE_FAILED_REGULARITY_TEST;

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

			// Replace lambda_k by lambda_prime_k.
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
		"***** NLEQ-RES Algorithm failed after %lld iterations.\n"
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
			// Make replacement.
			lambda_prime[k] = MIN(1.0, mu_prime[k]);

			// Reset l_A trial counter since we have found a proper lambda factor that makes Theta[k] < 1 - lambda[k] / 4.
			l_A = 0;
		}

		// Test if we are inside "local" region.
		// If lambda_prime_k = lambda_k = 1:
		if (lambda_prime[k] == lambda[k] && lambda[k] == 1.0 && Theta[k] < THETA_MAX)
		{
			if (qnres)
			{
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "ENTER QNRES");

				// Call QNRES with initial iterates u[k+1], f[k+1], ||f[k+1]||. J(u[k+1]) will be calculated inside QNRES.
				// Theta will remain the monitoring quantity, whereas mu is the gamma parameter inside QNERR.
				// At this point we have done k + 1 iterations, so QNRES is called with that number less iterations.
				// qnres_stop is the returned integer: it will be positive if QNRES succeeded, but negative otherwise.
				qnres_stop = nleq_res_qnres(&qnres_code, u + k + 1, f + k + 1,
					du + k + 1, norm_f + k + 1, Theta + k + 1, mu + k + 1,
					J, epsilon, max_newton_iterations - (k + 1),
					RHS_CALC, JACOBIAN_CALC, NORM, DOT, LINEAR_SOLVE_1, LINEAR_SOLVE_2);

				// Check for convergence.
				if (qnres_code == ERROR_CODE_SUCCESS)
				{
	printf(	"***** \n"
		"***** NLEQ-RES Algorithm converged successfully after %lld iterations. Converged on QNRES.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", k + 1 + qnres_stop);

					/* Clear auxiliary memory block. */
					SAFE_FREE(aux);

					/* No error code. */
					*err_code = ERROR_CODE_SUCCESS;

					/* Return positive index where solution is stored. */
					return k + 1 + qnres_stop;
				}
				/* Check if we exited by increase in Theta and still have iterations to go. */
				/* Also include the case where the matrix was ill-conditioned. */
				else if (qnres_code == ERROR_CODE_QNRES_THETA_INCREASE_EXIT || qnres_code == ERROR_CODE_QNRES_ILL_CONDITIONED)
				{
	printf(	"***** | NLEQ ITER  | ||du[k]||    | lambda[k]    | Theta[k]     | lambda'[k]   | STATUS      |\n"
	 	"***** |------------|--------------|--------------|--------------|--------------|-------------|\n");

					/* Prepare to restart NLEQ-RES. */
					/* In this case, the last update is stored in u[k + 1 - qnres_stop] since qnres_stop is negative. */
					/* Therefore, this will function as the initial iteration for NLEQ. We need, however, an initial */
					/* lambda damping factor which will be set to 1.0. */
					lambda[k + (-qnres_stop) + 1] = 1.0;

					/* Set k to one place before last update. */
					/* This k is not where the last solution is stored, but, rather, one place before. */
					/* This is due to the fact that the "continue" statement below will increase k via */
					/* the ++k statement at the for loop above. */
					k += (-qnres_stop);

					/* Increase prediction start. */
					/* This is because we do not have a lambda[k - 1] and have to start a prediction again. */
					prediction_start = k + 1;

					/* Goto 1. */
					/* Notice that if we have reached max_newton_iterations, we will immediately go to the */
					/* appropriate error code. */
					continue;
				}
				/* Check if QNRES spent all remaining iterations. */
				else if (qnres_code == ERROR_CODE_EXCEEDED_MAX_ITERATIONS)
				{
	printf(	"***** \n"
		"***** NLEQ-ERR Algorithm failed to converge after %lld maximum number of iterations. Failure on QNERR.\n"
		"***** \n"
		"***** Will exit after cleanup... \n"
		"***** \n", max_newton_iterations);

					/* Clear auxiliary memory block. */
					SAFE_FREE(aux);

					/* No error code. */
					*err_code = qnres_code;

					/* Return negative index where solution is stored. */
					return -(k + 1 + (-qnres_stop));
				}
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
		"***** NLEQ-RES Algorithm failed after %lld iterations.\n"
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
	printf(	"***** | %-10lld | %-12.5E | %-11.5E | %-12.5E | %-11.5E | %-11s |\n", k, norm_f[k], lambda[k], Theta[k], lambda_prime[k], "ACCEPT");

			/* Reset l_B counter. */
			l_B = 0;

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
	*err_code = -ERROR_CODE_EXCEEDED_MAX_ITERATIONS;

	/* Return negative index to last filled entry. */
	return -max_newton_iterations;
}
