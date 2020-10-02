#include "tools.h"
#define MAIN_FILE
#include "param.h"
#include "regularization_coupling.h"

#include "parser.h"
#include "io.h"
#include "initial.h"
#include "rhs.h"
#include "pardiso_start.h"
#include "pardiso_stop.h"
#include "omega_calc.h"
#include "csr.h"
#include "nleq_err.h"
#include "nleq_res.h"
#include "vector_algebra.h"
#include "pardiso_solve.h"
#include "low_rank.h"
#include "cart_to_pol.h"
#include "analysis.h"

#define MAX_INITIAL_GUESS_CHECKS 8
#define NORM_F0_TARGET 1.0E-05

int main(int argc, char *argv[])
{
	// Integer counter.
	MKL_INT i = 0, j = 0;

	// Stop index.
	MKL_INT k = 0;

	// Other counters.
	MKL_INT counter_i = 0;

	// Error code.
	MKL_INT errCode = 1;

	// Initial message.
	printf("******************************************************\n");
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***                    ROTBOSON                    \n");
	printf("***                                                \n");
	printf("***          Global Newton Method Version          \n");
	printf("***                                                \n");
	printf("***        Author: Santiago Ontanon Sanchez        \n");
	printf("***                                                \n");
	printf("***              ICN UNAM, Mexico City             \n");
	printf("***                                                \n");
	printf("***                                                \n");
	printf("***             First Revision: 01/08/2019         \n");
	printf("***                                                \n");
	printf("***             Last  Revision: 24/09/2020         \n");
	printf("***                                                \n");
	printf("******************************************************\n");

	// File name is in argv[1]. Check that we have at
	// least one argument.
	if (argc < 2)
	{
		printf("***                                                \n");
		printf("***           Usage: ./ROTBOSON file.par           \n");
		printf("***                                                \n");
		printf("***            Missing parameter file.             \n");
		printf("***                                                \n");
		printf("******************************************************\n");
		printf("******************************************************\n");
		return EXIT_FAILURE;
	}

	// Parse arguments and initial seed.
	// Generate directory name via format "l=?,w=X.XXXXXE-??,dr=X.XXXXXE-??,N=XXXX"
	// w will be unknown for now so set it to X.XXXXXE-01.
	parser(argv[1]);

	// Initial psi0 damper.
	double psi0_i = psi0;

	// Future scale factors.
	double next_scale[GNUM] = { 0.0 };

	// Print program start and parameters.
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***           Generating Rotating Boson           \n");
	printf("***            Star Initial Data For NR.           \n");
	printf("***                                                \n");
	printf("***           GRID:                                \n");
	printf("***            dr          = %-12.10E          \n", dr);
	printf("***            dz          = %-12.10E          \n", dz);
	printf("***            dim         = %-7lld               \n", dim);
	printf("***            NrInterior  = %-7lld               \n", NrInterior);
	printf("***            NzInterior  = %-7lld               \n", NzInterior);
	printf("***            order       = %lld                     \n", order);
	printf("***            ghost       = %lld                     \n", ghost);
	printf("***                                                \n");
	printf("***           SCALAR FIELD:                        \n");
	printf("***            l           = %-7lld               \n", l);
	printf("***            m           = %-12.10E          \n", m);
	if (fixedPhi)
	{
		printf("***            Scalar Field is Fixed at:           \n");
		printf("***            r(fixedPhi) = %-12.10E          \n", dr * (fixedPhiR - 0.5));
		printf("***            z(fixedPhi) = %-12.10E          \n", dz * (fixedPhiZ - 0.5));

	}
	else if (fixedOmega)
	{
		printf("***            Initial Omega is Fixed.             \n");
	}
	printf("***                                                \n");
	printf("***           INITIAL DATA:                        \n");
	printf("***            readInitialData = %lld     \n", readInitialData);
	if (readInitialData)
	{
		printf("***            log_alpha_i = %-18s     \n", log_alpha_i);
		printf("***            beta_i      = %-18s     \n", beta_i);
		printf("***            log_h_i     = %-18s     \n", log_h_i);
		printf("***            log_a_i     = %-18s     \n", log_a_i);
		printf("***            psi_i       = %-18s     \n", psi_i);
		printf("***            lambda_i    = %-18s     \n", lambda_i);
		printf("***            w_i         = %-18s     \n", w_i);
	}
	else
	{
		printf("***            psi0        = %-12.10E          \n", psi0);
		printf("***            sigmaR      = %-12.10E          \n", sigmaR);
		printf("***            sigmaZ      = %-12.10E          \n", sigmaZ);
		printf("***            rExt        = %-12.10E          \n", rExt);
	}
	if (!w_i)
	{
		printf("***            w0          = %-12.10E          \n", w0);
	}
	printf("***                                                \n");
	printf("***           SOLVER:                              \n");
	printf("***            solverType    = %-18s  \n", (solverType == 1) ? "Error" : "Residual");
	printf("***            epsilon       = %-12.10E        \n", epsilon);
	printf("***            maxNewtonIter = %-4lld                \n", maxNewtonIter);
	printf("***            lambda0       = %-12.10E        \n", lambda0);
	printf("***            lambdaMin     = %-12.10E        \n", lambdaMin);
	printf("***            useLowRank    = %lld       \n", useLowRank);
	printf("***                                                \n");
	printf("******************************************************\n");

	#pragma omp parallel
	{
		#pragma omp master
		{
			// Determine OMP threads.
			printf("******************************************************\n");
			printf("***                                                \n");
			printf("***            Maximum OMP threads = %d             \n", omp_get_max_threads());
			printf("***            Currently running on %d              \n", omp_get_num_threads());
			printf("***                                                \n");
			printf("******************************************************\n");
			mkl_set_dynamic(0);
			mkl_set_num_threads(omp_get_num_threads());
		}
	}

	// Allocate memory.
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***               Allocating memory...             \n");
	printf("***                                                \n");

	// Allocate pointer to double pointers.
	double **u      = (double **)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double *));
	double **f      = (double **)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double *));
	double **du     = (double **)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double *));
	double **du_bar = (double **)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double *));

	// Allocate memory.
	for (i = 0; i < maxNewtonIter + 1; i++)
	{
		u[i]      = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
		f[i]      = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
		du[i]     = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
		du_bar[i] = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
	}

	// Also include grids.
	double *r = (double *)SAFE_MALLOC(dim * sizeof(double));
	double *z = (double *)SAFE_MALLOC(dim * sizeof(double));

	// Since these grids never change, fill them once and for all.
	// Fill coordinate grids.
	double aux_r;
	#pragma omp parallel shared(r, z) private(i, j, aux_r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < NrTotal; i++)
		{
			// Calculate rho value.
			aux_r = ((double)(i - ghost) + 0.5) * dr;
			// Loop over z points.
			for (j = 0; j < NzTotal; j++)
			{
				r[IDX(i, j)] = aux_r;
				z[IDX(i, j)] = ((double)(j - ghost) + 0.5) * dz;
			}
		}
	}

	// Auxiliary global derivative pointers.
	Dr_u  = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
	Dz_u  = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
	Drr_u = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
	Dzz_u  = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));
	Drz_u  = (double *)SAFE_MALLOC((GNUM * dim + 1) * sizeof(double));

	// Auxiliary variables.
	u_aux  = (double *)SAFE_MALLOC(2 * dim * sizeof(double));
	Dr_u_aux  = (double *)SAFE_MALLOC(2 * dim * sizeof(double));

	// Newton output parameters.
	double *norm_f		= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *norm_du		= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *norm_du_bar	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *lambda	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *Theta	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *mu	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *lambda_prime	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));
	double *mu_prime	= (double *)SAFE_MALLOC((maxNewtonIter + 1) * sizeof(double));

	// Initial guess norms.
	double f_norms[GNUM];

	// Pointers to memory for spherical variables.
	double *i_rr = NULL;
	double *i_th = NULL;
	double *i_u = NULL;
	double M_KOMAR, J_KOMAR, GRV2, GRV3;
	double phi_max = 1.0, rr_phi_max = 0.0;
	MKL_INT k_rr_max = 0;

	printf("***               Finished allocation!             \n");
	printf("***                                                \n");
	printf("******************************************************\n");

	// Allocate PARDISO memory.
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***           Allocating PARDISO memory...         \n");
	printf("***                                                \n");

	// Initialize PARDISO memory and paramters.
	// Square matrix dimension is (GNUM * dim + 1).
	pardiso_start(GNUM * dim + 1);

	// Allocate CSR matrix.
	csr_matrix J;
	MKL_INT nnz = nnz_jacobian();
	csr_allocate(&J, GNUM * dim + 1, GNUM * dim + 1, nnz);

	printf("***                                                \n");
	printf("***            Allocated CSR matrix with:          \n");
	printf("***             Rows      = %-6lld                 \n", J.nrows);
	printf("***             Columns   = %-6lld                 \n", J.ncols);
	printf("***             Non-zeros = %-12lld           \n", J.nnz);
	printf("***                                                \n");
	printf("***           Finished PARDISO allocation!         \n");
	printf("***                                                \n");
	printf("******************************************************\n");

	// LOW RANK UPDATE and linear solver subroutines.
	// Linear Solver Subroutine.
	void (*linear_solve_1)(double *, csr_matrix *, double *);
	if (useLowRank)
	{
		linear_solve_1 = pardiso_solve_low_rank;
		diff_gen();
	}
	else
		linear_solve_1 = pardiso_simple_solve;

	/* NO LOW-RANK UPDATE
	void (*linear_solve_1)(double *, csr_matrix *, double *);
	linear_solve_1 = pardiso_simple_solve;
	*/
	
	// Repeated solver.
	void (*linear_solve_2)(double *, csr_matrix *, double *);
	linear_solve_2 = pardiso_repeated_solve;
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***          Setting initial guess and RHS.        \n");

	// Set initial guess
	initial_guess(u[0]);

	// Loop over sweep.
	while (1)
	{
		// First do initial guess subroutine.
		for (counter_i = 0; counter_i <= max_initial_guess_checks; ++counter_i)
		{
			// Print initial guess.
			if (counter_i == 0)
			{
				// Do I/O: create ouput directory, copy parameter file, change to output directory.
				io(initial_dirname, argv[1]);

				// Print main variables.
				write_single_file_2d(u[0]          , "log_alpha_i.asc", 	NrTotal, NzTotal);
				write_single_file_2d(u[0] +     dim, "beta_i.asc",		NrTotal, NzTotal);
				write_single_file_2d(u[0] + 2 * dim, "log_h_i.asc", 	NrTotal, NzTotal);
				write_single_file_2d(u[0] + 3 * dim, "log_a_i.asc", 	NrTotal, NzTotal);
				write_single_file_2d(u[0] + 4 * dim, "psi_i.asc", 	NrTotal, NzTotal);
				write_single_file_2d(u[0] + 5 * dim, "lambda_i.asc", 	NrTotal, NzTotal);
				write_single_file_1d(&w0, "w_i.asc", 1);
				
				// Also print r, z grids.
				write_single_file_2d(r, "r.asc", NrTotal, NzTotal);
				write_single_file_2d(z, "z.asc", NrTotal, NzTotal);
			}
			else
			{
				// If going back into checks, reprint psi.
				write_single_file_2d(u[0] + 4 * dim, "psi_i.asc", 	NrTotal, NzTotal);
			}

			// Do initial guess check.
			// First calculate initial RHS.
			rhs(f[0], u[0]);

			// Print initial RHS.
			write_single_file_2d(f[0]          , "f0_i.asc", NrTotal, NzTotal);
			write_single_file_2d(f[0] +     dim, "f1_i.asc", NrTotal, NzTotal);
			write_single_file_2d(f[0] + 2 * dim, "f2_i.asc", NrTotal, NzTotal);
			write_single_file_2d(f[0] + 3 * dim, "f3_i.asc", NrTotal, NzTotal);
			write_single_file_2d(f[0] + 4 * dim, "f4_i.asc", NrTotal, NzTotal);
			write_single_file_2d(f[0] + 5 * dim, "f5_i.asc", NrTotal, NzTotal);

			// Calculate 2-norms.
			f_norms[0] = norm2_interior(f[0]          );
			f_norms[1] = norm2_interior(f[0] +     dim);
			f_norms[2] = norm2_interior(f[0] + 2 * dim);
			f_norms[3] = norm2_interior(f[0] + 3 * dim);
			f_norms[4] = norm2_interior(f[0] + 4 * dim);
			f_norms[5] = norm2_interior(f[0] + 5 * dim);
			
			printf("***                                                \n");
			printf("***        INITIAL GUESS:                          \n");
			printf("***           counter_i  = %lld\n", counter_i);
			printf("***           || f0 ||   = %-12.10E           \n", f_norms[0]);
			printf("***           || f1 ||   = %-12.10E           \n", f_norms[1]);
			printf("***           || f2 ||   = %-12.10E           \n", f_norms[2]);
			printf("***           || f3 ||   = %-12.10E           \n", f_norms[3]);
			printf("***           || f4 ||   = %-12.10E           \n", f_norms[4]);
			printf("***           || f5 ||   = %-12.10E           \n", f_norms[5]);
			printf("***                                                \n");

			printf("***                                                \n");
			printf("******************************************************\n");

			// Calculate f_norms average.
			double f_norms_avg = (f_norms[0] + f_norms[1] + f_norms[2] + f_norms[3] + f_norms[4] + f_norms[5]) / 6.0;

			// Do main check.
			if (f_norms_avg < norm_f0_target)
			{
				// Exit loop.
				printf("******************************************************\n");
				printf("***\n");
				printf("**** Initial guess has small-enough norm. Will continue...\n");
				printf("***\n");
				printf("******************************************************\n");
				break;
			}
			else
			{
				if (counter_i < max_initial_guess_checks - 1)
				{
					// Change psi0 and rescale.
					printf("******************************************************\n");
					printf("***\n");
					printf("*** Initial guess norm is not small enough with psi0 = %3.5E.\n", psi0);

					cblas_dscal(dim, 0.5 * (1.0 + 1.0 / psi0), u[0] + 4 * dim, 1);
					psi0 = 0.5 * (1.0 + psi0);

					printf("*** Trying again with psi0 = %.5E.\n", psi0);
					printf("***\n");
					printf("******************************************************\n");
					continue;
				}
				else
				{
					// Change psi0 and rescale.
					printf("******************************************************\n");
					printf("***\n");
					printf("*** WARNING: MAX_INITIAL_GUESS_CHECKS were not enough to lower initial guess norm!\n");
					printf("***\n");
					printf("******************************************************\n");
					continue;
				}
			}
		}
		

		// Set initial damping factor lambda[0].
		lambda[0] = lambda0;

		// Set k = 0.
		k = 0;

		/* MAIN ALGORITHM: NEWTON SOLVER */
		// Start Newton iterations.
		if (maxNewtonIter > 0)
		{
			switch (solverType)
			{
				// Error-based algorithm.
				case 1:
					// Call algorithm.
					k = nleq_err(&errCode, u, f, lambda,
							du, du_bar, norm_du, norm_du_bar,
							Theta, mu, lambda_prime, mu_prime,
							&J, epsilon, maxNewtonIter, 5, 5,
							lambdaMin, localSolver,
							rhs, csr_gen_jacobian, 
							//norm2_interior, dot_interior,
							norm2_interior_all_variables, dot_interior_all_variables,
							linear_solve_1, linear_solve_2);
					break;
				// Residual-based algorithm.
				case 2:
					// ||f[0]|| is also an input parameter.
					norm_f[0] = norm2_interior_all_variables(u[0]);
					// Calle algorithm.
					k = nleq_res(&errCode, u, f, lambda,
							du, norm_f, Theta, mu, lambda_prime, mu_prime,
							&J, epsilon, maxNewtonIter, 5, 5,
							lambdaMin, localSolver, 
							rhs, csr_gen_jacobian, 
							//norm2_interior, dot_interior,
							norm2_interior_all_variables, dot_interior_all_variables,
							linear_solve_1, linear_solve_2);
					break;
			}

			// Write errCode to file.
			write_single_integer_file_1d(&errCode, "error_code.asc", 1);

			// Check for convergence.
			if (errCode != 0)
			{
				printf("******************************************************\n");
				printf("***                                                \n");
				printf("***    Warning! Did not converge: Error Code = %lld  \n", errCode);
				printf("***    Will output anyway. Do not trust results!   \n");
				printf("***                                                \n");
				printf("******************************************************\n");
				k = -k;
			}
		}
		else
		{
			printf("******************************************************\n");
			printf("***                                                \n");
			printf("***    Warning! User did not specify any Newton Iterations.  \n");
			printf("***                                                \n");
			printf("******************************************************\n");
			k = 0;
		}

		// Get omega.
		double w = omega_calc(u[k][w_idx], m);

		// Print final solutions
		write_single_file_2d(u[k]          , "log_alpha_f.asc", 	NrTotal, NzTotal);
		write_single_file_2d(u[k] +     dim, "beta_f.asc",		NrTotal, NzTotal);
		write_single_file_2d(u[k] + 2 * dim, "log_h_f.asc", 	NrTotal, NzTotal);
		write_single_file_2d(u[k] + 3 * dim, "log_a_f.asc", 	NrTotal, NzTotal);
		write_single_file_2d(u[k] + 4 * dim, "psi_f.asc", 	NrTotal, NzTotal);
		write_single_file_2d(u[k] + 5 * dim, "lambda_f.asc", 	NrTotal, NzTotal);
		write_single_file_1d(&w, "w_f.asc", 1);

		// Print final update.
		if (k > 0)
		{
			write_single_file_2d(du[k - 1]          , "du0_f.asc", NrTotal, NzTotal);
			write_single_file_2d(du[k - 1] +     dim, "du1_f.asc", NrTotal, NzTotal);
			write_single_file_2d(du[k - 1] + 2 * dim, "du2_f.asc", NrTotal, NzTotal);
			write_single_file_2d(du[k - 1] + 3 * dim, "du3_f.asc", NrTotal, NzTotal);
			write_single_file_2d(du[k - 1] + 4 * dim, "du4_f.asc", NrTotal, NzTotal);
			write_single_file_2d(du[k - 1] + 5 * dim, "du5_f.asc", NrTotal, NzTotal);
		}

		// Print final RHS.
		write_single_file_2d(f[k]          , "f0_f.asc", NrTotal, NzTotal);
		write_single_file_2d(f[k] +     dim, "f1_f.asc", NrTotal, NzTotal);
		write_single_file_2d(f[k] + 2 * dim, "f2_f.asc", NrTotal, NzTotal);
		write_single_file_2d(f[k] + 3 * dim, "f3_f.asc", NrTotal, NzTotal);
		write_single_file_2d(f[k] + 4 * dim, "f4_f.asc", NrTotal, NzTotal);
		write_single_file_2d(f[k] + 5 * dim, "f5_f.asc", NrTotal, NzTotal);

		// Also print Newton parameters.
		switch (solverType)
		{
			case 1:
				write_single_file_1d(norm_du,		"norm_du.asc",	 	k);
				write_single_file_1d(norm_du_bar,	"norm_du_bar.asc",	k);
				break;
			case 2:
				write_single_file_1d(norm_f,	"norm_f.asc",	 	k);
				break;
		}

		write_single_file_1d(lambda,		"lambda.asc",	 k);
		write_single_file_1d(Theta,		"Theta.asc",	 k);
		write_single_file_1d(mu,		"mu.asc",	 k);
		write_single_file_1d(lambda_prime,	"lambda_prime.asc",	 k);
		write_single_file_1d(mu_prime,	"mu_prime.asc",		 k);

		// Print final iteration's RHS's norms.
		f_norms[0] = norm2_interior(f[k]          );
		f_norms[1] = norm2_interior(f[k] +     dim);
		f_norms[2] = norm2_interior(f[k] + 2 * dim);
		f_norms[3] = norm2_interior(f[k] + 3 * dim);
		f_norms[4] = norm2_interior(f[k] + 4 * dim);
		f_norms[5] = norm2_interior(f[k] + 5 * dim);
		printf("***                                                \n");
		printf("***        FINAL ITERATION:                        \n");
		printf("***           || f0 ||   = %-12.10E           \n", f_norms[0]);
		printf("***           || f1 ||   = %-12.10E           \n", f_norms[1]);
		printf("***           || f2 ||   = %-12.10E           \n", f_norms[2]);
		printf("***           || f3 ||   = %-12.10E           \n", f_norms[3]);
		printf("***           || f4 ||   = %-12.10E           \n", f_norms[4]);
		printf("***           || f5 ||   = %-12.10E           \n", f_norms[5]);
		printf("***                                                \n");

		printf("***                                                \n");
		printf("******************************************************\n");

		// Also print omega.
		printf("******************************************************\n");
		printf("***                                                \n");
		printf("***           FINAL OMEGA:                         \n");
		printf("***            w          = %-12.10E            \n", w);
		printf("***                                                \n");
		printf("******************************************************\n");

		// ANALYSIS PHASE.
		// Most analysis or global quantities are calcualted in spherical coordinates.
		// Interpolate. Memory will be allocated in this subroutine.
		cart_to_pol(&i_u, &i_rr, &i_th, r, z, u[k], Dr_u, Dz_u, Drz_u, GNUM);
		// Write to file.
		write_single_file_2d_polar(i_rr           , "sph_rr.asc", 		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_th           , "sph_th.asc", 		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u            , "sph_log_alpha_f.asc", 	NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u +     p_dim, "sph_beta_f.asc",		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u + 2 * p_dim, "sph_log_h_f.asc", 		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u + 3 * p_dim, "sph_log_a_f.asc", 		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u + 4 * p_dim, "sph_psi_f.asc", 		NrrTotal, NthTotal);
		write_single_file_2d_polar(i_u + 5 * p_dim, "sph_lambda_f.asc", 	NrrTotal, NthTotal);
		// Do analysis.
		analysis(i_u, i_rr, i_th, w);
		// Calculate rr(phi_max).
		ex_phi_analysis(1, &phi_max, &rr_phi_max, &k_rr_max, i_u, i_rr, i_th, l, ghost, order, NrrTotal, NthTotal, p_dim, drr, dth, rr_inf);

		// Exit directory by going up one level (executable level).
		chdir(work_dirname);

		// Rename directory to include w.
		snprintf(final_dirname, MAX_STR_LEN, "l=%lld,w=%.5E,dr=%.5E,N=%04lld", l, w, dr, NrInterior);
		rename(initial_dirname, final_dirname);

		// Sweep continuation if sanity checks first.
		if (errCode == 0)
		{
			// Check if sweep should continue on this resolution.
			if (rr_phi_max >= rr_phi_max_minimum)
			{
#define NEXT_SCALE 0.0
				#pragma omp parallel shared(u)
				{
					#pragma omp for schedule(dynamic, 1)
					for (counter_i = 0; counter_i < GNUM * dim; ++counter_i)
					{
						u[0][counter_i] *= -NEXT_SCALE;
						u[0][counter_i] += (1.0 + NEXT_SCALE) * u[k][counter_i];
					}
				}

				// Scale psi.
				cblas_dscal(dim, psi0, u[0] + 4 * dim, 1);
			
				/*
				// Get "scaling factors".
				for (counter_i = 0; counter_i < GNUM; ++counter_i)
				{
					next_scale[counter_i] = u[k][cblas_idamax(dim, u[k] + counter_i * dim, 1)] / u[0][cblas_idamax(dim, u[0] + counter_i * dim, 1)];
				}
				next_scale[4] = psi0;

				// Now, set initial guess to final solution and continue.
				memcpy(u[0], u[k], (GNUM * dim + 1) * sizeof(double));

				// Rescale variables.
				for (counter_i = 0; counter_i < GNUM; ++counter_i)
				{
					cblas_dscal(dim, next_scale[counter_i], u[0] + counter_i * dim, 1);
				}
				*/

				// Set initial omega.
				w0 = w;

				// Set analysis phase to 0 again.
				if (!useLowRank)
					J.analysis_phase = 0;

				printf("******************************************************\n");
				printf("***                                                \n");
				printf("***   Setting initial data to last solution and continuing...\n");
				printf("***                                                \n");
				/*
				for (counter_i = 0; counter_i < GNUM; ++counter_i)
					printf("***   next_scale[%lld] = %.5E                         \n", counter_i, next_scale[counter_i]);
				*/
				printf("***                                                \n");
				printf("******************************************************\n");
			}
			// Else, exit.
			else
			{
				printf("******************************************************\n");
				printf("***                                                \n");
				printf("***   Sweep cannot continue because rr(max(phi)) < min(rr(max(phi))) = %.5E !\n", rr_phi_max_minimum);
				printf("***                                                \n");
				printf("******************************************************\n");
				break;
			}
		}
		else
		{
			// Cannot continue sweep because errCode != 0.
			printf("******************************************************\n");
			printf("***                                                \n");
			printf("***   Sweep cannot continue because errCode = %lld !\n", errCode);
			printf("***                                                \n");
			printf("******************************************************\n");
			break;
		}
		
	}

	// Clear memory.
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***              Deallocating memory...            \n");
	printf("***                                                \n");

	pardiso_stop();
	csr_deallocate(&J);

	// Free main variables with full maxNewtonIter size by looping inside them.
	for (i = 0; i < maxNewtonIter + 1; i++)
	{
		SAFE_FREE(u[i]);
		SAFE_FREE(f[i]);
		SAFE_FREE(du[i]);
		SAFE_FREE(du_bar[i]);
	}
	// Once all clear, free top pointer.
	SAFE_FREE(u);
	SAFE_FREE(f);
	SAFE_FREE(du);
	SAFE_FREE(du_bar);

	// Coordiante grids.
	SAFE_FREE(r);
	SAFE_FREE(z);

	// Derivatives.
	SAFE_FREE(Dr_u);
	SAFE_FREE(Dz_u);
	SAFE_FREE(Drr_u);
	SAFE_FREE(Dzz_u);
	SAFE_FREE(Drz_u);

	// Auxiliary variables.
	SAFE_FREE(u_aux);
	SAFE_FREE(Dr_u_aux);

	// Newton variables.
	SAFE_FREE(norm_f);
	SAFE_FREE(norm_du);
	SAFE_FREE(norm_du_bar);
	SAFE_FREE(lambda);
	SAFE_FREE(Theta);
	SAFE_FREE(mu);
	SAFE_FREE(lambda_prime);
	SAFE_FREE(mu_prime);

	// Analysis spherical variables.
	SAFE_FREE(i_rr);
	SAFE_FREE(i_th);
	SAFE_FREE(i_u);

	// Clear libconfig configuration.
	config_destroy(&cfg);

	printf("***              Finished deallocation!            \n");
	printf("***                                                \n");
	printf("******************************************************\n");

	// Print final message.
	printf("******************************************************\n");
	printf("***                                                \n");
	printf("***           All done! Have a nice day!           \n");
	printf("***                                                \n");
	printf("******************************************************\n");
	printf("******************************************************\n");

	// All done.
	return 0;
}