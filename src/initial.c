#include "tools.h"
#include "context.h"

#include "omega_calc.h"
#include "initial_interpolation.h"

#include "derivatives.h"

#undef BDRY_DEBUG
#undef I_DEBUG

void initial_guess(rb_context *ctx, double *u)
{
	// Local alias for the IDX macro, which indexes by row-major stride NzTotal.
	const MKL_INT NzTotal = ctx->NzTotal;

	// The main idea is to set:
	// 1. Lapse alpha to one.
	// 2. Shift beta to zero.
	// 3. h function to one.
	// 4. a function to one.
	// 5. Scalar field to Gaussian profile (see below).
	// 6. Omega to one.
	//
	// In terms of the five functions u1, u2, u3, u4, u5.
	//  u1 = 0.0
	//  u2 = 0.0
	//  u3 = 0.0
	//  u4 = 0.0
	//  u5 = Gaussian profile..

	// Integer counter.
	MKL_INT i = 0;
	MKL_INT j = 0;
	MKL_INT k = 0;

	// Auxiliary variables.
	double r, z, rr;

	// Set omega variable.
	if (ctx->w_i)
	{
		read_single_file_1d(&ctx->w0, ctx->w_i, 1, __FILE__, __LINE__);
		printf("***          Read omega initial data.       \n");
	}

	// Scale omega.
	ctx->w0 *= ctx->scale_u6;

	u[ctx->w_idx] = inverse_omega_calc(ctx->w0, ctx->m);

	double m2 = ctx->m * ctx->m;
	double w2 = ctx->w0 * ctx->w0;
	double chi = sqrt(m2 - w2);

	if (ctx->readInitialData == 3)
	{
		// Allocate memory for initial data.
		double *u_0 = (double *)SAFE_MALLOC((GNUM * ctx->NrTotalInitial * ctx->NzTotalInitial + 1) * sizeof(double));

		// Read initial data.
		read_single_file_2d(u_0 + 0 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->log_alpha_i	, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 1 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->beta_i		, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 2 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->log_h_i		, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 3 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->log_a_i		, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 4 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->psi_i		, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		if (ctx->lambda_i)
		{
			read_single_file_2d(u_0 + 5 * ctx->NrTotalInitial * ctx->NzTotalInitial, ctx->lambda_i		, ctx->NrTotalInitial, ctx->NzTotalInitial, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
		}
		else
		{
			k = (MKL_INT)floor(0.5 / ctx->dr_i + ctx->ghost_i - 0.5);
			#pragma omp parallel shared(u) private(i, j, r) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = k; i < ctx->NrTotalInitial; ++i)
				{
					r = ctx->dr_i * (i + 0.5 - ctx->ghost_i);
					//rl = pow(r, l);

					for (j = ctx->ghost_i; j < ctx->NzTotalInitial; ++j)
					{
						u_0[5 * ctx->NrTotalInitial * ctx->NzTotalInitial + i * ctx->NzTotalInitial + j] = (exp(2.0 * u_0[3 * ctx->NrTotalInitial * ctx->NzTotalInitial + i * ctx->NzTotalInitial + j]) - exp(2.0 * u_0[2 * ctx->NrTotalInitial * ctx->NzTotalInitial + i * ctx->NzTotalInitial + j])) / (r * r);
					}
				}
			}
			for (i = 0; i < k; ++i)
			{
				for (j = ctx->ghost_i; j < ctx->NzTotalInitial; ++j)
				{
					u_0[5 * ctx->NrTotalInitial * ctx->NzTotalInitial + i * ctx->NzTotalInitial + j] = u_0[5 * ctx->NrTotalInitial * ctx->NzTotalInitial + k * ctx->NzTotalInitial + j];
				}
			}
		}
		u_0[GNUM * ctx->NrTotalInitial * ctx->NzTotalInitial] = u[ctx->w_idx];

#ifdef I_DEBUG
		//fprintf(stderr, "NrTotalInital = %lld, NzTotalInitial = %lld, ghost_i = %lld, order_i = %lld, dr_i = %E, dz_i = %E.\n", NrTotalInitial, NzTotalInitial, ghost_i, order_i, dr_i, dz_i);
		write_single_file_2d(u_0 + 0 * ctx->NrTotalInitial * ctx->NzTotalInitial, "log_alpha_0.asc"	, ctx->NrTotalInitial, ctx->NzTotalInitial);
		write_single_file_2d(u_0 + 1 * ctx->NrTotalInitial * ctx->NzTotalInitial, "beta_0.asc"		, ctx->NrTotalInitial, ctx->NzTotalInitial);
		write_single_file_2d(u_0 + 2 * ctx->NrTotalInitial * ctx->NzTotalInitial, "log_h_0.asc"		, ctx->NrTotalInitial, ctx->NzTotalInitial);
		write_single_file_2d(u_0 + 3 * ctx->NrTotalInitial * ctx->NzTotalInitial, "log_a_0.asc"		, ctx->NrTotalInitial, ctx->NzTotalInitial);
		write_single_file_2d(u_0 + 4 * ctx->NrTotalInitial * ctx->NzTotalInitial, "psi_0.asc"		, ctx->NrTotalInitial, ctx->NzTotalInitial);
		write_single_file_2d(u_0 + 5 * ctx->NrTotalInitial * ctx->NzTotalInitial, "lambda_0.asc"		, ctx->NrTotalInitial, ctx->NzTotalInitial);
#endif

		// Interpolate u0 into u.
		initial_interpolator(u, u_0, ctx->NrTotalInitial - 2 * ctx->ghost_i, ctx->NzTotalInitial - 2 * ctx->ghost_i, ctx->ghost_i, ctx->order_i, ctx->dr_i, ctx->dz_i,
			ctx->NrInterior, ctx->NzInterior, ctx->ghost, ctx->order, ctx->dr, ctx->dz, ctx->w0, ctx->m, ctx->l);


		// Free initial data.
		SAFE_FREE(u_0);
	}
	else
	{
		// log(alpha)	= 0.0
		// beta 	= 0.0
		// log(h)	= 0.0
		// log(a)	= 0.0
		// lambda	= 0.0

		if (!ctx->log_alpha_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 0 * ctx->dim; i <  1 * ctx->dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 0 * ctx->dim, ctx->log_alpha_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_alpha initial data.        \n");
		}

		if (!ctx->beta_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 1 * ctx->dim; i <  2 * ctx->dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 1 * ctx->dim, ctx->beta_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read beta initial data.        \n");
		}

		if (!ctx->log_h_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 2 * ctx->dim; i <  3 * ctx->dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 2 * ctx->dim, ctx->log_h_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_h initial data.        \n");
		}

		if (!ctx->log_a_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 3 * ctx->dim; i <  4 * ctx->dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 3 * ctx->dim, ctx->log_a_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_a initial data.        \n");
		}

		if (!ctx->psi_i)
		{
			// Now do initial guess for phi.
			#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = ctx->ghost; i < ctx->NrTotal; ++i)
				{
					r = ctx->dr * (i + 0.5 - ctx->ghost);
					//rl = pow(r, l);

					for (j = ctx->ghost; j < ctx->NzTotal; ++j)
					{
					z = ctx->dz * (j + 0.5 - ctx->ghost);
					rr = sqrt(r * r + z * z);

					u[4 * ctx->dim + IDX(i, j)] = ctx->psi0 * exp(-0.5 * r * r / (ctx->sigmaR * ctx->sigmaR)) * exp(-0.5 * z * z / (ctx->sigmaZ * ctx->sigmaZ))
						+ (ctx->psi0 * exp(-chi * rr) / pow(rr, ctx->l + 1)) * (0.5 + 0.5 * erf(2.0 * (rr - ctx->rExt) / M_2_SQRTPI));	
					}
				}
			}
		}
		else
		{
			// Rescale scalar field by constant psi0.
			read_single_file_2d(u + 4 * ctx->dim, ctx->psi_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read psi initial data.        \n");
		}

		if (!ctx->lambda_i)
		{
			/*
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 5 * dim; i <  5 * dim; ++i)
				{
					u[i] = 0.0;
				}
			}
			*/
			k = (MKL_INT)floor(0.5 / ctx->dr + ctx->ghost - 0.5);
			#pragma omp parallel shared(u) private(i, j, r) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = k; i < ctx->NrTotal; ++i)
				{
					r = ctx->dr * (i + 0.5 - ctx->ghost);
					//rl = pow(r, l);

					for (j = ctx->ghost; j < ctx->NzTotal; ++j)
					{
						u[5 * ctx->dim + IDX(i, j)] = (exp(2.0 * u[3 * ctx->dim + IDX(i, j)]) - exp(2.0 * u[2 * ctx->dim + IDX(i, j)])) / (r * r);
					}
				}
			}
			for (i = 0; i < k; ++i)
			{
				for (j = ctx->ghost; j < ctx->NzTotal; ++j)
				{
					u[5 * ctx->dim + IDX(i, j)] = u[5 * ctx->dim + IDX(k, j)];
				}
			}
		}
		else
		{
			read_single_file_2d(u + 5 * ctx->dim, ctx->lambda_i, ctx->NrTotal, ctx->NzTotal, ctx->NrTotalInitial, ctx->NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read lambda_i initial data.         \n");
		}
	}

	// Assert symmetries since they might not be automatic.
	// All functions are even with respect to the axis and equator.
	// Corner.
	for (i = 0; i < ctx->ghost; ++i)
	{
		for (j = 0; j < ctx->ghost; ++j)
		{
			u[0 * ctx->dim + IDX(i, j)] = u[0 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
			u[1 * ctx->dim + IDX(i, j)] = u[1 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
			u[2 * ctx->dim + IDX(i, j)] = u[2 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
			u[3 * ctx->dim + IDX(i, j)] = u[3 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
			u[4 * ctx->dim + IDX(i, j)] = u[4 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
			u[5 * ctx->dim + IDX(i, j)] = u[5 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), 2 * ctx->ghost - (j + 1))];
		}
	}
	// Axis.
	#pragma omp parallel shared(u) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ctx->ghost; j < ctx->NzTotal; ++j)
		{
			for (i = 0; i < ctx->ghost; ++i)
			{
				u[0 * ctx->dim + IDX(i, j)] = u[0 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
				u[1 * ctx->dim + IDX(i, j)] = u[1 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
				u[2 * ctx->dim + IDX(i, j)] = u[2 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
				u[3 * ctx->dim + IDX(i, j)] = u[3 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
				u[4 * ctx->dim + IDX(i, j)] = u[4 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
				u[5 * ctx->dim + IDX(i, j)] = u[5 * ctx->dim + IDX(2 * ctx->ghost - (i + 1), j)];
			}
		}
	}
	// Equator.
	#pragma omp parallel shared(u) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ctx->ghost; i < ctx->NrTotal; ++i)
		{
			for (j = 0; j < ctx->ghost; ++j)
			{
				u[0 * ctx->dim + IDX(i, j)] = u[0 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
				u[1 * ctx->dim + IDX(i, j)] = u[1 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
				u[2 * ctx->dim + IDX(i, j)] = u[2 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
				u[3 * ctx->dim + IDX(i, j)] = u[3 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
				u[4 * ctx->dim + IDX(i, j)] = u[4 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
				u[5 * ctx->dim + IDX(i, j)] = u[5 * ctx->dim + IDX(i, 2 * ctx->ghost - (j + 1))];
			}
		}
	}

	// Before scaling, copy seed to u_seed.
	memcpy(ctx->u_seed + 0 * ctx->dim, u + 0 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 1 * ctx->dim, u + 1 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 2 * ctx->dim, u + 2 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 3 * ctx->dim, u + 3 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 4 * ctx->dim, u + 4 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 5 * ctx->dim, u + 5 * ctx->dim, ctx->dim * sizeof(double));
	memcpy(ctx->u_seed + 6 * ctx->dim, u + 6 * ctx->dim,   1 * sizeof(double));
	
	// Scale initial data.
	cblas_dscal(ctx->dim, ctx->scale_u0, u + 0 * ctx->dim, 1);
	cblas_dscal(ctx->dim, ctx->scale_u1, u + 1 * ctx->dim, 1);
	cblas_dscal(ctx->dim, ctx->scale_u2, u + 2 * ctx->dim, 1);
	cblas_dscal(ctx->dim, ctx->scale_u3, u + 3 * ctx->dim, 1);
	cblas_dscal(ctx->dim, ctx->scale_u4, u + 4 * ctx->dim, 1);
	cblas_dscal(ctx->dim, ctx->scale_u5, u + 5 * ctx->dim, 1);
	// Omega has already been scaled.
	//cblas_dscal(  1, scale_u6, u + 6 * dim, 1);

	// All done.
	return;
}
