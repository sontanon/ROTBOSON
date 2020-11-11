#include "tools.h"
#include "param.h"

#include "omega_calc.h"
#include "initial_interpolation.h"

#include "derivatives.h"

#undef BDRY_DEBUG
#undef I_DEBUG

void initial_guess(double *u)
{
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
	if (w_i)
	{
		read_single_file_1d(&w0, w_i, 1, __FILE__, __LINE__);
		printf("***          Read omega initial data.       \n");
	}

	// Scale omega.
	w0 *= scale_u6;

	u[w_idx] = inverse_omega_calc(w0, m);

	double m2 = m * m;
	double w2 = w0 * w0;
	double chi = sqrt(m2 - w2);

	if (readInitialData == 3)
	{
		// Allocate memory for initial data.
		double *u_0 = (double *)SAFE_MALLOC((GNUM * NrTotalInitial * NzTotalInitial + 1) * sizeof(double));

		// Read initial data.
		read_single_file_2d(u_0 + 0 * NrTotalInitial * NzTotalInitial, log_alpha_i	, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 1 * NrTotalInitial * NzTotalInitial, beta_i		, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 2 * NrTotalInitial * NzTotalInitial, log_h_i		, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 3 * NrTotalInitial * NzTotalInitial, log_a_i		, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		read_single_file_2d(u_0 + 4 * NrTotalInitial * NzTotalInitial, psi_i		, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		if (lambda_i)
		{
			read_single_file_2d(u_0 + 5 * NrTotalInitial * NzTotalInitial, lambda_i		, NrTotalInitial, NzTotalInitial, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		}
		else
		{
			k = (MKL_INT)floor(0.5 / dr_i + ghost_i - 0.5);
			#pragma omp parallel shared(u) private(i, j, r) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = k; i < NrTotalInitial; ++i)
				{
					r = dr_i * (i + 0.5 - ghost_i);
					//rl = pow(r, l);

					for (j = ghost_i; j < NzTotalInitial; ++j)
					{
						u_0[5 * NrTotalInitial * NzTotalInitial + i * NzTotalInitial + j] = (exp(2.0 * u_0[3 * NrTotalInitial * NzTotalInitial + i * NzTotalInitial + j]) - exp(2.0 * u_0[2 * NrTotalInitial * NzTotalInitial + i * NzTotalInitial + j])) / (r * r);
					}
				}
			}
			for (i = 0; i < k; ++i)
			{
				for (j = ghost_i; j < NzTotalInitial; ++j)
				{
					u_0[5 * NrTotalInitial * NzTotalInitial + i * NzTotalInitial + j] = u_0[5 * NrTotalInitial * NzTotalInitial + k * NzTotalInitial + j];
				}
			}
		}
		u_0[GNUM * NrTotalInitial * NzTotalInitial] = u[w_idx];

#ifdef I_DEBUG
		//fprintf(stderr, "NrTotalInital = %lld, NzTotalInitial = %lld, ghost_i = %lld, order_i = %lld, dr_i = %E, dz_i = %E.\n", NrTotalInitial, NzTotalInitial, ghost_i, order_i, dr_i, dz_i);
		write_single_file_2d(u_0 + 0 * NrTotalInitial * NzTotalInitial, "log_alpha_0.asc"	, NrTotalInitial, NzTotalInitial);
		write_single_file_2d(u_0 + 1 * NrTotalInitial * NzTotalInitial, "beta_0.asc"		, NrTotalInitial, NzTotalInitial);
		write_single_file_2d(u_0 + 2 * NrTotalInitial * NzTotalInitial, "log_h_0.asc"		, NrTotalInitial, NzTotalInitial);
		write_single_file_2d(u_0 + 3 * NrTotalInitial * NzTotalInitial, "log_a_0.asc"		, NrTotalInitial, NzTotalInitial);
		write_single_file_2d(u_0 + 4 * NrTotalInitial * NzTotalInitial, "psi_0.asc"		, NrTotalInitial, NzTotalInitial);
		write_single_file_2d(u_0 + 5 * NrTotalInitial * NzTotalInitial, "lambda_0.asc"		, NrTotalInitial, NzTotalInitial);
#endif

		// Interpolate u0 into u.
		initial_interpolator(u, u_0, NrTotalInitial - 2 * ghost_i, NzTotalInitial - 2 * ghost_i, ghost_i, order_i, dr_i, dz_i,
			NrInterior, NzInterior, ghost, order, dr, dz, w0, m, l);


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

		if (!log_alpha_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 0 * dim; i <  1 * dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 0 * dim, log_alpha_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_alpha initial data.        \n");
		}

		if (!beta_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 1 * dim; i <  2 * dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 1 * dim, beta_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read beta initial data.        \n");
		}

		if (!log_h_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 2 * dim; i <  3 * dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 2 * dim, log_h_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_h initial data.        \n");
		}

		if (!log_a_i)
		{
			#pragma omp parallel shared(u)
			{
				#pragma omp for schedule(guided)
				for (i = 3 * dim; i <  4 * dim; ++i)
				{
					u[i] = 0.0;
				}
			}
		}
		else
		{
			read_single_file_2d(u + 3 * dim, log_a_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read log_a initial data.        \n");
		}

		if (!psi_i)
		{
			// Now do initial guess for phi.
			#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = ghost; i < NrTotal; ++i)
				{
					r = dr * (i + 0.5 - ghost);
					//rl = pow(r, l);

					for (j = ghost; j < NzTotal; ++j)
					{
					z = dz * (j + 0.5 - ghost);
					rr = sqrt(r * r + z * z);

					u[4 * dim + IDX(i, j)] = psi0 * exp(-0.5 * r * r / (sigmaR * sigmaR)) * exp(-0.5 * z * z / (sigmaZ * sigmaZ))
						+ (psi0 * exp(-chi * rr) / pow(rr, l + 1)) * (0.5 + 0.5 * erf(2.0 * (rr - rExt) / M_2_SQRTPI));	
					}
				}
			}
		}
		else
		{
			// Rescale scalar field by constant psi0.
			read_single_file_2d(u + 4 * dim, psi_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read psi initial data.        \n");
		}

		if (!lambda_i)
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
			k = (MKL_INT)floor(0.5 / dr + ghost - 0.5);
			#pragma omp parallel shared(u) private(i, j, r) // rr.
			{
				#pragma omp for schedule(dynamic, 1)
				for (i = k; i < NrTotal; ++i)
				{
					r = dr * (i + 0.5 - ghost);
					//rl = pow(r, l);

					for (j = ghost; j < NzTotal; ++j)
					{
						u[5 * dim + IDX(i, j)] = (exp(2.0 * u[3 * dim + IDX(i, j)]) - exp(2.0 * u[2 * dim + IDX(i, j)])) / (r * r);
					}
				}
			}
			for (i = 0; i < k; ++i)
			{
				for (j = ghost; j < NzTotal; ++j)
				{
					u[5 * dim + IDX(i, j)] = u[5 * dim + IDX(k, j)];
				}
			}
		}
		else
		{
			read_single_file_2d(u + 5 * dim, lambda_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
			printf("***           Read lambda_i initial data.         \n");
		}
	}

	// Assert symmetries since they might not be automatic.
	// All functions are even with respect to the axis and equator.
	// Corner.
	for (i = 0; i < ghost; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[5 * dim + IDX(i, j)] = u[5 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
		}
	}
	// Axis.
	#pragma omp parallel shared(u) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ghost; j < NzTotal; ++j)
		{
			for (i = 0; i < ghost; ++i)
			{
				u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(2 * ghost - (i + 1), j)];
				u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(2 * ghost - (i + 1), j)];
				u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(2 * ghost - (i + 1), j)];
				u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(2 * ghost - (i + 1), j)];
				u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(2 * ghost - (i + 1), j)];
				u[5 * dim + IDX(i, j)] = u[5 * dim + IDX(2 * ghost - (i + 1), j)];
			}
		}
	}
	// Equator.
	#pragma omp parallel shared(u) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrTotal; ++i)
		{
			for (j = 0; j < ghost; ++j)
			{
				u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[5 * dim + IDX(i, j)] = u[5 * dim + IDX(i, 2 * ghost - (j + 1))];
			}
		}
	}

	// Before scaling, copy seed to u_seed.
	memcpy(u_seed + 0 * dim, u + 0 * dim, dim * sizeof(double));
	memcpy(u_seed + 1 * dim, u + 1 * dim, dim * sizeof(double));
	memcpy(u_seed + 2 * dim, u + 2 * dim, dim * sizeof(double));
	memcpy(u_seed + 3 * dim, u + 3 * dim, dim * sizeof(double));
	memcpy(u_seed + 4 * dim, u + 4 * dim, dim * sizeof(double));
	memcpy(u_seed + 5 * dim, u + 5 * dim, dim * sizeof(double));
	memcpy(u_seed + 6 * dim, u + 6 * dim,   1 * sizeof(double));
	
	// Scale initial data.
	cblas_dscal(dim, scale_u0, u + 0 * dim, 1);
	cblas_dscal(dim, scale_u1, u + 1 * dim, 1);
	cblas_dscal(dim, scale_u2, u + 2 * dim, 1);
	cblas_dscal(dim, scale_u3, u + 3 * dim, 1);
	cblas_dscal(dim, scale_u4, u + 4 * dim, 1);
	cblas_dscal(dim, scale_u5, u + 5 * dim, 1);
	// Omega has already been scaled.
	//cblas_dscal(  1, scale_u6, u + 6 * dim, 1);

	// All done.
	return;
}
