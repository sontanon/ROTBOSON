#include "tools.h"
#include "derivatives.h"
#include "regularization_tools.h"

#undef DEBUG

void regularization_calc(double *lambda,
	const MKL_INT i_max,
	double *u, 
	double *Dr_u,
	double *Dz_u, 
	double *Drr_u, 
	double *Dzz_u,
	const double dr, const double dz,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT ghost, const MKL_INT order,
	const double w, const double m, const MKL_INT l)
{
	// Declare variables.
	double *log_alpha, *alpha, *Dr_alpha, *Dz_alpha, *Drr_alpha, *Dzz_alpha;
	double *alpha00, *alpha01, *alpha02, *alpha03, *alpha20, *alpha21, *alpha22, *alpha40;
	double *beta, *Dr_beta, *Drr_beta, *beta00, *beta01;
	double *log_h, *H, *Dr_H, *Dz_H, *Drr_H, *Dzz_H;
	double *H00, *H01, *H02, *H03, *H20, *H21, *H22, *H40;
	double *log_a, *A;
	double *psi, *Dr_psi, *Drr_psi, *psi00, *psi01, *psi02;
	double *lambda00, *lambda20;

	// Loop counters.
	MKL_INT i = 0, j = 0, k = 0;

	// Radial coordinate.
	double r = 0.0;

	// Determine if we have to do regularization.
	if (i_max)
	{
		// Allocate physical variables.
		log_alpha = u;
		alpha = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dr_alpha = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dz_alpha = (double *)SAFE_MALLOC(dim * sizeof(double));
		Drr_alpha = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dzz_alpha = (double *)SAFE_MALLOC(dim * sizeof(double));
		alpha00 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha01 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha02 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha03 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha20 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha21 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha22 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		alpha40 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));

		beta = u + dim;
		Dr_beta = Dr_u + dim;
		Drr_beta = Drr_u + dim;
		beta00 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		beta01 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));

		log_h = u + 2 * dim;
		H = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dr_H = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dz_H = (double *)SAFE_MALLOC(dim * sizeof(double));
		Drr_H = (double *)SAFE_MALLOC(dim * sizeof(double));
		Dzz_H = (double *)SAFE_MALLOC(dim * sizeof(double));
		H00 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H01 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H02 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H03 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H20 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H21 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H22 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		H40 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));

		log_a = u + 3 * dim;
		A = (double *)SAFE_MALLOC(dim * sizeof(double));

		psi = u + 4 * dim;
		Dr_psi = Dr_u + 4 * dim;
		Drr_psi = Drr_u + 4 * dim;
		psi00 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		psi01 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		psi02 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));

		lambda00 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));
		lambda20 = (double *)SAFE_MALLOC(NzTotal * sizeof(double));

		// Calculate physical variables alpha, H.
		#pragma omp parallel shared(alpha, H, A) private(k)
		{
			#pragma omp for schedule(dynamic, 1)
			for (k = 0; k < dim; ++k)
			{
				alpha[k] = exp(log_alpha[k]);
				H[k] = exp(2.0 * log_h[k]);
				A[k] = exp(2.0 * log_a[k]);
			}
		}

		// Non-calculated derivatives.
		diff1r(Dr_alpha, alpha, 1);
		diff1z(Dz_alpha, alpha, 1);
		diff2r(Drr_alpha, alpha, 1);
		diff2z(Dzz_alpha, alpha, 1);
		diff1r(Dr_H, H, 1);
		diff1z(Dz_H, H, 1);
		diff2r(Drr_H, H, 1);
		diff2z(Dzz_H, H, 1);

		// Interpolate into axis.
		#pragma omp parallel shared(alpha00, beta00, H00, psi00) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				alpha00[j] 	= axis_i(alpha[IDX(ghost,j)], 		Dr_alpha[IDX(ghost,j)], 	Drr_alpha[IDX(ghost,j)],	dr);
				beta00[j] 	= axis_i(beta[IDX(ghost,j)], 		Dr_beta[IDX(ghost,j)], 		Drr_beta[IDX(ghost,j)], 	dr);
				H00[j] 		= axis_i(H[IDX(ghost,j)], 		Dr_H[IDX(ghost,j)], 		Drr_H[IDX(ghost,j)], 	dr);
				psi00[j] 	= axis_i(psi[IDX(ghost,j)], 		Dr_psi[IDX(ghost,j)], 		Drr_psi[IDX(ghost,j)], 	dr);
			}
		}

	#ifdef DEBUG
		write_single_file_1d(alpha00, "alpha00.asc", NzTotal);
		write_single_file_1d(beta00, "beta00.asc", NzTotal);
		write_single_file_1d(H00, "H00.asc", NzTotal);
		write_single_file_1d(psi00, "psi00.asc", NzTotal);
	#endif

		// Calculate z derivatives along axis.
		ex_diff1(alpha01, alpha00, 1, dz, NzTotal, ghost, order);
		ex_diff2(alpha02, alpha00, 1, dz, NzTotal, ghost, order);
		ex_diff3(alpha03, alpha00, 1, dz, NzTotal, ghost, order);
		ex_diff1(H01, H00, 1, dz, NzTotal, ghost, order);
		ex_diff2(H02, H00, 1, dz, NzTotal, ghost, order);
		ex_diff3(H03, H00, 1, dz, NzTotal, ghost, order);
		ex_diff1(beta01, beta00, 1, dz, NzTotal, ghost, order);
		ex_diff1(psi01, psi00, 1, dz, NzTotal, ghost, order);
		ex_diff2(psi02, psi00, 1, dz, NzTotal, ghost, order);

	#ifdef DEBUG
		write_single_file_1d(alpha01, "alpha01.asc", NzTotal);
		write_single_file_1d(alpha02, "alpha02.asc", NzTotal);
		write_single_file_1d(alpha03, "alpha03.asc", NzTotal);
		write_single_file_1d(beta01, "beta01.asc", NzTotal);
		write_single_file_1d(H01, "H01.asc", NzTotal);
		write_single_file_1d(H02, "H02.asc", NzTotal);
		write_single_file_1d(H03, "H03.asc", NzTotal);
		write_single_file_1d(psi01, "psi01.asc", NzTotal);
		write_single_file_1d(psi02, "psi02.asc", NzTotal);
	#endif

		// Calculate r derivatives along axis.
		#pragma omp parallel shared(alpha20, alpha21, alpha22, alpha40, H20, H21, H22, H40) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				alpha20[j] = axis_Drr_u(alpha00[j], alpha, 	dr, j, ghost, NrTotal, NzTotal);
				alpha21[j] = axis_Drr_u(alpha01[j], Dz_alpha, 	dr, j, ghost, NrTotal, NzTotal);
				alpha22[j] = axis_Drr_u(alpha02[j], Dzz_alpha, 	dr, j, ghost, NrTotal, NzTotal);
				alpha40[j] = axis_Drrrr_u(alpha00[j], alpha, dr, j, ghost, NrTotal, NzTotal);
				H20[j] = axis_Drr_u(H00[j], H, 		dr, j, ghost, NrTotal, NzTotal);
				H21[j] = axis_Drr_u(H01[j], Dz_H, 	dr, j, ghost, NrTotal, NzTotal);
				H22[j] = axis_Drr_u(H02[j], Dzz_H, 	dr, j, ghost, NrTotal, NzTotal);
				H40[j] = axis_Drrrr_u(H00[j], H, dr, j, ghost, NrTotal, NzTotal);
			}
		}

	#ifdef DEBUG
		write_single_file_1d(alpha20, "alpha20.asc", NzTotal);
		write_single_file_1d(alpha21, "alpha21.asc", NzTotal);
		write_single_file_1d(alpha22, "alpha22.asc", NzTotal);
		write_single_file_1d(alpha40, "alpha40.asc", NzTotal);
		write_single_file_1d(H20, "H20.asc", NzTotal);
		write_single_file_1d(H21, "H21.asc", NzTotal);
		write_single_file_1d(H22, "H22.asc", NzTotal);
		write_single_file_1d(H40, "H40.asc", NzTotal);
	#endif
		// Now we can actually calculate lambda and Drr_lambda on the r axis.
		#pragma omp parallel shared(lambda00, lambda20) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				lambda00[j] = lambda_A(H00[j], H01[j], H20[j], alpha00[j], alpha01[j], alpha20[j]);
				lambda20[j] =	  Drr_lambda_A(H00[j], H01[j], H02[j], H03[j], H20[j], H21[j], H22[j], H40[j])
						+ Drr_lambda_B(H00[j], H01[j], H02[j], H03[j], H20[j], H21[j], H22[j], H40[j],
								alpha00[j], alpha01[j], alpha02[j], alpha03[j], alpha20[j], alpha21[j], alpha22[j], alpha40[j], beta01[j]);
			}
		}

		// Add terms for different l's.
		if (l == 2)
		{
			#pragma omp parallel shared(lambda20) private(j)
			{
				#pragma omp for schedule(dynamic, 1)
				for (j = 0; j < NzTotal; ++j)
				{
					lambda20[j] += Drr_lambda_C(H00[j], psi00[j]);
				}
			}
		}
		else if (l == 1)
		{
			#pragma omp parallel shared(lambda00, lambda20) private(j)
			{
				#pragma omp for schedule(dynamic, 1)
				for (j = 0; j < NzTotal; ++j)
				{
					lambda00[j] += lambda_B(H00[j], psi00[j]);
					lambda20[j] += Drr_lambda_D(H00[j], H01[j], H02[j], H03[j], H20[j], H21[j], H22[j], H40[j],
							alpha00[j], alpha01[j], alpha02[j], alpha03[j], alpha20[j], alpha21[j], alpha22[j], alpha40[j],
							beta00[j], psi00[j], psi01[j], psi02[j], w, m);
				}
			}
		}

		// Having lambda and Drr_lambda at the origin, we can Taylor series-expand up to a determined r.
		for (i = 0; i < i_max; ++i)
		{
			r = dr * (i - ghost + 0.5);

			#pragma omp parallel shared(lambda) private(j)
			{
				#pragma omp for schedule(dynamic, 1) 
				for (j = 0; j < NzTotal; ++j)
				{
					lambda[IDX(i,j)] = lambda00[j] + 0.5 * lambda20[j] * r * r;
				}
			}

		}
	}

	// Fill remaining points with lambda = (A - H) / r**2.
	#pragma omp parallel shared(lambda) private(i, j, k, r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = i_max; i < NrTotal; ++i)
		{
			r = dr * (i - ghost + 0.5);

			for (j = 0; j < NzTotal; ++j)
			{
				k = IDX(i, j);

				lambda[k] = (exp(2.0 * u[3 * dim + k]) - exp(2.0 * u[2 * dim + k])) / (r * r);
			}
		}
	}

	// Release memory.
	if (i_max)
	{
		SAFE_FREE(alpha);
		SAFE_FREE(Dr_alpha);
		SAFE_FREE(Dz_alpha);
		SAFE_FREE(Drr_alpha);
		SAFE_FREE(Dzz_alpha);
		SAFE_FREE(alpha00);
		SAFE_FREE(alpha01);
		SAFE_FREE(alpha02);
		SAFE_FREE(alpha03);
		SAFE_FREE(alpha20);
		SAFE_FREE(alpha21);
		SAFE_FREE(alpha22);
		SAFE_FREE(alpha40);
		SAFE_FREE(beta00);
		SAFE_FREE(beta01);
		SAFE_FREE(H);
		SAFE_FREE(Dr_H);
		SAFE_FREE(Dz_H);
		SAFE_FREE(Drr_H);
		SAFE_FREE(Dzz_H);
		SAFE_FREE(H00);
		SAFE_FREE(H01);
		SAFE_FREE(H02);
		SAFE_FREE(H03);
		SAFE_FREE(H20);
		SAFE_FREE(H21);
		SAFE_FREE(H22);
		SAFE_FREE(H40);
		SAFE_FREE(A);
		SAFE_FREE(psi00);
		SAFE_FREE(psi01);
		SAFE_FREE(psi02);
		SAFE_FREE(lambda00);
		SAFE_FREE(lambda20);
	}

	// All done.
	return;
}