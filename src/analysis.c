#include "tools.h"
#include "param.h"
#include "derivatives.h"
#include "simpson.h"

#define EVEN 1

void analysis(double *sph_u, const double *sph_rr, const double *sph_th, const double w)
{
	// Loop counter.
	MKL_INT k = 0;

	// Rename or extract main variables.
	double *sph_log_alpha 	= sph_u;
	double *sph_beta 	= sph_u + p_dim;
	double *sph_log_h 	= sph_u + 2 * p_dim;
	double *sph_log_a 	= sph_u + 3 * p_dim;
	double *sph_psi 	= sph_u + 4 * p_dim;

	// Memory.
	// Derivatives.
	double *sph_Drr_log_alpha 	= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_beta 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_log_h 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_log_a 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_psi 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);

	// Auxiliary arrays.
	// Integrands.
	double *i0 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i1 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i2 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	// Integrals.
	double *I0 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I1 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I2 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);

	// Masses.
	// Schwarzschild Pseudomass.
	double *M_Schwarz = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	// Komar masses.
	double *M_Komar1 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *M_Komar2 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	// ADM mass.
	double *M_ADM = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);

	// Angular momentum.
	double *J_Komar1 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *J_Komar2 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);

	// Calculate radial derivatives.
	diff1rr(sph_Drr_log_alpha, 	sph_log_alpha, 	EVEN);
	diff1rr(sph_Drr_beta, 		sph_beta, 	EVEN);
	diff1rr(sph_Drr_log_h, 		sph_log_h, 	EVEN);
	diff1rr(sph_Drr_log_a, 		sph_log_a, 	EVEN);
	diff1rr(sph_Drr_psi, 		sph_psi, 	EVEN);

	// Schwarzschild Psuedomass.
	// First compute integrands.
	#pragma omp parallel for schedule(dynamic, 1) shared(i0, i1, i2) private(k)
	for (k = 0; k < p_dim; ++k)
	{
		// i0 is volumen element.
		i0[k] = exp(sph_log_a[k] + sph_log_h[k]) * sin(sph_th[k]);
		// Radial metric times volume element.
		i1[k] = exp(2.0 * sph_log_a[k]) * i0[k];
		// Area derivative.
		i2[k] = (1.0 + 0.5 * sph_rr[k] * (sph_Drr_log_a[k] + sph_Drr_log_h[k])) * i0[k];
	}
	#pragma omp parallel for schedule(dynamic, 1) shared(I0, I1, I2, M_Schwarz) private(k)
	for (k = 0; k < NrrTotal; ++k)
	{
		I0[k] = simps(&i0[P_IDX(k, 0)], dth, NthTotal);
		I1[k] = simps(&i1[P_IDX(k, 0)], dth, NthTotal);
		I2[k] = simps(&i2[P_IDX(k, 0)], dth, NthTotal);
		M_Schwarz[k] = 0.5 * sph_rr[P_IDX(k, 0)] * sqrt(I0[k]) * (1.0 - I2[k] * I2[k] / I1[k]);
	}

	// Surface integrals.
	// Komar mass 1, ADM mass, Komar angular momentum 1.
	// Compute integrands.
	#pragma omp parallel for schedule(dynamic, 1) shared(i0, i1, i2) private(k)
	for (k = 0; k < p_dim; ++k)
	{
		i0[k] = (exp(sph_log_alpha[k]) * sph_Drr_log_alpha[k] - 0.5 * exp(2.0 * sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]) * sin(sph_th[k]) * sph_beta[k] * sph_Drr_beta[k] / exp(sph_log_alpha[k])) * exp(sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]);
		i1[k] = -0.25 * (exp(2.0 * sph_log_a[k]) * (2.0 * sph_rr[k] * sph_Drr_log_a[k] - 1.0) + exp(2.0 * sph_log_h[k]) * (2.0 * sph_rr[k] * sph_Drr_log_h[k] + 1.0)) * sph_rr[k] * sin(sph_th[k]);
		i2[k] = 0.25 * exp(3.0 * sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sph_rr[k] * sph_rr[k] * sin(sph_th[k]) * sin(sph_th[k]) * sin(sph_th[k]) * sph_Drr_beta[k] / exp(sph_log_alpha[k]);
	}
	#pragma omp parallel for schedule(dynamic, 1) shared(M_Komar1, M_ADM, J_Komar2) private(k)
	for (k = 0; k < NrrTotal; ++k)
	{
		M_Komar1[k] 	= simps(&i0[P_IDX(k, 0)], dth, NthTotal);
		M_ADM[k]	= simps(&i1[P_IDX(k, 0)], dth, NthTotal);
		J_Komar1[k]	= simps(&i2[P_IDX(k, 0)], dth, NthTotal);
	}

	// Volume integrals.
	// Komar mass 2, Komar angular momentum 2.
	#pragma omp parallel for schedule(dynamic, 1) shared(i0, i1) private(k)
	for (k = 0; k < p_dim; ++k)
	{
		i0[k] = 4.0 * M_PI * (2.0 * w * (w + l * sph_beta[k]) / exp(sph_log_alpha[k]) - exp(sph_log_alpha[k]) * m * m) * (sph_psi[k] * sph_psi[k] * pow(sph_rr[k] * sin(sph_th[k]), 2 * l)) * exp(2.0 * sph_log_a[k] + sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]);
		i1[k] = 4.0 * M_PI * l * (w + l * sph_beta[k]) * (sph_psi[k] * sph_psi[k] * pow(sph_rr[k] * sin(sph_th[k]), 2 * l)) * exp(2.0 * sph_log_a[k] + sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]) / exp(sph_log_alpha[k]);
	}
	// Integrate angle.
	#pragma omp parallel for schedule(dynamic, 1) shared(I0, I1) private(k)
	for (k = 0; k < NrrTotal; ++k)
	{
		I0[k] = simps(&i0[P_IDX(k, 0)], dth, NthTotal);
		I1[k] = simps(&i1[P_IDX(k, 0)], dth, NthTotal);
	}
	// Integrate radius.
	for (k = 0; k < 10; ++k)
	{
		M_Komar2[k] = 0.0;
		J_Komar2[k] = 0.0;
	}
	for (k = 10; k < NrrTotal; ++k)
	{
		M_Komar2[k] = simps(I0, drr, k + 1);
		J_Komar2[k] = simps(I1, drr, k + 1);
	}

	// Write files.
	write_single_file_1d(M_Schwarz, "M_Schwarz.asc", NrrTotal);
	write_single_file_1d(M_Komar1, "M_Komar1.asc", NrrTotal);
	write_single_file_1d(M_Komar2, "M_Komar2.asc", NrrTotal);
	write_single_file_1d(M_ADM, "M_ADM.asc", NrrTotal);
	write_single_file_1d(J_Komar1, "J_Komar1.asc", NrrTotal);
	write_single_file_1d(J_Komar2, "J_Komar2.asc", NrrTotal);

	// Print information to screen.
	printf("*** \n");
	printf("*** GLOBAL QUANTITIES ANALYSIS\n");
	printf("*** \n");
	printf("*** Final radius is rr_inf = %6.5e.\n", rr_inf);
	printf("*** \n");
	printf("***  -------------------------- ----------------------- ----------------- ----------------- -------------------------- ------------------------ \n");
	printf("*** | Komar M Geometry Surface | Komar M Matter Volume |      ADM M      | Schwarzschild M | Komar J Geometry Surface | Komar J Matter Surface |\n");
	printf("***  -------------------------- ----------------------- ----------------- ----------------- -------------------------- ------------------------ \n");
	//printf("*** |      1234567890123       |     1234567890123     |  1234567890123  |  1234567890123  |      1234567890123       |     1234567890123      |\n");
	printf("*** |       %-6.5e        |      %-6.5e      |   %-6.5e   |   %-6.5e   |       %-6.5e        |      %-6.5e       |\n", M_Komar1[NrrTotal - 1], M_Komar2[NrrTotal - 1], M_ADM[NrrTotal - 1], M_Schwarz[NrrTotal - 1], J_Komar1[NrrTotal - 1], J_Komar2[NrrTotal - 1]);
	printf("***  -------------------------- ----------------------- ----------------- ----------------- -------------------------- ------------------------ \n");
	printf("*** \n");

	// Free memory.
	SAFE_FREE(sph_Drr_log_alpha);
	SAFE_FREE(sph_Drr_beta);
	SAFE_FREE(sph_Drr_log_h);
	SAFE_FREE(sph_Drr_log_a);
	SAFE_FREE(sph_Drr_psi);
	SAFE_FREE(I0);
	SAFE_FREE(I1);
	SAFE_FREE(I2);
	SAFE_FREE(i0);
	SAFE_FREE(i1);
	SAFE_FREE(i2);
	SAFE_FREE(M_Schwarz);
	SAFE_FREE(M_Komar1);
	SAFE_FREE(M_Komar2);
	SAFE_FREE(J_Komar1);
	SAFE_FREE(J_Komar2);

	// All done.
	return;
}