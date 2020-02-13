#include "tools.h"
#include "param.h"
#include "derivatives.h"
#include "simpson.h"

#define EVEN 1

void analysis(const double *sph_u, const double *sph_rr, const double *sph_th,
	const double w)
{
	// Loop counter.
	MKL_INT k = 0;

	// Rename or extract main variables.
	double *sph_log_alpha 	= sph_u;
	double *sph_beta 	= sph_u + p_dim;
	double *sph_log_h 	= sph_u + 2 * p_dim;
	double *sph_log_a 	= sph_u + 3 * p_dim;
	double *sph_psi 	= sph_u + 4 * dim;

	// Memory.
	// Derivatives.
	double *sph_Drr_log_alpha 	= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_beta 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_log_h 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_log_a 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Drr_psi 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);

	// Auxiliary arrays.
	// Integrals.
	double *I0 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I1 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I2 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	// Integrands.
	double *i0 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i1 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i2 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);

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
		i0[k] = (exp(sph_log_alpha[k]) * sph_Drr_log_alpha[k] - 0.5 * exp(2.0 * sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]) * sin(sph_th[k]) / exp(sph_log_alpha[k]) * sph_beta[k] * sph_Drr_beta[k]) * exp(sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sin(sph_th[k]);
		i1[k] = -0.25 * (exp(2.0 * sph_log_a[k]) * (2.0 * sph_rr[k] * sph_Drr_log_a[k] - 1.0) + exp(2.0 * sph_log_h[l]) * (2.0 * sph_rr[k] * sph_Drr_log_h[k] + 1.0)) * sph_rr[k] * sin(sph_th[k]);
		i2[k] = 0.25 * exp(3.0 * sph_log_h[k]) * sph_rr[k] * sph_rr[k] * sph_rr[k] * sph_rr[k] * sin(sph_th[k]) * sin(sph_th[k]) * sin(sph_th[k]) * sph_Drr_beta[k] / exp(sph_log_alpha[k]);
	}
	#pragma omp parallel for schedule(dynamic, 1) shared(M_Komar1, M_ADM, J_Komar2)
	for (k = 0; k < p_dim; ++k)
	{
		M_Komar1[k] 	= simps(&i0[P_IDX(k, 0)], dth, NthTotal);
		M_ADM[k]	= simps(&i1[P_IDX(k, 0)], dth, NthTotal);
		J_Komar1[k]	= simps(&i2[P_IDX(k, 0)], dth, NthTotal);
	}

	// Volume integrals.
	// Komar mass 2, Komar angular momentum 2.


	// Free memory.
	SAFE_FREE(sph_Drr_log_alpha);
	SAFE_FREE(sph_Drr_beta);
	SAFE_FREE(sph_Drr_log_h);
	SAFE_FREE(sph_Drr_log_a);
	SAFE_FREE(sph_psi);
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