#include "tools.h"
#include "derivatives.h"
#include "simpson.h"

#define EVEN 1

void ex_phi_analysis(const MKL_INT print, 
	double *phi_max, double *rr_phi_max, MKL_INT *f_res,
	double *sph_u, double *sph_rr, double *sph_th, 
	const MKL_INT l,
	const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT p_dim, const double drr, const double dth, const double rr_inf)
{
	// Grid variable maximum location.
	MKL_INT k = 0;

	// Extract variables.
	double *sph_psi 	= sph_u + 4 * p_dim;

	// Add scalar field.
	double *sph_phi = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	#pragma omp parallel for schedule(dynamic, 1) shared(sph_phi)	
	for (k = 0; k < p_dim; ++k)
	{
		sph_phi[k] = pow(sph_rr[k] * sin(sph_th[k]), l) * sph_psi[k];
	}

	// Minimum value.
	double phi_min = sph_phi[cblas_idamin(p_dim, sph_phi, 1)];

	// Calculate maximum index.
	k = cblas_idamax(p_dim, sph_phi, 1);

	// 2D indices.
	MKL_INT i, j;
	i = k / NthTotal;
	j = k % NthTotal;
	
	// Use parabolic interpolation.
	double fa = sph_phi[k - NthTotal];
	double fb = sph_phi[k];
	double fc = sph_phi[k + NthTotal];
	double a = sph_rr[k - NthTotal];
	double b = sph_rr[k];
	double c = sph_rr[k + NthTotal];
	double x = b;

	if (((fb - fa) + (fb - fc)) != 0.0)
		*rr_phi_max = x = b + 0.5 * drr * ((fb - fa) - (fb - fc)) / ((fb - fa) + (fb - fc));
	else
		*rr_phi_max = x = b;
	if (x != b)
		*phi_max = fa * ((x - b) / (a - b)) * ((x - c) / (a - c)) + fb * ((x - c) / (b - c)) * ((x - a) / (b - a)) + fc * ((x - a) / (c - a)) * ((x - b) / (c - b));
	else
		*phi_max = fb;

	// Now count number of points at half-width-length.
	double hwl = 0.5 * *phi_max;
	MKL_INT l_res = 0;
	MKL_INT r_res = 0;
	MKL_INT counter = 0;

	for (counter = i; counter >= 0; --counter)
	{
		if (sph_phi[P_IDX(counter, j)] > hwl)
			++l_res;
		else
			break;
	}

	for (counter = i + 1; counter < NrrTotal; ++counter)
	{
		if (sph_phi[P_IDX(counter, j)] > hwl)
			++r_res;
		else
			break;
	}

	*f_res = l_res + r_res;

		
	if (print)
	{
		// Write files.
		write_single_file_1d(phi_max, "phi_max.asc", 1);
		write_single_file_1d(rr_phi_max, "rr_phi_max.asc", 1);
		write_single_integer_file_1d(f_res, "hwl_resolution.asc", 1);
		printf("***\n");
		printf("*** Scalar Field Analysis: Maximum coordinates k = %lld, i = %lld, j = %lld .\n", k, i, j);
		printf("***\n");
		printf("***  -------------------------- ----------------------- ----------------------- \n");
		printf("*** | max(phi)                 | rr(max(phi))          | min(phi)              |\n");
		printf("***  -------------------------- ----------------------- ----------------------- \n");	
		printf("*** |       %-6.5e        |      %-6.5e      |      %-6.5e      |\n", *phi_max, *rr_phi_max, phi_min);
		printf("***  -------------------------- ----------------------- ----------------------- \n");
		printf("***\n");
		printf("***  -------------------------- ----------------------- ----------------------- \n");
		printf("*** | psi(0)                   | psi(rr(max(phi))      | HWL Resolution        |\n");
		printf("***  -------------------------- ----------------------- ----------------------- \n");
		printf("*** |       %-6.5e        |      %-6.5e      |        % 4lld          |\n", sph_psi[0], sph_psi[k], *f_res);
		printf("***  -------------------------- ----------------------- ----------------------- \n");
		printf("**** \n");
	}

	// Free memory.
	SAFE_FREE(sph_phi);

	// All done.
	return;
}

void ex_analysis(
	const MKL_INT print, double *M, double *J, double *GRV2, double *GRV3,
	double *sph_u, double *sph_rr, double *sph_th, 
	const double w, const double m, const MKL_INT l,
	const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT p_dim, const double drr, const double dth, const double rr_inf)
{
	// Loop counter.
	MKL_INT k = 0;
	
	// Rename or extract main variables.
	double *sph_log_alpha 	= sph_u + 0 * p_dim;
	double *sph_beta 	= sph_u + 1 * p_dim;
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
	double *sph_Dth_log_alpha 	= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Dth_beta 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Dth_log_h 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Dth_log_a 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *sph_Dth_psi 		= (double *)SAFE_MALLOC(sizeof(double) * p_dim);

	// Auxiliary arrays.
	// Integrands.
	double *i0 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i1 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i2 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	double *i3 = (double *)SAFE_MALLOC(sizeof(double) * p_dim);
	// Integrals.
	double *I0 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I1 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I2 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);
	double *I3 = (double *)SAFE_MALLOC(sizeof(double) * NrrTotal);

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
	// Calculate angular derivatives.
	diff1th(sph_Dth_log_alpha, 	sph_log_alpha, 	EVEN, EVEN);
	diff1th(sph_Dth_beta, 		sph_beta, 	EVEN, EVEN);
	diff1th(sph_Dth_log_h, 		sph_log_h, 	EVEN, EVEN);
	diff1th(sph_Dth_log_a, 		sph_log_a, 	EVEN, EVEN);
	diff1th(sph_Dth_psi, 		sph_psi, 	EVEN, EVEN);

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

	// Baryon number, mass and binding energy.
	double baryon_number = J_Komar2[NrrTotal - 1] / (double)l;
	double baryon_mass = baryon_number * m;
	double binding_energy = M_Komar2[NrrTotal - 1] - baryon_mass;

	// Virial identities.
	// Auxiliary doubles for expressions below.
	double aux_rr;
	double aux_th;
	double aux_r;
	double aux_rlm1;
	//double aux_rl;
	double aux_alpha2;
	double aux_beta;
	double aux_a2;
	double aux_h2;
	//double aux_phi;
	double aux_phi_o_r;
	//double aux_phi2;
	double aux_phi2_o_r2;
	// GRV2.
	*GRV2 = 0.0;
	// First compute auxiliary integrands.
	// At the origin, the integrand is simply zero.
	for (k = 0; k < NthTotal; ++k)
	{
		i0[k] = i1[k] = i2[k] = i3[k] = 0.0;
	}
	// Beyond the origin, the integrand must be calculated carefully.
	#pragma omp parallel for schedule(dynamic, 1) shared(i0, i1, i2, i3) private(k,\
	aux_rr, aux_th, aux_r, aux_rlm1, aux_alpha2, aux_beta, aux_a2, aux_h2, aux_phi_o_r, aux_phi2_o_r2)
	// aux_rl, axu_phi, aux_phi2)
	for (k = NthTotal; k < p_dim; ++k)
	{
		// Coordinates.
		aux_rr = sph_rr[k];
		aux_th = sph_th[k];
		aux_r = aux_rr * sin(aux_th);
		aux_rlm1 = (l == 1) ? 1.0 : pow(aux_r, l - 1);
		//aux_rl = aux_rlm1 * aux_r;
		// Metric functions.
		aux_alpha2 = exp(2.0 * sph_log_alpha[k]);
		aux_beta = sph_beta[k];
		aux_a2 = exp(2.0 * sph_log_a[k]);
		aux_h2 = exp(2.0 * sph_log_h[k]);
		// Scalar field over rho.
		aux_phi_o_r = sph_psi[k] * aux_rlm1;
		aux_phi2_o_r2 = aux_phi_o_r * aux_phi_o_r;
		// Scalar field proper.
		//aux_phi = aux_phi_o_r * aux_r;
		//aux_phi2 = aux_phi * aux_phi;

		// 8 * PI * A**2 * rr * S**phi_phi.
		i0[k] = 4.0 * M_PI * aux_rr * (aux_a2 * (((w + l * aux_beta) * (w + l * aux_beta) / aux_alpha2 - m * m) * aux_r * aux_r + l * l / aux_h2) * aux_phi2_o_r2
				- (l * l * aux_phi2_o_r2 + aux_rlm1 * aux_rlm1 * sin(aux_th) * sin(aux_th) * ((aux_rr * sph_Drr_psi[k]) * (aux_rr * sph_Drr_psi[k]) + sph_Dth_psi[k] * sph_Dth_psi[k])
					+ 2.0 * l * aux_phi_o_r * aux_rlm1 * sin(aux_th) * (sin(aux_th) * (aux_rr * sph_Drr_psi[k]) + cos(aux_th) * sph_Dth_psi[k])));
		// 0.75 * H**2 * rr * (r**2 * D_beta_D_beta) / alpha**2.
		i1[k] = 0.75 * aux_h2 * aux_rr  * sin(aux_th) * sin(aux_th) * ((aux_rr * sph_Drr_beta[k]) * (aux_rr * sph_Drr_beta[k]) + sph_Dth_beta[k] * sph_Dth_beta[k]) / aux_alpha2;
		// -r * D_log_alpha_D_log_alpha.
		i2[k] = -(sph_Drr_log_alpha[k] * (aux_rr * sph_Drr_log_alpha[k]) + sph_Dth_log_alpha[k] * (sph_Dth_log_alpha[k] / aux_rr));
		// Sum all components.
		i3[k] = i0[k] + i1[k] + i2[k];
	}
	// Integrate angles: theta from 0 to PI.
	I3[0] = 0.0;
	#pragma omp parallel for schedule(dynamic, 1) shared(I3) private(k)
	for (k = 1; k < NrrTotal; ++k)
	{
		I3[k] = 2.0 * simps(&i3[P_IDX(k, 0)], dth, NthTotal);
	}
	// Integrate radius.
	*GRV2 = simps(I3, drr, NrrTotal);

	// GRV3
	*GRV3 = 0.0;
	// At the origin, the integrand is simply zero.
	for (k = 0; k < NthTotal; ++k)
	{
		i0[k] = i1[k] = i2[k] = i3[k] = 0.0;
	}
	// Beyond the origin, the integrand must be calculated carefully.
	#pragma omp parallel for schedule(dynamic, 1) shared(i0, i1, i2, i3) private(k,\
	aux_rr, aux_th, aux_r, aux_rlm1, aux_alpha2, aux_beta, aux_a2, aux_h2, aux_phi_o_r, aux_phi2_o_r2)
	// aux_rl, aux_phi, aux_phi2)
	for (k = NthTotal; k < p_dim; ++k)
	{
		// Coordinates.
		aux_rr = sph_rr[k];
		aux_th = sph_th[k];
		aux_r = aux_rr * sin(aux_th);
		aux_rlm1 = (l == 1) ? 1.0 : pow(aux_r, l - 1);
		//aux_rl = aux_rlm1 * aux_r;
		// Metric functions.
		aux_alpha2 = exp(2.0 * sph_log_alpha[k]);
		aux_beta = sph_beta[k];
		aux_a2 = exp(2.0 * sph_log_a[k]);
		aux_h2 = exp(2.0 * sph_log_h[k]);
		// Scalar field over rho.
		aux_phi_o_r = sph_psi[k] * aux_rlm1;
		aux_phi2_o_r2 = aux_phi_o_r * aux_phi_o_r;
		// Scalar field proper.
		//aux_phi = aux_phi_o_r * aux_r;
		//aux_phi2 = aux_phi * aux_phi;

		// 4 * PI * S * A**2 * H * rr**2 * sin(th).
		i0[k] = 4.0 * M_PI * exp(sph_log_h[k]) * aux_rr * aux_rr * sin(aux_th) * 
				(aux_a2 * (1.5 * ((w + l * aux_beta) * (w + l * aux_beta) / aux_alpha2 - m * m) * aux_r * aux_r - 0.5 * l * l / aux_h2) * aux_phi2_o_r2
				-0.5 * (l * l * aux_phi2_o_r2 + aux_rlm1 * aux_rlm1 * sin(aux_th) * sin(aux_th) * ((aux_rr * sph_Drr_psi[k]) * (aux_rr * sph_Drr_psi[k]) + sph_Dth_psi[k] * sph_Dth_psi[k])
					+ 2.0 * l * aux_phi_o_r * aux_rlm1 * sin(aux_th) * (sin(aux_th) * (aux_rr * sph_Drr_psi[k]) + cos(aux_th) * sph_Dth_psi[k])));
		// 0.375 * H**3 * rr**2 * sin(th) * (r**2 * D_beta_D_beta) / alpha**2.
		i1[k] = 0.375 * exp(3.0 * sph_log_h[k]) * sin(aux_th)  * sin(aux_th) * sin(aux_th) * aux_rr * aux_rr * ((aux_rr * sph_Drr_beta[k]) * (aux_rr * sph_Drr_beta[k]) + sph_Dth_beta[k] * sph_Dth_beta[k]) / aux_alpha2;
		// Metric derivative terms.
		i2[k] = -sin(aux_th) * exp(sph_log_h[k]) * (((aux_rr * sph_Drr_log_alpha[k]) * (aux_rr * sph_Drr_log_alpha[k]) + sph_Dth_log_alpha[k] * sph_Dth_log_alpha[k])
				-0.5 * ((aux_rr * sph_Drr_log_h[k]) * (aux_rr * sph_Drr_log_a[k]) + sph_Dth_log_h[k] * sph_Dth_log_a[k]))
			+ 0.5 * (aux_h2 - aux_a2) * (sin(aux_th) * (aux_rr * sph_Drr_log_a[k]) + cos(aux_th) * sph_Dth_log_a[k] - 0.5 * (sin(aux_th) * (aux_rr * sph_Drr_log_h[k]) + cos(aux_th) * sph_Dth_log_h[k])) / exp(sph_log_h[k]);
		// Sum all contributions.
		i3[k] = i0[k] + i1[k] + i2[k];
	}
	// Integrate angles.
	I3[0] = 0.0;
	#pragma omp parallel for schedule(dynamic, 1) shared(I3) private(k)
	for (k = 1; k < NrrTotal; ++k)
	{
		I3[k] = 4.0 * M_PI * simps(&i3[P_IDX(k, 0)], dth, NthTotal);
	}
	// Integrate radius.
	*GRV3 = simps(I3, drr, NrrTotal);

	// Set pointer values.
	*M = 0.5 * (M_Komar1[NrrTotal - 1] + M_Komar2[NrrTotal - 1]);
	*J = 0.5 * (J_Komar1[NrrTotal - 1] + J_Komar2[NrrTotal - 1]);

	// Radius 99.
	double r99 = 0.0;
	for (k = NrrTotal - 2; k > 0; --k)
	{
		if (0.5 * (M_Komar1[k] + M_Komar2[k]) < 0.99 * *M)
		{
			// Linear interpolation.
			r99 = sph_rr[P_IDX(k, 0)] + drr * (0.99 * *M - 0.5 * (M_Komar1[k] + M_Komar2[k])) / (0.5 * (M_Komar1[k + 1] + M_Komar2[k + 1] - M_Komar1[k] - M_Komar2[k]));
			break;
		}
	}

	// Complementary terms for GRV2 and GRV3 using Kerr extrapolation.
	double GRV2_c = 0.0;
	double GRV3_c = 0.0;

	// Kerr ratio.
	double a = *J / *M;

	GRV2_c = -M_PI * (*M / rr_inf) * (*M / rr_inf) * (0.5 
		+ (*M / rr_inf) * ((4.0 / 3.0) 
		+ (*M / rr_inf) * (3.0 - (33.0 / 8.0) * a * a / (*M * *M) 
		+ (*M / rr_inf) * ((32.0 / 5.0) - (31.0 / 5.0) * a * a / (*M * *M)))));

	GRV3_c = M_PI * *M * (*M / rr_inf) * (4.0
		+ (*M / rr_inf) * (8.0
		+ (*M / rr_inf) * (8.0 * (86.0 - 15.0 * a * a / (*M * *M)) / 45.0
		+ (*M / rr_inf) * (2.0 * (1526.0 - 379.0 * a * a / (*M * *M)) / 105.0
		+ (*M / rr_inf) * (4.0 * (21576.0 - 9256.0 * a * a / (*M * *M) + 1365.0 * a * a * a * a / (*M * *M * *M * *M)) / 1575.0)))));

	// Determine if ergoregion exists.
	MKL_INT ergoregion_flag = 0;
	for (k = 0; k < p_dim; ++k)
	{
		// Metric component -g_{tt}.
		i0[k] = exp(2.0 * sph_log_alpha[k]) - sph_rr[k] * sph_rr[k] * sin(sph_th[k]) * sin(sph_th[k]) * exp(2.0 * sph_log_h[k]) * sph_beta[k] * sph_beta[k];
		if (i0[k] < 0.0 && !ergoregion_flag)
		{
			ergoregion_flag = 1;
		}
	}

	if (print)
	{
		// Print information to screen.
		printf("*** \n");
		printf("*** GLOBAL QUANTITIES ANALYSIS\n");
		printf("*** \n");
		printf("*** Final radius is rr_inf = %6.5e.\n", rr_inf);
		printf("*** \n");
		printf("***  -------------------------- ----------------------- ----------------- ----------------- \n");
		printf("*** | Komar M Geometry Surface | Komar M Matter Volume |      ADM M      | Schwarzschild M |\n");
		printf("***  -------------------------- ----------------------- ----------------- ----------------- \n");	
		//printf("*** |      1234567890123       |     1234567890123     |  1234567890123  |  1234567890123  |      1234567890123       |     1234567890123      |\n");
		printf("*** |       %-6.5e        |      %-6.5e      |   %-6.5e   |   %-6.5e   |\n", M_Komar1[NrrTotal - 1], M_Komar2[NrrTotal - 1], M_ADM[NrrTotal - 1], M_Schwarz[NrrTotal - 1]);
		printf("***  -------------------------- ----------------------- ----------------- ----------------- \n");
		printf("*** \n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** | Komar J Geometry Surface | Komar J Matter Surface |\n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** |       %-6.5e        |      %-6.5e       |\n", J_Komar1[NrrTotal - 1], J_Komar2[NrrTotal - 1]);
		printf("***  -------------------------- ------------------------ \n");
		printf("*** \n");
		printf("***  -------------------------- ----------------------- ----------------- \n");
		printf("*** | Baryon Number            | Baryon Mass           | Binding Energy  |\n");
		printf("***  -------------------------- ----------------------- ----------------- \n");	
		printf("*** |       %-6.5e        |      %-6.5e      |   %-6.5e  |\n", baryon_number, baryon_mass, binding_energy);
		printf("***  -------------------------- ----------------------- ----------------- \n");
		printf("*** \n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** | Radius 99%% M             | Ergoregion Exists      |\n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** |       %-6.5e        |      %lld               |\n", r99, ergoregion_flag);
		printf("***  -------------------------- ------------------------ \n");
		printf("*** \n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** | GRV2 Virital Identity    | GRV3 Virial Identity   |\n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** |       %- 6.5e       |      %- 6.5e      |\n", *GRV2, *GRV3);
		printf("***  -------------------------- ------------------------ \n");
		printf("*** | GRV2 Kerr Extrapolat.    | GRV3 Kerr Extrapolat.  |\n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** |       %- 6.5e       |      %- 6.5e      |\n", GRV2_c, GRV3_c);
		printf("***  -------------------------- ------------------------ \n");
		printf("*** | GRV2 Total               | GRV3 Total             |\n");
		printf("***  -------------------------- ------------------------ \n");
		printf("*** |       %- 6.5e       |      %- 6.5e      |\n", *GRV2 + GRV2_c, *GRV3 + GRV3_c);
		printf("***  -------------------------- ------------------------ \n");
		printf("**** \n");

		// Write files.
		*GRV2 += GRV2_c;
		*GRV3 += GRV3_c;

		write_single_file_1d(M_Schwarz, "M_Schwarz.asc", NrrTotal);
		write_single_file_1d(M_Komar1, "M_Komar1.asc", NrrTotal);
		write_single_file_1d(M_Komar2, "M_Komar2.asc", NrrTotal);
		write_single_file_1d(M_ADM, "M_ADM.asc", NrrTotal);
		write_single_file_1d(J_Komar1, "J_Komar1.asc", NrrTotal);
		write_single_file_1d(J_Komar2, "J_Komar2.asc", NrrTotal);
		write_single_file_1d(GRV2, "GRV2.asc", 1);
		write_single_file_1d(GRV3, "GRV3.asc", 1);
		write_single_file_1d(&r99, "r99.asc", 1);
		write_single_integer_file_1d(&ergoregion_flag, "ergoregion_flag.asc", 1);
	}
	// Free memory.
	SAFE_FREE(sph_Drr_log_alpha);
	SAFE_FREE(sph_Drr_beta);
	SAFE_FREE(sph_Drr_log_h);
	SAFE_FREE(sph_Drr_log_a);
	SAFE_FREE(sph_Drr_psi);
	SAFE_FREE(sph_Dth_log_alpha);
	SAFE_FREE(sph_Dth_beta);
	SAFE_FREE(sph_Dth_log_h);
	SAFE_FREE(sph_Dth_log_a);
	SAFE_FREE(sph_Dth_psi);
	SAFE_FREE(I0);
	SAFE_FREE(I1);
	SAFE_FREE(I2);
	SAFE_FREE(I3);
	SAFE_FREE(i0);
	SAFE_FREE(i1);
	SAFE_FREE(i2);
	SAFE_FREE(i3);
	SAFE_FREE(M_Schwarz);
	SAFE_FREE(M_Komar1);
	SAFE_FREE(M_Komar2);
	SAFE_FREE(J_Komar1);
	SAFE_FREE(J_Komar2);

	// All done.
	return;
}