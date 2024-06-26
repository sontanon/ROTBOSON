#include "tools.h"
#include "param.h"

#include "derivatives.h"
#include "omega_calc.h"

#include "rhs_vars.h"
#include "cart_to_pol.h"
#include "analysis.h"

// All functions are even about the equator and the axis.
#define EVEN 1

#undef DERIVATIVE_DEBUG

#define REGULARIZATION_AUXILIARIES

void rhs(double *f, double *u)
{
	// Omega.
	double w = omega_calc(u[w_idx], m);

	// Loop counters.
	MKL_INT i = 0;
	MKL_INT j = 0;
	MKL_INT k = 0;

	// Axis coordinate.
	double r = 0.0;

	// Calculate derivatives.
	for (k = 0; k < GNUM; ++k)
	{
		diff1r(Dr_u  + k * dim, u + k * dim, EVEN);
		diff2r(Drr_u + k * dim, u + k * dim, EVEN);
		diff1z(Dz_u  + k * dim, u + k * dim, EVEN);
		diff2z(Dzz_u + k * dim, u + k * dim, EVEN);
		/* Mixed derivatives are only used for interpolation */
		diff2rz(Drz_u + k * dim , u + k * dim, EVEN, EVEN);
	}

	// Regularization Auxiliaries.
#ifdef REGULARIZATION_AUXILIARIES
	// First calculate derivatives: Dr(log(alpha)) and Dr(log(h)).
	// Notive that derivatives are calculate to greater order.
	// For 4th order, they are calculated at 6th order.
	ex_diff1r(u_aux + 0 * dim, u + 0 * dim, EVEN, dr, NrTotal, NzTotal, ghost, order + 2);
	ex_diff1r(u_aux + 1 * dim, u + 2 * dim, EVEN, dr, NrTotal, NzTotal, ghost, order + 2);
	// Rescale.
	#pragma omp parallel shared(u_aux) private(i, j, r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < NrTotal; ++i)
		{
			r = ((double)(i - ghost) + 0.5) * dr;
			for (j = 0; j < NzTotal; ++j)
			{
				// u6 = (Dr(alpha) / r) = alpha * (Dr(log(alpha)) / r)
				u_aux[0 * dim + IDX(i, j)] *= exp(u[0 * dim + IDX(i, j)]) / r;
				// u7 = (Dr(H) / r) = 2.0 * H * (Dr(log(h)) / r)
				u_aux[1 * dim + IDX(i, j)] *= 2.0 * exp(2.0 * u[2 * dim + IDX(i, j)]) / r;
			}
		}
	}
	// Now calculate auxiliary derivatives. Similarily, to greater order.
	ex_diff1r(Dr_u_aux + 0 * dim, u_aux + 0 * dim, EVEN, dr, NrTotal, NzTotal, ghost, order + 2);
	ex_diff1r(Dr_u_aux + 1 * dim, u_aux + 1 * dim, EVEN, dr, NrTotal, NzTotal, ghost, order + 2);
#else
	#pragma omp parallel shared(u_aux, Dr_u_aux) private(i, j, r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < NrTotal; ++i)
		{
			r = ((double)(i - ghost) + 0.5) * dr;
			for (j = 0; j < NzTotal; ++j)
			{
				// Dr(alpha) / r.
				u_aux[0 * dim + IDX(i, j)] = exp(u[0 * dim + IDX(i, j)]) * (Dr_u[0 * dim + IDX(i, j)] / r);
				// Dr(H) / r.
				u_aux[1 * dim + IDX(i, j)] = 2.0 * exp(2.0 * u[2 * dim + IDX(i, j)]) * (Dr_u[2 * dim + IDX(i, j)] / r);
				// Dr(Dr(alpha) / r) = (Drr(alpha) - Dr(alpha) / r) / r = alpha * ((Drr(log(alpha)) - Dr(log(alpha)) / r) + Dr(log(alpha))**2) / r.
				Dr_u_aux[0 * dim + IDX(i, j)] = exp(u[0 * dim + IDX(i, j)]) * ((Drr_u[0 * dim + IDX(i, j)] - Dr_u[0 * dim + IDX(i, j)] / r) + Dr_u[0 * dim + IDX(i, j)] * Dr_u[0 * dim + IDX(i, j)]) / r;
				// Dr(Dr(H) / r) = 2.0 * H * ((Drr(log(h)) - Dr(log(h)) / r) + 2.0 * Dr(log(h))**2) / r.
				Dr_u_aux[1 * dim + IDX(i, j)] = 2.0 * exp(2.0 * u[2 * dim + IDX(i, j)]) * ((Drr_u[2 * dim + IDX(i, j)] - Dr_u[2 * dim + IDX(i, j)] / r) + 2.0 * Dr_u[2 * dim + IDX(i, j)] * Dr_u[2 * dim + IDX(i, j)]) / r;
			}
		}
	}
#endif

#ifdef DERIVATIVE_DEBUG
	write_single_file_2d(Dr_u, "Dr_log_alpha.asc", NrTotal, NzTotal);
	write_single_file_2d(Dr_u + dim, "Dr_beta.asc", NrTotal, NzTotal);
	write_single_file_2d(Dr_u + 2 * dim, "Dr_log_h.asc", NrTotal, NzTotal);
	write_single_file_2d(Dr_u + 3 * dim, "Dr_log_a.asc", NrTotal, NzTotal);
	write_single_file_2d(Dr_u + 4 * dim, "Dr_psi.asc", NrTotal, NzTotal);

	write_single_file_2d(Dz_u, "Dz_log_alpha.asc", NrTotal, NzTotal);
	write_single_file_2d(Dz_u + dim, "Dz_beta.asc", NrTotal, NzTotal);
	write_single_file_2d(Dz_u + 2 * dim, "Dz_log_h.asc", NrTotal, NzTotal);
	write_single_file_2d(Dz_u + 3 * dim, "Dz_log_a.asc", NrTotal, NzTotal);
	write_single_file_2d(Dz_u + 4 * dim, "Dz_psi.asc", NrTotal, NzTotal);

	write_single_file_2d(Drr_u, "Drr_log_alpha.asc", NrTotal, NzTotal);
	write_single_file_2d(Drr_u + dim, "Drr_beta.asc", NrTotal, NzTotal);
	write_single_file_2d(Drr_u + 2 * dim, "Drr_log_h.asc", NrTotal, NzTotal);
	write_single_file_2d(Drr_u + 3 * dim, "Drr_log_a.asc", NrTotal, NzTotal);
	write_single_file_2d(Drr_u + 4 * dim, "Drr_psi.asc", NrTotal, NzTotal);

	write_single_file_2d(Dzz_u, "Dzz_log_alpha.asc", NrTotal, NzTotal);
	write_single_file_2d(Dzz_u + dim, "Dzz_beta.asc", NrTotal, NzTotal);
	write_single_file_2d(Dzz_u + 2 * dim, "Dzz_log_h.asc", NrTotal, NzTotal);
	write_single_file_2d(Dzz_u + 3 * dim, "Dzz_log_a.asc", NrTotal, NzTotal);
	write_single_file_2d(Dzz_u + 4 * dim, "Dzz_psi.asc", NrTotal, NzTotal);
#endif

	// Lower-left corner with parity.
	for (i = 0; i < ghost; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			for (k = 0; k < GNUM; ++k)
			{
				f[k * dim + IDX(i, j)] = 0.0;
			}
		}
	}

	// Parity on r axis.
	for (i = 0; i < ghost; ++i)
	{
		#pragma omp parallel shared(f) private(j, k)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost; j < NzTotal; ++j)
			{
				for (k = 0; k < GNUM; ++k)
				{
					f[k * dim + IDX(i, j)] = 0.0;
				}
			}
		}
	}

	// Parity on z axis.
	for (j = 0; j < ghost; ++j)
	{
		#pragma omp parallel shared(f) private(i, k)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
				for (k = 0; k < GNUM; ++k)
				{
					f[k * dim + IDX(i, j)] = 0.0;
				}
			}
		}
	}
	// Main interior points.
	#pragma omp parallel shared(f) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrTotal - 1; ++i)
		{
			for (j = ghost; j < NzTotal - 1; ++j)
			{
				rhs_vars(f, u, Dr_u, Dz_u, Drr_u, Dzz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0, u_aux, Dr_u_aux);
			}
		}
	}

	// Before doing boundary, analysis must be done.
	/*
	ex_cart_to_pol(&i_u, &i_rr, &i_th, NULL, NULL, u, Dr_u, Dz_u, Drz_u, GNUM, dr, dz, NrInterior, NzInterior, ghost, &NrrTotal, &NthTotal, &p_dim, &drr, &dth, &rr_inf);
	ex_analysis(0, &M_KOMAR, &J_KOMAR, &GRV2, &GRV3, i_u, i_rr, i_th, w, m, l, ghost, order, NrrTotal, NthTotal, p_dim, drr, dth, rr_inf);

	//printf("Hi there, friend! M_KOMAR = %.5E, J_KOMAR = %.5E.\n", M_KOMAR, J_KOMAR);

	SAFE_FREE(i_u);
	SAFE_FREE(i_rr);
	SAFE_FREE(i_th);
	*/

	// Z boundary condition.
	j = NzTotal - 1;
	#pragma omp parallel shared(f) private(i)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrTotal - 1; ++i)
		{
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, M_KOMAR, J_KOMAR, -1.0);
		}
	}

	/*
	// Lower-right corner with parity.
	for (i = ghost + NzInterior; i < NzTotal; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}
	*/

	// R boundary condition.
	i = NrTotal - 1;
	#pragma omp parallel shared(f) private(j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ghost; j < NzTotal - 1; ++j)
		{
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, M_KOMAR, J_KOMAR, -1.0);
		}
	}

	// Top-right corner boundary condition.
	i = NrTotal - 1;
	j = NzTotal - 1;
	rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, M_KOMAR, J_KOMAR, -1.0);

	// Omega constraint.
	f[w_idx] = 0.0;

	// All done. 
	return;
}