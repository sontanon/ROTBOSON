#include "tools.h"
#include "param.h"

#include "derivatives.h"
#include "omega_calc.h"

#include "rhs_vars.h"

#include "regularization.h"

#define EVEN 1

#undef DERIVATIVE_DEBUG

void rhs(double *f, double *u)
{
	// Omega.
	double w = omega_calc(u[w_idx], m);

	// Loop counters.
	MKL_INT i = 0;
	MKL_INT j = 0;

	// Calculate derivatives.
	diff1r(Dr_u          , u          , EVEN);
	diff1r(Dr_u +     dim, u +     dim, EVEN);
	diff1r(Dr_u + 2 * dim, u + 2 * dim, EVEN);
	diff1r(Dr_u + 3 * dim, u + 3 * dim, EVEN);
	diff1r(Dr_u + 4 * dim, u + 4 * dim, EVEN);

	diff2r(Drr_u          , u          , EVEN);
	diff2r(Drr_u +     dim, u +     dim, EVEN);
	diff2r(Drr_u + 2 * dim, u + 2 * dim, EVEN);
	diff2r(Drr_u + 3 * dim, u + 3 * dim, EVEN);
	diff2r(Drr_u + 4 * dim, u + 4 * dim, EVEN);

	diff1z(Dz_u          , u          , EVEN);
	diff1z(Dz_u +     dim, u +     dim, EVEN);
	diff1z(Dz_u + 2 * dim, u + 2 * dim, EVEN);
	diff1z(Dz_u + 3 * dim, u + 3 * dim, EVEN);
	diff1z(Dz_u + 4 * dim, u + 4 * dim, EVEN);

	diff2z(Dzz_u          , u          , EVEN);
	diff2z(Dzz_u +     dim, u +     dim, EVEN);
	diff2z(Dzz_u + 2 * dim, u + 2 * dim, EVEN);
	diff2z(Dzz_u + 3 * dim, u + 3 * dim, EVEN);
	diff2z(Dzz_u + 4 * dim, u + 4 * dim, EVEN);

	/* Mixed derivatives are only used for interpolation */
	diff2rz(Drz_u           , u          , EVEN, EVEN);
	diff2rz(Drz_u +     dim , u +     dim, EVEN, EVEN);
	diff2rz(Drz_u + 2 * dim , u + 2 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 3 * dim , u + 3 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 4 * dim , u + 4 * dim, EVEN, EVEN);

	// Calculate regularization lambda.
	if (regularization)
	{
		regularization_calc(lambda, regularization_i_stop,
			u, Dr_u, Dz_u, Drr_u, Dzz_u,
			dr, dz, NrTotal, NzTotal, dim, ghost, order,
			w, m, l);
	}
	// Do dumb calculation with no respect for regularization.
	else
	{
		#pragma omp parallel shared(lambda) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrTotal; ++i)
			{
				for (j = 0; j < NzTotal; ++j)
				{
					// lambda = (A - H) / r**2.
					lambda[IDX(i, j)] = (exp(2.0 * u[3 * dim + IDX(i, j)]) - exp(2.0 * u[2 * dim + IDX(i, j)])) / pow(dr * (i + 0.5 - ghost), 2);
				}
			}
		}
	}

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
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}

	// Parity on r axis.
	for (i = 0; i < ghost; ++i)
	{
		#pragma omp parallel shared(f) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost; j < NzTotal; ++j)
			{
				f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
			}
		}
	}

	/*
	// Top-left corner with parity.
	for (i = 0; i < ghost; ++i)
	{
		for (j = ghost + NzInterior; j < NzTotal; ++j)
		{
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}
	*/

	// Parity on z axis.
	for (j = 0; j < ghost; ++j)
	{
		#pragma omp parallel shared(f) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
				f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
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
				rhs_vars(f, u, Dr_u, Dz_u, Drr_u, Dzz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0, lambda);
			}
		}
	}

	// Z boundary condition.
	j = NzTotal - 1;
	#pragma omp parallel shared(f) private(i)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrTotal - 1; ++i)
		{
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0);
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
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0);
		}
	}

	// Top-right corner boundary condition.
	i = NrTotal - 1;
	j = NzTotal - 1;
	rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0);

	// Omega constraint.
	f[w_idx] = 0.0;

	// All done. 
	return;
}