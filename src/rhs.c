#include "tools.h"
#include "param.h"

#include "derivatives.h"
#include "omega_calc.h"

#include "rhs_vars.h"

#define EVEN 1

void rhs(double *f, const double *u)
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

	/* Mixed derivatives should not be used.
	diff2rz(Drz_u           , u          , EVEN, EVEN);
	diff2rz(Drz_u +     dim , u +     dim, EVEN, EVEN);
	diff2rz(Drz_u + 2 * dim , u + 2 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 3 * dim , u + 3 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 4 * dim , u + 4 * dim, EVEN, EVEN);
	*/

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
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
			}
		}
	}

	// Top-left corner with parity.
	for (i = 0; i < ghost; ++i)
	{
		for (j = ghost + NzInterior; j < NzTotal; ++j)
		{
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}

	// Parity on z axis.
	for (j = 0; j < ghost; ++j)
	{
		#pragma omp parallel shared(f) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < ghost + NrInterior; ++i)
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
				rhs_vars(f, u, Dr_u, Dz_u, Drr_u, Dzz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, m, w, -1.0);
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
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, -1.0);
		}
	}

	// Lower-right corner with parity.
	for (i = ghost + NzInterior; i < NzTotal; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}

	// R boundary condition.
	i = NrTotal - 1;
	#pragma omp parallel shared(f) private(j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ghost; j < ghost + NzInterior; ++j)
		{
			rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, -1.0);
		}
	}

	// Top-right corner boundary condition.
	i = NrTotal - 1;
	j = NzTotal - 1;
	rhs_bdry(f, u, Dr_u, Dz_u, NrTotal, NzTotal, dim, ghost, i, j, dr, dz, l, -1.0);

	// Omega constraint.
	f[w_idx] = 0.0;

	// All done. 
	return;
}