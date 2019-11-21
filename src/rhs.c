#include "tools.h"
#include "param.h"

#include "derivatives.h"
#include "omega_calc.h"

#include "rhs_vars.h"
#include "rhs_pseudo_robin.h"
#include "rhs_pseudo_exp_decay.h"

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

	diff2rz(Drz_u           , u          , EVEN, EVEN);
	diff2rz(Drz_u +     dim , u +     dim, EVEN, EVEN);
	diff2rz(Drz_u + 2 * dim , u + 2 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 3 * dim , u + 3 * dim, EVEN, EVEN);
	diff2rz(Drz_u + 4 * dim , u + 4 * dim, EVEN, EVEN);

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
        for (i = ghost; i < ghost + NrInterior; ++i)
        {
            for (j = ghost; j < ghost + NzInterior; ++j)
            {
                rhs_vars(f, u, Dr_u, Dz_u, Drr_u, Dzz_u, NrTotal, NzTotal, dim, i, j, dr, dz, l, m, w, -1.0);
            }
        }
    }

    // Z boundary condition.
    #pragma omp parallel shared(f) private(i, j)
    {
        #pragma omp for schedule(dynamic, 1)
        for (i = ghost; i < ghost + NrInterior; ++i)
        {
            for (j = ghost + NzInterior; j < NzTotal; ++j)
            {
                rhs_z_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 1, i, j, dr, dz, 1, alphaBoundOrder, 0.0, -1.0);
                rhs_z_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, betaBoundOrder, 0.0, -1.0);
                rhs_z_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, aBoundOrder, 0.0, -1.0);
                rhs_z_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 4, i, j, dr, dz, 1, hBoundOrder, 0.0, -1.0);
                rhs_z_pseudo_exp_decay_2nd(f, u, NrTotal, NzTotal, dim, 5, i, j, dr, dz, phiBoundOrder, -1.0, w_idx, m, l);
            }
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
    #pragma omp parallel shared(f) private(i, j)
    {
        #pragma omp for schedule(dynamic, 1)
        for (j = ghost; j < ghost + NzInterior; ++j)
        {
            for (i = ghost + NrInterior; i < NzTotal; ++i)
            {
                rhs_r_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 1, i, j, dr, dz, 1, alphaBoundOrder, 0.0, -1.0);
                rhs_r_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, betaBoundOrder, 0.0, -1.0);
                rhs_r_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, aBoundOrder, 0.0, -1.0);
                rhs_r_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 4, i, j, dr, dz, 1, hBoundOrder, 0.0, -1.0);
                rhs_r_pseudo_exp_decay_2nd(f, u, NrTotal, NzTotal, dim, 5, i, j, dr, dz, phiBoundOrder, -1.0, w_idx, m, l);
            }
        }
    }

    // Top-right corner boundary condition.
    for (i = ghost + NrInterior; i < NrTotal; ++i)
    {
        for (j = ghost + NzInterior; j < NzTotal; ++j)
        {
            rhs_corner_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 1, i, j, dr, dz, 1, alphaBoundOrder, 0.0, -1.0);
            rhs_corner_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, betaBoundOrder, 0.0, -1.0);
            rhs_corner_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, aBoundOrder, 0.0, -1.0);
            rhs_corner_pseudo_robin_2nd(f, u, NrTotal, NzTotal, dim, 4, i, j, dr, dz, 1, hBoundOrder, 0.0, -1.0);
            rhs_corner_pseudo_exp_decay_2nd(f, u, NrTotal, NzTotal, dim, 5, i, j, dr, dz, phiBoundOrder, -1.0, w_idx, m, l);
        }
    }

    // Omega constraint.
    f[w_idx] = 0.0;

    // All done. 
    return;
}