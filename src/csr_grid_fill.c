#include "tools.h"
#include "csr_symmetry.h"
#include "csr_robin.h"
#include "csr_exp_decay.h"

#define EVEN 1
#define ODD -1

#define DIRICHLET_TYPE -1
#define EXP_DECAY_TYPE	0
#define ROBIN_TYPE_1	1
#define ROBIN_TYPE_3	3

void csr_grid_fill_2nd(
	csr_matrix A,
	const MKL_INT NrInterior, 
	const MKL_INT NzInterior, 
	const double dr, 
	const double dz,
	const double *u,
	const MKL_INT l,
	const double m,
	const MKL_INT r_sym[5],
	const MKL_INT z_sym[5],
	const MKL_INT bound_order[5],
	const MKL_INT nnz[5],
	const MKL_INT p_center[5],
	const MKL_INT p_bound[5],
	void (*j_cc)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const double, const double, const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT)
)
{
	// Auxiliary integers.
	MKL_INT i = 0, j = 0, k = 0;

	// Number of ghost zones for second order.
	MKL_INT ghost = 1;

	// Grid extensions.
	MKL_INT NrTotal = NrInterior + 2;
	MKL_INT NzTotal = NzInterior + 2;

	// Grid function size
	MKL_INT dim = NrTotal * NzTotal;

	// Omega index.
	MKL_INT w_idx = 5 * dim;

	// Integer arrays for offsets and start indices.
	MKL_INT offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT t_offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT start_offset[5] = { 0, 0, 0, 0, 0};

	// Set start/initial offsets.
	/*
	for (k = 1; k < 5; ++k) 
		offset[k] = start_offset[k] = nnz[k - 1];
	*/
	offset[0] = start_offset[0] = 0;
	offset[1] = start_offset[1] = nnz[0];
	offset[2] = start_offset[2] = nnz[0] + nnz[1];
	offset[3] = start_offset[3] = nnz[0] + nnz[1] + nnz[2];
	offset[4] = start_offset[4] = nnz[0] + nnz[1] + nnz[2] + nnz[3];

	// Fetch omega variable once.
	double xi = u[w_idx];

	// Lower-left corner: diagonal symmetry.
	i = 0;
	j = 0;
	corner_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, r_sym[0], z_sym[0]);
	corner_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, r_sym[1], z_sym[1]);
	corner_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, r_sym[2], z_sym[2]);
	corner_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, r_sym[3], z_sym[3]);
	corner_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, r_sym[4], z_sym[4]);
	// Increase offsets by 2.
	for (k = 0; k < 5; ++k)
		offset[k] += 2;

	// Set temporary offsets.
	for (k = 0; k < 5; ++k)
		t_offset[k] = offset[k];

	// Fill left-boundary using axis symmetry.
	#pragma omp parallel shared(A) private(offset)
	{
		#pragma omp for schedule(dynamic, 1) private(k, j)
		for (j = ghost; j < NzTotal; ++j)
		{
			// Each j iteration fills 2 elements.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + 2 * (j - 1);

			r_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, r_sym[0]);
			r_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, r_sym[1]);
			r_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, r_sym[2]);
			r_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, r_sym[3]);
			r_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, r_sym[4]);
			// Increase offsets by 2.
			for (k = 0; k < 5; ++k)
				offset[k] += 2;
		}
	}

	// We have now filled:
	for (k = 0; k < 5; ++k)
		t_offset[k] = offset[k] = start_offset[k] + 2 + 2 * NzInterior + 2;

	// Now come the interior points plus the top and bottom boundaries with
	// Robin and equatorial symmetry, respectively.
	#pragma omp parallel shared(A) private(offset, i, j, k)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 1; i < NrInterior + 1; ++i)
		{
			// Each iteration of i loop will fill p_center * NzInterior + (2 + p_bound) values.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + (i - 1) * (2 + p_bound[k] + p_center[k] * NzInterior);

			// Do bottom boundary first with equatorial symmetry.
			j = 0;
			z_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, z_sym[0]);
			z_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, z_sym[1]);
			z_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, z_sym[2]);
			z_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, z_sym[3]);
			z_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, z_sym[4]);
			// Increase offsets by 2.
			for (k = 0; k < 5; ++k)
				offset[k] += 2;

			// Now loop over interior points.
			for (j = 1; j < NzInterior + 1; ++j)
			{
				// Fill matrix coefficients.
				(*j_cc)(A.a, A.ia, A.ja, 
					NrTotal, NzTotal, dim, i, j,
					dr, dz, l, m, xi,
					u[      IDX(i-1,j  )], u[      IDX(i  ,j-1)], u[      IDX(i  ,j  )], u[      IDX(i  ,j+1)], u[      IDX(i+1,j  )],
					u[  dim+IDX(i-1,j  )], u[  dim+IDX(i  ,j-1)], u[  dim+IDX(i  ,j  )], u[  dim+IDX(i  ,j+1)], u[  dim+IDX(i+1,j  )],
					u[2*dim+IDX(i-1,j  )], u[2*dim+IDX(i  ,j-1)], u[2*dim+IDX(i  ,j  )], u[2*dim+IDX(i  ,j+1)], u[2*dim+IDX(i+1,j  )],
					u[3*dim+IDX(i-1,j  )], u[3*dim+IDX(i  ,j-1)], u[3*dim+IDX(i  ,j  )], u[3*dim+IDX(i  ,j+1)], u[3*dim+IDX(i+1,j  )],
					u[4*dim+IDX(i-1,j  )], u[4*dim+IDX(i  ,j-1)], u[4*dim+IDX(i  ,j  )], u[4*dim+IDX(i  ,j+1)], u[4*dim+IDX(i+1,j  )],
					offset[0], offset[1], offset[2], offset[3], offset[4]);
				// Increase offsets.
				for (k = 0; k < 5; ++k)
					offset[k] += p_center[k];
			}

			// Now fill top boundary with Robin.
			j = NzInterior + 1;
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			z_exp_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, dr, dz);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}
	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 4 + 2 * NzInterior + NrInterior * (p_center[k] * NzInterior + 2 + p_bound[k]);

	// Lower-right corner: equatorial symmetry.
	i = NrInterior + 1;
	j = 0;
	z_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, z_sym[0]);
	z_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, z_sym[1]);
	z_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, z_sym[2]);
	z_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, z_sym[3]);
	z_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, z_sym[4]);
	// Increase offsets by 2.
	for (k = 0; k < 5; ++k)
		t_offset[k] = offset[k] += 2;

	// Robin Boundary.
	#pragma omp parallel shared(A), private(j, k, offset)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = 1; j < NzInterior + 1; ++j)
		{
			// Each iteration of the loop fills p_bound elements.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + p_bound[k] * (j - 1);

			r_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			r_exp_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, dr, dz);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}

	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 6 + (2 + p_bound[k]) * (NrInterior + NzInterior) + p_center[k] * NrInterior * NzInterior;

	// Upper-right corner: fill with Robin.
	j = NzInterior + 1;
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
	corner_exp_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, dr, dz);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_bound[k];

	// All done.
	return;

}