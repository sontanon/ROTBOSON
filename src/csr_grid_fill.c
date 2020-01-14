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

void csr_grid_fill_4th(
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
	const MKL_INT p_cc[5],
	const MKL_INT p_cs[5],
	const MKL_INT p_sc[5],
	const MKL_INT p_ss[5],
	const MKL_INT p_bound[5],
	void (*j_cc)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const MKL_INT, const MKL_INT, const double, const double,
		const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT),
	void (*j_cs)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const MKL_INT, const MKL_INT, const double, const double,
		const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT),
	void (*j_sc)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const MKL_INT, const MKL_INT, const double, const double,
		const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT),
	void (*j_ss)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const MKL_INT, const MKL_INT, const double, const double,
		const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT)
)
{
	// Auxiliary integer.
	MKL_INT i = 0, j = 0, k = 0;

	// Number of ghost zones for fourth order.
	MKL_INT ghost = 2;

	// Grid extensions.
	MKL_INT NrTotal = NrInterior + 2 * ghost;
	MKL_INT NzTotal = NzInterior + 2 * ghost;

	// Grid function size.
	MKL_INT dim = NrTotal * NzTotal;

	// Omega index.
	MKL_INT w_idx = 5 * dim;

	// Integer arrays for offsets and start indices.
	MKL_INT offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT t_offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT start_offset[5] = { 0, 0, 0, 0, 0};

	// Set start/initial offsets.
	offset[0] = start_offset[0] = 0;
	offset[1] = start_offset[1] = nnz[0];
	offset[2] = start_offset[2] = nnz[0] + nnz[1];
	offset[3] = start_offset[3] = nnz[0] + nnz[1] + nnz[2];
	offset[4] = start_offset[4] = nnz[0] + nnz[1] + nnz[2] + nnz[3];

	// Fetch xi variable once.
	double xi = u[w_idx];

	// Left band 2 * NzTotal points.
	for (i = 0; i < ghost; ++i)
	{
		// Corner symmetries.
		for (j = 0; j < ghost; ++j)
		{
			corner_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, r_sym[0], z_sym[0]);
			corner_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, r_sym[1], z_sym[1]);
			corner_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, r_sym[2], z_sym[2]);
			corner_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, r_sym[3], z_sym[3]);
			corner_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, r_sym[4], z_sym[4]);
			// Increase offsets by 2.
			for (k = 0; k < 5; ++k)
				offset[k] += 2;
		}

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
			t_offset[k] = offset[k] = start_offset[k] + (i + 1) * (4 + 2 * NzInterior + 4);
	}

	// Now come the interior points plus the top and bottom boundaries 
	// with Robin, semi-onesided stencil and equatorial symmetry.
	#pragma omp parallel shared(A) private(offset, i, j, k)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrInterior + ghost; ++i)
		{
			// Each iteration of i loop will fill p_cc * NzInterior + 4 + p_cs + p_bound values.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + (i - ghost) * (4 + p_bound[k] + p_cs[k] + p_cc[k] * NzInterior);

			// Do bottom boundary first with equatorial symmetry.
			for (j = 0; j < ghost; ++j)
			{
				z_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, z_sym[0]);
				z_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, z_sym[1]);
				z_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, z_sym[2]);
				z_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, z_sym[3]);
				z_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, z_sym[4]);
				// Increase offsets by 2.
				for (k = 0; k < 5; ++k)
					offset[k] += 2;
			}

			// Now loop over interior points.
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Fill matrix coeffients.
				(*j_cc)(A.a, A.ia, A.ja,
					NrTotal, NzTotal, dim, ghost,
					i, j, dr, dz,
					l, m, xi, 
					u[          IDX(i-2,j)], u[          IDX(i-1,j)], u[          IDX(i,j-2)], u[          IDX(i,j-1)], u[          IDX(i,j)], u[          IDX(i,j+1)], u[          IDX(i,j+2)], u[          IDX(i+1,j)], u[          IDX(i+2,j)], 
					u[    dim + IDX(i-2,j)], u[    dim + IDX(i-1,j)], u[    dim + IDX(i,j-2)], u[    dim + IDX(i,j-1)], u[    dim + IDX(i,j)], u[    dim + IDX(i,j+1)], u[    dim + IDX(i,j+2)], u[    dim + IDX(i+1,j)], u[    dim + IDX(i+2,j)], 
					u[2 * dim + IDX(i-2,j)], u[2 * dim + IDX(i-1,j)], u[2 * dim + IDX(i,j-2)], u[2 * dim + IDX(i,j-1)], u[2 * dim + IDX(i,j)], u[2 * dim + IDX(i,j+1)], u[2 * dim + IDX(i,j+2)], u[2 * dim + IDX(i+1,j)], u[2 * dim + IDX(i+2,j)], 
					u[3 * dim + IDX(i-2,j)], u[3 * dim + IDX(i-1,j)], u[3 * dim + IDX(i,j-2)], u[3 * dim + IDX(i,j-1)], u[3 * dim + IDX(i,j)], u[3 * dim + IDX(i,j+1)], u[3 * dim + IDX(i,j+2)], u[3 * dim + IDX(i+1,j)], u[3 * dim + IDX(i+2,j)], 
					u[4 * dim + IDX(i-2,j)], u[4 * dim + IDX(i-1,j)], u[4 * dim + IDX(i,j-2)], u[4 * dim + IDX(i,j-1)], u[4 * dim + IDX(i,j)], u[4 * dim + IDX(i,j+1)], u[4 * dim + IDX(i,j+2)], u[4 * dim + IDX(i+1,j)], u[4 * dim + IDX(i+2,j)], 
					offset[0], offset[1], offset[2], offset[3], offset[4]);
				// Increase offsets.
				for (k = 0; k < 5; ++k)
					offset[k] += p_cc[k];
			}

			// Fill cs stencil.
			j = ghost + NzInterior;
			(*j_cs)(A.a, A.ia, A.ja,
				NrTotal, NzTotal, dim, ghost,
				i, j, dr, dz,
				l, m, xi, 
				u[          IDX(i-2,j)], u[          IDX(i-1,j)], u[          IDX(i,j-4)], u[          IDX(i,j-3)], u[          IDX(i,j-2)], u[          IDX(i,j-1)], u[          IDX(i,j)], u[          IDX(i,j+1)], u[          IDX(i+1,j)], u[          IDX(i+2,j)],
				u[    dim + IDX(i-2,j)], u[    dim + IDX(i-1,j)], u[    dim + IDX(i,j-4)], u[    dim + IDX(i,j-3)], u[    dim + IDX(i,j-2)], u[    dim + IDX(i,j-1)], u[    dim + IDX(i,j)], u[    dim + IDX(i,j+1)], u[    dim + IDX(i+1,j)], u[    dim + IDX(i+2,j)],
				u[2 * dim + IDX(i-2,j)], u[2 * dim + IDX(i-1,j)], u[2 * dim + IDX(i,j-4)], u[2 * dim + IDX(i,j-3)], u[2 * dim + IDX(i,j-2)], u[2 * dim + IDX(i,j-1)], u[2 * dim + IDX(i,j)], u[2 * dim + IDX(i,j+1)], u[2 * dim + IDX(i+1,j)], u[2 * dim + IDX(i+2,j)],
				u[3 * dim + IDX(i-2,j)], u[3 * dim + IDX(i-1,j)], u[3 * dim + IDX(i,j-4)], u[3 * dim + IDX(i,j-3)], u[3 * dim + IDX(i,j-2)], u[3 * dim + IDX(i,j-1)], u[3 * dim + IDX(i,j)], u[3 * dim + IDX(i,j+1)], u[3 * dim + IDX(i+1,j)], u[3 * dim + IDX(i+2,j)],
				u[4 * dim + IDX(i-2,j)], u[4 * dim + IDX(i-1,j)], u[4 * dim + IDX(i,j-4)], u[4 * dim + IDX(i,j-3)], u[4 * dim + IDX(i,j-2)], u[4 * dim + IDX(i,j-1)], u[4 * dim + IDX(i,j)], u[4 * dim + IDX(i,j+1)], u[4 * dim + IDX(i+1,j)], u[4 * dim + IDX(i+2,j)],
				offset[0], offset[1], offset[2], offset[3], offset[4]);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_cs[k];

			// Last point with top boundary condtion.
			j = NzTotal - 1;
			z_robin_4th_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			z_robin_4th_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			z_robin_4th_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			z_robin_4th_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			z_decay_4th_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}
	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 16 + 4 * (NrInterior + NzInterior) + p_cc[k] * NrInterior * NzInterior + NrInterior * (p_cs[k] + p_bound[k]);

	// Right band.
	i = ghost + NrInterior;
	// Equatorial symmetry.
	for (j = 0; j < ghost; ++j)
	{
		z_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, z_sym[0]);
		z_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, z_sym[1]);
		z_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, z_sym[2]);
		z_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, z_sym[3]);
		z_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, z_sym[4]);
		// Increase offsets by 2.
		for (k = 0; k < 5; ++k)
			t_offset[k] = offset[k] += 2;
	}

	// Main sc stencil.
	#pragma omp parallel shared(A), private(j, k, offset)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ghost; j < ghost + NzInterior; ++j)
		{
			// Each iteration of the j loop fills p_sc elements.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + p_sc[k] * (j - ghost);

			// Fill matrix coefficients.
			(*j_sc)(A.a, A.ia, A.ja,
				NrTotal, NzTotal, dim, ghost,
				i, j, dr, dz,
				l, m, xi, 
				u[          IDX(i-4,j)], u[          IDX(i-3,j)], u[          IDX(i-2,j)], u[          IDX(i-1,j)], u[          IDX(i,j-2)], u[          IDX(i,j-1)], u[          IDX(i,j)], u[          IDX(i,j+1)], u[          IDX(i,j+2)],  u[          IDX(i+1,j)],
				u[    dim + IDX(i-4,j)], u[    dim + IDX(i-3,j)], u[    dim + IDX(i-2,j)], u[    dim + IDX(i-1,j)], u[    dim + IDX(i,j-2)], u[    dim + IDX(i,j-1)], u[    dim + IDX(i,j)], u[    dim + IDX(i,j+1)], u[    dim + IDX(i,j+2)],  u[    dim + IDX(i+1,j)],
				u[2 * dim + IDX(i-4,j)], u[2 * dim + IDX(i-3,j)], u[2 * dim + IDX(i-2,j)], u[2 * dim + IDX(i-1,j)], u[2 * dim + IDX(i,j-2)], u[2 * dim + IDX(i,j-1)], u[2 * dim + IDX(i,j)], u[2 * dim + IDX(i,j+1)], u[2 * dim + IDX(i,j+2)],  u[2 * dim + IDX(i+1,j)],
				u[3 * dim + IDX(i-4,j)], u[3 * dim + IDX(i-3,j)], u[3 * dim + IDX(i-2,j)], u[3 * dim + IDX(i-1,j)], u[3 * dim + IDX(i,j-2)], u[3 * dim + IDX(i,j-1)], u[3 * dim + IDX(i,j)], u[3 * dim + IDX(i,j+1)], u[3 * dim + IDX(i,j+2)],  u[3 * dim + IDX(i+1,j)],
				u[4 * dim + IDX(i-4,j)], u[4 * dim + IDX(i-3,j)], u[4 * dim + IDX(i-2,j)], u[4 * dim + IDX(i-1,j)], u[4 * dim + IDX(i,j-2)], u[4 * dim + IDX(i,j-1)], u[4 * dim + IDX(i,j)], u[4 * dim + IDX(i,j+1)], u[4 * dim + IDX(i,j+2)],  u[4 * dim + IDX(i+1,j)],
				offset[0], offset[1], offset[2], offset[3], offset[4]);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_sc[k];
		}
	}

	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 20 + 4 * (NrInterior + NzInterior) + p_cc[k] * NrInterior * NrInterior + p_cs[k] * NrInterior + p_sc[k] * NzInterior + p_bound[k] * NrInterior;

	// Main ss stencil.
	j = ghost + NzInterior;
	// Fill matrix coefficients.
	(*j_ss)(A.a, A.ia, A.ja,
		NrTotal, NzTotal, dim, ghost,
		i, j, dr, dz,
		l, m, xi, 
		u[          IDX(i-4,j)], u[          IDX(i-3,j)], u[          IDX(i-2,j)], u[          IDX(i-1,j)], u[          IDX(i,j-4)], u[          IDX(i,j-3)], u[          IDX(i,j-2)], u[          IDX(i,j-1)], u[          IDX(i,j)], u[          IDX(i,j+1)], u[          IDX(i+1,j)],
		u[    dim + IDX(i-4,j)], u[    dim + IDX(i-3,j)], u[    dim + IDX(i-2,j)], u[    dim + IDX(i-1,j)], u[    dim + IDX(i,j-4)], u[    dim + IDX(i,j-3)], u[    dim + IDX(i,j-2)], u[    dim + IDX(i,j-1)], u[    dim + IDX(i,j)], u[    dim + IDX(i,j+1)], u[    dim + IDX(i+1,j)],
		u[2 * dim + IDX(i-4,j)], u[2 * dim + IDX(i-3,j)], u[2 * dim + IDX(i-2,j)], u[2 * dim + IDX(i-1,j)], u[2 * dim + IDX(i,j-4)], u[2 * dim + IDX(i,j-3)], u[2 * dim + IDX(i,j-2)], u[2 * dim + IDX(i,j-1)], u[2 * dim + IDX(i,j)], u[2 * dim + IDX(i,j+1)], u[2 * dim + IDX(i+1,j)],
		u[3 * dim + IDX(i-4,j)], u[3 * dim + IDX(i-3,j)], u[3 * dim + IDX(i-2,j)], u[3 * dim + IDX(i-1,j)], u[3 * dim + IDX(i,j-4)], u[3 * dim + IDX(i,j-3)], u[3 * dim + IDX(i,j-2)], u[3 * dim + IDX(i,j-1)], u[3 * dim + IDX(i,j)], u[3 * dim + IDX(i,j+1)], u[3 * dim + IDX(i+1,j)],
		u[4 * dim + IDX(i-4,j)], u[4 * dim + IDX(i-3,j)], u[4 * dim + IDX(i-2,j)], u[4 * dim + IDX(i-1,j)], u[4 * dim + IDX(i,j-4)], u[4 * dim + IDX(i,j-3)], u[4 * dim + IDX(i,j-2)], u[4 * dim + IDX(i,j-1)], u[4 * dim + IDX(i,j)], u[4 * dim + IDX(i,j+1)], u[4 * dim + IDX(i+1,j)],
		offset[0], offset[1], offset[2], offset[3], offset[4]);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_ss[k];

	// Boundary conditions.
	j = NzTotal - 1;
	z_so_robin_4th_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
	z_so_robin_4th_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
	z_so_robin_4th_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
	z_so_robin_4th_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
	z_so_decay_4th_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_bound[k];

	// Left-most band.
	i = NrTotal - 1;
	// Equatorial symmetry.
	for (j = 0; j < ghost; ++j)
	{
		z_symmetry(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, ghost, z_sym[0]);
		z_symmetry(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, ghost, z_sym[1]);
		z_symmetry(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, ghost, z_sym[2]);
		z_symmetry(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, ghost, z_sym[3]);
		z_symmetry(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, 4, i, j, ghost, z_sym[4]);
		// Increase offsets by 2.
		for (k = 0; k < 5; ++k)
			t_offset[k] = offset[k] += 2;
	}

	// Boundary.
	#pragma omp parallel shared(A), private(j, k, offset)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ghost; j < ghost + NzInterior; ++j)
		{
			// Each iteration of the j loop fills p_bound elements.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + p_bound[k] * (j - ghost);

			// Fill matrix coefficients.
			r_robin_4th_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			r_robin_4th_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			r_robin_4th_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			r_robin_4th_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			r_decay_4th_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}

	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 24 + 4 * (NrInterior + NzInterior) + p_cc[k] * NrInterior * NrInterior + p_cs[k] * NrInterior + p_sc[k] * NzInterior + p_bound[k] * (NrInterior + NzInterior) + p_ss[k] + p_bound[k];

	// Second to last point.
	j = ghost + NzInterior;
	// Fill matrix coefficients.
	r_so_robin_4th_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
	r_so_robin_4th_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
	r_so_robin_4th_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
	r_so_robin_4th_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
	r_so_decay_4th_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_bound[k];

	// Boundary conditions.
	j = NzTotal - 1;
	corner_robin_4th_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
	corner_robin_4th_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
	corner_robin_4th_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
	corner_robin_4th_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
	corner_decay_4th_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_bound[k];

	// All done.
	return;
}

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
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
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
	MKL_INT NrTotal = NrInterior + 2 * ghost;
	MKL_INT NzTotal = NzInterior + 2 * ghost;

	// Grid function size
	MKL_INT dim = NrTotal * NzTotal;

	// Omega index.
	MKL_INT w_idx = 5 * dim;

	// Integer arrays for offsets and start indices.
	MKL_INT offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT t_offset[5] = { 0, 0, 0, 0, 0 };
	MKL_INT start_offset[5] = { 0, 0, 0, 0, 0};

	// Set start/initial offsets.
	offset[0] = start_offset[0] = 0;
	offset[1] = start_offset[1] = nnz[0];
	offset[2] = start_offset[2] = nnz[0] + nnz[1];
	offset[3] = start_offset[3] = nnz[0] + nnz[1] + nnz[2];
	offset[4] = start_offset[4] = nnz[0] + nnz[1] + nnz[2] + nnz[3];

	// Fetch xi variable once.
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
		for (i = ghost; i < NrInterior + ghost; ++i)
		{
			// Each iteration of i loop will fill p_center * NzInterior + (2 + p_bound) values.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + (i - ghost) * (2 + p_bound[k] + p_center[k] * NzInterior);

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
			for (j = ghost; j < NzInterior + ghost; ++j)
			{
				// Fill matrix coefficients.
				(*j_cc)(A.a, A.ia, A.ja, 
					NrTotal, NzTotal, dim, ghost, i, j,
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
			j = NzInterior + ghost;
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			z_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			z_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}
	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 4 + 2 * (NrInterior + NzInterior) + p_center[k] * NrInterior * NzInterior + p_bound[k] * NrInterior;

	// Lower-right corner: equatorial symmetry.
	i = NrInterior + ghost;
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
		for (j = ghost; j < NzInterior + 1; ++j)
		{
			// Each iteration of the loop fills p_bound elements.
			for (k = 0; k < 5; ++k)
				offset[k] = t_offset[k] + p_bound[k] * (j - ghost);

			r_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
			r_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
			r_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
			// Increase offsets.
			for (k = 0; k < 5; ++k)
				offset[k] += p_bound[k];
		}
	}

	// At this point we have filled:
	for (k = 0; k < 5; ++k)
		offset[k] = start_offset[k] + 6 + (2 + p_bound[k]) * (NrInterior + NzInterior) + p_center[k] * NrInterior * NzInterior;

	// Upper-right corner: fill with Robin.
	j = NzInterior + ghost;
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[0], NrTotal, NzTotal, dim, 0, i, j, dr, dz, 1, bound_order[0]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[1], NrTotal, NzTotal, dim, 1, i, j, dr, dz, 3, bound_order[1]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[2], NrTotal, NzTotal, dim, 2, i, j, dr, dz, 1, bound_order[2]);
	corner_robin_2nd_order(A.a, A.ia, A.ja, offset[3], NrTotal, NzTotal, dim, 3, i, j, dr, dz, 1, bound_order[3]);
	corner_decay_2nd_order(A.a, A.ia, A.ja, offset[4], NrTotal, NzTotal, dim, ghost, 4, i, j, dr, dz, u, w_idx, m, l);
	// Increase offsets.
	for (k = 0; k < 5; ++k)
		offset[k] += p_bound[k];

	// All done.
	return;
}