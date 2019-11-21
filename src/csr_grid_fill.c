#include "tools.h"
#include "csr_symmetry.h"
#include "csr_pseudo_robin.h"
#include "csr_pseudo_exp_decay.h"

#define EVEN 1
#define ODD -1

#define DIRICHLET_TYPE -1
#define EXP_DECAY_TYPE	0
#define ROBIN_TYPE_1	1
#define ROBIN_TYPE_3	3

void csr_grid_fill_2nd(csr_matrix A, const MKL_INT start_offset, 
		const MKL_INT NrInterior, const MKL_INT NzInterior,
		const double dr, const double dz, 
		const double *u, const MKL_INT l, const double m,
		const MKL_INT gnum, const MKL_INT r_sym, const MKL_INT z_sym, 
		const MKL_INT bound_type, const MKL_INT bound_order,
		void (*f_center)(double *, MKL_INT *, MKL_INT *,
			const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
			const MKL_INT, const MKL_INT, const double, const double,
			const MKL_INT, const double, const double,
			const double, const double, const double, const double, const double,
			const double, const double, const double, const double, const double,
			const double, const double, const double, const double, const double,
			const double, const double, const double, const double, const double,
			const double, const double, const double, const double, const double),
		const MKL_INT p_center)
{
	// Grid extensions.
	MKL_INT NrTotal = NrInterior + 2;
	MKL_INT NzTotal = NzInterior + 2;

	// Grid function size.
	MKL_INT dim = NrTotal * NzTotal;

	// Omega index.
	MKL_INT w_idx = 5 * dim;

	// Start index.
	MKL_INT offset = start_offset;

	// Auxiliary variables.
	MKL_INT i, j, t_offset;

	// Number of nonzero elements on boundary.
	MKL_INT p_bound = 0;

	// Pseudo-exponential decay.
	// Notice that case 2 falls into case 3.
	if (bound_type == EXP_DECAY_TYPE)
	{
		switch (bound_order)
		{
			case 0:
				p_bound = 2;
				break;
			case 1:
				p_bound = 4;
				break;
			case 2:
			case 3:
				p_bound = 5;
				break;
		}
	}
	// Else select pseudo-Robin decay.
	else
	{
		switch (bound_order)
		{
			case 0:
				p_bound = 1;
				break;
			case 1:
				p_bound = 3;
				break;
			case 2:
				p_bound = 4;
				break;
			case 3:
				p_bound = 5;
				break;
		}
	}
	
	// Auxiliary doubles.
	// Fetch omega variable once.
	double chi = u[w_idx];

	// Lower-left corner: diagonal symmetry.
	i = 0;
	j = 0;
	corner_symmetry(A.a, A.ia, A.ja, offset,
		NrTotal, NzTotal, dim,
		gnum, i, j, r_sym, z_sym);
	offset += 2;

	// Set temporary offset.
	t_offset = offset;

	// Fill left-boundary using axis symmetry.
	#pragma omp parallel shared(A) private(offset)
	{
		#pragma omp for schedule(guided)
		for (j = 1; j < NzInterior + 2; ++j)
		{
			// Each j iteration fills 2 elements.
			offset = t_offset + 2 * (j - 1);
			r_symmetry(A.a, A.ia, A.ja, offset,
				NrTotal, NzTotal, dim,
				gnum, i, j, r_sym);
		}
	}

	// We have now filled:
	offset = start_offset + 2 + 2 * NzInterior + 2;

	// Set temporary offset.
	t_offset = offset;

	// Now come the interior points plus the top and bottom boundaries with
	// Robin and equatorial symmetry, respectively.
	#pragma omp parallel shared(A) private(offset, j)
	{
		#pragma omp for schedule(guided)
		for (i = 1; i < NrInterior + 1; ++i)
		{
			// Each iteration of i loop will fill p_center * NzInterior + (2 + p_bound) values.
			offset = t_offset + (i - 1) * (2 + p_bound + p_center * NzInterior);

			// Do bottom boundary first with equatorial symmetry.
			j = 0;;
			z_symmetry(A.a, A.ia, A.ja, offset,
				NrTotal, NzTotal, dim,
				gnum, i, j, z_sym);
			offset += 2;

			// Now loop over interior points.
			for (j = 1; j < NzInterior + 1; ++j)
			{
				// Fill matrix coefficients.
				(*f_center)(A.a, A.ia, A.ja,
					offset, NrTotal, NzTotal, dim,
					i, j, dr, dz, l, m, chi,
					u[      IDX(i-1,j  )], u[      IDX(i  ,j-1)], u[      IDX(i  ,j  )], u[      IDX(i  ,j+1)], u[      IDX(i+1,j  )],
					u[  dim+IDX(i-1,j  )], u[  dim+IDX(i  ,j-1)], u[  dim+IDX(i  ,j  )], u[  dim+IDX(i  ,j+1)], u[  dim+IDX(i+1,j  )],
					u[2*dim+IDX(i-1,j  )], u[2*dim+IDX(i  ,j-1)], u[2*dim+IDX(i  ,j  )], u[2*dim+IDX(i  ,j+1)], u[2*dim+IDX(i+1,j  )],
					u[3*dim+IDX(i-1,j  )], u[3*dim+IDX(i  ,j-1)], u[3*dim+IDX(i  ,j  )], u[3*dim+IDX(i  ,j+1)], u[3*dim+IDX(i+1,j  )],
					u[4*dim+IDX(i-1,j  )], u[4*dim+IDX(i  ,j-1)], u[4*dim+IDX(i  ,j  )], u[4*dim+IDX(i  ,j+1)], u[4*dim+IDX(i+1,j  )]);
				// Increase offset.
				offset += p_center;
			}

			// Now fill top boundary with Robin.
			j = NzInterior + 1;
			if (bound_type == EXP_DECAY_TYPE)
			{
				csr_z_pseudo_exp_decay_2nd(A.a, A.ia, A.ja,
					offset, NrTotal, NzTotal, dim, gnum,
					i, j, dr, dz, bound_order, 
					u, w_idx, m, l);
			}
			else
			{
				csr_z_pseudo_robin_2nd(A.a, A.ia, A.ja,
					offset, NrTotal, NzTotal, dim, gnum,
					i, j, dr, dz, bound_type, bound_order);
			}
			offset += p_bound;
		}
	}
	// At this point we have filled:
	offset = start_offset + 4 + 2 * NzInterior + NrInterior * (p_center * NzInterior + 2 + p_bound);

	// Lower-right corner: equatorial symmetry.
	i = NrInterior + 1;
	j = 0;
	z_symmetry(A.a, A.ia, A.ja, offset,
		NrTotal, NzTotal, dim,
		gnum, i, j, z_sym);
	offset += 2;

	// Set temporary offset.
	t_offset = offset;

	// Robin Boundary.
	#pragma omp parallel shared(A), private(offset)
	{
		#pragma omp for schedule(guided)
		for (j = 1; j < NzInterior + 1; ++j)
		{
			// Each iteration of the loop fills p_bound elements.
			offset = t_offset + p_bound * (j - 1);
			if (bound_type == EXP_DECAY_TYPE)
			{
				csr_r_pseudo_exp_decay_2nd(A.a, A.ia, A.ja,
					offset, NrTotal, NzTotal, dim, gnum,
					i, j, dr, dz, bound_order, 
					u, w_idx, m, l);
			}
			else
			{
				csr_r_pseudo_robin_2nd(A.a, A.ia, A.ja,
					offset, NrTotal, NzTotal, dim, gnum,
					i, j, dr, dz, bound_type, bound_order);
			}
		}
	}

	// At this point we have filled:
	offset = start_offset + 6 + (2 + p_bound) * (NrInterior + NzInterior) + p_center * NrInterior * NzInterior;

	// Upper-right corner: fill with Robin.
	j = NzInterior + 1;
	if (bound_type == EXP_DECAY_TYPE)
	{
		csr_corner_pseudo_exp_decay_2nd(A.a, A.ia, A.ja,
			offset, NrTotal, NzTotal, dim, gnum,
			i, j, dr, dz, bound_order, 
			u, w_idx, m, l);
	}
	else
	{
		csr_corner_pseudo_robin_2nd(A.a, A.ia, A.ja,
			offset, NrTotal, NzTotal, dim, gnum,
			i, j, dr, dz, bound_type, bound_order);
	}

	// All done.
	return;
}
