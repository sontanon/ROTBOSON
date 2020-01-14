#include "tools.h"
#include "param.h"

#include "csr_grid_fill.h"
#include "csr_vars.h"
#include "csr_omega_constraint.h"

#define EVEN 1
#define ODD -1

#define DIRICHLET_TYPE	-1
#define EXP_DECAY_TYPE	 0
#define ROBIN_TYPE_1  	 1
#define ROBIN_TYPE_3  	 3

//
// FOR NOW, ONLY SECOND ORDER IS SUPPORTED!
//

void nnz_jacobian_get_nnzs(MKL_INT *p_nnz1, MKL_INT *p_nnz2, MKL_INT *p_nnz3, MKL_INT *p_nnz4,MKL_INT *p_nnz5)
{
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Select order.
	if (order == 4)
	{
		// Interior points plus parity boundaries.
		nnz1 = 30 * NrInterior * NzInterior 
			+ 31 * (NrInterior + NzInterior) + 32
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz2 = 29 * NrInterior * NzInterior 
			+ 31 * (NrInterior + NzInterior) + 32
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz3 = 28 * NrInterior * NzInterior 
			+ 30 * (NrInterior + NzInterior) + 31
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz4 = 46 * NrInterior * NzInterior 
			+ 47 * (NrInterior + NzInterior) + 48
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz5 = 30 * NrInterior * NzInterior 
			+ 31 * (NrInterior + NzInterior) + 32
			+ 4 * (NrInterior + NzInterior) + 24;

		// Select boundary conditions.
		switch (alphaBoundOrder)
		{
			case 0:
				nnz1 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz1 += 9 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (betaBoundOrder)
		{
			case 0:
				nnz2 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz2 += 9 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (hBoundOrder)
		{
			case 0:
				nnz3 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz3 += 9 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (aBoundOrder)
		{
			case 0:
				nnz4 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz4 += 9 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (phiBoundOrder)
		{
			case 0:
				nnz5 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz5 += 10 * (1 + (NrInterior + NzInterior));
				break;
		}
	}
	else
	{
		// Interior points plus parity boundaries.
		nnz1 = 18 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz2 = 17 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz3 = 16 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz4 = 26 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz5 = 18 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;

		// Select boundary conditions.
		switch (alphaBoundOrder)
		{
			case 0:
				nnz1 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz1 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (betaBoundOrder)
		{
			case 0:
				nnz2 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz2 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (hBoundOrder)
		{
			case 0:
				nnz3 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz3 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (aBoundOrder)
		{
			case 0:
				nnz4 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz4 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (phiBoundOrder)
		{
			case 0:
				nnz5 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz5 += 6 * (1 + (NrInterior + NzInterior));
				break;
		}
	}

	*p_nnz1 = nnz1;
	*p_nnz2 = nnz2;
	*p_nnz3 = nnz3;
	*p_nnz4 = nnz4;
	*p_nnz5 = nnz5;

	return;
}

int nnz_jacobian(void)
{
	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	nnz_jacobian_get_nnzs(&nnz1, &nnz2, &nnz3, &nnz4, &nnz5);

	// Total number of nonzeros.
	return nnz1 + nnz2 + nnz3 + nnz4 + nnz5 + 1;
}

void csr_gen_jacobian(csr_matrix A, const double *u, const int print)
{
	// Number of elements we have filled in.
	MKL_INT offset = 0;

	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Calculate nonzeros.
	nnz_jacobian_get_nnzs(&nnz1, &nnz2, &nnz3, &nnz4, &nnz5);

	// Integer arrays.
	MKL_INT r_sym[5] = {EVEN, EVEN, EVEN, EVEN, EVEN};
	MKL_INT z_sym[5] = {EVEN, EVEN, EVEN, EVEN, EVEN};
	MKL_INT bound_order[5] = {alphaBoundOrder, betaBoundOrder, hBoundOrder, aBoundOrder, phiBoundOrder};
	MKL_INT nnzs[5] = {nnz1, nnz2, nnz3, nnz4, nnz5};

	MKL_INT p_center[5] = {0, 0, 0, 0, 0};
	MKL_INT p_bound[5] = {0, 0, 0, 0, 0};

	if (order == 4)
	{
		// TO DO.
	}
	else
	{
		p_center[0] = 18;
		p_center[1] = 17;
		p_center[2] = 16;
		p_center[3] = 26;
		p_center[4] = 18;

		p_bound[0] = p_bound[1] = p_bound[2] = p_bound[3] = 5;
		p_bound[4] = 6;
	}


	// Check for order and fill matrix.
	if (order == 4)
	{
		// TO DO.
	}
	// Second order default.
	else
	{
		csr_grid_fill_2nd(A, 
			NrInterior, NzInterior, dr, dz, u, l, m, 
			r_sym, z_sym, bound_order, nnzs, p_center, p_bound,
			jacobian_2nd_order_variable_omega_cc);
	}

	// FINALLY FILL OMEGA EQUATION OR u5(1,1) CONSTRAINT.
	//printf("ROTBOSON-JACOBIAN: Starting omega...\n");
	offset = nnz1 + nnz2 + nnz3 + nnz4 + nnz5;
	omega_constraint(A.a, A.ia, A.ja, offset, NrTotal, NzTotal, dim, 5, w_idx, fixedPhi, fixedPhiR, fixedPhiZ);
	//printf("ROTBOSON-JACOBIAN: Done omega.\n");

	// FILL LAST ELEMENT.
	A.ia[w_idx + 1] = BASE + A.nnz;

	// PRINT MATRIX.
	if (print)
		csr_print(&A, "a.asc", "ia.asc", "ja.asc");

	// All done.
	return;
}
