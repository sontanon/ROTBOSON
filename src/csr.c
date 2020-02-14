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

// Print nnz cumulative sum.
#undef DEBUG

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
		nnz1 = 29 * NrInterior * NzInterior 
			+ 30 * (NrInterior + NzInterior) + 31
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz2 = 28 * NrInterior * NzInterior 
			+ 30 * (NrInterior + NzInterior) + 31
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz3 = 28 * NrInterior * NzInterior 
			+ 30 * (NrInterior + NzInterior) + 31
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz4 = 45 * NrInterior * NzInterior 
			+ 46 * (NrInterior + NzInterior) + 47
			+ 4 * (NrInterior + NzInterior) + 24;
		nnz5 = 29 * NrInterior * NzInterior 
			+ 30 * (NrInterior + NzInterior) + 31
			+ 4 * (NrInterior + NzInterior) + 24;

		// Select boundary conditions.
		switch (alphaBoundOrder)
		{
			case 0:
				nnz1 += 3 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz1 += 9 * (3 + (NrInterior + NzInterior));
				break;
		}
		switch (betaBoundOrder)
		{
			case 0:
				nnz2 += 3 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz2 += 9 * (3 + (NrInterior + NzInterior));
				break;
		}
		switch (hBoundOrder)
		{
			case 0:
				nnz3 += 3 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz3 += 9 * (3 + (NrInterior + NzInterior));
				break;
		}
		switch (aBoundOrder)
		{
			case 0:
				nnz4 += 3 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz4 += 9 * (3 + (NrInterior + NzInterior));
				break;
		}
		switch (phiBoundOrder)
		{
			case 0:
				nnz5 += 3 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz5 += 9 * (3 + (NrInterior + NzInterior));
				break;
		}
	}
	else
	{
		// Interior points plus parity boundaries.
		nnz1 = 17 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz2 = 16 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz3 = 16 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz4 = 25 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;
		nnz5 = 17 * NrInterior * NzInterior + 2 * (NrInterior + NzInterior) + 6;

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
				nnz5 += 5 * (1 + (NrInterior + NzInterior));
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

MKL_INT nnz_jacobian(void)
{
	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	nnz_jacobian_get_nnzs(&nnz1, &nnz2, &nnz3, &nnz4, &nnz5);
#ifdef DEBUG
	printf("nnz1 = %lld,\n", nnz1);
	printf("nnz2 = %lld,\n", nnz2);
	printf("nnz3 = %lld,\n", nnz3);
	printf("nnz4 = %lld,\n", nnz4);
	printf("nnz5 = %lld,\n", nnz5);
	printf("\n");
	printf("nnz1 + nnz2 = %lld,\n", nnz1 + nnz2);
	printf("nnz1 + nnz2 + nnz3 = %lld,\n", nnz1 + nnz2 + nnz3);
	printf("nnz1 + nnz2 + nnz3 + nnz4 = %lld.\n", nnz1 + nnz2 + nnz3 + nnz4);
#endif

	// Total number of nonzeros.
	return nnz1 + nnz2 + nnz3 + nnz4 + nnz5;
}

void csr_gen_jacobian(csr_matrix A, const double *u, const int print)
{
	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Calculate nonzeros.
	nnz_jacobian_get_nnzs(&nnz1, &nnz2, &nnz3, &nnz4, &nnz5);

	// Integer arrays.
	MKL_INT r_sym[5] = {EVEN, EVEN, EVEN, EVEN, EVEN};
	MKL_INT z_sym[5] = {EVEN, EVEN, EVEN, EVEN, EVEN};
	MKL_INT bound_order[5] = {alphaBoundOrder, betaBoundOrder, hBoundOrder, aBoundOrder, phiBoundOrder};
	MKL_INT nnzs[5] = {nnz1, nnz2, nnz3, nnz4, nnz5};

	MKL_INT p_cc[5] = {0, 0, 0, 0, 0};
	MKL_INT p_cs[5] = {0, 0, 0, 0, 0};
	MKL_INT p_sc[5] = {0, 0, 0, 0, 0};
	MKL_INT p_ss[5] = {0, 0, 0, 0, 0};
	MKL_INT p_bound[5] = {0, 0, 0, 0, 0};

	// Set integer arrays according to order.
	if (order == 4)
	{
		p_cc[0] = 29;
		p_cc[1] = 28;
		p_cc[2] = 28;
		p_cc[3] = 45;
		p_cc[4] = 29;

		p_cs[0] = p_sc[0] = 30;
		p_cs[1] = p_sc[1] = 30;
		p_cs[2] = p_sc[2] = 30;
		p_cs[3] = p_sc[3] = 46;
		p_cs[4] = p_sc[4] = 30;

		p_ss[0] = 31;
		p_ss[1] = 31;
		p_ss[2] = 31;
		p_ss[3] = 47;
		p_ss[4] = 31;

		p_bound[0] = p_bound[1] = p_bound[2] = p_bound[3] = 9;
		p_bound[4] = 9;
	}
	else
	{
		p_cc[0] = 17;
		p_cc[1] = 16;
		p_cc[2] = 16;
		p_cc[3] = 25;
		p_cc[4] = 17;

		p_bound[0] = p_bound[1] = p_bound[2] = p_bound[3] = 5;
		p_bound[4] = 5;
	}


	// Check for order and fill matrix.
	if (order == 4)
	{
		csr_grid_fill_4th(A,
			NrInterior, NzInterior, dr, dz, u,l, m,
			r_sym, z_sym, bound_order, nnzs, p_cc, p_cs, p_sc, p_ss, p_bound,
			jacobian_4th_order_variable_omega_cc,
			jacobian_4th_order_variable_omega_cs,
			jacobian_4th_order_variable_omega_sc,
			jacobian_4th_order_variable_omega_ss);
	}
	// Second order default.
	else
	{
		csr_grid_fill_2nd(A, 
			NrInterior, NzInterior, dr, dz, u, l, m, 
			r_sym, z_sym, bound_order, nnzs, p_cc, p_bound,
			jacobian_2nd_order_variable_omega_cc);
	}
	//printf("ROTBOSON-JACOBIAN: Done omega.\n");

	// FILL LAST ELEMENT.
	A.ia[A.nrows + 1] = BASE + A.nnz;

	// PRINT MATRIX.
	if (print)
		csr_print(&A, "a.asc", "ia.asc", "ja.asc");

	// All done.
	return;
}
