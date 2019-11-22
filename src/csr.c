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

int nnz_jacobian(void)
{
	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Select order.
	if (order == 4)
	{
		// TODO.
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
				nnz1 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz1 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz1 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (betaBoundOrder)
		{
			case 0:
				nnz2 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz2 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz2 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz2 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (hBoundOrder)
		{
			case 0:
				nnz3 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz3 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz3 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz3 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (aBoundOrder)
		{
			case 0:
				nnz4 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz4 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz4 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz4 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (phiBoundOrder)
		{
			case 0:
				nnz5 += 2 * (1 + (NrInterior + NzInterior));
				break;
			case 1:
				nnz5 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
			case 3:
				nnz5 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
	}

	// Total number of nonzeros.
	return nnz1 + nnz2 + nnz3 + nnz4 + nnz5 + 1;
}

void nnz_jacobian_get_nnzs(MKL_INT *p_nnz1, MKL_INT *p_nnz2, MKL_INT *p_nnz3, MKL_INT *p_nnz4,MKL_INT *p_nnz5)
{
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Select order.
	if (order == 4)
	{
		// TODO.
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
				nnz1 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz1 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz1 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (betaBoundOrder)
		{
			case 0:
				nnz2 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz2 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz2 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz2 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (hBoundOrder)
		{
			case 0:
				nnz3 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz3 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz3 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz3 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (aBoundOrder)
		{
			case 0:
				nnz4 += 1 + (NrInterior + NzInterior);
				break;
			case 1:
				nnz4 += 3 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
				nnz4 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 3:
				nnz4 += 5 * (1 + (NrInterior + NzInterior));
				break;
		}
		switch (phiBoundOrder)
		{
			case 0:
				nnz5 += 2 * (1 + (NrInterior + NzInterior));
				break;
			case 1:
				nnz5 += 4 * (1 + (NrInterior + NzInterior));
				break;
			case 2:
			case 3:
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

void csr_gen_jacobian(csr_matrix A, const double *u, const int print)
{
	// Number of elements we have filled in.
	MKL_INT offset = 0;

	// Number of nonzero elements per grid function.
	MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0;

	// Calculate nonzeros.
	nnz_jacobian_get_nnzs(&nnz1, &nnz2, &nnz3, &nnz4, &nnz5);

	// Number of points in interior equation.
	MKL_INT p_center;

	// Grid number.
	MKL_INT gnum;

	// Check for order and fill matrix.
	if (order == 4)
	{
		// TODO.
	}
	// Second order default.
	else
	{
		// START BY FILLING IN u1 = alpha.
		//printf("ROTBOSON-JACOBIAN: Starting u1...\n");
		gnum = 1;
		offset = 0;
		p_center = 18;
		csr_grid_fill_2nd(A, offset, NrInterior, NzInterior, dr, dz, u, l, m,
				gnum, EVEN, EVEN, ROBIN_TYPE_1, alphaBoundOrder, f1, p_center);
		//printf("ROTBOSON-JACOBIAN: Done u1.\n");

		// NEXT IS u2 = beta.
		//printf("ROTBOSON-JACOBIAN: Starting u2...\n");
		gnum = 2;
		offset = nnz1;
		p_center = 17;
		csr_grid_fill_2nd(A, offset, NrInterior, NzInterior, dr, dz, u, l, m,
				gnum, EVEN, EVEN, ROBIN_TYPE_3, betaBoundOrder, f2, p_center);

		//printf("ROTBOSON-JACOBIAN: Done u2.\n");

		// NEXT IS u3 = h.
		//printf("ROTBOSON-JACOBIAN: Starting u3...\n");
		gnum = 3;
		p_center = 16;
		offset = nnz1 + nnz2;
		csr_grid_fill_2nd(A, offset, NrInterior, NzInterior, dr, dz, u, l, m,
				gnum, EVEN, EVEN, ROBIN_TYPE_1, hBoundOrder, f3, p_center);
		//printf("ROTBOSON-JACOBIAN: Done u3.\n");

		// NEXT IS u4 = a.
		//printf("ROTBOSON-JACOBIAN: Starting u4...\n");
		gnum = 4;
		p_center = 26;
		offset = nnz1 + nnz2 + nnz3;
		csr_grid_fill_2nd(A, offset, NrInterior, NzInterior, dr, dz, u, l, m,
				gnum, EVEN, EVEN, ROBIN_TYPE_1, aBoundOrder, f4, p_center);
		//printf("ROTBOSON-JACOBIAN: Done u4.\n");

		// NEXT IS u5 = phi / r**l.
		//printf("ROTBOSON-JACOBIAN: Starting u5...\n");
		gnum = 5;
		p_center = 18;
		offset = nnz1 + nnz2 + nnz3 + nnz4;
		csr_grid_fill_2nd(A, offset, NrInterior, NzInterior, dr, dz, u, l, m,
				gnum, EVEN, EVEN, EXP_DECAY_TYPE, phiBoundOrder, f5, p_center);
		//printf("ROTBOSON-JACOBIAN: Done u5.\n");
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
