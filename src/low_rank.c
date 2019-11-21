#include "tools.h"
#include "param.h"
#include "pardiso_param.h"

void diff_gen(void)
{
	// Auxiliary integers.
	int i, j, offset1, offset2, offset3, offset4, offset5, offset6, offset7, offset8, offset9;

	// Number of different elements after update.
	int ndiff = 0;

	// Elements at boundary.
	int p_bound = 0;

	// Base ndiff with Dirichlet.
	if (order == 4)
	{
		// TODO.
	}
	else
	{
		// Main interior points.
		// 18 + 22 + 16 + 17 + 18 = 91.
		ndiff = 91 * NrInterior * NzInterior;

		// Add phiBoundOrder.
		switch (phiBoundOrder)
		{
			case 0:
				ndiff += (NrInterior + NzInterior) + 1;
				p_bound = 1;
				break;
			case 1:
			case 2:
			case 3:
				ndiff += 2 * (NrInterior + NzInterior) + 2;
				p_bound = 2;
				break;
		}
	}

	// Allocate memory for diff array.
	diff = (MKL_INT *)SAFE_MALLOC(sizeof(MKL_INT) * (2 * ndiff + 1));

	// First element of diff is ndiff.
	diff[0] = ndiff;

	// Fill in rest of array.
	if (order == 4)
	{
		// TODO.
	}
	else
	{
		// Interior points.
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j,\
				offset1, offset2, offset3,\
				offset4, offset5, offset6)
		for (i = 1; i < NrInterior + 1; i++)
		{
			// log_alpha: 18 different points.
			offset1 = 1 + 36 * NzInterior * (i - 1);
			for (j = 1; j < NzInterior + 1; j++)
			{
				// Row indices are all row IDX(i, j).
				diff[offset1 +  0] = IDX(i, j);
				diff[offset1 +  2] = IDX(i, j);
				diff[offset1 +  4] = IDX(i, j);
				diff[offset1 +  6] = IDX(i, j);
				diff[offset1 +  8] = IDX(i, j);
				diff[offset1 + 10] = IDX(i, j);
				diff[offset1 + 12] = IDX(i, j);
				diff[offset1 + 14] = IDX(i, j);
				diff[offset1 + 16] = IDX(i, j);
				diff[offset1 + 18] = IDX(i, j);
				diff[offset1 + 20] = IDX(i, j);
				diff[offset1 + 22] = IDX(i, j);
				diff[offset1 + 24] = IDX(i, j);
				diff[offset1 + 26] = IDX(i, j);
				diff[offset1 + 28] = IDX(i, j);
				diff[offset1 + 30] = IDX(i, j);
				diff[offset1 + 32] = IDX(i, j);
				diff[offset1 + 34] = IDX(i, j);
				// Column indices.
				diff[offset1 +  1] = IDX(i - 1, j    );
				diff[offset1 +  3] = IDX(i    , j - 1);
				diff[offset1 +  5] = IDX(i    , j    );
				diff[offset1 +  7] = IDX(i    , j + 1);
				diff[offset1 +  9] = IDX(i + 1, j    );
				diff[offset1 + 11] = dim + IDX(i - 1, j    );
				diff[offset1 + 13] = dim + IDX(i    , j - 1);
				diff[offset1 + 15] = dim + IDX(i    , j    );
				diff[offset1 + 17] = dim + IDX(i    , j + 1);
				diff[offset1 + 19] = dim + IDX(i + 1, j    );
				diff[offset1 + 21] = 2 * dim + IDX(i - 1, j    );
				diff[offset1 + 23] = 2 * dim + IDX(i    , j - 1);
				diff[offset1 + 25] = 2 * dim + IDX(i    , j    );
				diff[offset1 + 27] = 2 * dim + IDX(i    , j + 1);
				diff[offset1 + 29] = 2 * dim + IDX(i + 1, j    );
				diff[offset1 + 31] = 3 * dim + IDX(i    , j    );
				diff[offset1 + 33] = 4 * dim + IDX(i    , j    );
				diff[offset1 + 35] = 5 * dim;
			}

			// beta: 17 different points.
			offset2 = 1 + 36 * NrInterior * NzInterior + 34 * NzInterior * (i - 1);
			for (j = 1; j < NzInterior + 1; j++)
			{
				// Row indices are all row dim + IDX(i, j).
				diff[offset2 +  0] = dim + IDX(i, j);
				diff[offset2 +  2] = dim + IDX(i, j);
				diff[offset2 +  4] = dim + IDX(i, j);
				diff[offset2 +  6] = dim + IDX(i, j);
				diff[offset2 +  8] = dim + IDX(i, j);
				diff[offset2 + 10] = dim + IDX(i, j);
				diff[offset2 + 12] = dim + IDX(i, j);
				diff[offset2 + 14] = dim + IDX(i, j);
				diff[offset2 + 16] = dim + IDX(i, j);
				diff[offset2 + 18] = dim + IDX(i, j);
				diff[offset2 + 20] = dim + IDX(i, j);
				diff[offset2 + 22] = dim + IDX(i, j);
				diff[offset2 + 24] = dim + IDX(i, j);
				diff[offset2 + 26] = dim + IDX(i, j);
				diff[offset2 + 28] = dim + IDX(i, j);
				diff[offset2 + 30] = dim + IDX(i, j);
				diff[offset2 + 32] = dim + IDX(i, j);
				// Column indices.
				diff[offset2 +  1] = IDX(i - 1, j    );
				diff[offset2 +  3] = IDX(i    , j - 1);
				diff[offset2 +  5] = IDX(i    , j + 1);
				diff[offset2 +  7] = IDX(i + 1, j    );
				diff[offset2 +  9] = dim + IDX(i - 1, j    );
				diff[offset2 + 11] = dim + IDX(i    , j - 1);
				diff[offset2 + 13] = dim + IDX(i    , j    );
				diff[offset2 + 15] = dim + IDX(i    , j + 1);
				diff[offset2 + 17] = dim + IDX(i + 1, j    );
				diff[offset2 + 19] = 2 * dim + IDX(i - 1, j    );
				diff[offset2 + 21] = 2 * dim + IDX(i    , j - 1);
				diff[offset2 + 23] = 2 * dim + IDX(i    , j    );
				diff[offset2 + 25] = 2 * dim + IDX(i    , j + 1);
				diff[offset2 + 27] = 2 * dim + IDX(i + 1, j    );
				diff[offset2 + 29] = 3 * dim + IDX(i    , j    );
				diff[offset2 + 31] = 4 * dim + IDX(i    , j    );
				diff[offset2 + 33] = 5 * dim;
			}

			// log_h: 16 different points.
			offset3 = 1 + 70 * NrInterior * NzInterior + 32 * NzInterior * (i - 1); 
			for (j = 1; j < NzInterior + 1; j++)
			{
				// Row indices are all row 2 * dim + IDX(i, j).
				diff[offset3 +  0] = 2 * dim + IDX(i, j);
				diff[offset3 +  2] = 2 * dim + IDX(i, j);
				diff[offset3 +  4] = 2 * dim + IDX(i, j);
				diff[offset3 +  6] = 2 * dim + IDX(i, j);
				diff[offset3 +  8] = 2 * dim + IDX(i, j);
				diff[offset3 + 10] = 2 * dim + IDX(i, j);
				diff[offset3 + 12] = 2 * dim + IDX(i, j);
				diff[offset3 + 14] = 2 * dim + IDX(i, j);
				diff[offset3 + 16] = 2 * dim + IDX(i, j);
				diff[offset3 + 18] = 2 * dim + IDX(i, j);
				diff[offset3 + 20] = 2 * dim + IDX(i, j);
				diff[offset3 + 22] = 2 * dim + IDX(i, j);
				diff[offset3 + 24] = 2 * dim + IDX(i, j);
				diff[offset3 + 26] = 2 * dim + IDX(i, j);
				diff[offset3 + 28] = 2 * dim + IDX(i, j);
				diff[offset3 + 30] = 2 * dim + IDX(i, j);
				// Column indices.
				diff[offset3 +  1] = IDX(i - 1, j    );
				diff[offset3 +  3] = IDX(i    , j - 1);
				diff[offset3 +  5] = IDX(i    , j    );
				diff[offset3 +  7] = IDX(i    , j + 1);
				diff[offset3 +  9] = IDX(i + 1, j    );
				diff[offset3 + 11] = dim + IDX(i - 1, j    );
				diff[offset3 + 13] = dim + IDX(i    , j - 1);
				diff[offset3 + 15] = dim + IDX(i    , j + 1);
				diff[offset3 + 17] = dim + IDX(i + 1, j    );
				diff[offset3 + 19] = 2 * dim + IDX(i - 1, j    );
				diff[offset3 + 21] = 2 * dim + IDX(i    , j - 1);
				diff[offset3 + 23] = 2 * dim + IDX(i    , j    );
				diff[offset3 + 25] = 2 * dim + IDX(i    , j + 1);
				diff[offset3 + 27] = 2 * dim + IDX(i + 1, j    );
				diff[offset3 + 29] = 3 * dim + IDX(i    , j    );
				diff[offset3 + 31] = 4 * dim + IDX(i    , j    );
			}

			// log_a: 22 different points.
			offset4 = 1 + 102 * NrInterior * NzInterior + 44 * NzInterior * (i - 1); 
			for (j = 1; j < NzInterior + 1; j++)
			{
				// Row indices are all row 3 * dim + IDX(i, j).
				diff[offset4 +  0] = 3 * dim + IDX(i, j);
				diff[offset4 +  2] = 3 * dim + IDX(i, j);
				diff[offset4 +  4] = 3 * dim + IDX(i, j);
				diff[offset4 +  6] = 3 * dim + IDX(i, j);
				diff[offset4 +  8] = 3 * dim + IDX(i, j);
				diff[offset4 + 10] = 3 * dim + IDX(i, j);
				diff[offset4 + 12] = 3 * dim + IDX(i, j);
				diff[offset4 + 14] = 3 * dim + IDX(i, j);
				diff[offset4 + 16] = 3 * dim + IDX(i, j);
				diff[offset4 + 18] = 3 * dim + IDX(i, j);
				diff[offset4 + 20] = 3 * dim + IDX(i, j);
				diff[offset4 + 22] = 3 * dim + IDX(i, j);
				diff[offset4 + 24] = 3 * dim + IDX(i, j);
				diff[offset4 + 26] = 3 * dim + IDX(i, j);
				diff[offset4 + 28] = 3 * dim + IDX(i, j);
				diff[offset4 + 30] = 3 * dim + IDX(i, j);
				diff[offset4 + 32] = 3 * dim + IDX(i, j);
				diff[offset4 + 34] = 3 * dim + IDX(i, j);
				diff[offset4 + 36] = 3 * dim + IDX(i, j);
				diff[offset4 + 38] = 3 * dim + IDX(i, j);
				diff[offset4 + 40] = 3 * dim + IDX(i, j);
				diff[offset4 + 42] = 3 * dim + IDX(i, j);
				// Column indices.
				diff[offset4 +  1] = IDX(i - 1, j    );
				diff[offset4 +  3] = IDX(i    , j - 1);
				diff[offset4 +  5] = IDX(i    , j    );
				diff[offset4 +  7] = IDX(i    , j + 1);
				diff[offset4 +  9] = IDX(i + 1, j    );
				diff[offset4 + 11] = dim + IDX(i - 1, j    );
				diff[offset4 + 13] = dim + IDX(i    , j - 1);
				diff[offset4 + 15] = dim + IDX(i    , j    );
				diff[offset4 + 17] = dim + IDX(i    , j + 1);
				diff[offset4 + 19] = dim + IDX(i + 1, j    );
				diff[offset4 + 21] = 2 * dim + IDX(i - 1, j    );
				diff[offset4 + 23] = 2 * dim + IDX(i    , j - 1);
				diff[offset4 + 25] = 2 * dim + IDX(i    , j    );
				diff[offset4 + 27] = 2 * dim + IDX(i    , j + 1);
				diff[offset4 + 29] = 2 * dim + IDX(i + 1, j    );
				diff[offset4 + 31] = 3 * dim + IDX(i    , j    );
				diff[offset4 + 33] = 4 * dim + IDX(i - 1, j    );
				diff[offset4 + 35] = 4 * dim + IDX(i    , j - 1);
				diff[offset4 + 37] = 4 * dim + IDX(i    , j    );
				diff[offset4 + 39] = 4 * dim + IDX(i    , j + 1);
				diff[offset4 + 41] = 4 * dim + IDX(i + 1, j    );
				diff[offset4 + 43] = 5 * dim;
			}

			// psi: 18 different points, plus p_bound points.
			offset5 = 1 + 146 * NrInterior * NzInterior + (36 * NzInterior + 2 * p_bound) * (i - 1);
			for (j = 1; j < NzInterior + 1; j++)
			{
				// Row indices are all row 4 * dim + IDX(i, j).
				diff[offset5 +  0] = 4 * dim + IDX(i, j);
				diff[offset5 +  2] = 4 * dim + IDX(i, j);
				diff[offset5 +  4] = 4 * dim + IDX(i, j);
				diff[offset5 +  6] = 4 * dim + IDX(i, j);
				diff[offset5 +  8] = 4 * dim + IDX(i, j);
				diff[offset5 + 10] = 4 * dim + IDX(i, j);
				diff[offset5 + 12] = 4 * dim + IDX(i, j);
				diff[offset5 + 14] = 4 * dim + IDX(i, j);
				diff[offset5 + 16] = 4 * dim + IDX(i, j);
				diff[offset5 + 18] = 4 * dim + IDX(i, j);
				diff[offset5 + 20] = 4 * dim + IDX(i, j);
				diff[offset5 + 22] = 4 * dim + IDX(i, j);
				diff[offset5 + 24] = 4 * dim + IDX(i, j);
				diff[offset5 + 26] = 4 * dim + IDX(i, j);
				diff[offset5 + 28] = 4 * dim + IDX(i, j);
				diff[offset5 + 30] = 4 * dim + IDX(i, j);
				diff[offset5 + 32] = 4 * dim + IDX(i, j);
				diff[offset5 + 34] = 4 * dim + IDX(i, j);
				// Column indices.
				diff[offset5 +  1] = IDX(i - 1, j    );
				diff[offset5 +  3] = IDX(i    , j - 1);
				diff[offset5 +  5] = IDX(i    , j    );
				diff[offset5 +  7] = IDX(i    , j + 1);
				diff[offset5 +  9] = IDX(i + 1, j    );
				diff[offset5 + 11] = dim + IDX(i    , j    );
				diff[offset5 + 13] = 2 * dim + IDX(i - 1, j    );
				diff[offset5 + 15] = 2 * dim + IDX(i    , j - 1);
				diff[offset5 + 17] = 2 * dim + IDX(i    , j    );
				diff[offset5 + 19] = 2 * dim + IDX(i    , j + 1);
				diff[offset5 + 21] = 2 * dim + IDX(i + 1, j    );
				diff[offset5 + 23] = 3 * dim + IDX(i    , j    );
				diff[offset5 + 25] = 4 * dim + IDX(i - 1, j    );
				diff[offset5 + 27] = 4 * dim + IDX(i    , j - 1);
				diff[offset5 + 29] = 4 * dim + IDX(i    , j    );
				diff[offset5 + 31] = 4 * dim + IDX(i    , j + 1);
				diff[offset5 + 33] = 4 * dim + IDX(i + 1, j    );
				diff[offset5 + 35] = 5 * dim;
			}
			offset6 = offset5 + 36 * NzInterior;
			j = NzInterior + 1;
			switch (phiBoundOrder)
			{
				// Dirichlet.
				case 0:
					// Row.
					diff[offset6 + 0] = 4 * dim + IDX(i, j);
					// Columns.
					diff[offset6 + 1] = 5 * dim;
					break;

				// Exponential decay 1.
				case 1:
				case 2:
				case 3:
					// Row.
					diff[offset6 +  0] = 4 * dim + IDX(i, j);
					diff[offset6 +  2] = 4 * dim + IDX(i, j);
					// Columns.
					diff[offset6 +  1] = 4 * dim + IDX(i    , j    );
					diff[offset6 +  3] = 5 * dim;
					break;
			}
			
		}
		
		// Boundary points.
		offset7 = 1 + 182 * NrInterior * NzInterior + 2 * p_bound * NrInterior;
		i = NrInterior + 1;
		for (j = 1; j < NzInterior + 1; j++)
		{
			offset8 = offset7 + 2 * p_bound * (j - 1);
			switch (phiBoundOrder)
			{
				// Dirichlet.
				case 0:
					// Row.
					diff[offset8 + 0] = 4 * dim + IDX(i, j);
					// Columns.
					diff[offset8 + 1] = 5 * dim;
					break;

				// Exponential decay 1.
				case 1:
				case 2:
				case 3:
					// Row.
					diff[offset8 +  0] = 4 * dim + IDX(i, j);
					diff[offset8 +  2] = 4 * dim + IDX(i, j);
					// Columns.
					diff[offset8 +  1] = 4 * dim + IDX(i    , j    );
					diff[offset8 +  3] = 5 * dim;
					break;
			}
		}

		// Corner.
		offset9 = 1 + 182 * NrInterior * NzInterior + 2 * p_bound * (NrInterior + NzInterior);
		j = NzInterior + 1;
		switch (phiBoundOrder)
		{
			// Dirichlet.
			case 0:
				// Row.
				diff[offset9 + 0] = 4 * dim + IDX(i, j);
				// Columns.
				diff[offset9 + 1] = 5 * dim;
				break;

			// Exponential decay 1.
			case 1:
			case 2:
			case 3:
				// Row.
				diff[offset9 +  0] = 4 * dim + IDX(i, j);
				diff[offset9 +  2] = 4 * dim + IDX(i, j);
				// Columns.
				diff[offset9 +  1] = 4 * dim + IDX(i    , j    );
				diff[offset9 +  3] = 5 * dim;
				break;
		}
	}

	// All done.
	return;
}
