#include "tools.h"
#include "param.h"
#include "pardiso_param.h"

// Debug diff printer.
#undef DEBUG

void diff_gen(void)
{
	// Auxiliary integers.
	MKL_INT i, j, k, offset1, offset2, offset3, offset4, offset5;

	// Number of different elements after update.
	MKL_INT ndiff = 0;

	// Base ndiff with Dirichlet.
	if (order == 4)
	{
		// Main interior points.
		// 29 + 28 + 28 + 37 + 29 = 151.
		ndiff = 151 * NrInterior * NzInterior;
		// Add boundary strip.
		// 29 + 29 + 29 + 37 + 29 = 153.
		ndiff += 153 * (NrInterior + NzInterior + 1);
	}
	else
	{
		// Main interior points.
		// 17 + 16 + 16 + 21 + 17 = 87.
		ndiff = 87 * NrInterior * NzInterior;
	}

	// Allocate memory for diff array.
	diff = (MKL_INT *)SAFE_MALLOC(sizeof(MKL_INT) * (2 * ndiff + 1));

	// First element of diff is ndiff.
	diff[0] = ndiff;

	// Fill in rest of array.
	if (order == 4)
	{
		// Interior points.
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j, k,\
			offset1, offset2, offset3, offset4, offset5)
		for (i = ghost; i < ghost + NrInterior; ++i)
		{
			// 1. log_alpha: 29 different points.
			offset1 = 1 + 2 * ((29 * NzInterior + 29) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row IDX(i, j).
				for (k = 0; k < 29; ++k)
				{
					diff[offset1 + 2 * k] = IDX(i, j);
				}
				// Column indices.
				diff[offset1 +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset1 +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset1 +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset1 +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset1 +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset1 +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset1 +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset1 +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset1 +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset1 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset1 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset1 + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset1 + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset1 + 13 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset1 + 14 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset1 + 15 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset1 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset1 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset1 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset1 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset1 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset1 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset1 + 22 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset1 + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset1 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset1 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset1 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset1 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset1 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
				// Update offset by 29.
				offset1 += 2 * 29;
			}
			// Semi-one-sided: 29 points.
			j = ghost + NzInterior;
			// Row indices are all row IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset1 + 2 * k] = IDX(i, j);
			}
			// Columns.
			diff[offset1 +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset1 +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset1 +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset1 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset1 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset1 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset1 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset1 +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset1 +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset1 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset1 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset1 + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset1 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset1 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset1 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset1 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset1 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset1 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset1 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset1 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset1 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset1 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset1 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset1 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset1 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset1 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset1 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset1 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset1 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 2. beta: 28 points.
			offset2 = 1 + 2 * (29 * NrInterior * NzInterior + 29 * (NrInterior + NzInterior + 1)
				+ (28 * NzInterior + 29) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row dim + IDX(i, j).
				for (k = 0; k < 28; ++k)
				{
					diff[offset2 + 2 * k] = dim + IDX(i, j);
				}
				// Column indices.
				diff[offset2 +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset2 +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset2 +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset2 +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset2 +  4 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset2 +  5 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset2 +  6 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset2 +  7 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset2 +  8 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset2 +  9 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset2 + 10 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset2 + 11 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset2 + 12 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset2 + 13 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset2 + 14 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset2 + 15 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset2 + 16 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset2 + 17 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset2 + 18 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset2 + 19 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset2 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset2 + 21 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset2 + 22 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset2 + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset2 + 24 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset2 + 25 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset2 + 26 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset2 + 27 * 2 + 1] = 4 * dim + IDX(i    , j    );
				// Update offset by 28.
				offset2 += 2 * 28;
			}
			// Semi-one-sided: 29 points.
			j = ghost + NzInterior;
			// Row indices are all row dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset2 + 2 * k] = dim + IDX(i, j);
			}
			// Columns.
			diff[offset2 +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset2 +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset2 +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset2 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset2 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset2 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset2 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset2 +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset2 +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset2 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset2 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset2 + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset2 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset2 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset2 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset2 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset2 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset2 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset2 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset2 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset2 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset2 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset2 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset2 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset2 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset2 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset2 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset2 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset2 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 3. log_h: 28 points.
			offset3 = 1 + 2 * ((29 + 28) * NrInterior * NzInterior + (29 + 29) * (NrInterior + NzInterior + 1)
				+ (28 * NzInterior + 29) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 2 * dim + IDX(i, j).
				for (k = 0; k < 28; ++k)
				{
					diff[offset3 + 2 * k] = 2 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset3 +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset3 +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset3 +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset3 +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset3 +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset3 +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset3 +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset3 +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset3 +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset3 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset3 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset3 + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset3 + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset3 + 13 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset3 + 14 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset3 + 15 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset3 + 16 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset3 + 17 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset3 + 18 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset3 + 19 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset3 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset3 + 21 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset3 + 22 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset3 + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset3 + 24 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset3 + 25 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset3 + 26 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset3 + 27 * 2 + 1] = 4 * dim + IDX(i    , j    );
				// Update offset by 28.
				offset3 += 2 * 28;
			}
			// Semi-one-sided: 29 points.
			j = ghost + NzInterior;
			// Row indices are all row 2 * dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset3 + 2 * k] = 2 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset3 +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset3 +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset3 +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset3 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset3 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset3 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset3 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset3 +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset3 +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset3 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset3 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset3 + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset3 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset3 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset3 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset3 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset3 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset3 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset3 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset3 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset3 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset3 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset3 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset3 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset3 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset3 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset3 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset3 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset3 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 4. log_a: 37 points.
			offset4 = 1 + 2 * ((29 + 28 + 28) * NrInterior * NzInterior + (29 + 29 + 29) * (NrInterior + NzInterior + 1)
				+ (37 * NzInterior + 37) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 3 * dim + IDX(i, j).
				for (k = 0; k < 37; ++k)
				{
					diff[offset4 + 2 * k] = 3 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset4 +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset4 +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset4 +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset4 +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset4 +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset4 +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset4 +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset4 +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset4 +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset4 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset4 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset4 + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset4 + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset4 + 13 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset4 + 14 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset4 + 15 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset4 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset4 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset4 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset4 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset4 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset4 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset4 + 22 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset4 + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset4 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset4 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset4 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset4 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset4 + 28 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
				diff[offset4 + 29 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
				diff[offset4 + 30 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
				diff[offset4 + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
				diff[offset4 + 32 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset4 + 33 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
				diff[offset4 + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
				diff[offset4 + 35 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
				diff[offset4 + 36 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
				// Update offset by 37.
				offset4 += 2 * 37;
			}
			// Semi-one-sided: 37 points.
			j = ghost + NzInterior;
			// Row indices are all row 3 * dim + IDX(i, j).
			for (k = 0; k < 37; ++k)
			{
				diff[offset4 + 2 * k] = 3 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset4 +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset4 +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset4 +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset4 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset4 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset4 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset4 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset4 +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset4 +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset4 +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset4 + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset4 + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset4 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset4 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset4 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset4 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset4 + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset4 + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset4 + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset4 + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset4 + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset4 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset4 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset4 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset4 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset4 + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset4 + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset4 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset4 + 28 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset4 + 29 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset4 + 30 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
			diff[offset4 + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset4 + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset4 + 33 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset4 + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset4 + 35 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset4 + 36 * 2 + 1] = 4 * dim + IDX(i + 2, j    );

			// 5. psi: 29 points.
			offset5 = 1 + 2 * ((29 + 28 + 28 + 37) * NrInterior * NzInterior + (29 + 29 + 29 + 37) * (NrInterior + NzInterior + 1)
				+ (29 * NzInterior + 29) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 4 * dim + IDX(i, j).
				for (k = 0; k < 29; ++k)
				{
					diff[offset5 + 2 * k] = 4 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset5 +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset5 +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset5 +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset5 +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset5 +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset5 +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset5 +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset5 +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset5 +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset5 +  9 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset5 + 10 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset5 + 11 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset5 + 12 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset5 + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset5 + 14 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset5 + 15 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset5 + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset5 + 17 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset5 + 18 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset5 + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset5 + 20 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
				diff[offset5 + 21 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
				diff[offset5 + 22 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
				diff[offset5 + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
				diff[offset5 + 24 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset5 + 25 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
				diff[offset5 + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
				diff[offset5 + 27 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
				diff[offset5 + 28 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
				// Update offset by 29.
				offset5 += 2 * 29;
			}
			// Semi-one-sided: 29 points.
			j = ghost + NzInterior;
			// Row indices are all row 4 * dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset5 + 2 * k] = 4 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset5 +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset5 +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset5 +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset5 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset5 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset5 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset5 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset5 +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset5 +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset5 +  9 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset5 + 10 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset5 + 11 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset5 + 12 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset5 + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset5 + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset5 + 15 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset5 + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset5 + 17 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset5 + 18 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset5 + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset5 + 20 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset5 + 21 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset5 + 22 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
			diff[offset5 + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset5 + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset5 + 25 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset5 + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset5 + 27 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset5 + 28 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
		}

		// Now next-to-last rho strip.
		i = ghost + NrInterior;
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(j, k,\
			offset1, offset2, offset3, offset4, offset5)
		for (j = ghost; j < ghost + NzInterior; ++j)
		{
			// 1. log_alpha: 29 points.
			offset1 = 1 + 2 * (29 * NrInterior * NzInterior 
				+ 29 * NrInterior 
				+ 29 * (j - ghost));
			// All rows are IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset1 + 2 * k] = IDX(i, j);
			}
			// Columns.
			diff[offset1 +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset1 +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset1 +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset1 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset1 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset1 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset1 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset1 +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset1 +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset1 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset1 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset1 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset1 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset1 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset1 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset1 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset1 + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset1 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset1 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset1 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset1 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset1 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset1 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset1 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset1 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset1 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset1 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset1 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset1 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 2. beta: 29 points.
			offset2 = 1 + 2 * ((29 + 28) * NrInterior * NzInterior + 29 * (NrInterior + NzInterior + 1)
				+ 29 * NrInterior
				+ 29 * (j - ghost));
			// All rows are dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset2 + 2 * k] = dim + IDX(i, j);
			}
			// Columns.
			diff[offset2 +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset2 +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset2 +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset2 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset2 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset2 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset2 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset2 +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset2 +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset2 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset2 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset2 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset2 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset2 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset2 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset2 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset2 + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset2 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset2 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset2 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset2 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset2 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset2 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset2 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset2 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset2 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset2 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset2 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset2 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 3. log_h: 29 points.
			offset3 = 1 + 2 * ((29 + 28 + 28) * NrInterior * NzInterior + (29 + 29) * (NrInterior + NzInterior + 1)
				+ 29 * NrInterior
				+ 29 * (j - ghost));
			// All rows are 2 * dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset3 + 2 * k] = 2 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset3 +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset3 +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset3 +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset3 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset3 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset3 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset3 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset3 +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset3 +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset3 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset3 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset3 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset3 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset3 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset3 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset3 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset3 + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset3 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset3 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset3 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset3 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset3 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset3 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset3 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset3 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset3 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset3 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset3 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset3 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 4. log_a: 37 points.
			offset4 = 1 + 2 * ((29 + 28 + 28 + 37) * NrInterior * NzInterior + (29 + 29 + 29) * (NrInterior + NzInterior + 1)
				+ 37 * NrInterior
				+ 37 * (j - ghost));
			// All rows are 3 * dim + IDX(i, j).
			for (k = 0; k < 37; ++k)
			{
				diff[offset4 + 2 * k] = 3 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset4 +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset4 +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset4 +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset4 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset4 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset4 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset4 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset4 +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset4 +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset4 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset4 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset4 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset4 + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset4 + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset4 + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset4 + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset4 + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset4 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset4 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset4 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset4 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset4 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset4 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset4 + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset4 + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset4 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset4 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset4 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset4 + 28 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
			diff[offset4 + 29 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset4 + 30 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset4 + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset4 + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset4 + 33 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset4 + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset4 + 35 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
			diff[offset4 + 36 * 2 + 1] = 4 * dim + IDX(i + 1, j    );

			// 5. psi: 29 points.
			offset5 = 1 + 2 * ((29 + 28 + 28 + 37 + 29) * NrInterior * NzInterior + (29 + 29 + 29 + 37) * (NrInterior + NzInterior + 1)
				+ 29 * NrInterior
				+ 29 * (j - ghost));
			// All rows are 4 * dim + IDX(i, j).
			for (k = 0; k < 29; ++k)
			{
				diff[offset5 + 2 * k] = 4 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset5 +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset5 +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset5 +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset5 +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset5 +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset5 +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset5 +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset5 +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset5 +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset5 +  9 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset5 + 10 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset5 + 11 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset5 + 12 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset5 + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset5 + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset5 + 15 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset5 + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset5 + 17 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset5 + 18 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset5 + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset5 + 20 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
			diff[offset5 + 21 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset5 + 22 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset5 + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset5 + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset5 + 25 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset5 + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset5 + 27 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
			diff[offset5 + 28 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
		}

		// Corner.
		j = ghost + NzInterior;
		// 1. log_alpha: 29 points.
		offset1 = 1 + 2 * (29 * NrInterior * NzInterior 
			+ 29 * (NrInterior + NzInterior)); 
		// All rows are IDX(i, j).
		for (k = 0; k < 29; ++k)
		{
			diff[offset1 + 2 * k] = IDX(i, j);
		}
		// Columns.
		diff[offset1 +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset1 +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset1 +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset1 +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset1 +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset1 +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset1 +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset1 +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset1 +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset1 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset1 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset1 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset1 + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset1 + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset1 + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset1 + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset1 + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset1 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset1 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset1 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset1 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset1 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset1 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset1 + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset1 + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset1 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset1 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset1 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset1 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

		// 2. beta: 29 points.
		offset2 = 1 + 2 * ((29 + 28) * NrInterior * NzInterior + 29 * (NrInterior + NzInterior + 1)
			+ 29 * (NrInterior + NzInterior));
		// All rows are dim + IDX(i, j).
		for (k = 0; k < 29; ++k)
		{
			diff[offset2 + 2 * k] = dim + IDX(i, j);
		}
		// Columns.
		diff[offset2 +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset2 +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset2 +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset2 +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset2 +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset2 +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset2 +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset2 +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset2 +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset2 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset2 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset2 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset2 + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset2 + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset2 + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset2 + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset2 + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset2 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset2 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset2 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset2 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset2 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset2 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset2 + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset2 + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset2 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset2 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset2 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset2 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

		// 3. log_h: 29 points.
		offset3 = 1 + 2 * ((29 + 28 + 28) * NrInterior * NzInterior + (29 + 29) * (NrInterior + NzInterior + 1)
			+ 29 * (NrInterior + NzInterior));
		// All rows are 2 * dim + IDX(i, j).
		for (k = 0; k < 29; ++k)
		{
			diff[offset3 + 2 * k] = 2 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset3 +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset3 +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset3 +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset3 +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset3 +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset3 +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset3 +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset3 +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset3 +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset3 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset3 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset3 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset3 + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset3 + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset3 + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset3 + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset3 + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset3 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset3 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset3 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset3 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset3 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset3 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset3 + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset3 + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset3 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset3 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset3 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset3 + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

		// 4. log_h: 37 points.
		offset4 = 1 + 2 * ((29 + 28 + 28 + 37) * NrInterior * NzInterior + (29 + 29 + 29) * (NrInterior + NzInterior + 1)
			+ 37 * (NrInterior + NzInterior));
		// All rows are 3 * dim + IDX(i, j).
		for (k = 0; k < 37; ++k)
		{
			diff[offset4 + 2 * k] = 3 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset4 +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset4 +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset4 +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset4 +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset4 +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset4 +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset4 +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset4 +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset4 +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset4 +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset4 + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset4 + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset4 + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset4 + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset4 + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset4 + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset4 + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset4 + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset4 + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset4 + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset4 + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset4 + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset4 + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset4 + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset4 + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset4 + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset4 + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset4 + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset4 + 28 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
		diff[offset4 + 29 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
		diff[offset4 + 30 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
		diff[offset4 + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
		diff[offset4 + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
		diff[offset4 + 33 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
		diff[offset4 + 34 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset4 + 35 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
		diff[offset4 + 36 * 2 + 1] = 4 * dim + IDX(i + 1, j    );

		// 5. psi: 29 points.
		offset5 = 1 + 2 * ((29 + 28 + 28 + 37 + 29) * NrInterior * NzInterior + (29 + 29 + 29 + 37) * (NrInterior + NzInterior + 1)
			+ 29 * (NrInterior + NzInterior));
		// All rows are 4 * dim + IDX(i, j).
		for (k = 0; k < 29; ++k)
		{
			diff[offset5 + 2 * k] = 4 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset5 +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset5 +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset5 +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset5 +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset5 +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset5 +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset5 +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset5 +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset5 +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset5 +  9 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset5 + 10 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset5 + 11 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset5 + 12 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset5 + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset5 + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset5 + 15 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset5 + 16 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset5 + 17 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset5 + 18 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset5 + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset5 + 20 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
		diff[offset5 + 21 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
		diff[offset5 + 22 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
		diff[offset5 + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
		diff[offset5 + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
		diff[offset5 + 25 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
		diff[offset5 + 26 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset5 + 27 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
		diff[offset5 + 28 * 2 + 1] = 4 * dim + IDX(i + 1, j    );

	}
	else
	{
		// Interior points.
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j,\
				offset1, offset2, offset3,\
				offset4, offset5)
		for (i = ghost; i < NrInterior + ghost; i++)
		{
			// log_alpha: 17 different points.
			offset1 = 1 + 34 * NzInterior * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
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
				offset1 += 34;
			}

			// beta: 16 different points.
			offset2 = 1 + 34 * NrInterior * NzInterior + 32 * NzInterior * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
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
				offset2 += 32;
			}

			// log_h: 16 different points.
			offset3 = 1 + 66 * NrInterior * NzInterior + 32 * NzInterior * (i - ghost); 
			for (j = ghost; j < NzInterior + ghost; j++)
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
				offset3 += 32;
			}

			// log_a: 21 different points.
			offset4 = 1 + 98 * NrInterior * NzInterior + 42 * NzInterior * (i - ghost); 
			for (j = ghost; j < NzInterior + ghost; j++)
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
				offset4 += 42;
			}

			// psi: 17 different points.
			offset5 = 1 + 140 * NrInterior * NzInterior + (34 * NzInterior) * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
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
				offset5 += 34;
			}
		}
	}

#ifdef DEBUG
	write_single_integer_file_1d(diff, "diff.asc", 2 * ndiff + 1);
#endif

	// All done.
	return;
}
