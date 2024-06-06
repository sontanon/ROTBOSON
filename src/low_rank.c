#include "tools.h"
#include "param.h"
#include "pardiso_param.h"

// Debug diff printer.
#undef DEBUG

void diff_gen(void)
{
	const MKL_INT P4_CC[6] = {30, 29, 28, 38, 31, 40};
	const MKL_INT P4_CS[6] = {30, 30, 29, 38, 31, 41};
	const MKL_INT P4_SC[6] = {30, 30, 29, 38, 31, 43};
	const MKL_INT P4_SS[6] = {30, 30, 29, 38, 31, 43};
	const MKL_INT P2_CC[6] = {18, 17, 16, 22, 19, 22};

	// Auxiliary integers.
	MKL_INT i, j, k;
	// MKL_INT offset1, offset2, offset3, offset4, offset5, offset6;
	MKL_INT offset[6] = {0, 0, 0, 0, 0, 0};

	// Number of different elements after update.
	MKL_INT ndiff = 0;

	// Base ndiff with Dirichlet.
	if (order == 4)
	{
		// Main interior points.
		// 30 + 29 + 28 + 38 + 31 + 40 = 196.
		ndiff = (P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) * NrInterior * NzInterior;
		// Add boundary strip.
		// cs: 30 + 30 + 29 + 38 + 31 + 41 = 199.
		// sc: 30 + 30 + 29 + 38 + 31 + 43 = 201.
		// ss: 30 + 30 + 29 + 38 + 31 + 43 = 201.
		ndiff += (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + P4_CS[5]) * NrInterior;
		ndiff += (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + P4_SC[5]) * NzInterior;
		ndiff += (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + P4_SS[5]);
		// Add phiBoundOrder.
		ndiff += 2 * (NrInterior + NzInterior + 3);
	}
	else
	{
		// Main interior points.
		// 18 + 17 + 16 + 22 + 19 + 22 = 114.
		ndiff = (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4] + P2_CC[5]) * NrInterior * NzInterior;
		// Add phiBoundOrder.
		ndiff += 2 * (NrInterior + NzInterior + 1);
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
			offset)
		for (i = ghost; i < ghost + NrInterior; ++i)
		{
			// 1. log_alpha: 30 different points.
			offset[0] = 1 + 2 * ((P4_CC[0] * NzInterior + P4_CS[0]) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row IDX(i, j).
				for (k = 0; k < P4_CC[0]; ++k)
				{
					diff[offset[0] + 2 * k] = IDX(i, j);
				}
				// Column indices.
				diff[offset[0] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[0] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[0] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[0] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[0] +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset[0] +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[0] +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[0] +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[0] +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[0] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset[0] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset[0] + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset[0] + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset[0] + 13 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset[0] + 14 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset[0] + 15 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset[0] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset[0] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset[0] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[0] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[0] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[0] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[0] + 22 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[0] + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[0] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[0] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[0] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[0] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset[0] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset[0] + 29 * 2 + 1] = GNUM * dim;
				// Update offset by 30.
				offset[0] += 2 * P4_CC[0];
			}
			// Semi-one-sided: 30 points.
			j = ghost + NzInterior;
			// Row indices are all row IDX(i, j).
			for (k = 0; k < P4_CS[0]; ++k)
			{
				diff[offset[0] + 2 * k] = IDX(i, j);
			}
			// Columns.
			diff[offset[0] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[0] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[0] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[0] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[0] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[0] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[0] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[0] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[0] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[0] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[0] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[0] + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset[0] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[0] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[0] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[0] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[0] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[0] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset[0] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[0] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[0] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[0] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[0] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[0] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[0] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[0] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[0] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[0] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[0] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[0] + 29 * 2 + 1] = GNUM * dim;

			// 2. beta: 29 points.
			offset[1] = 1 + 2 * (P4_CC[0] * NrInterior * NzInterior 
					+ P4_CS[0] * NrInterior 
					+ P4_SC[0] * NzInterior 
					+ P4_SS[0]
				+ (P4_CC[1] * NzInterior + P4_CS[1]) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row dim + IDX(i, j).
				for (k = 0; k < P4_CC[1]; ++k)
				{
					diff[offset[1] + 2 * k] = dim + IDX(i, j);
				}
				// Column indices.
				diff[offset[1] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[1] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[1] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[1] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[1] +  4 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[1] +  5 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[1] +  6 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[1] +  7 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[1] +  8 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset[1] +  9 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset[1] + 10 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset[1] + 11 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset[1] + 12 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset[1] + 13 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset[1] + 14 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset[1] + 15 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset[1] + 16 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset[1] + 17 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[1] + 18 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[1] + 19 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[1] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[1] + 21 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[1] + 22 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[1] + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[1] + 24 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[1] + 25 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[1] + 26 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset[1] + 27 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset[1] + 28 * 2 + 1] = GNUM * dim;
				// Update offset by 29.
				offset[1] += 2 * P4_CC[1];
			}
			// Semi-one-sided: 30 points.
			j = ghost + NzInterior;
			// Row indices are all row dim + IDX(i, j).
			for (k = 0; k < P4_CS[1]; ++k)
			{
				diff[offset[1] + 2 * k] = dim + IDX(i, j);
			}
			// Columns.
			diff[offset[1] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[1] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[1] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[1] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[1] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[1] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[1] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[1] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[1] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[1] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[1] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[1] + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset[1] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[1] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[1] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[1] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[1] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[1] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset[1] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[1] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[1] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[1] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[1] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[1] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[1] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[1] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[1] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[1] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[1] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[1] + 29 * 2 + 1] = GNUM * dim;

			// 3. log_h: 28 points.
			offset[2] = 1 + 2 * ((P4_CC[0] + P4_CC[1]) * NrInterior * NzInterior 
					+ (P4_CS[0] + P4_CS[1]) * NrInterior 
					+ (P4_SC[0] + P4_SC[1]) * NzInterior 
					+ (P4_SS[0] + P4_SS[1])
				+ (P4_CC[2] * NzInterior + P4_CS[2]) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 2 * dim + IDX(i, j).
				for (k = 0; k < P4_CC[2]; ++k)
				{
					diff[offset[2] + 2 * k] = 2 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset[2] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[2] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[2] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[2] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[2] +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset[2] +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[2] +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[2] +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[2] +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[2] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset[2] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset[2] + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset[2] + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset[2] + 13 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset[2] + 14 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset[2] + 15 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset[2] + 16 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset[2] + 17 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[2] + 18 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[2] + 19 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[2] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[2] + 21 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[2] + 22 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[2] + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[2] + 24 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[2] + 25 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[2] + 26 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset[2] + 27 * 2 + 1] = 4 * dim + IDX(i    , j    );
				// Update offset by 28.
				offset[2] += 2 * P4_CC[2];
			}
			// Semi-one-sided: 29 points.
			j = ghost + NzInterior;
			// Row indices are all row 2 * dim + IDX(i, j).
			for (k = 0; k < P4_CS[2]; ++k)
			{
				diff[offset[2] + 2 * k] = 2 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[2] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[2] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[2] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[2] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[2] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[2] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[2] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[2] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[2] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[2] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[2] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[2] + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset[2] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[2] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[2] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[2] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[2] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[2] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset[2] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[2] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[2] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[2] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[2] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[2] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[2] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[2] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[2] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[2] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[2] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 4. log_a: 38 points.
			offset[3] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * NrInterior * NzInterior 
					+ (P4_CS[0] + P4_CS[1] + P4_CS[2]) * NrInterior 
					+ (P4_SC[0] + P4_SC[1] + P4_SC[2]) * NzInterior 
					+ (P4_SS[0] + P4_SS[1] + P4_SS[2])
				+ (P4_CC[3] * NzInterior + P4_CS[3]) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 3 * dim + IDX(i, j).
				for (k = 0; k < P4_CC[3]; ++k)
				{
					diff[offset[3] + 2 * k] = 3 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset[3] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[3] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[3] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[3] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[3] +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset[3] +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[3] +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[3] +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[3] +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[3] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset[3] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset[3] + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset[3] + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset[3] + 13 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset[3] + 14 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset[3] + 15 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset[3] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset[3] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset[3] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[3] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[3] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[3] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[3] + 22 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[3] + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[3] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[3] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[3] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[3] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset[3] + 28 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
				diff[offset[3] + 29 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
				diff[offset[3] + 30 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
				diff[offset[3] + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
				diff[offset[3] + 32 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset[3] + 33 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
				diff[offset[3] + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
				diff[offset[3] + 35 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
				diff[offset[3] + 36 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
				diff[offset[3] + 37 * 2 + 1] = GNUM * dim;
				// Update offset by 38.
				offset[3] += 2 * P4_CC[3];
			}
			// Semi-one-sided: 38 points.
			j = ghost + NzInterior;
			// Row indices are all row 3 * dim + IDX(i, j).
			for (k = 0; k < P4_CS[3]; ++k)
			{
				diff[offset[3] + 2 * k] = 3 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[3] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[3] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[3] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[3] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[3] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[3] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[3] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[3] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[3] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[3] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[3] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[3] + 11 * 2 + 1] =     dim + IDX(i    , j - 3);
			diff[offset[3] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[3] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[3] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[3] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[3] + 16 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[3] + 17 * 2 + 1] =     dim + IDX(i + 2, j    );
			diff[offset[3] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[3] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[3] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[3] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[3] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[3] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[3] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[3] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[3] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[3] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[3] + 28 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[3] + 29 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[3] + 30 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
			diff[offset[3] + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset[3] + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset[3] + 33 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[3] + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset[3] + 35 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[3] + 36 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
			diff[offset[3] + 37 * 2 + 1] = GNUM * dim;

			// 5. psi: 31 points.
			offset[4] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * NrInterior * NzInterior 
					+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * NrInterior 
					+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * NzInterior 
					+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3])
				+ (P4_CC[4] * NzInterior + P4_CS[4] + 2) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 4 * dim + IDX(i, j).
				for (k = 0; k < P4_CC[4]; ++k)
				{
					diff[offset[4] + 2 * k] = 4 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset[4] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[4] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[4] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[4] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[4] +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset[4] +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[4] +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[4] +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[4] +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[4] +  9 * 2 + 1] =     dim + IDX(i    , j    );
				diff[offset[4] + 10 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[4] + 11 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[4] + 12 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[4] + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[4] + 14 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[4] + 15 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[4] + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[4] + 17 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[4] + 18 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[4] + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
				diff[offset[4] + 20 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
				diff[offset[4] + 21 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
				diff[offset[4] + 22 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
				diff[offset[4] + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
				diff[offset[4] + 24 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset[4] + 25 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
				diff[offset[4] + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
				diff[offset[4] + 27 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
				diff[offset[4] + 28 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
				diff[offset[4] + 29 * 2 + 1] = 5 * dim + IDX(i, j);
				diff[offset[4] + 30 * 2 + 1] = GNUM * dim;
				// Update offset by 31.
				offset[4] += 2 * P4_CC[4];
			}
			// Semi-one-sided: 31 points.
			j = ghost + NzInterior;
			// Row indices are all row 4 * dim + IDX(i, j).
			for (k = 0; k < P4_CS[4]; ++k)
			{
				diff[offset[4] + 2 * k] = 4 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[4] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[4] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[4] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[4] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[4] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[4] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[4] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[4] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[4] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[4] +  9 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[4] + 10 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[4] + 11 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[4] + 12 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[4] + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[4] + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[4] + 15 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[4] + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[4] + 17 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[4] + 18 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[4] + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[4] + 20 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[4] + 21 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[4] + 22 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
			diff[offset[4] + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset[4] + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset[4] + 25 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[4] + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset[4] + 27 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[4] + 28 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
			diff[offset[4] + 29 * 2 + 1] = 5 * dim + IDX(i, j);
			diff[offset[4] + 30 * 2 + 1] = GNUM * dim;
			// Update offset by 31.
			offset[4] += 2 * P4_CS[4];
			// Boundary: 2 points.
			j = NzTotal - 1;
			diff[offset[4] + 0] = 4 * dim + IDX(i, j);
			diff[offset[4] + 2] = 4 * dim + IDX(i, j);
			diff[offset[4] + 1] = 4 * dim + IDX(i, j);
			diff[offset[4] + 3] = GNUM * dim;

			// 6. lambda: 40 points.
			offset[5] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * NrInterior * NzInterior 
					+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4]) * NrInterior 
					+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * NzInterior 
					+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4])
					+ 2 * (NrInterior + NzInterior + 3)
				+ (P4_CC[5] * NzInterior + P4_CS[5]) * (i - ghost));
			for (j = ghost; j < ghost + NzInterior; ++j)
			{
				// Row indices are all row 5 * dim + IDX(i, j).
				for (k = 0; k < P4_CC[5]; ++k)
				{
					diff[offset[5] + 2 * k] = 5 * dim + IDX(i, j);
				}
				// Column indices.
				diff[offset[5] +  0 * 2 + 1] =           IDX(i - 2, j    );
				diff[offset[5] +  1 * 2 + 1] =           IDX(i - 1, j    );
				diff[offset[5] +  2 * 2 + 1] =           IDX(i    , j - 2);
				diff[offset[5] +  3 * 2 + 1] =           IDX(i    , j - 1);
				diff[offset[5] +  4 * 2 + 1] =           IDX(i    , j    );
				diff[offset[5] +  5 * 2 + 1] =           IDX(i    , j + 1);
				diff[offset[5] +  6 * 2 + 1] =           IDX(i    , j + 2);
				diff[offset[5] +  7 * 2 + 1] =           IDX(i + 1, j    );
				diff[offset[5] +  8 * 2 + 1] =           IDX(i + 2, j    );
				diff[offset[5] +  9 * 2 + 1] =     dim + IDX(i - 2, j    );
				diff[offset[5] + 10 * 2 + 1] =     dim + IDX(i - 1, j    );
				diff[offset[5] + 11 * 2 + 1] =     dim + IDX(i    , j - 2);
				diff[offset[5] + 12 * 2 + 1] =     dim + IDX(i    , j - 1);
				diff[offset[5] + 13 * 2 + 1] =     dim + IDX(i    , j + 1);
				diff[offset[5] + 14 * 2 + 1] =     dim + IDX(i    , j + 2);
				diff[offset[5] + 15 * 2 + 1] =     dim + IDX(i + 1, j    );
				diff[offset[5] + 16 * 2 + 1] =     dim + IDX(i + 2, j    );
				diff[offset[5] + 17 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
				diff[offset[5] + 18 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
				diff[offset[5] + 19 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
				diff[offset[5] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
				diff[offset[5] + 21 * 2 + 1] = 2 * dim + IDX(i    , j    );
				diff[offset[5] + 22 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
				diff[offset[5] + 23 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
				diff[offset[5] + 24 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
				diff[offset[5] + 25 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
				diff[offset[5] + 26 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
				diff[offset[5] + 27 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
				diff[offset[5] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
				diff[offset[5] + 29 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
				diff[offset[5] + 30 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
				diff[offset[5] + 31 * 2 + 1] = 5 * dim + IDX(i - 2, j);
				diff[offset[5] + 32 * 2 + 1] = 5 * dim + IDX(i - 1, j);
				diff[offset[5] + 33 * 2 + 1] = 5 * dim + IDX(i, j - 2);
				diff[offset[5] + 34 * 2 + 1] = 5 * dim + IDX(i, j - 1);
				diff[offset[5] + 35 * 2 + 1] = 5 * dim + IDX(i, j);
				diff[offset[5] + 36 * 2 + 1] = 5 * dim + IDX(i, j + 1);
				diff[offset[5] + 37 * 2 + 1] = 5 * dim + IDX(i, j + 2);
				diff[offset[5] + 38 * 2 + 1] = 5 * dim + IDX(i + 1, j);
				diff[offset[5] + 39 * 2 + 1] = 5 * dim + IDX(i + 2, j);
				// Update offset by 40.
				offset[5] += 2 * P4_CC[5];
			}
			// Semi-one-sided: 41 points.
			j = ghost + NzInterior;
			// Row indices are all row 5 * dim + IDX(i, j).
			for (k = 0; k < P4_CS[5]; ++k)
			{
				diff[offset[5] + 2 * k] = 5 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[5] +  0 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[5] +  1 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[5] +  2 * 2 + 1] =           IDX(i    , j - 3);
			diff[offset[5] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[5] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[5] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[5] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[5] +  7 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[5] +  8 * 2 + 1] =           IDX(i + 2, j    );
			diff[offset[5] +  9 * 2 + 1] = 1 * dim + IDX(i - 2, j    );
			diff[offset[5] + 10 * 2 + 1] = 1 * dim + IDX(i - 1, j    );
			diff[offset[5] + 11 * 2 + 1] = 1 * dim + IDX(i    , j - 3);
			diff[offset[5] + 12 * 2 + 1] = 1 * dim + IDX(i    , j - 2);
			diff[offset[5] + 13 * 2 + 1] = 1 * dim + IDX(i    , j - 1);
			diff[offset[5] + 14 * 2 + 1] = 1 * dim + IDX(i    , j    );
			diff[offset[5] + 15 * 2 + 1] = 1 * dim + IDX(i    , j + 1);
			diff[offset[5] + 16 * 2 + 1] = 1 * dim + IDX(i + 1, j    );
			diff[offset[5] + 17 * 2 + 1] = 1 * dim + IDX(i + 2, j    );
			diff[offset[5] + 18 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[5] + 19 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[5] + 20 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
			diff[offset[5] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[5] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[5] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[5] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[5] + 25 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[5] + 26 * 2 + 1] = 2 * dim + IDX(i + 2, j    );
			diff[offset[5] + 27 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[5] + 28 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[5] + 29 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[5] + 30 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[5] + 31 * 2 + 1] = 4 * dim + IDX(i + 2, j    );
			diff[offset[5] + 32 * 2 + 1] = 5 * dim + IDX(i - 2, j    );
			diff[offset[5] + 33 * 2 + 1] = 5 * dim + IDX(i - 1, j    );
			diff[offset[5] + 34 * 2 + 1] = 5 * dim + IDX(i    , j - 3);
			diff[offset[5] + 35 * 2 + 1] = 5 * dim + IDX(i    , j - 2);
			diff[offset[5] + 36 * 2 + 1] = 5 * dim + IDX(i    , j - 1);
			diff[offset[5] + 37 * 2 + 1] = 5 * dim + IDX(i    , j    );
			diff[offset[5] + 38 * 2 + 1] = 5 * dim + IDX(i    , j + 1);
			diff[offset[5] + 39 * 2 + 1] = 5 * dim + IDX(i + 1, j    );
			diff[offset[5] + 40 * 2 + 1] = 5 * dim + IDX(i + 2, j    );
			// Update offset by 41.
			offset[5] += 2 * P4_CS[5];
		}

		// Now next-to-last rho strip.
		i = ghost + NrInterior;
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(j, k,\
			offset)
		for (j = ghost; j < ghost + NzInterior; ++j)
		{
			// 1. log_alpha: 30 points.
			offset[0] = 1 + 2 * (P4_CC[0] * NrInterior * NzInterior 
				+ P4_CS[0] * NrInterior 
				+ P4_SC[0] * (j - ghost));
			// All rows are IDX(i, j).
			for (k = 0; k < P4_SC[0]; ++k)
			{
				diff[offset[0] + 2 * k] = IDX(i, j);
			}
			// Columns.
			diff[offset[0] +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[0] +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[0] +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[0] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[0] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[0] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[0] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[0] +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[0] +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[0] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset[0] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[0] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[0] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[0] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[0] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[0] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[0] + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset[0] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[0] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[0] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[0] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[0] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[0] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[0] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[0] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[0] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[0] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[0] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[0] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[0] + 29 * 2 + 1] = GNUM * dim;

			// 2. beta: 30 points.
			offset[1] = 1 + 2 * ((P4_CC[0] + P4_CC[1]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1]) * NrInterior
				+ (P4_SC[0]) * NzInterior
				+ (P4_SS[0])
				+ P4_SC[1] * (j - ghost));
			// All rows are dim + IDX(i, j).
			for (k = 0; k < P4_SC[1]; ++k)
			{
				diff[offset[1] + 2 * k] = dim + IDX(i, j);
			}
			// Columns.
			diff[offset[1] +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[1] +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[1] +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[1] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[1] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[1] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[1] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[1] +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[1] +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[1] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset[1] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[1] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[1] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[1] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[1] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[1] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[1] + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset[1] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[1] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[1] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[1] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[1] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[1] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[1] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[1] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[1] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[1] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[1] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[1] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[1] + 29 * 2 + 1] = GNUM * dim;

			// 3. log_h: 29 points.
			offset[2] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1] + P4_CS[2]) * NrInterior
				+ (P4_SC[0] + P4_SC[1]) * NzInterior
				+ (P4_SS[0] + P4_SS[1])
				+ P4_SC[2] * (j - ghost));
			// All rows are 2 * dim + IDX(i, j).
			for (k = 0; k < P4_SC[2]; ++k)
			{
				diff[offset[2] + 2 * k] = 2 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[2] +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[2] +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[2] +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[2] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[2] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[2] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[2] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[2] +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[2] +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[2] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset[2] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[2] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[2] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[2] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[2] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[2] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[2] + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset[2] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[2] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[2] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[2] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[2] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[2] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[2] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[2] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[2] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[2] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[2] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[2] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

			// 4. log_h: 38 points.
			offset[3] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * NrInterior
				+ (P4_SC[0] + P4_SC[1] + P4_SC[2]) * NzInterior
				+ (P4_SS[0] + P4_SS[1] + P4_SS[2])
				+ P4_SC[3] * (j - ghost));
			// All rows are 3 * dim + IDX(i, j).
			for (k = 0; k < P4_SC[3]; ++k)
			{
				diff[offset[3] + 2 * k] = 3 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[3] +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[3] +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[3] +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[3] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[3] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[3] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[3] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[3] +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[3] +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[3] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
			diff[offset[3] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
			diff[offset[3] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
			diff[offset[3] + 12 * 2 + 1] =     dim + IDX(i    , j - 2);
			diff[offset[3] + 13 * 2 + 1] =     dim + IDX(i    , j - 1);
			diff[offset[3] + 14 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[3] + 15 * 2 + 1] =     dim + IDX(i    , j + 1);
			diff[offset[3] + 16 * 2 + 1] =     dim + IDX(i    , j + 2);
			diff[offset[3] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
			diff[offset[3] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[3] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[3] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[3] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[3] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[3] + 23 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[3] + 24 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[3] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[3] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[3] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[3] + 28 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
			diff[offset[3] + 29 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[3] + 30 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[3] + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset[3] + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset[3] + 33 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[3] + 34 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset[3] + 35 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
			diff[offset[3] + 36 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[3] + 37 * 2 + 1] = GNUM * dim;

			// 5. psi: 31 points.
			offset[4] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * NrInterior
				+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * NzInterior
				+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3])
				+ P4_SC[4] * (j - ghost));
			// All rows are 4 * dim + IDX(i, j).
			for (k = 0; k < P4_SC[4]; ++k)
			{
				diff[offset[4] + 2 * k] = 4 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[4] +  0 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[4] +  1 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[4] +  2 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[4] +  3 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[4] +  4 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[4] +  5 * 2 + 1] =           IDX(i    , j    );
			diff[offset[4] +  6 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[4] +  7 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[4] +  8 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[4] +  9 * 2 + 1] =     dim + IDX(i    , j    );
			diff[offset[4] + 10 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[4] + 11 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[4] + 12 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[4] + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[4] + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[4] + 15 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[4] + 16 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[4] + 17 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[4] + 18 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[4] + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
			diff[offset[4] + 20 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
			diff[offset[4] + 21 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[4] + 22 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[4] + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
			diff[offset[4] + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
			diff[offset[4] + 25 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[4] + 26 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
			diff[offset[4] + 27 * 2 + 1] = 4 * dim + IDX(i    , j + 2);
			diff[offset[4] + 28 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[4] + 29 * 2 + 1] = 5 * dim + IDX(i, j);
			diff[offset[4] + 30 * 2 + 1] = GNUM * dim;

			// 6. lambda: 43 points.
			offset[5] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2 + P4_CS[5]) * NrInterior
				+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + 2) * NzInterior
				+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 6)
				+ P4_SC[5] * (j - ghost));
			// All rows are 5 * dim + IDX(i, j).
			for (k = 0; k < P4_SC[5]; ++k)
			{
				diff[offset[5] + 2 * k] = 5 * dim + IDX(i, j);
			}
			// Columns.
			diff[offset[5] +  0 * 2 + 1] =           IDX(i - 4, j    );
			diff[offset[5] +  1 * 2 + 1] =           IDX(i - 3, j    );
			diff[offset[5] +  2 * 2 + 1] =           IDX(i - 2, j    );
			diff[offset[5] +  3 * 2 + 1] =           IDX(i - 1, j    );
			diff[offset[5] +  4 * 2 + 1] =           IDX(i    , j - 2);
			diff[offset[5] +  5 * 2 + 1] =           IDX(i    , j - 1);
			diff[offset[5] +  6 * 2 + 1] =           IDX(i    , j    );
			diff[offset[5] +  7 * 2 + 1] =           IDX(i    , j + 1);
			diff[offset[5] +  8 * 2 + 1] =           IDX(i    , j + 2);
			diff[offset[5] +  9 * 2 + 1] =           IDX(i + 1, j    );
			diff[offset[5] + 10 * 2 + 1] = 1 * dim + IDX(i - 3, j    );
			diff[offset[5] + 11 * 2 + 1] = 1 * dim + IDX(i - 2, j    );
			diff[offset[5] + 12 * 2 + 1] = 1 * dim + IDX(i - 1, j    );
			diff[offset[5] + 13 * 2 + 1] = 1 * dim + IDX(i    , j - 2);
			diff[offset[5] + 14 * 2 + 1] = 1 * dim + IDX(i    , j - 1);
			diff[offset[5] + 15 * 2 + 1] = 1 * dim + IDX(i    , j    );
			diff[offset[5] + 16 * 2 + 1] = 1 * dim + IDX(i    , j + 1);
			diff[offset[5] + 17 * 2 + 1] = 1 * dim + IDX(i    , j + 2);
			diff[offset[5] + 18 * 2 + 1] = 1 * dim + IDX(i + 1, j    );
			diff[offset[5] + 19 * 2 + 1] = 2 * dim + IDX(i - 4, j    );
			diff[offset[5] + 20 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
			diff[offset[5] + 21 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
			diff[offset[5] + 22 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
			diff[offset[5] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
			diff[offset[5] + 24 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
			diff[offset[5] + 25 * 2 + 1] = 2 * dim + IDX(i    , j    );
			diff[offset[5] + 26 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
			diff[offset[5] + 27 * 2 + 1] = 2 * dim + IDX(i    , j + 2);
			diff[offset[5] + 28 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
			diff[offset[5] + 29 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
			diff[offset[5] + 30 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
			diff[offset[5] + 31 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
			diff[offset[5] + 32 * 2 + 1] = 4 * dim + IDX(i    , j    );
			diff[offset[5] + 33 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
			diff[offset[5] + 34 * 2 + 1] = 5 * dim + IDX(i - 3, j    );
			diff[offset[5] + 35 * 2 + 1] = 5 * dim + IDX(i - 2, j    );
			diff[offset[5] + 36 * 2 + 1] = 5 * dim + IDX(i - 1, j    );
			diff[offset[5] + 37 * 2 + 1] = 5 * dim + IDX(i    , j - 2);
			diff[offset[5] + 38 * 2 + 1] = 5 * dim + IDX(i    , j - 1);
			diff[offset[5] + 39 * 2 + 1] = 5 * dim + IDX(i    , j    );
			diff[offset[5] + 40 * 2 + 1] = 5 * dim + IDX(i    , j + 1);
			diff[offset[5] + 41 * 2 + 1] = 5 * dim + IDX(i    , j + 2);
			diff[offset[5] + 42 * 2 + 1] = 5 * dim + IDX(i + 1, j    );
		}

		// Corner.
		j = ghost + NzInterior;
		// 1. log_alpha: 30 points.
		offset[0] = 1 + 2 * (P4_CC[0] * NrInterior * NzInterior 
			+ P4_CS[0] * NrInterior
			+ P4_SC[0] * NzInterior);
		// All rows are IDX(i, j).
		for (k = 0; k < P4_SS[0]; ++k)
		{
			diff[offset[0] + 2 * k] = IDX(i, j);
		}
		// Columns.
		diff[offset[0] +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[0] +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[0] +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[0] +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[0] +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[0] +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[0] +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset[0] +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[0] +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[0] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset[0] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset[0] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset[0] + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset[0] + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset[0] + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset[0] + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset[0] + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset[0] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset[0] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[0] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[0] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[0] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[0] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[0] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[0] + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[0] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[0] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[0] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset[0] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset[0] + 29 * 2 + 1] = GNUM * dim;

		// 2. beta: 30 points.
		offset[1] = 1 + 2 * ((P4_CC[0] + P4_CC[1]) * NrInterior * NzInterior 
			+ (P4_CS[0] + P4_CS[1]) * NrInterior
			+ (P4_SC[0] + P4_SC[1]) * NzInterior
			+ (P4_SS[0]));
		// All rows are dim + IDX(i, j).
		for (k = 0; k < P4_SS[1]; ++k)
		{
			diff[offset[1] + 2 * k] = dim + IDX(i, j);
		}
		// Columns.
		diff[offset[1] +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[1] +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[1] +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[1] +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[1] +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[1] +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[1] +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset[1] +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[1] +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[1] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset[1] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset[1] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset[1] + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset[1] + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset[1] + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset[1] + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset[1] + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset[1] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset[1] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[1] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[1] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[1] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[1] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[1] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[1] + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[1] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[1] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[1] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset[1] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset[1] + 29 * 2 + 1] = GNUM * dim;

		// 3. log_h: 29 points.
		offset[2] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * NrInterior * NzInterior 
			+ (P4_CS[0] + P4_CS[1] + P4_CS[2]) * NrInterior
			+ (P4_SC[0] + P4_SC[1] + P4_SC[2]) * NzInterior
			+ (P4_SS[0] + P4_SS[1]));
		// All rows are 2 * dim + IDX(i, j).
		for (k = 0; k < P4_SS[2]; ++k)
		{
			diff[offset[2] + 2 * k] = 2 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset[2] +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[2] +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[2] +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[2] +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[2] +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[2] +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[2] +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset[2] +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[2] +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[2] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset[2] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset[2] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset[2] + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset[2] + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset[2] + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset[2] + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset[2] + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset[2] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset[2] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[2] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[2] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[2] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[2] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[2] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[2] + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[2] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[2] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[2] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset[2] + 28 * 2 + 1] = 4 * dim + IDX(i    , j    );

		// 4. log_h: 38 points.
		offset[3] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * NrInterior * NzInterior 
			+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * NrInterior
			+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * NzInterior
			+ (P4_SS[0] + P4_SS[1] + P4_SS[2]));
		// All rows are 3 * dim + IDX(i, j).
		for (k = 0; k < P4_SS[3]; ++k)
		{
			diff[offset[3] + 2 * k] = 3 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset[3] +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[3] +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[3] +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[3] +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[3] +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[3] +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[3] +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset[3] +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[3] +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[3] +  9 * 2 + 1] =     dim + IDX(i - 3, j    );
		diff[offset[3] + 10 * 2 + 1] =     dim + IDX(i - 2, j    );
		diff[offset[3] + 11 * 2 + 1] =     dim + IDX(i - 1, j    );
		diff[offset[3] + 12 * 2 + 1] =     dim + IDX(i    , j - 3);
		diff[offset[3] + 13 * 2 + 1] =     dim + IDX(i    , j - 2);
		diff[offset[3] + 14 * 2 + 1] =     dim + IDX(i    , j - 1);
		diff[offset[3] + 15 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset[3] + 16 * 2 + 1] =     dim + IDX(i    , j + 1);
		diff[offset[3] + 17 * 2 + 1] =     dim + IDX(i + 1, j    );
		diff[offset[3] + 18 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[3] + 19 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[3] + 20 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[3] + 21 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[3] + 22 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[3] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[3] + 24 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[3] + 25 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[3] + 26 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[3] + 27 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset[3] + 28 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
		diff[offset[3] + 29 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
		diff[offset[3] + 30 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
		diff[offset[3] + 31 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
		diff[offset[3] + 32 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
		diff[offset[3] + 33 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
		diff[offset[3] + 34 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset[3] + 35 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
		diff[offset[3] + 36 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
		diff[offset[3] + 37 * 2 + 1] = GNUM * dim;

		// 5. psi: 31 points.
		offset[4] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * NrInterior * NzInterior 
			+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * NrInterior
			+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * NzInterior
			+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3]));
		// All rows are 4 * dim + IDX(i, j).
		for (k = 0; k < P4_SS[4]; ++k)
		{
			diff[offset[4] + 2 * k] = 4 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset[4] +  0 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[4] +  1 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[4] +  2 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[4] +  3 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[4] +  4 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[4] +  5 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[4] +  6 * 2 + 1] =           IDX(i    , j    );
		diff[offset[4] +  7 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[4] +  8 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[4] +  9 * 2 + 1] =     dim + IDX(i    , j    );
		diff[offset[4] + 10 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[4] + 11 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[4] + 12 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[4] + 13 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[4] + 14 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[4] + 15 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[4] + 16 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[4] + 17 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[4] + 18 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[4] + 19 * 2 + 1] = 3 * dim + IDX(i    , j    );
		diff[offset[4] + 20 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
		diff[offset[4] + 21 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
		diff[offset[4] + 22 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
		diff[offset[4] + 23 * 2 + 1] = 4 * dim + IDX(i    , j - 3);
		diff[offset[4] + 24 * 2 + 1] = 4 * dim + IDX(i    , j - 2);
		diff[offset[4] + 25 * 2 + 1] = 4 * dim + IDX(i    , j - 1);
		diff[offset[4] + 26 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset[4] + 27 * 2 + 1] = 4 * dim + IDX(i    , j + 1);
		diff[offset[4] + 28 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
		diff[offset[4] + 29 * 2 + 1] = 5 * dim + IDX(i, j);
		diff[offset[4] + 30 * 2 + 1] = GNUM * dim;

		// Phi boundary: i = NrTotal - 2, j = NzTotal - 1.
		offset[4] += 2 * P4_SS[4];
		j = NzTotal - 1;
		diff[offset[4] + 0] = 4 * dim + IDX(i, j);
		diff[offset[4] + 2] = 4 * dim + IDX(i, j);
		diff[offset[4] + 1] = 4 * dim + IDX(i, j);
		diff[offset[4] + 3] = GNUM * dim;
		offset[4] += 4;

		// 6. lambda: 43 points.
		offset[5] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) * NrInterior * NzInterior 
			+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2 + P4_CS[5]) * NrInterior
			+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + 2 + P4_SC[5]) * NzInterior
			+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 6));
		// All rows are 5 * dim + IDX(i, j).
		for (k = 0; k < P4_SS[5]; ++k)
		{
			diff[offset[5] + 2 * k] = 5 * dim + IDX(i, j);
		}
		// Columns.
		diff[offset[5] +  0 * 2 + 1] =           IDX(i - 4, j    );
		diff[offset[5] +  1 * 2 + 1] =           IDX(i - 3, j    );
		diff[offset[5] +  2 * 2 + 1] =           IDX(i - 2, j    );
		diff[offset[5] +  3 * 2 + 1] =           IDX(i - 1, j    );
		diff[offset[5] +  4 * 2 + 1] =           IDX(i    , j - 3);
		diff[offset[5] +  5 * 2 + 1] =           IDX(i    , j - 2);
		diff[offset[5] +  6 * 2 + 1] =           IDX(i    , j - 1);
		diff[offset[5] +  7 * 2 + 1] =           IDX(i    , j    );
		diff[offset[5] +  8 * 2 + 1] =           IDX(i    , j + 1);
		diff[offset[5] +  9 * 2 + 1] =           IDX(i + 1, j    );
		diff[offset[5] + 10 * 2 + 1] = 1 * dim + IDX(i - 3, j    );
		diff[offset[5] + 11 * 2 + 1] = 1 * dim + IDX(i - 2, j    );
		diff[offset[5] + 12 * 2 + 1] = 1 * dim + IDX(i - 1, j    );
		diff[offset[5] + 13 * 2 + 1] = 1 * dim + IDX(i    , j - 3);
		diff[offset[5] + 14 * 2 + 1] = 1 * dim + IDX(i    , j - 2);
		diff[offset[5] + 15 * 2 + 1] = 1 * dim + IDX(i    , j - 1);
		diff[offset[5] + 16 * 2 + 1] = 1 * dim + IDX(i    , j    );
		diff[offset[5] + 17 * 2 + 1] = 1 * dim + IDX(i    , j + 1);
		diff[offset[5] + 18 * 2 + 1] = 1 * dim + IDX(i + 1, j    );
		diff[offset[5] + 19 * 2 + 1] = 2 * dim + IDX(i - 4, j    );
		diff[offset[5] + 20 * 2 + 1] = 2 * dim + IDX(i - 3, j    );
		diff[offset[5] + 21 * 2 + 1] = 2 * dim + IDX(i - 2, j    );
		diff[offset[5] + 22 * 2 + 1] = 2 * dim + IDX(i - 1, j    );
		diff[offset[5] + 23 * 2 + 1] = 2 * dim + IDX(i    , j - 3);
		diff[offset[5] + 24 * 2 + 1] = 2 * dim + IDX(i    , j - 2);
		diff[offset[5] + 25 * 2 + 1] = 2 * dim + IDX(i    , j - 1);
		diff[offset[5] + 26 * 2 + 1] = 2 * dim + IDX(i    , j    );
		diff[offset[5] + 27 * 2 + 1] = 2 * dim + IDX(i    , j + 1);
		diff[offset[5] + 28 * 2 + 1] = 2 * dim + IDX(i + 1, j    );
		diff[offset[5] + 29 * 2 + 1] = 4 * dim + IDX(i - 3, j    );
		diff[offset[5] + 30 * 2 + 1] = 4 * dim + IDX(i - 2, j    );
		diff[offset[5] + 31 * 2 + 1] = 4 * dim + IDX(i - 1, j    );
		diff[offset[5] + 32 * 2 + 1] = 4 * dim + IDX(i    , j    );
		diff[offset[5] + 33 * 2 + 1] = 4 * dim + IDX(i + 1, j    );
		diff[offset[5] + 34 * 2 + 1] = 5 * dim + IDX(i - 3, j    );
		diff[offset[5] + 35 * 2 + 1] = 5 * dim + IDX(i - 2, j    );
		diff[offset[5] + 36 * 2 + 1] = 5 * dim + IDX(i - 1, j    );
		diff[offset[5] + 37 * 2 + 1] = 5 * dim + IDX(i    , j - 3);
		diff[offset[5] + 38 * 2 + 1] = 5 * dim + IDX(i    , j - 2);
		diff[offset[5] + 39 * 2 + 1] = 5 * dim + IDX(i    , j - 1);
		diff[offset[5] + 40 * 2 + 1] = 5 * dim + IDX(i    , j    );
		diff[offset[5] + 41 * 2 + 1] = 5 * dim + IDX(i    , j + 1);
		diff[offset[5] + 42 * 2 + 1] = 5 * dim + IDX(i + 1, j    );

		// Last boundary points.
		i = NrTotal - 1;
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(j,\
			offset)
		for (j = ghost; j < NzTotal; ++j)
		{
			// 5. psi: 2 points.
			offset[4] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * NrInterior * NzInterior 
				+ (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * NrInterior
				+ (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * NzInterior
				+ (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 2)
				+ 2 * (j - ghost));

			diff[offset[4] + 0] = 4 * dim + IDX(i, j);
			diff[offset[4] + 2] = 4 * dim + IDX(i, j);
			diff[offset[4] + 1] = 4 * dim + IDX(i, j);
			diff[offset[4] + 3] = GNUM * dim;
		}
	}
	else
	{
		// Interior points.
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j,\
				offset)
		for (i = ghost; i < NrInterior + ghost; i++)
		{
			// log_alpha: 18 different points.
			offset[0] = 1 + 2 * P2_CC[0] * NzInterior * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row IDX(i, j).
				diff[offset[0] +  0] = IDX(i, j);
				diff[offset[0] +  2] = IDX(i, j);
				diff[offset[0] +  4] = IDX(i, j);
				diff[offset[0] +  6] = IDX(i, j);
				diff[offset[0] +  8] = IDX(i, j);
				diff[offset[0] + 10] = IDX(i, j);
				diff[offset[0] + 12] = IDX(i, j);
				diff[offset[0] + 14] = IDX(i, j);
				diff[offset[0] + 16] = IDX(i, j);
				diff[offset[0] + 18] = IDX(i, j);
				diff[offset[0] + 20] = IDX(i, j);
				diff[offset[0] + 22] = IDX(i, j);
				diff[offset[0] + 24] = IDX(i, j);
				diff[offset[0] + 26] = IDX(i, j);
				diff[offset[0] + 28] = IDX(i, j);
				diff[offset[0] + 30] = IDX(i, j);
				diff[offset[0] + 32] = IDX(i, j);
				diff[offset[0] + 34] = IDX(i, j);
				// Column indices.
				diff[offset[0] +  1] = IDX(i - 1, j    );
				diff[offset[0] +  3] = IDX(i    , j - 1);
				diff[offset[0] +  5] = IDX(i    , j    );
				diff[offset[0] +  7] = IDX(i    , j + 1);
				diff[offset[0] +  9] = IDX(i + 1, j    );
				diff[offset[0] + 11] = dim + IDX(i - 1, j    );
				diff[offset[0] + 13] = dim + IDX(i    , j - 1);
				diff[offset[0] + 15] = dim + IDX(i    , j    );
				diff[offset[0] + 17] = dim + IDX(i    , j + 1);
				diff[offset[0] + 19] = dim + IDX(i + 1, j    );
				diff[offset[0] + 21] = 2 * dim + IDX(i - 1, j    );
				diff[offset[0] + 23] = 2 * dim + IDX(i    , j - 1);
				diff[offset[0] + 25] = 2 * dim + IDX(i    , j    );
				diff[offset[0] + 27] = 2 * dim + IDX(i    , j + 1);
				diff[offset[0] + 29] = 2 * dim + IDX(i + 1, j    );
				diff[offset[0] + 31] = 3 * dim + IDX(i    , j    );
				diff[offset[0] + 33] = 4 * dim + IDX(i    , j    );
				diff[offset[0] + 35] = GNUM * dim;
				offset[0] += 2 * P2_CC[0];
			}

			// beta: 17 different points.
			offset[1] = 1 + 2 * (P2_CC[0]) * NrInterior * NzInterior + 2 * P2_CC[1] * NzInterior * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row dim + IDX(i, j).
				diff[offset[1] +  0] = dim + IDX(i, j);
				diff[offset[1] +  2] = dim + IDX(i, j);
				diff[offset[1] +  4] = dim + IDX(i, j);
				diff[offset[1] +  6] = dim + IDX(i, j);
				diff[offset[1] +  8] = dim + IDX(i, j);
				diff[offset[1] + 10] = dim + IDX(i, j);
				diff[offset[1] + 12] = dim + IDX(i, j);
				diff[offset[1] + 14] = dim + IDX(i, j);
				diff[offset[1] + 16] = dim + IDX(i, j);
				diff[offset[1] + 18] = dim + IDX(i, j);
				diff[offset[1] + 20] = dim + IDX(i, j);
				diff[offset[1] + 22] = dim + IDX(i, j);
				diff[offset[1] + 24] = dim + IDX(i, j);
				diff[offset[1] + 26] = dim + IDX(i, j);
				diff[offset[1] + 28] = dim + IDX(i, j);
				diff[offset[1] + 30] = dim + IDX(i, j);
				diff[offset[1] + 32] = dim + IDX(i, j);
				// Column indices.
				diff[offset[1] +  1] = IDX(i - 1, j    );
				diff[offset[1] +  3] = IDX(i    , j - 1);
				diff[offset[1] +  5] = IDX(i    , j + 1);
				diff[offset[1] +  7] = IDX(i + 1, j    );
				diff[offset[1] +  9] = dim + IDX(i - 1, j    );
				diff[offset[1] + 11] = dim + IDX(i    , j - 1);
				diff[offset[1] + 13] = dim + IDX(i    , j    );
				diff[offset[1] + 15] = dim + IDX(i    , j + 1);
				diff[offset[1] + 17] = dim + IDX(i + 1, j    );
				diff[offset[1] + 19] = 2 * dim + IDX(i - 1, j    );
				diff[offset[1] + 21] = 2 * dim + IDX(i    , j - 1);
				diff[offset[1] + 23] = 2 * dim + IDX(i    , j    );
				diff[offset[1] + 25] = 2 * dim + IDX(i    , j + 1);
				diff[offset[1] + 27] = 2 * dim + IDX(i + 1, j    );
				diff[offset[1] + 29] = 3 * dim + IDX(i    , j    );
				diff[offset[1] + 31] = 4 * dim + IDX(i    , j    );
				diff[offset[1] + 33] = GNUM * dim;
				offset[1] += 2 * P2_CC[1];
			}

			// log_h: 16 different points.
			offset[2] = 1 + 2 * (P2_CC[0] + P2_CC[1]) * NrInterior * NzInterior + 2 * P2_CC[2] * NzInterior * (i - ghost); 
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row 2 * dim + IDX(i, j).
				diff[offset[2] +  0] = 2 * dim + IDX(i, j);
				diff[offset[2] +  2] = 2 * dim + IDX(i, j);
				diff[offset[2] +  4] = 2 * dim + IDX(i, j);
				diff[offset[2] +  6] = 2 * dim + IDX(i, j);
				diff[offset[2] +  8] = 2 * dim + IDX(i, j);
				diff[offset[2] + 10] = 2 * dim + IDX(i, j);
				diff[offset[2] + 12] = 2 * dim + IDX(i, j);
				diff[offset[2] + 14] = 2 * dim + IDX(i, j);
				diff[offset[2] + 16] = 2 * dim + IDX(i, j);
				diff[offset[2] + 18] = 2 * dim + IDX(i, j);
				diff[offset[2] + 20] = 2 * dim + IDX(i, j);
				diff[offset[2] + 22] = 2 * dim + IDX(i, j);
				diff[offset[2] + 24] = 2 * dim + IDX(i, j);
				diff[offset[2] + 26] = 2 * dim + IDX(i, j);
				diff[offset[2] + 28] = 2 * dim + IDX(i, j);
				diff[offset[2] + 30] = 2 * dim + IDX(i, j);
				// Column indices.
				diff[offset[2] +  1] = IDX(i - 1, j    );
				diff[offset[2] +  3] = IDX(i    , j - 1);
				diff[offset[2] +  5] = IDX(i    , j    );
				diff[offset[2] +  7] = IDX(i    , j + 1);
				diff[offset[2] +  9] = IDX(i + 1, j    );
				diff[offset[2] + 11] = dim + IDX(i - 1, j    );
				diff[offset[2] + 13] = dim + IDX(i    , j - 1);
				diff[offset[2] + 15] = dim + IDX(i    , j + 1);
				diff[offset[2] + 17] = dim + IDX(i + 1, j    );
				diff[offset[2] + 19] = 2 * dim + IDX(i - 1, j    );
				diff[offset[2] + 21] = 2 * dim + IDX(i    , j - 1);
				diff[offset[2] + 23] = 2 * dim + IDX(i    , j    );
				diff[offset[2] + 25] = 2 * dim + IDX(i    , j + 1);
				diff[offset[2] + 27] = 2 * dim + IDX(i + 1, j    );
				diff[offset[2] + 29] = 3 * dim + IDX(i    , j    );
				diff[offset[2] + 31] = 4 * dim + IDX(i    , j    );
				offset[2] += 2 * P2_CC[2];
			}

			// log_a: 22 different points.
			offset[3] = 1 + 2 * (P2_CC[0] + P2_CC[1] + P2_CC[2]) * NrInterior * NzInterior + 2 * P2_CC[3] * NzInterior * (i - ghost); 
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row 3 * dim + IDX(i, j).
				diff[offset[3] +  0] = 3 * dim + IDX(i, j);
				diff[offset[3] +  2] = 3 * dim + IDX(i, j);
				diff[offset[3] +  4] = 3 * dim + IDX(i, j);
				diff[offset[3] +  6] = 3 * dim + IDX(i, j);
				diff[offset[3] +  8] = 3 * dim + IDX(i, j);
				diff[offset[3] + 10] = 3 * dim + IDX(i, j);
				diff[offset[3] + 12] = 3 * dim + IDX(i, j);
				diff[offset[3] + 14] = 3 * dim + IDX(i, j);
				diff[offset[3] + 16] = 3 * dim + IDX(i, j);
				diff[offset[3] + 18] = 3 * dim + IDX(i, j);
				diff[offset[3] + 20] = 3 * dim + IDX(i, j);
				diff[offset[3] + 22] = 3 * dim + IDX(i, j);
				diff[offset[3] + 24] = 3 * dim + IDX(i, j);
				diff[offset[3] + 26] = 3 * dim + IDX(i, j);
				diff[offset[3] + 28] = 3 * dim + IDX(i, j);
				diff[offset[3] + 30] = 3 * dim + IDX(i, j);
				diff[offset[3] + 32] = 3 * dim + IDX(i, j);
				diff[offset[3] + 34] = 3 * dim + IDX(i, j);
				diff[offset[3] + 36] = 3 * dim + IDX(i, j);
				diff[offset[3] + 38] = 3 * dim + IDX(i, j);
				diff[offset[3] + 40] = 3 * dim + IDX(i, j);
				diff[offset[3] + 42] = 3 * dim + IDX(i, j);
				// Column indices.
				diff[offset[3] +  1] = IDX(i - 1, j    );
				diff[offset[3] +  3] = IDX(i    , j - 1);
				diff[offset[3] +  5] = IDX(i    , j    );
				diff[offset[3] +  7] = IDX(i    , j + 1);
				diff[offset[3] +  9] = IDX(i + 1, j    );
				diff[offset[3] + 11] = dim + IDX(i - 1, j    );
				diff[offset[3] + 13] = dim + IDX(i    , j - 1);
				diff[offset[3] + 15] = dim + IDX(i    , j    );
				diff[offset[3] + 17] = dim + IDX(i    , j + 1);
				diff[offset[3] + 19] = dim + IDX(i + 1, j    );
				diff[offset[3] + 21] = 2 * dim + IDX(i - 1, j    );
				diff[offset[3] + 23] = 2 * dim + IDX(i    , j - 1);
				diff[offset[3] + 25] = 2 * dim + IDX(i    , j    );
				diff[offset[3] + 27] = 2 * dim + IDX(i    , j + 1);
				diff[offset[3] + 29] = 2 * dim + IDX(i + 1, j    );
				diff[offset[3] + 31] = 3 * dim + IDX(i    , j    );
				diff[offset[3] + 33] = 4 * dim + IDX(i - 1, j    );
				diff[offset[3] + 35] = 4 * dim + IDX(i    , j - 1);
				diff[offset[3] + 37] = 4 * dim + IDX(i    , j    );
				diff[offset[3] + 39] = 4 * dim + IDX(i    , j + 1);
				diff[offset[3] + 41] = 4 * dim + IDX(i + 1, j    );
				diff[offset[3] + 43] = GNUM * dim;
				offset[3] += 2 * P2_CC[3];
			}

			// psi: 19 different points, plus p_bound points.
			offset[4] = 1 + 2 * (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3]) * NrInterior * NzInterior + 2 * (P2_CC[4] * NzInterior + 2) * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row 4 * dim + IDX(i, j).
				diff[offset[4] +  0] = 4 * dim + IDX(i, j);
				diff[offset[4] +  2] = 4 * dim + IDX(i, j);
				diff[offset[4] +  4] = 4 * dim + IDX(i, j);
				diff[offset[4] +  6] = 4 * dim + IDX(i, j);
				diff[offset[4] +  8] = 4 * dim + IDX(i, j);
				diff[offset[4] + 10] = 4 * dim + IDX(i, j);
				diff[offset[4] + 12] = 4 * dim + IDX(i, j);
				diff[offset[4] + 14] = 4 * dim + IDX(i, j);
				diff[offset[4] + 16] = 4 * dim + IDX(i, j);
				diff[offset[4] + 18] = 4 * dim + IDX(i, j);
				diff[offset[4] + 20] = 4 * dim + IDX(i, j);
				diff[offset[4] + 22] = 4 * dim + IDX(i, j);
				diff[offset[4] + 24] = 4 * dim + IDX(i, j);
				diff[offset[4] + 26] = 4 * dim + IDX(i, j);
				diff[offset[4] + 28] = 4 * dim + IDX(i, j);
				diff[offset[4] + 30] = 4 * dim + IDX(i, j);
				diff[offset[4] + 32] = 4 * dim + IDX(i, j);
				diff[offset[4] + 34] = 4 * dim + IDX(i, j);
				diff[offset[4] + 36] = 4 * dim + IDX(i, j);
				// Column indices.
				diff[offset[4] +  1] = IDX(i - 1, j    );
				diff[offset[4] +  3] = IDX(i    , j - 1);
				diff[offset[4] +  5] = IDX(i    , j    );
				diff[offset[4] +  7] = IDX(i    , j + 1);
				diff[offset[4] +  9] = IDX(i + 1, j    );
				diff[offset[4] + 11] = dim + IDX(i    , j    );
				diff[offset[4] + 13] = 2 * dim + IDX(i - 1, j    );
				diff[offset[4] + 15] = 2 * dim + IDX(i    , j - 1);
				diff[offset[4] + 17] = 2 * dim + IDX(i    , j    );
				diff[offset[4] + 19] = 2 * dim + IDX(i    , j + 1);
				diff[offset[4] + 21] = 2 * dim + IDX(i + 1, j    );
				diff[offset[4] + 23] = 3 * dim + IDX(i    , j    );
				diff[offset[4] + 25] = 4 * dim + IDX(i - 1, j    );
				diff[offset[4] + 27] = 4 * dim + IDX(i    , j - 1);
				diff[offset[4] + 29] = 4 * dim + IDX(i    , j    );
				diff[offset[4] + 31] = 4 * dim + IDX(i    , j + 1);
				diff[offset[4] + 33] = 4 * dim + IDX(i + 1, j    );
				diff[offset[4] + 35] = 5 * dim + IDX(i    , j    );
				diff[offset[4] + 37] = GNUM * dim;
				offset[4] += 2 * P2_CC[4];
			}
			j = NzInterior + ghost;
			// Row.
			diff[offset[4] +  0] = 4 * dim + IDX(i, j);
			diff[offset[4] +  2] = 4 * dim + IDX(i, j);
			// Columns.
			diff[offset[4] +  1] = 4 * dim + IDX(i, j);
			diff[offset[4] +  3] = GNUM * dim;
		}
			
		// Boundary points.
		offset[4] = 1 + 2 * (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4]) * NrInterior * NzInterior + 4 * NrInterior;
		i = NrInterior + ghost;
		for (j = ghost; j < NzTotal; j++)
		{
			// Row.
			diff[offset[4] +  0] = 4 * dim + IDX(i, j);
			diff[offset[4] +  2] = 4 * dim + IDX(i, j);
			// Columns.
			diff[offset[4] +  1] = 4 * dim + IDX(i    , j    );
			diff[offset[4] +  3] = GNUM * dim;
			offset[4] += 4;
		}

		// Lambda interior points.
		#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j,\
				offset)
		for (i = ghost; i < NrInterior + ghost; i++)
		{
			// lambda: 22 different points.
			offset[5] = 1 + 2 * ((P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4]) * NrInterior * NzInterior + 2 * (NrInterior + NzInterior + 1)) + 2 * P2_CC[5] * NzInterior * (i - ghost);
			for (j = ghost; j < NzInterior + ghost; j++)
			{
				// Row indices are all row 5 * dim + IDX(i, j).
				diff[offset[5] +  0] = 5 * dim + IDX(i, j);
				diff[offset[5] +  2] = 5 * dim + IDX(i, j);
				diff[offset[5] +  4] = 5 * dim + IDX(i, j);
				diff[offset[5] +  6] = 5 * dim + IDX(i, j);
				diff[offset[5] +  8] = 5 * dim + IDX(i, j);
				diff[offset[5] + 10] = 5 * dim + IDX(i, j);
				diff[offset[5] + 12] = 5 * dim + IDX(i, j);
				diff[offset[5] + 14] = 5 * dim + IDX(i, j);
				diff[offset[5] + 16] = 5 * dim + IDX(i, j);
				diff[offset[5] + 18] = 5 * dim + IDX(i, j);
				diff[offset[5] + 20] = 5 * dim + IDX(i, j);
				diff[offset[5] + 22] = 5 * dim + IDX(i, j);
				diff[offset[5] + 24] = 5 * dim + IDX(i, j);
				diff[offset[5] + 26] = 5 * dim + IDX(i, j);
				diff[offset[5] + 28] = 5 * dim + IDX(i, j);
				diff[offset[5] + 30] = 5 * dim + IDX(i, j);
				diff[offset[5] + 32] = 5 * dim + IDX(i, j);
				diff[offset[5] + 34] = 5 * dim + IDX(i, j);
				diff[offset[5] + 36] = 5 * dim + IDX(i, j);
				diff[offset[5] + 38] = 5 * dim + IDX(i, j);
				diff[offset[5] + 40] = 5 * dim + IDX(i, j);
				// Column indices.
				diff[offset[5] +  1] = IDX(i - 1, j    );
				diff[offset[5] +  3] = IDX(i    , j - 1);
				diff[offset[5] +  5] = IDX(i    , j    );
				diff[offset[5] +  7] = IDX(i    , j + 1);
				diff[offset[5] +  9] = IDX(i + 1, j    );
				diff[offset[5] + 11] = dim + IDX(i - 1, j    );
				diff[offset[5] + 13] = dim + IDX(i    , j - 1);
				diff[offset[5] + 15] = dim + IDX(i - 1, j    );
				diff[offset[5] + 17] = dim + IDX(i    , j - 1);
				diff[offset[5] + 19] = 2 * dim + IDX(i - 1, j    );
				diff[offset[5] + 21] = 2 * dim + IDX(i    , j - 1);
				diff[offset[5] + 23] = 2 * dim + IDX(i    , j    );
				diff[offset[5] + 25] = 2 * dim + IDX(i    , j + 1);
				diff[offset[5] + 27] = 2 * dim + IDX(i + 1, j    );
				diff[offset[5] + 29] = 4 * dim + IDX(i - 1, j    );
				diff[offset[5] + 31] = 4 * dim + IDX(i    , j    );
				diff[offset[5] + 33] = 4 * dim + IDX(i + 1, j    );
				diff[offset[5] + 35] = 5 * dim + IDX(i    , j    );
				diff[offset[5] + 37] = 5 * dim + IDX(i    , j    );
				diff[offset[5] + 39] = 5 * dim + IDX(i    , j    );
				diff[offset[5] + 41] = 5 * dim + IDX(i    , j    );
				diff[offset[5] + 43] = 5 * dim + IDX(i    , j    );
				offset[5] += 2 * P2_CC[5];
			}
		}
	}

#ifdef DEBUG
	write_single_integer_file_1d(diff, "diff.asc", 2 * ndiff + 1);
#endif

	// All done.
	return;
}
