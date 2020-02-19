#include "tools.h"
#include "param.h"

void diff1r(double *dvar, const double *var, const MKL_INT symr)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse spatial step.
	double idr = 1.0 / dr;

	// Constants.
	const double half = 0.5;
	const double twelfth = 1.0 / 12.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Parity on axis and boundary.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				dvar[IDX(ghost, j)] = idr * half * (var[IDX(ghost + 1, j)] - (double)symr * var[IDX(ghost, j)]);
				dvar[IDX(NrTotal - 1, j)] = idr * half * (3.0 * var[IDX(NrTotal - 1, j)] - 4.0 * var[IDX(NrTotal - 2, j)] + var[IDX(NrTotal - 3, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost + 1; i < NrTotal - 1; ++i)
			{
				for (j = 0; j < NzTotal; ++j)
				{
					dvar[IDX(i, j)] = idr * half * (var[IDX(i + 1, j)] - var[IDX(i - 1, j)]);
				}
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Parity on axis and boundary.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				// First interior point.
				dvar[IDX(ghost, j)] = idr * twelfth * (-var[IDX(ghost + 2, j)] + (8.0 + (double)symr) * var[IDX(ghost + 1, j)] - (double)symr * 8.0 * var[IDX(ghost, j)]);
				// Second interior point.
				dvar[IDX(ghost + 1, j)] = idr * twelfth * (-var[IDX(ghost + 3, j)] + 8.0 * var[IDX(ghost + 2, j)] + (-8.0 + (double)symr) * var[IDX(ghost, j)]);
				// Second-to-last point.
				dvar[IDX(NrTotal - 2, j)] = idr * twelfth * (3.0 * var[IDX(NrTotal - 1, j)] + 10.0 * var[IDX(NrTotal - 2, j)] - 18.0 * var[IDX(NrTotal - 3, j)] + 6.0 * var[IDX(NrTotal - 4, j)] - var[IDX(NrTotal - 5, j)]);
				// Last point.
				dvar[IDX(NrTotal - 1, j)] = idr * twelfth * (25.0 * var[IDX(NrTotal - 1, j)] - 48.0 * var[IDX(NrTotal - 2, j)] + 36.0 * var[IDX(NrTotal - 3, j)] - 16.0 * var[IDX(NrTotal - 4, j)] + 3.0 * var[IDX(NrTotal - 5, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost + 2; i < NrTotal - 2; ++i)
			{
				for (j = 0; j < NzTotal; ++j)
				{
					dvar[IDX(i, j)] = idr * twelfth *  (8.0 * (var[IDX(i + 1, j)] - var[IDX(i - 1, j)]) - (var[IDX(i + 2, j)] - var[IDX(i - 2, j)]));
				}
			}
		}
	}
	// Symmetries on axis.
	#pragma omp parallel shared(dvar) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = 0; j < NzTotal; ++j)
		{
			for (i = 0; i < ghost; ++i)
			{
				dvar[IDX(ghost - 1 - i, j)] = -(double)(symr) * dvar[IDX(ghost + i, j)];
			}
		}
	}
	// All done.
	return;
}

void diff1z(double *dvar, const double *var, const MKL_INT symz)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse spatial step.
	double idz = 1.0 / dz;

	// Constants.
	const double half = 0.5;
	const double twelfth = 1.0 / 12.0;

	// Second-oder derivatives.
	if (order == 2)
	{
		// Parity on equator and boundary.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrTotal; ++i)
			{
				dvar[IDX(i, ghost)] = idz * half * (var[IDX(i, ghost + 1)] - (double)symz * var[IDX(i, ghost)]);
				dvar[IDX(i, NzTotal - 1)] = idz * half * (3.0 * var[IDX(i, NzTotal - 1)] - 4.0 * var[IDX(i, NzTotal - 2)] + var[IDX(i, NzTotal - 3)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost + 1; j < NzTotal - 1; ++j)
			{
				for (i = 0; i < NrTotal; ++i)
				{
					dvar[IDX(i, j)] = idz * half * (var[IDX(i, j + 1)] - var[IDX(i, j - 1)]);
				}
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Parity on equator and boundary.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrTotal; ++i)
			{
				// First interior point.
				dvar[IDX(i, ghost)] = idz * twelfth * (-var[IDX(i, ghost + 2)] + (8.0 + (double)symz) * var[IDX(i, ghost + 1)] - (double)symz * 8.0 * var[IDX(i, ghost)]);
				// Second interior point.
				dvar[IDX(i, ghost + 1)] = idz * twelfth * (-var[IDX(i, ghost + 3)] + 8.0 * var[IDX(i, ghost + 2)] + (-8.0 + (double)symz) * var[IDX(i, ghost)]);
				// Second-to-last point.
				dvar[IDX(i, NzTotal - 2)] = idz * twelfth * (3.0 * var[IDX(i, NzTotal - 1)] + 10.0 * var[IDX(i, NzTotal - 2)] - 18.0 * var[IDX(i, NzTotal - 3)] + 6.0 * var[IDX(i, NzTotal - 4)] - var[IDX(i, NzTotal - 5)]);
				// Last point.
				dvar[IDX(i, NzTotal - 1)] = idz * twelfth * (25.0 * var[IDX(i, NzTotal - 1)] - 48.0 * var[IDX(i, NzTotal - 2)] + 36.0 * var[IDX(i, NzTotal - 3)] - 16.0 * var[IDX(i, NzTotal - 4)] + 3.0 * var[IDX(i, NzTotal - 5)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(j, i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost + 2; j < NzTotal - 2; ++j)
			{
				for (i = 0; i < NrTotal; ++i)
				{
					dvar[IDX(i, j)] = idz * twelfth *  (8.0 * (var[IDX(i, j + 1)] - var[IDX(i, j - 1)]) - (var[IDX(i, j + 2)] - var[IDX(i, j - 2)]));
				}
			}
		}
	}
	// Symmetries on equator.
	#pragma omp parallel shared(dvar) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < NrTotal; ++i)
		{
			for (j = 0; j < ghost; ++j)
			{
				dvar[IDX(i, ghost - 1 - j)] = -(double)(symz)* dvar[IDX(i, ghost + j)];
			}
		}
	}
	// All done.
	return;
}

void diff2r(double *dvar, const double *var, const MKL_INT symr)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse step squared.
	double idr2 = 1.0 / (dr * dr);

	// Constant.
	const double twelfth = 1.0 / 12.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Parity on axis and boundary.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				dvar[IDX(ghost, j)] = idr2 * (var[IDX(ghost + 1, j)] + (-2.0 + (double)symr) * var[IDX(ghost, j)]);
				dvar[IDX(NrTotal - 1, j)] = idr2 * (2.0 * var[IDX(NrTotal - 1, j)] - 5.0 * var[IDX(NrTotal - 2, j)] + 4.0 * var[IDX(NrTotal - 3, j)] - var[IDX(NrTotal - 4, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost + 1; i < NrTotal - 1; ++i)
			{
				for (j = 0; j < NzTotal; ++j)
				{
					dvar[IDX(i, j)] = idr2 * (var[IDX(i + 1, j)] - 2.0 * var[IDX(i, j)] + var[IDX(i - 1, j)]);
				}
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Parity on axis and boundary.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NzTotal; ++j)
			{
				// First interior point.
				dvar[IDX(ghost, j)] = idr2 * twelfth * (-var[IDX(ghost + 2, j)] + (16.0 - (double)symr) * var[IDX(ghost + 1, j)] +  (-30.0 + 16.0 * (double)symr) * var[IDX(ghost, j)]);
				// Second interior point.
				dvar[IDX(ghost + 1, j)] = idr2 * twelfth * (-var[IDX(ghost + 3, j)] + 16.0 * var[IDX(ghost + 2, j)] - 30.0 * var[IDX(ghost + 1, j)] + (16.0 - (double)symr) * var[IDX(ghost, j)]);
				// Second-to-last point.
				dvar[IDX(NrTotal - 2, j)] = idr2 * twelfth * (10.0 * var[IDX(NrTotal - 1, j)] - 15.0 * var[IDX(NrTotal - 2, j)] - 4.0 * var[IDX(NrTotal - 3, j)] + 14.0 * var[IDX(NrTotal - 4, j)] - 6.0 * var[IDX(NrTotal - 5, j)] + var[IDX(NrTotal - 6, j)]);
				// Last point.
				dvar[IDX(NrTotal - 1, j)] = idr2 * twelfth * (45.0 * var[IDX(NrTotal - 1, j)] - 154.0 * var[IDX(NrTotal - 2, j)] + 214.0 * var[IDX(NrTotal - 3, j)] - 156.0 * var[IDX(NrTotal - 4, j)] + 61.0 * var[IDX(NrTotal - 5, j)] - 10.0 * var[IDX(NrTotal - 6, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost + 2; i < NrTotal - 2; ++i)
			{
				for (j = 0; j < NzTotal; ++j)
				{
					dvar[IDX(i, j)] = -idr2 * twelfth * (30.0 * var[IDX(i, j)] - 16.0 * (var[IDX(i + 1, j)] + var[IDX(i - 1, j)]) + var[IDX(i + 2, j)] + var[IDX(i - 2, j)]);
				}
			}
		}
	}
	// Symmetries on axis.
	#pragma omp parallel shared(dvar) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = 0; j < NzTotal; ++j)
		{
			for (i = 0; i < ghost; ++i)
			{
				dvar[IDX(ghost - 1 - i, j)] = (double)(symr)* dvar[IDX(ghost + i, j)];
			}
		}
	}
	// All done.
	return;
}

void diff2z(double *dvar, const double *var, const MKL_INT symz)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse step squared.
	double idz2 = 1.0 / (dz * dz);

	// Constant.
	const double twelfth = 1.0 / 12.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Parity on equator and boundary.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrTotal; ++i)
			{
				dvar[IDX(i, ghost)] = idz2 * (var[IDX(i, ghost + 1)] + (-2.0 + (double)symz) * var[IDX(i, ghost)]);
				dvar[IDX(i, NzTotal - 1)] = idz2 * (2.0 * var[IDX(i, NzTotal - 1)] - 5.0 * var[IDX(i, NzTotal - 2)] + 4.0 * var[IDX(i, NzTotal - 3)] - var[IDX(i, NzTotal - 4)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(j, i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost + 1; j < NzTotal - 1; ++j)
			{
				for (i = 0; i < NrTotal; ++i)
				{
					dvar[IDX(i, j)] = idz2 * (var[IDX(i, j + 1)] - 2.0 * var[IDX(i, j)] + var[IDX(i, j - 1)]);
				}
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Parity on equator and boundary.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrTotal; ++i)
			{
				// First interior point.
				dvar[IDX(i, ghost)] = idz2 * twelfth * (-var[IDX(i, ghost + 2)] + (16.0 - (double)symz) * var[IDX(i, ghost + 1)] +  (-30.0 + 16.0 * (double)symz) * var[IDX(i, ghost)]);
				// Second interior point.
				dvar[IDX(i, ghost + 1)] = idz2 * twelfth * (-var[IDX(i, ghost + 3)] + 16.0 * var[IDX(i, ghost + 2)] - 30.0 * var[IDX(i, ghost + 1)] + (16.0 - (double)symz) * var[IDX(i, ghost)]);
				// Second-to-last point.
				dvar[IDX(i, NzTotal - 2)] = idz2 * twelfth * (10.0 * var[IDX(i, NzTotal - 1)] - 15.0 * var[IDX(i, NzTotal - 2)] - 4.0 * var[IDX(i, NzTotal - 3)] + 14.0 * var[IDX(i, NzTotal - 4)] - 6.0 * var[IDX(i, NzTotal - 5)] + var[IDX(i, NzTotal - 6)]);
				// Last point.
				dvar[IDX(i, NzTotal - 1)] = idz2 * twelfth * (45.0 * var[IDX(i, NzTotal - 1)] - 154.0 * var[IDX(i, NzTotal - 2)] + 214.0 * var[IDX(i, NzTotal - 3)] - 156.0 * var[IDX(i, NzTotal - 4)] + 61.0 * var[IDX(i, NzTotal - 5)] - 10.0 * var[IDX(i, NzTotal - 6)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(j, i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost + 2; j < NzTotal - 2; ++j)
			{
				for (i = 0; i < NrTotal; ++i)
				{
					dvar[IDX(i, j)] = -idz2 * twelfth * (30.0 * var[IDX(i, j)] - 16.0 * (var[IDX(i, j + 1)] + var[IDX(i, j - 1)]) + var[IDX(i, j + 2)] + var[IDX(i, j - 2)]);
				}
			}
		}
	}
	// Symmetries on equator.
	#pragma omp parallel shared(dvar) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < NrTotal; ++i)
		{
			for (j = 0; j < ghost; ++j)
			{
				dvar[IDX(i, ghost - 1 - j)] = (double)(symz)*dvar[IDX(i, ghost + j)];
			}
		}
	}
	// All done.
	return;
}

void diff2rz(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz)
{
	MKL_INT i, j;

	double idr = 1.0 / dr, idz = 1.0 / dz;
	double idrz = idr * idz;

	const double quarter = 0.25;
	const double i48 = 1.0 / 48.0;
	const double i144 = 1.0 / 144.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal - 1; ++i)
			{
				for (j = ghost; j < NzTotal - 1; ++j)
				{
					dvar[IDX(i, j)] = idrz * quarter * (var[IDX(i + 1, j + 1)] + var[IDX(i - 1, j - 1)] - var[IDX(i + 1, j - 1)] - var[IDX(i - 1, j + 1)]);
				}

				// Last point on z.
				dvar[IDX(i, NzTotal - 1)] = idrz * quarter * ((3.0 * var[IDX(i + 1, NzTotal - 1)] - 4.0 * var[IDX(i + 1, NzTotal - 2)] + var[IDX(i + 1, NzTotal - 3)]) - (3.0 * var[IDX(i - 1, NzTotal - 1)] - 4.0 * var[IDX(i - 1, NzTotal - 2)] + var[IDX(i - 1, NzTotal - 3)]));
			}
		}

		// Last point on r
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost; j < NzTotal - 1; ++j)
			{
				dvar[IDX(NrTotal - 1, j)] = idrz * quarter * ((3.0 * var[IDX(NrTotal - 1, j + 1)] - 4.0 * var[IDX(NrTotal - 2, j + 1)] + var[IDX(NrTotal - 3, j + 1)]) - (3.0 * var[IDX(NrTotal - 1, j - 1)] - 4.0 * var[IDX(NrTotal - 2, j - 1)] + var[IDX(NrTotal - 3, j - 1)]));
			}
		}

		// Corner.
		dvar[IDX(NrTotal - 1, NzTotal - 1)] = idrz * quarter * (9.0 * var[IDX(NrTotal - 1, NzTotal - 1)] + 16.0 * var[IDX(NrTotal - 2, NzTotal - 2)] + var[IDX(NrTotal - 3, NzTotal - 3)] - 12.0 * (var[IDX(NrTotal - 2, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 2)]) + 3.0 * (var[IDX(NrTotal - 1, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 1)]) - 4.0 * (var[IDX(NrTotal - 2, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 2)]));
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal - 2; ++i)
			{
				for (j = ghost; j < NzTotal - 2; ++j)
				{
					dvar[IDX(i, j)] = idrz * i48 * (16.0 * (var[IDX(i + 1, j + 1)] + var[IDX(i - 1, j - 1)] - var[IDX(i + 1, j - 1)] - var[IDX(i - 1, j + 1)]) - (var[IDX(i + 2, j + 2)] + var[IDX(i - 2, j - 2)] - var[IDX(i + 2, j - 2)] - var[IDX(i - 2, j + 2)]));
				}

				// Second-to-last and last points in Z.
				dvar[IDX(i, NzTotal - 2)] = -idrz * i144 * ((var[IDX(i - 2, NzTotal - 5)] - var[IDX(i + 2, NzTotal - 5)]) - 6.0 * (var[IDX(i - 2, NzTotal - 4)] - var[IDX(i + 2, NzTotal - 4)]) + 18.0 * (var[IDX(i - 2, NzTotal - 3)] - var[IDX(i + 2, NzTotal - 3)]) - 10.0 * (var[IDX(i - 2, NzTotal - 2)] - var[IDX(i + 2, NzTotal - 2)]) - 3.0 * (var[IDX(i - 2, NzTotal - 1)] - var[IDX(i + 2, NzTotal - 1)]) - 8.0 * (var[IDX(i - 1, NzTotal - 5)] - var[IDX(i + 1, NzTotal - 5)]) + 48.0 * (var[IDX(i - 1, NzTotal - 4)] - var[IDX(i + 1, NzTotal - 4)]) - 144.0 * (var[IDX(i - 1, NzTotal - 3)] - var[IDX(i + 1, NzTotal - 3)]) + 80.0 * (var[IDX(i - 1, NzTotal - 2)] - var[IDX(i + 1, NzTotal - 2)]) + 24.0 * (var[IDX(i - 1, NzTotal - 1)] - var[IDX(i + 1, NzTotal - 1)]));

				dvar[IDX(i, NzTotal - 1)] = -idrz * i144 * (-3.0 * (var[IDX(i - 2, NzTotal - 5)] - var[IDX(i + 2, NzTotal - 5)]) + 16.0 * (var[IDX(i - 2, NzTotal - 4)] - var[IDX(i + 2, NzTotal - 4)]) - 36.0 * (var[IDX(i - 2, NzTotal - 3)] - var[IDX(i + 2, NzTotal - 3)]) + 48.0 * (var[IDX(i - 2, NzTotal - 2)] - var[IDX(i + 2, NzTotal - 2)]) - 25.0 * (var[IDX(i - 2, NzTotal - 1)] - var[IDX(i + 2, NzTotal - 1)]) + 24.0 * (var[IDX(i - 1, NzTotal - 5)] - var[IDX(i + 1, NzTotal - 5)]) - 128.0 * (var[IDX(i - 1, NzTotal - 4)] - var[IDX(i + 1, NzTotal - 4)]) + 288.0 * (var[IDX(i - 1, NzTotal - 3)] - var[IDX(i + 1, NzTotal - 3)]) - 384.0 * (var[IDX(i - 1, NzTotal - 2)] - var[IDX(i + 1, NzTotal - 2)]) + 200.0 * (var[IDX(i - 1, NzTotal - 1)] - var[IDX(i + 1, NzTotal - 1)]));

			}

		}

		// Second-to-last and last points in R.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost; j < NzTotal - 2; ++j)
			{
				dvar[IDX(NrTotal - 2, j)] = -idrz * i144 * ((var[IDX(NrTotal - 5, j - 2)] - var[IDX(NrTotal - 5, j + 2)]) - 6.0 * (var[IDX(NrTotal - 4, j - 2)] - var[IDX(NrTotal - 4, j + 2)]) + 18.0 * (var[IDX(NrTotal - 3, j - 2)] - var[IDX(NrTotal - 3, j + 2)]) - 10.0 * (var[IDX(NrTotal - 2, j - 2)] - var[IDX(NrTotal - 2, j + 2)]) - 3.0 * (var[IDX(NrTotal - 1, j - 2)] - var[IDX(NrTotal - 1, j + 2)]) - 8.0 * (var[IDX(NrTotal - 5, j - 1)] - var[IDX(NrTotal - 5, j + 1)]) + 48.0 * (var[IDX(NrTotal - 4, j - 1)] - var[IDX(NrTotal - 4, j + 1)]) - 144.0 * (var[IDX(NrTotal - 3, j - 1)] - var[IDX(NrTotal - 3, j + 1)]) + 80.0 * (var[IDX(NrTotal - 2, j - 1)] - var[IDX(NrTotal - 2, j + 1)]) + 24.0 * (var[IDX(NrTotal - 1, j - 1)] - var[IDX(NrTotal - 1, j + 1)]));

				dvar[IDX(NrTotal - 1, j)] = -idrz * i144 * (-3.0 * (var[IDX(NrTotal - 5, j - 2)] - var[IDX(NrTotal - 5, j + 2)]) + 16.0 * (var[IDX(NrTotal - 4, j - 2)] - var[IDX(NrTotal - 4, j + 2)]) - 36.0 * (var[IDX(NrTotal - 3, j - 2)] - var[IDX(NrTotal - 3, j + 2)]) + 48.0 * (var[IDX(NrTotal - 2, j - 2)] - var[IDX(NrTotal - 2, j + 2)]) - 25.0 * (var[IDX(NrTotal - 1, j - 2)] - var[IDX(NrTotal - 1, j + 2)]) + 24.0 * (var[IDX(NrTotal - 5, j - 1)] - var[IDX(NrTotal - 5, j + 1)]) - 128.0 * (var[IDX(NrTotal - 4, j - 1)] - var[IDX(NrTotal - 4, j + 1)]) + 288.0 * (var[IDX(NrTotal - 3, j - 1)] - var[IDX(NrTotal - 3, j + 1)]) - 384.0 * (var[IDX(NrTotal - 2, j - 1)] - var[IDX(NrTotal - 2, j + 1)]) + 200.0 * (var[IDX(NrTotal - 1, j - 1)] - var[IDX(NrTotal - 1, j + 1)]));
			}
		}

		// Corner: 4 points.
		dvar[IDX(NrTotal - 2, NzTotal - 2)] = idrz * i144 * (var[IDX(NrTotal - 5, NzTotal - 5)] + 36.0 * var[IDX(NrTotal - 4, NzTotal - 4)] + 324.0 * var[IDX(NrTotal - 3, NzTotal - 3)] + 100.0 * var[IDX(NrTotal - 2, NzTotal - 2)] + 9 * var[IDX(NrTotal - 1, NzTotal - 1)] - 6.0 * (var[IDX(NrTotal - 5, NzTotal - 4)] + var[IDX(NrTotal - 4, NzTotal - 5)]) + 18.0 * (var[IDX(NrTotal - 5, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 5)]) - 10.0 * (var[IDX(NrTotal - 5, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 5)]) - 3.0 * (var[IDX(NrTotal - 5, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 5)]) - 108.0 * (var[IDX(NrTotal - 4, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 4)]) + 60.0 * (var[IDX(NrTotal - 4, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 4)]) + 18.0 * (var[IDX(NrTotal - 4, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 4)]) - 180.0 * (var[IDX(NrTotal - 3, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 3)]) - 54.0 * (var[IDX(NrTotal - 3, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 3)]) + 30.0 * (var[IDX(NrTotal - 2, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 2)]));

		dvar[IDX(NrTotal - 2, NzTotal - 1)] = idrz * i144 * (-3.0 * var[IDX(NrTotal - 5, NzTotal - 5)] - 96.0 * var[IDX(NrTotal - 4, NzTotal - 4)] - 648.0 * var[IDX(NrTotal - 3, NzTotal - 3)] - 480.0 * var[IDX(NrTotal - 2, NzTotal - 2)] + 75.0 * var[IDX(NrTotal - 1, NzTotal - 1)] + 16.0 * var[IDX(NrTotal - 5, NzTotal - 4)] + 18.0 * var[IDX(NrTotal - 4, NzTotal - 5)] - 36.0 * var[IDX(NrTotal - 5, NzTotal - 3)] - 54.0 * var[IDX(NrTotal - 3, NzTotal - 5)] + 48.0 * var[IDX(NrTotal - 5, NzTotal - 2)] + 30.0 * var[IDX(NrTotal - 2, NzTotal - 5)] - 25.0 * var[IDX(NrTotal - 5, NzTotal - 1)] + 9.0 * var[IDX(NrTotal - 1, NzTotal - 5)] + 216.0 * var[IDX(NrTotal - 4, NzTotal - 3)] + 288.0 * var[IDX(NrTotal - 3, NzTotal - 4)] - 288.0 * var[IDX(NrTotal - 4, NzTotal - 2)] - 160.0 * var[IDX(NrTotal - 2, NzTotal - 4)] + 150.0 * var[IDX(NrTotal - 4, NzTotal - 1)] - 48.0 * var[IDX(NrTotal - 1, NzTotal - 4)] + 864.0 * var[IDX(NrTotal - 3, NzTotal - 2)] + 360.0 * var[IDX(NrTotal - 2, NzTotal - 3)] - 450.0 * var[IDX(NrTotal - 3, NzTotal - 1)] + 108.0 * var[IDX(NrTotal - 1, NzTotal - 3)] + 250.0 * var[IDX(NrTotal - 2, NzTotal - 1)] - 144.0 * var[IDX(NrTotal - 1, NzTotal - 2)]);

		dvar[IDX(NrTotal - 1, NzTotal - 2)] = idrz * i144 * (-3.0 * var[IDX(NrTotal - 5, NzTotal - 5)] - 96.0 * var[IDX(NrTotal - 4, NzTotal - 4)] - 648.0 * var[IDX(NrTotal - 3, NzTotal - 3)] - 480.0 * var[IDX(NrTotal - 2, NzTotal - 2)] + 75.0 * var[IDX(NrTotal - 1, NzTotal - 1)] + 18.0 * var[IDX(NrTotal - 5, NzTotal - 4)] + 16.0 * var[IDX(NrTotal - 4, NzTotal - 5)] - 54.0 * var[IDX(NrTotal - 5, NzTotal - 3)] - 36.0 * var[IDX(NrTotal - 3, NzTotal - 5)] + 30.0 * var[IDX(NrTotal - 5, NzTotal - 2)] + 48.0 * var[IDX(NrTotal - 2, NzTotal - 5)] + 9.0 * var[IDX(NrTotal - 5, NzTotal - 1)] - 25.0 * var[IDX(NrTotal - 1, NzTotal - 5)] + 288.0 * var[IDX(NrTotal - 4, NzTotal - 3)] + 216.0 * var[IDX(NrTotal - 3, NzTotal - 4)] - 160.0 * var[IDX(NrTotal - 4, NzTotal - 2)] - 288.0 * var[IDX(NrTotal - 2, NzTotal - 4)] - 48.0 * var[IDX(NrTotal - 4, NzTotal - 1)] + 150.0 * var[IDX(NrTotal - 1, NzTotal - 4)] + 360.0 * var[IDX(NrTotal - 3, NzTotal - 2)] + 864.0 * var[IDX(NrTotal - 2, NzTotal - 3)] + 108.0 * var[IDX(NrTotal - 3, NzTotal - 1)] - 450.0 * var[IDX(NrTotal - 1, NzTotal - 3)] - 144.0 * var[IDX(NrTotal - 2, NzTotal - 1)] + 250.0 * var[IDX(NrTotal - 1, NzTotal - 2)]);

		dvar[IDX(NrTotal - 1, NzTotal - 1)] = idrz * i144 * (9.0 * var[IDX(NrTotal - 5, NzTotal - 5)] + 256.0 * var[IDX(NrTotal - 4, NzTotal - 4)] + 1296.0 * var[IDX(NrTotal - 3, NzTotal - 3)] + 2304.0 * var[IDX(NrTotal - 2, NzTotal - 2)] + 625.0 * var[IDX(NrTotal - 1, NzTotal - 1)] - 48.0 * (var[IDX(NrTotal - 5, NzTotal - 4)] + var[IDX(NrTotal - 4, NzTotal - 5)]) + 108.0 * (var[IDX(NrTotal - 5, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 5)]) - 144.0 * (var[IDX(NrTotal - 5, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 5)]) + 75.0 * (var[IDX(NrTotal - 5, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 5)]) - 576.0 * (var[IDX(NrTotal - 4, NzTotal - 3)] + var[IDX(NrTotal - 3, NzTotal - 4)]) + 768.0 * (var[IDX(NrTotal - 4, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 4)]) - 400.0 * (var[IDX(NrTotal - 4, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 4)]) - 1728.0 * (var[IDX(NrTotal - 3, NzTotal - 2)] + var[IDX(NrTotal - 2, NzTotal - 3)]) + 900.0 * (var[IDX(NrTotal - 3, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 3)]) - 1200.0 * (var[IDX(NrTotal - 2, NzTotal - 1)] + var[IDX(NrTotal - 1, NzTotal - 2)]));

	}

	// Symmetries on axis and equator.
	MKL_INT k;
	for (k = 0; k < ghost; ++k)
	{
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost - k; i < NrTotal; ++i)
			{
				dvar[IDX(i, ghost - 1 - k)] = -(double)(symz)*dvar[IDX(i, ghost + k)];
			}
		}

		#pragma omp parallel shared (dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ghost - k; j < NzTotal; ++j)
			{
				dvar[IDX(ghost - 1 - k, j)] = -(double)(symr)*dvar[IDX(ghost + k, j)];
			}
		}
		// Corner.
		dvar[IDX(ghost - 1 - k, ghost - 1 - k)] = (double)(symr*symz)*dvar[IDX(ghost + k, ghost + k)];
	}

	// All done.
	return;
}

// Angular differentiation.
void diff1th(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse angular step.
	double idth = 1.0 / dth;

	// Constants.
	const double half = 0.5;
	const double twelfth = 1.0 / 12.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Axial symmetry.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrrTotal; ++i)
			{
				dvar[P_IDX(i, 0)] = half * idth * var[P_IDX(i, 1)] * (1.0 - (double)symr);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 1; j < NthTotal; ++j)
			{
				for (i = 0; i < NrrTotal; ++i)
				{
					dvar[P_IDX(i, j)] = half * idth * (var[P_IDX(i, j + 1)] - var[P_IDX(i, j - 1)]);
				}
			}
		}
		// Equatorial symmetry.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrrTotal; ++i)
			{
				dvar[P_IDX(i, NthTotal - 1)] = -half * idth * var[P_IDX(i, NthTotal - 2)] * (1.0 - (double)symz);
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Axial symmetry.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrrTotal; ++i)
			{
				dvar[P_IDX(i, 0)] = twelfth * idth * (-var[P_IDX(i, 2)] + 8.0 * var[P_IDX(i, 1)]) * (1.0 - (double)symr);
				dvar[P_IDX(i, 1)] = twelfth * idth * (-8.0 * var[P_IDX(i, 0)] + (double)symr * var[P_IDX(i, 1)] + 8.0 * var[P_IDX(i, 2)] - var[P_IDX(i, 3)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 2; j < NthTotal - 2; ++j)
			{
				for (i = 0; i < NrrTotal; ++i)
				{
					dvar[P_IDX(i, j)] = twelfth * idth * (-(var[P_IDX(i, j + 2)] - var[P_IDX(i, j - 2)]) + 8.0 * (var[P_IDX(i, j + 1)] - var[P_IDX(i, j - 1)]));
				}
			}
		}
		// Equatorial symmetry.
		#pragma omp parallel shared(dvar) private(i)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 0; i < NrrTotal; ++i)
			{
				dvar[P_IDX(i, NthTotal - 2)] = -twelfth * idth * (-8.0 * var[P_IDX(i, NthTotal - 1)] + (double)symr * var[P_IDX(i, NthTotal - 2)] + 8.0 * var[P_IDX(i, NthTotal - 3)] - var[P_IDX(i, NthTotal - 4)]);
				dvar[P_IDX(i, NthTotal - 1)] = -twelfth * idth * (-var[P_IDX(i, NthTotal - 3)] + 8.0 * var[P_IDX(i, NthTotal - 2)]) * (1.0 - (double)symz);
			}
		}
	}

	// All done.
	return;
}

// Radial differentiation.
void diff1rr(double *dvar, const double *var, const MKL_INT symrr)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Inverse spatial step.
	double idrr = 1.0 / drr;

	// Constants.
	const double half = 0.5;
	const double twelfth = 1.0 / 12.0;

	// Second-order derivatives.
	if (order == 2)
	{
		// Derivative on the origin and on the last point.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NthTotal; ++j)
			{
				dvar[P_IDX(0, j)] = half * idrr * var[P_IDX(1, j)] * (1.0 - (double)symrr);
				dvar[P_IDX(NrrTotal - 1, j)] = half * idrr * (3.0 * var[P_IDX(NrrTotal - 1, j)] - 4.0 * var[P_IDX(NrrTotal - 2, j)] + var[P_IDX(NrrTotal - 3, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 1; i < NrrTotal - 1; ++i)
			{
				for (j = 0; j < NthTotal; ++j)
				{
					dvar[P_IDX(i, j)] = half * idrr * (var[P_IDX(i + 1, j)] - var[P_IDX(i - 1, j)]);
				}
			}
		}
	}
	// Fourth-order derivatives.
	else if (order == 4)
	{
		// Derivative near the origin and on the last points.
		#pragma omp parallel shared(dvar) private(j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = 0; j < NthTotal; ++j)
			{
				dvar[P_IDX(0, j)] = twelfth * idrr * (-var[P_IDX(2, j)] + 8.0 * var[P_IDX(1, j)]) * (1.0 - symrr);
				dvar[P_IDX(1, j)] = twelfth * idrr * (-var[P_IDX(3, j)] + 8.0 * var[P_IDX(2, j)] - 8.0 * var[P_IDX(0, j)] + symrr * var[P_IDX(1, j)]);
				dvar[P_IDX(NrrTotal - 2, j)] = twelfth * idrr * (3.0 * var[P_IDX(NrrTotal - 1, j)] + 10.0 * var[P_IDX(NrrTotal - 2, j)] - 18.0 * var[P_IDX(NrrTotal - 3, j)] + 6.0 * var[P_IDX(NrrTotal - 4, j)] - var[P_IDX(NrrTotal - 5, j)]);
				dvar[P_IDX(NrrTotal - 1, j)] = twelfth * idrr * (25.0 * var[P_IDX(NrrTotal - 1, j)] - 48.0 * var[P_IDX(NrrTotal - 2, j)] + 36.0 * var[P_IDX(NrrTotal - 3, j)] - 16.0 * var[P_IDX(NrrTotal - 4, j)] + 3.0 * var[P_IDX(NrrTotal - 5, j)]);
			}
		}
		// Main interior points.
		#pragma omp parallel shared(dvar) private(i, j)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = 2; i < NrrTotal - 2; ++i)
			{
				for (j = 0; j < NthTotal; ++j)
				{
					dvar[P_IDX(i, j)] = twelfth * idrr * (-(var[P_IDX(i + 2, j)] - var[P_IDX(i - 2, j)]) + 8.0 * (var[P_IDX(i + 1, j)] - var[P_IDX(i - 1, j)]));
				}
			}
		}
	}

	// All done.
	return;
}