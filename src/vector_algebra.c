#include "tools.h"
#include "param.h"

#define GNUM_VECTOR 6

double norm2(double *x)
{
	return cblas_dnrm2(dim, x, 1) / sqrt(dim);
}

double dot(double *x, double *y)
{
	return cblas_ddot(dim, x, 1, y, 1)  / dim;
}

double norm2_interior(double *x)
{
	double sum = cblas_ddot(NzTotal * NrInterior, x + ghost * NzTotal, 1, x + ghost * NzTotal, 1);

	MKL_INT k = 0;

	for (k = 0; k < ghost; ++k)
	{
		sum -= cblas_ddot(NrInterior, x + ghost * NzTotal + k, NzTotal, x + ghost * NzTotal + k, NzTotal);
		sum -= cblas_ddot(NrInterior, x + (ghost + 1) * NzTotal - ghost + k, NzTotal, x + (ghost + 1) * NzTotal - ghost + k, NzTotal);
	}

	return sqrt(sum) / sqrt(NrInterior * NzInterior);
}

double dot_interior(double *x, double *y)
{
	double sum = cblas_ddot(NzTotal * NrInterior, x + ghost * NzTotal, 1, y + ghost * NzTotal, 1);

	MKL_INT k = 0;

	for (k = 0; k < ghost; ++k)
	{
		sum -= cblas_ddot(NrInterior, x + ghost * NzTotal + k, NzTotal, y + ghost * NzTotal + k, NzTotal);
		sum -= cblas_ddot(NrInterior, x + (ghost + 1) * NzTotal - ghost + k, NzTotal, y + (ghost + 1) * NzTotal - ghost + k, NzTotal);
	}

	return sum / (NrInterior * NzInterior);
}

double dot_interior_all_variables(double *x, double *y)
{
	double sum = 0.0;
	MKL_INT k = 0;

	// Add all dot products.
	for (k = 0; k < GNUM_VECTOR; ++k)
	{
		sum += dot_interior(x + k * dim, y + k * dim);
	}

	// Rescale.
	return sum / ((double)GNUM_VECTOR);
}

double norm2_interior_all_variables(double *x)
{
	double sum = dot_interior_all_variables(x, x);

	return sqrt(sum);
}