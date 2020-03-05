#include "tools.h"
#include "param.h"

double norm2(const double *x)
{
	return cblas_dnrm2(dim, x, 1) / sqrt(dim);
}

double dot(const double *x, const double *y)
{
	return cblas_ddot(dim, x, 1, y, 1)  / dim;
}

double norm2_interior(const double *x)
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

double dot_interior(const double *x, const double *y)
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

double dot_interior_all_variables(const double *x, const double *y)
{
	double sum = 0.0;
	MKL_INT k = 0;

	// Add five dot products.
	for (k = 0; k < 5; ++k)
	{
		sum += dot_interior(x + k * dim, y + k * dim);
	}

	// Rescale.
	return sum / 5.0;
}

double norm2_interior_all_variables(const double *x)
{
	double sum = dot_interior_all_variables(x, x);

	return sqrt(sum);
}