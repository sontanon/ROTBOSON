#include "tools.h"

double norm2(const double *x, const MKL_INT dim)
{
	return cblas_dnrm2(dim, x, 1) / sqrt(dim);
}

double dot(const double *x, const double *y, const MKL_INT dim)
{
	return cblas_ddot(dim, x, 1, y, 1)  / dim;
}
