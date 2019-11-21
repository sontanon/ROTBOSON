#include "tools.h"

// Equatorial symmetry.
void z_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT z_sym)
{
	// Row starts at offset.
	ia[(g_num - 1) * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset    ] = 1.0;
	a[offset + 1] = -(double)z_sym;

	// Set column values.
	ja[offset    ] = BASE + (g_num - 1) * dim + IDX(i, j);
	ja[offset + 1] = BASE + (g_num - 1) * dim + IDX(i, j + 1);

	// All done.
	return;
}

// Radial symmetry.
void r_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT r_sym)
{
	// Row starts at offset.
	ia[(g_num - 1) * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset    ] = 1.0;
	a[offset + 1] = -(double)r_sym;

	// Set column values.
	ja[offset    ] = BASE + (g_num - 1) * dim + IDX(i, j);
	ja[offset + 1] = BASE + (g_num - 1) * dim + IDX(i + 1, j);

	// All done.
	return;
}

// Corner symmetry, i.e. radial plus equatorial symmetry.
void corner_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT r_sym, const MKL_INT z_sym)
{
	// Row starts at offset.
	ia[(g_num - 1) * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset    ] = 1.0;
	a[offset + 1] = -(double)(r_sym * z_sym);

	// Set column values.
	ja[offset    ] = BASE + (g_num - 1) * dim + IDX(i, j);
	ja[offset + 1] = BASE + (g_num - 1) * dim + IDX(i + 1, j + 1);

	// All done.
	return;
}
