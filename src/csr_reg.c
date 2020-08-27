#include "tools.h"

#define D10 (-0.5)
#define D11 (+0.0)
#define D12 (+0.5)

#define S10 (+0.5)
#define S11 (-2.0)
#define S12 (+1.5)

void c_reg_2nd_order
(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM
)
{
	double dzodr = dz / dr;
	double dr2 = dr * dr;
	double ri = (double)i - 0.5;

	// Row starts at offset.
	ia[U_GNUM * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	aa[offset + 0] = dzodr * (v / ri) * D10;
	aa[offset + 1] = dzodr * (v / ri) * D12;
	aa[offset + 2] = -dzodr * dr2;

	// Columns.
	ja[offset + 0] = BASE + V_GNUM * dim + IDX(i - 1, j);
	ja[offset + 1] = BASE + V_GNUM * dim + IDX(i + 1, j);
	ja[offset + 2] = BASE + U_GNUM * dim + IDX(i, j);

	// All done.
	return;
}

void s_reg_2nd_order
(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM
)
{
	double dzodr = dz / dr;
	double dr2 = dr * dr;
	double ri = (double)i - 0.5;

	// Row starts at offset.
	ia[U_GNUM * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	aa[offset + 0] = dzodr * (v / ri) * S10;
	aa[offset + 1] = dzodr * (v / ri) * S11;
	aa[offset + 2] = dzodr * (v / ri) * S12;
	aa[offset + 3] = -dzodr * dr2;

	// Columns.
	ja[offset + 0] = BASE + V_GNUM * dim + IDX(i - 2, j);
	ja[offset + 1] = BASE + V_GNUM * dim + IDX(i - 1, j);
	ja[offset + 2] = BASE + V_GNUM * dim + IDX(i    , j);
	ja[offset + 3] = BASE + U_GNUM * dim + IDX(i, j);

	// All done.
	return;
}

#define D1_4_0 (+1.0 / 12.0)
#define D1_4_1 (-2.0 / 3.0)
#define D1_4_2 (+0.0)
#define D1_4_3 (+2.0 / 3.0)
#define D1_4_4 (-1.0 / 12.0)

#define S1_4_0 (+0.25)
#define S1_4_1 (-4.0 / 3.0)
#define S1_4_2 (+3.0)
#define S1_4_3 (-4.0)
#define S1_4_4 (25.0 / 12.0)

#define SO1_4_0 (-1.0 / 12.0)
#define SO1_4_1 (+0.5)
#define SO1_4_2 (-1.5)
#define SO1_4_3 (+5.0 / 6.0)
#define SO1_4_4 (+0.25)

void c_reg_4th_order
(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM
)
{
	double dzodr = dz / dr;
	double dr2 = dr * dr;
	double ri = (double)i - 1.5;

	// Row starts at offset.
	ia[U_GNUM * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	aa[offset + 0] = dzodr * (v / ri) * D1_4_0;
	aa[offset + 1] = dzodr * (v / ri) * D1_4_1;
	aa[offset + 2] = dzodr * (v / ri) * D1_4_3;
	aa[offset + 3] = dzodr * (v / ri) * D1_4_4;
	aa[offset + 4] = -dzodr * dr2;

	// Columns.
	ja[offset + 0] = BASE + V_GNUM * dim + IDX(i - 2, j);
	ja[offset + 1] = BASE + V_GNUM * dim + IDX(i - 1, j);
	ja[offset + 2] = BASE + V_GNUM * dim + IDX(i + 1, j);
	ja[offset + 3] = BASE + V_GNUM * dim + IDX(i + 2, j);
	ja[offset + 4] = BASE + U_GNUM * dim + IDX(i, j);

	// All done.
	return;
}

void s_reg_4th_order
(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM
)
{
	double dzodr = dz / dr;
	double dr2 = dr * dr;
	double ri = (double)i - 1.5;

	// Row starts at offset.
	ia[U_GNUM * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	aa[offset + 0] = dzodr * (v / ri) * SO1_4_0;
	aa[offset + 1] = dzodr * (v / ri) * SO1_4_1;
	aa[offset + 2] = dzodr * (v / ri) * SO1_4_2;
	aa[offset + 3] = dzodr * (v / ri) * SO1_4_3;
	aa[offset + 4] = dzodr * (v / ri) * SO1_4_4;
	aa[offset + 5] = -dzodr * dr2;

	// Columns.
	ja[offset + 0] = BASE + V_GNUM * dim + IDX(i - 3, j);
	ja[offset + 1] = BASE + V_GNUM * dim + IDX(i - 2, j);
	ja[offset + 2] = BASE + V_GNUM * dim + IDX(i - 1, j);
	ja[offset + 3] = BASE + V_GNUM * dim + IDX(i    , j);
	ja[offset + 4] = BASE + V_GNUM * dim + IDX(i + 1, j);
	ja[offset + 5] = BASE + U_GNUM * dim + IDX(i, j);

	// All done.
	return;
}

void o_reg_4th_order
(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM
)
{
	double dzodr = dz / dr;
	double dr2 = dr * dr;
	double ri = (double)i - 1.5;

	// Row starts at offset.
	ia[U_GNUM * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	aa[offset + 0] = dzodr * (v / ri) * S1_4_0;
	aa[offset + 1] = dzodr * (v / ri) * S1_4_1;
	aa[offset + 2] = dzodr * (v / ri) * S1_4_2;
	aa[offset + 3] = dzodr * (v / ri) * S1_4_3;
	aa[offset + 4] = dzodr * (v / ri) * S1_4_4;
	aa[offset + 5] = -dzodr * dr2;

	// Columns.
	ja[offset + 0] = BASE + V_GNUM * dim + IDX(i - 4, j);
	ja[offset + 1] = BASE + V_GNUM * dim + IDX(i - 3, j);
	ja[offset + 2] = BASE + V_GNUM * dim + IDX(i - 2, j);
	ja[offset + 3] = BASE + V_GNUM * dim + IDX(i - 1, j);
	ja[offset + 4] = BASE + V_GNUM * dim + IDX(i    , j);
	ja[offset + 5] = BASE + U_GNUM * dim + IDX(i, j);

	// All done.
	return;
}