#include "tools.h"

const double D10 = -0.5;
const double D11 = 0.0;
const double D12 = +0.5;

const double S10 = +0.5;
const double S11 = -2.0;
const double S12 = +1.5;

// This boundary condition is derived from a variable u that has the following
// asymptotic behavior:
//                  n
// exp(u) -> u  / rr  ,
//            1
// where u_1 is a constant and n is an integer.
// Thus, exp(u) has a Robin boundary condition and we have the equation:
//                    -(n+1)
// rr d   u + n  = O(rr   ) .
//     rr              inf
//
// When calcuting the r,z derivatives it may be necessary to use one-sided stencils.

// Robin along z direction.
void z_exp_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;

	// Set values.
	aa[offset + 0] = (D10) * r;
	aa[offset + 1] = (S10) * z;
	aa[offset + 2] = (S11) * z;
	aa[offset + 3] = (S12) * z;
	aa[offset + 4] = (D12) * r;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 1, j);
	ja[offset + 1] = BASE + k + IDX(i, j - 2);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i + 1, j);

	// All done.
	return;
}

// Robin along r direction.
void r_exp_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;

	// Set values.
	aa[offset + 0] = (S10) * r;
	aa[offset + 1] = (S11) * r;
	aa[offset + 2] = (D10) * z;
	aa[offset + 3] = (S12) * r;
	aa[offset + 4] = (D12) * z;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i, j + 1);

	// All done.
	return;
}

// Robin along corner direction.
void corner_exp_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;

	// Set values.
	aa[offset + 0] = (S10) * r;
	aa[offset + 1] = (S11) * r;
	aa[offset + 2] = (S10) * z;
	aa[offset + 3] = (S11) * z;
	aa[offset + 4] = (S12) * (r + z);

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 2);
	ja[offset + 3] = BASE + k + IDX(i, j - 1);
	ja[offset + 4] = BASE + k + IDX(i, j);

	// All done.
	return;
}

const double D1_4_0 = +1.0 / 12.0;
const double D1_4_1 = -2.0 / 3.0;
const double D1_4_2 = 0.0;
const double D1_4_3 = +2.0 / 3.0;
const double D1_4_4 = -1.0 / 12.0;

const double S1_4_0 = +0.25;
const double S1_4_1 = -4.0 / 3.0;
const double S1_4_2 = +3.0;
const double S1_4_3 = -4.0;
const double S1_4_4 = 25.0 / 12.0;

const double SO1_4_0 = -1.0 / 12.0;
const double SO1_4_1 = +0.5;
const double SO1_4_2 = -1.5;
const double SO1_4_3 = +5.0 / 6.0;
const double SO1_4_4 = +0.25;

// Robin along z direction.
void z_exp_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;

	// Set values.
	aa[offset + 0] = (D1_4_0) * r;
	aa[offset + 1] = (D1_4_1) * r;
	aa[offset + 2] = (S1_4_0) * z;
	aa[offset + 3] = (S1_4_1) * z;
	aa[offset + 4] = (S1_4_2) * z;
	aa[offset + 5] = (S1_4_3) * z;
	aa[offset + 6] = (S1_4_4) * z;
	aa[offset + 7] = (D1_4_3) * r;
	aa[offset + 8] = (D1_4_4) * r;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 4);
	ja[offset + 3] = BASE + k + IDX(i, j - 3);
	ja[offset + 4] = BASE + k + IDX(i, j - 2);
	ja[offset + 5] = BASE + k + IDX(i, j - 1);
	ja[offset + 6] = BASE + k + IDX(i, j    );
	ja[offset + 7] = BASE + k + IDX(i + 1, j);
	ja[offset + 8] = BASE + k + IDX(i + 2, j);

	// All done.
	return;
}

// Robin along r direction.
void r_exp_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;

	// Set values.
	aa[offset + 0] = (S1_4_0) * r;
	aa[offset + 1] = (S1_4_1) * r;
	aa[offset + 2] = (S1_4_2) * r;
	aa[offset + 3] = (S1_4_3) * r;
	aa[offset + 4] = (D1_4_0) * z;
	aa[offset + 5] = (D1_4_1) * z;
	aa[offset + 6] = (S1_4_4) * r;
	aa[offset + 7] = (D1_4_3) * z;
	aa[offset + 8] = (D1_4_4) * z;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 4, j);
	ja[offset + 1] = BASE + k + IDX(i - 3, j);
	ja[offset + 2] = BASE + k + IDX(i - 2, j);
	ja[offset + 3] = BASE + k + IDX(i - 1, j);
	ja[offset + 4] = BASE + k + IDX(i, j - 2);
	ja[offset + 5] = BASE + k + IDX(i, j - 1);
	ja[offset + 6] = BASE + k + IDX(i, j);
	ja[offset + 7] = BASE + k + IDX(i, j + 1);
	ja[offset + 8] = BASE + k + IDX(i, j + 2);

	// All done.
	return;
}

// Robin along corner direction.
void corner_exp_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;

	// Set values.
	aa[offset + 0] = (S1_4_0) * r;
	aa[offset + 1] = (S1_4_1) * r;
	aa[offset + 2] = (S1_4_2) * r;
	aa[offset + 3] = (S1_4_3) * r;
	aa[offset + 4] = (S1_4_0) * z;
	aa[offset + 5] = (S1_4_1) * z;
	aa[offset + 7] = (S1_4_2) * z;
	aa[offset + 8] = (S1_4_3) * z;
	aa[offset + 6] = (S1_4_4) * (r + z);

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 4, j);
	ja[offset + 1] = BASE + k + IDX(i - 3, j);
	ja[offset + 2] = BASE + k + IDX(i - 2, j);
	ja[offset + 3] = BASE + k + IDX(i - 1, j);
	ja[offset + 4] = BASE + k + IDX(i, j - 4);
	ja[offset + 5] = BASE + k + IDX(i, j - 3);
	ja[offset + 6] = BASE + k + IDX(i, j - 2);
	ja[offset + 7] = BASE + k + IDX(i, j - 1);
	ja[offset + 8] = BASE + k + IDX(i, j);

	// All done.
	return;
}

// Robin along z but with semi-onesided r derivative.
void z_so_exp_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error	// Whether to use Dirichlet (0) or Robin (1).
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;

	// Set values.
	aa[offset + 0] = (SO1_4_0) * r;
	aa[offset + 1] = (SO1_4_1) * r;
	aa[offset + 2] = (SO1_4_2) * r;
	aa[offset + 3] = (S1_4_0) * z;
	aa[offset + 4] = (S1_4_1) * z;
	aa[offset + 5] = (S1_4_2) * z;
	aa[offset + 6] = (S1_4_3) * z;
	aa[offset + 7] = (S1_4_4) * z + (SO1_4_3) * r;
	aa[offset + 8] = (SO1_4_4) * r;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 3, j);
	ja[offset + 1] = BASE + k + IDX(i - 2, j);
	ja[offset + 2] = BASE + k + IDX(i - 1, j);
	ja[offset + 3] = BASE + k + IDX(i, j - 4);
	ja[offset + 4] = BASE + k + IDX(i, j - 3);
	ja[offset + 5] = BASE + k + IDX(i, j - 2);
	ja[offset + 6] = BASE + k + IDX(i, j - 1);
	ja[offset + 7] = BASE + k + IDX(i, j    );
	ja[offset + 8] = BASE + k + IDX(i + 1, j);

	// All done.
	return;
}

// Robin along r but with semi-onesided z derivative.
void r_so_exp_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz 	// Z spatial step.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;

	// Set values.
	aa[offset + 0] = (S1_4_0) * r;
	aa[offset + 1] = (S1_4_1) * r;
	aa[offset + 2] = (S1_4_2) * r;
	aa[offset + 3] = (S1_4_3) * r;
	aa[offset + 4] = (SO1_4_0) * z;
	aa[offset + 5] = (SO1_4_1) * z;
	aa[offset + 7] = (SO1_4_2) * z;
	aa[offset + 6] = (S1_4_4) * r + (SO1_4_3) * z;
	aa[offset + 8] = (SO1_4_4) * z;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 4, j);
	ja[offset + 1] = BASE + k + IDX(i - 3, j);
	ja[offset + 2] = BASE + k + IDX(i - 2, j);
	ja[offset + 3] = BASE + k + IDX(i - 1, j);
	ja[offset + 4] = BASE + k + IDX(i, j - 3);
	ja[offset + 5] = BASE + k + IDX(i, j - 2);
	ja[offset + 6] = BASE + k + IDX(i, j - 1);
	ja[offset + 7] = BASE + k + IDX(i, j);			
	ja[offset + 8] = BASE + k + IDX(i, j + 1);

	// All done.
	return;
}