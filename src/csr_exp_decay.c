#include "tools.h"
#include "omega_calc.h"

// This boundary condition is derived from a variable u that has the following 
// asymptotic behavior:
//
//                          l+1
// u -> u exp(-chi * rr) / r
//       1                                 
//                                         
// Where u_1 is a constant and chi = sqrt(m*m - w*w).
// Thus, u * exp(+chi * rr) has a Robin boundary condition and we have the equation:
//
//                                                                -(l+2)
// f = exp(+chi * rr) * (rr d   u + (l + 1 + chi * rr) * u ) = O(rr   ).
//                           rr                                    inf
//
// Notice that the Jacobian calculation will require a derivative with respect to w.
// This is
// 
//
// d  f = (d  chi) * (rr f + exp(+chi * rr) * (rr * u)).
//  w       w
//
// However, when calculating the RHS, we eliminate the factor exp(+chi * rr), since it
// can give rise to numerical problems.
// When calculating the r,z derivatives, it may be necessary to use one-sided stencils.
// Finally, we have also chosen to scale the entire solution by a factor of dr * dz / (rr * rr).
// This really serves no purpose except to make the global RHS a smoother function, though this
// has not really been thoroughly tested.

// Finite difference coefficients for second order.
// Centered
const double D10 = -0.5;
const double D11 = 0.0;
const double D12 = +0.5;
// One-sided.
const double S10 = +0.5;
const double S11 = -2.0;
const double S12 = +1.5;

// Decay along z direction.
void z_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i + 0.5 - ghost;
	z = (double)j + 0.5 - ghost;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double f = r * (D10 * u[k + IDX(i - 1, j)] + D12 * u[k + IDX(i + 1, j)]) 
		+ z * (S10 * u[k + IDX(i, j - 2)] + S11 * u[k + IDX(i, j - 1)] + S12 * psi) 
		+ (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((D10) * r) * scale;
	aa[offset + 1] = ((S10) * z) * scale;
	aa[offset + 2] = ((S11) * z) * scale;
	aa[offset + 3] = ((S12) * z + (rr * chi + l + 1.0)) * scale;
	aa[offset + 4] = ((D12) * r) * scale;
	aa[offset + 5] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 1, j);
	ja[offset + 1] = BASE + k + IDX(i, j - 2);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i + 1, j);
	ja[offset + 5] = BASE + w_idx;

	// All done.
	return;
}

// Decay along r direction.
void r_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i + 0.5 - ghost;
	z = (double)j + 0.5 - ghost;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double f = r * (S10 * u[k + IDX(i - 2, j)] + S11 * u[k + IDX(i - 1, j)] + S12 * psi) 
		+ z * (D10 * u[k + IDX(i, j - 1)] + D12 * u[k + IDX(i, j + 1)]) 
		+ (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((S10) * r) * scale;
	aa[offset + 1] = ((S11) * r) * scale;
	aa[offset + 2] = ((D10) * z) * scale;
	aa[offset + 3] = ((S12) * r + (rr * chi + l + 1.0)) * scale;
	aa[offset + 4] = ((D12) * z) * scale;
	aa[offset + 5] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i, j + 1);
	ja[offset + 5] = BASE + w_idx;

	// All done.
	return;
}

// Decay along corner.
void corner_decay_2nd_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i + 0.5 - ghost;
	z = (double)j + 0.5 - ghost;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double f = r * (S10 * u[k + IDX(i - 2, j)] + S11 * u[k + IDX(i - 1, j)] + S12 * psi) 
		+ z * (S10 * u[k + IDX(i, j - 2)] + S11 * u[k + IDX(i, j - 1)] + S12 * psi) 
		+ (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((S10) * r) * scale;
	aa[offset + 1] = ((S11) * r) * scale;
	aa[offset + 2] = ((S10) * z) * scale;
	aa[offset + 3] = ((S11) * z) * scale;
	aa[offset + 4] = ((S12) * (r + z) + (rr * chi + l + 1.0)) * scale;
	aa[offset + 5] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 2);
	ja[offset + 3] = BASE + k + IDX(i, j - 1);
	ja[offset + 4] = BASE + k + IDX(i, j);
	ja[offset + 5] = BASE + w_idx;

	// All done.
	return;
}

// Now comes the fourth order implementation.
// Finite difference coefficients for fourth order.
// Centered.
const double D1_4_0 = +1.0 / 12.0;
const double D1_4_1 = -2.0 / 3.0;
const double D1_4_2 = 0.0;
const double D1_4_3 = +2.0 / 3.0;
const double D1_4_4 = -1.0 / 12.0;
// One-sided.
const double S1_4_0 = +0.25;
const double S1_4_1 = -4.0 / 3.0;
const double S1_4_2 = +3.0;
const double S1_4_3 = -4.0;
const double S1_4_4 = 25.0 / 12.0;
// Semi-one-sided.
const double SO1_4_0 = -1.0 / 12.0;
const double SO1_4_1 = +0.5;
const double SO1_4_2 = -1.5;
const double SO1_4_3 = +5.0 / 6.0;
const double SO1_4_4 = +0.25;

// Decay along z direction.
void z_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double ri, zi;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	ri = (double)i + 0.5 - ghost;
	zi = (double)j + 0.5 - ghost;
	rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double Dr_psi = D1_4_0 * u[k + IDX(i - 2, j)] + D1_4_1 * u[k + IDX(i - 1, j)] + D1_4_3 * u[k + IDX(i + 1, j)] + D1_4_4 * u[k + IDX(i + 2, j)];
	double Dz_psi = S1_4_0 * u[k + IDX(i, j - 4)] + S1_4_1 * u[k + IDX(i, j - 3)] + S1_4_2 * u[k + IDX(i, j - 2)] + S1_4_3 * u[k + IDX(i, j - 1)] + S1_4_4 * psi;
	double f = ri * Dr_psi + zi * Dz_psi + (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((D1_4_0) * ri) * scale;
	aa[offset + 1] = ((D1_4_1) * ri) * scale;
	aa[offset + 2] = ((S1_4_0) * zi) * scale;
	aa[offset + 3] = ((S1_4_1) * zi) * scale;
	aa[offset + 4] = ((S1_4_2) * zi) * scale;
	aa[offset + 5] = ((S1_4_3) * zi) * scale;
	aa[offset + 6] = ((S1_4_4) * zi + (rr * chi + l + 1.0)) * scale;
	aa[offset + 7] = ((D1_4_3) * ri) * scale;
	aa[offset + 8] = ((D1_4_4) * ri) * scale;
	aa[offset + 9] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

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
	ja[offset + 9] = BASE + w_idx;

	// All done.
	return;
}

// Decay along r direction.
void r_decay_4th_order
(	
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double ri, zi;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	ri = (double)i + 0.5 - ghost;
	zi = (double)j + 0.5 - ghost;
	rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double Dr_psi = S1_4_0 * u[k + IDX(i - 4, j)] + S1_4_1 * u[k + IDX(i - 3, j)] + S1_4_2 * u[k + IDX(i - 2, j)] + S1_4_3 * u[k + IDX(i - 1, j)] + S1_4_4 * psi;
	double Dz_psi = D1_4_0 * u[k + IDX(i, j - 2)] + D1_4_1 * u[k + IDX(i, j - 1)] + D1_4_3 * u[k + IDX(i, j + 1)] + D1_4_4 * u[k + IDX(i, j + 2)];
	double f = ri * Dr_psi + zi * Dz_psi + (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * ri) * scale;
	aa[offset + 1] = ((S1_4_1) * ri) * scale;
	aa[offset + 2] = ((S1_4_2) * ri) * scale;
	aa[offset + 3] = ((S1_4_3) * ri) * scale;
	aa[offset + 4] = ((D1_4_0) * zi) * scale;
	aa[offset + 5] = ((D1_4_1) * zi) * scale;
	aa[offset + 6] = ((S1_4_4) * ri + (rr * chi + l + 1.0)) * scale;
	aa[offset + 7] = ((D1_4_3) * zi) * scale;
	aa[offset + 8] = ((D1_4_4) * zi) * scale;
	aa[offset + 9] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

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
	ja[offset + 9] = BASE + w_idx;

	// All done.
	return;
}

// Decay along corner direction.
void corner_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double ri, zi;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	ri = (double)i + 0.5 - ghost;
	zi = (double)j + 0.5 - ghost;
	rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double Dr_psi = S1_4_0 * u[k + IDX(i - 4, j)] + S1_4_1 * u[k + IDX(i - 3, j)] + S1_4_2 * u[k + IDX(i - 2, j)] + S1_4_3 * u[k + IDX(i - 1, j)] + S1_4_4 * psi;
	double Dz_psi = S1_4_0 * u[k + IDX(i, j - 4)] + S1_4_1 * u[k + IDX(i, j - 3)] + S1_4_2 * u[k + IDX(i, j - 2)] + S1_4_3 * u[k + IDX(i, j - 1)] + S1_4_4 * psi;
	double f = ri * Dr_psi + zi * Dz_psi + (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * ri) * scale;
	aa[offset + 1] = ((S1_4_1) * ri) * scale;
	aa[offset + 2] = ((S1_4_2) * ri) * scale;
	aa[offset + 3] = ((S1_4_3) * ri) * scale;
	aa[offset + 4] = ((S1_4_0) * zi) * scale;
	aa[offset + 5] = ((S1_4_1) * zi) * scale;
	aa[offset + 6] = ((S1_4_2) * zi) * scale;
	aa[offset + 7] = ((S1_4_3) * zi) * scale;
	aa[offset + 8] = ((S1_4_4) * (ri + zi) + (chi * rr + l + 1.0)) * scale;
	aa[offset + 9] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

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
	ja[offset + 9] = BASE + w_idx;

	// All done.
	return;
}

// Decay along z but with semi-onesided r derivative.
void z_so_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double ri, zi;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	ri = (double)i + 0.5 - ghost;
	zi = (double)j + 0.5 - ghost;
	rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double Dr_psi = SO1_4_0 * u[k + IDX(i - 3, j)] + SO1_4_1 * u[k + IDX(i - 2, j)] + SO1_4_2 * u[k + IDX(i - 1, j)] + SO1_4_3 * psi + SO1_4_4 * u[k + IDX(i + 1, j)];
	double Dz_psi = S1_4_0 * u[k + IDX(i, j - 4)] + S1_4_1 * u[k + IDX(i, j - 3)] + S1_4_2 * u[k + IDX(i, j - 2)] + S1_4_3 * u[k + IDX(i, j - 1)] + S1_4_4 * psi;
	double f = ri * Dr_psi + zi * Dz_psi + (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((SO1_4_0) * ri) * scale;
	aa[offset + 1] = ((SO1_4_1) * ri) * scale;
	aa[offset + 2] = ((SO1_4_2) * ri) * scale;
	aa[offset + 3] = ((S1_4_0) * zi) * scale;
	aa[offset + 4] = ((S1_4_1) * zi) * scale;
	aa[offset + 5] = ((S1_4_2) * zi) * scale;
	aa[offset + 6] = ((S1_4_3) * zi) * scale;
	aa[offset + 7] = ((S1_4_4) * zi + (SO1_4_3) * ri + (rr * chi + l + 1.0)) * scale;
	aa[offset + 8] = ((SO1_4_4) * ri) * scale;
	aa[offset + 9] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

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
	ja[offset + 9] = BASE + w_idx;

	// All done.
	return;
}

// Decay along r but with semi-onesided z derivative.
void r_so_decay_4th_order
(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost, 	// Number of ghost zones.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz, 	// Z spatial step.
	double *u,	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l 	// Scalar field rotation number.
)
{
	// Grid offset.
	MKL_INT k = g_num * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double ri, zi;
	double rr2, rr;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	ri = (double)i + 0.5 - ghost;
	zi = (double)j + 0.5 - ghost;
	rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
	rr = sqrt(rr2);
	// Scale factor.
	scale = dr * dz / rr2;

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// RHS.
	double psi = u[k + IDX(i, j)];
	double Dr_psi = S1_4_0 * u[k + IDX(i - 4, j)] + S1_4_1 * u[k + IDX(i - 3, j)] + S1_4_2 * u[k + IDX(i - 2, j)] + S1_4_3 * u[k + IDX(i - 1, j)] + S1_4_4 * psi;
	double Dz_psi = SO1_4_0 * u[k + IDX(i, j - 3)] + SO1_4_1 * u[k + IDX(i, j - 2)] + SO1_4_2 * u[k + IDX(i, j - 1)] + SO1_4_3 * psi + SO1_4_4 * u[k + IDX(i, j + 1)];
	double f = ri * Dr_psi + zi * Dz_psi + (rr * chi + l + 1.0) * psi;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * ri) * scale;
	aa[offset + 1] = ((S1_4_1) * ri) * scale;
	aa[offset + 2] = ((S1_4_2) * ri) * scale;
	aa[offset + 3] = ((S1_4_3) * ri) * scale;
	aa[offset + 4] = ((SO1_4_0) * zi) * scale;
	aa[offset + 5] = ((SO1_4_1) * zi) * scale;
	aa[offset + 6] = ((SO1_4_2) * zi) * scale;
	aa[offset + 7] = ((S1_4_4) * ri + (SO1_4_3) * zi + (rr * chi + l + 1.0)) * scale;
	aa[offset + 8] = ((SO1_4_4) * zi) * scale;
	aa[offset + 9] = dw_du(v, m) * (rr * f + rr * psi) * (-w / chi) * scale;

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
	ja[offset + 9] = BASE + w_idx;

	// All done.
	return;
}

/*  DEPRECATED: OLD BOUNDARY CONDITION
//
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

// Finite difference coefficients for second order.
// Centered
const double D10 = -0.5;
const double D11 = 0.0;
const double D12 = +0.5;
// One-sided.
const double S10 = +0.5;
const double S11 = -2.0;
const double S12 = +1.5;

// Exponential decay along z direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((D10) * r) * scale;
	aa[offset + 1] = ((S10) * z) * scale;
	aa[offset + 2] = ((S11) * z) * scale;
	aa[offset + 3] = ((S12) * z) * scale;
	aa[offset + 4] = ((D12) * r) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 1, j);
	ja[offset + 1] = BASE + k + IDX(i, j - 2);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i + 1, j);

	// All done.
	return;
}

// Exponential decay along r direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((S10) * r) * scale;
	aa[offset + 1] = ((S11) * r) * scale;
	aa[offset + 2] = ((D10) * z) * scale;
	aa[offset + 3] = ((S12) * r) * scale;
	aa[offset + 4] = ((D12) * z) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 1);
	ja[offset + 3] = BASE + k + IDX(i, j    );
	ja[offset + 4] = BASE + k + IDX(i, j + 1);

	// All done.
	return;
}

// Exponential decay along corner direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((S10) * r) * scale;
	aa[offset + 1] = ((S11) * r) * scale;
	aa[offset + 2] = ((S10) * z) * scale;
	aa[offset + 3] = ((S11) * z) * scale;
	aa[offset + 4] = ((S12) * (r + z)) * scale;

	// Column indices.
	ja[offset + 0] = BASE + k + IDX(i - 2, j);
	ja[offset + 1] = BASE + k + IDX(i - 1, j);
	ja[offset + 2] = BASE + k + IDX(i, j - 2);
	ja[offset + 3] = BASE + k + IDX(i, j - 1);
	ja[offset + 4] = BASE + k + IDX(i, j);

	// All done.
	return;
}

// Now comes the fourth order implementation.
// Finite difference coefficients for fourth order.
// Centered.
const double D1_4_0 = +1.0 / 12.0;
const double D1_4_1 = -2.0 / 3.0;
const double D1_4_2 = 0.0;
const double D1_4_3 = +2.0 / 3.0;
const double D1_4_4 = -1.0 / 12.0;
// One-sided.
const double S1_4_0 = +0.25;
const double S1_4_1 = -4.0 / 3.0;
const double S1_4_2 = +3.0;
const double S1_4_3 = -4.0;
const double S1_4_4 = 25.0 / 12.0;
// Semi-one-sided.
const double SO1_4_0 = -1.0 / 12.0;
const double SO1_4_1 = +0.5;
const double SO1_4_2 = -1.5;
const double SO1_4_3 = +5.0 / 6.0;
const double SO1_4_4 = +0.25;

// Exponential decay along z direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((D1_4_0) * r) * scale;
	aa[offset + 1] = ((D1_4_1) * r) * scale;
	aa[offset + 2] = ((S1_4_0) * z) * scale;
	aa[offset + 3] = ((S1_4_1) * z) * scale;
	aa[offset + 4] = ((S1_4_2) * z) * scale;
	aa[offset + 5] = ((S1_4_3) * z) * scale;
	aa[offset + 6] = ((S1_4_4) * z) * scale;
	aa[offset + 7] = ((D1_4_3) * r) * scale;
	aa[offset + 8] = ((D1_4_4) * r) * scale;

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

// Exponential decay along r direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * r) * scale;
	aa[offset + 1] = ((S1_4_1) * r) * scale;
	aa[offset + 2] = ((S1_4_2) * r) * scale;
	aa[offset + 3] = ((S1_4_3) * r) * scale;
	aa[offset + 4] = ((D1_4_0) * z) * scale;
	aa[offset + 5] = ((D1_4_1) * z) * scale;
	aa[offset + 6] = ((S1_4_4) * r) * scale;
	aa[offset + 7] = ((D1_4_3) * z) * scale;
	aa[offset + 8] = ((D1_4_4) * z) * scale;

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

// Exponential decay along corner direction.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * r) * scale;
	aa[offset + 1] = ((S1_4_1) * r) * scale;
	aa[offset + 2] = ((S1_4_2) * r) * scale;
	aa[offset + 3] = ((S1_4_3) * r) * scale;
	aa[offset + 4] = ((S1_4_0) * z) * scale;
	aa[offset + 5] = ((S1_4_1) * z) * scale;
	aa[offset + 6] = ((S1_4_2) * z) * scale;
	aa[offset + 7] = ((S1_4_3) * z) * scale;
	aa[offset + 8] = ((S1_4_4) * (r + z)) * scale;

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

// Exponential decay along z but with semi-onesided r derivative.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((SO1_4_0) * r) * scale;
	aa[offset + 1] = ((SO1_4_1) * r) * scale;
	aa[offset + 2] = ((SO1_4_2) * r) * scale;
	aa[offset + 3] = ((S1_4_0) * z) * scale;
	aa[offset + 4] = ((S1_4_1) * z) * scale;
	aa[offset + 5] = ((S1_4_2) * z) * scale;
	aa[offset + 6] = ((S1_4_3) * z) * scale;
	aa[offset + 7] = ((S1_4_4) * z + (SO1_4_3) * r) * scale;
	aa[offset + 8] = ((SO1_4_4) * r) * scale;

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

// Expoential decay along r but with semi-onesided z derivative.
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
	double rr2;
	double scale;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Coordinates.
	r = (double)i - 1.5;
	z = (double)j - 1.5;
	rr2 = r * r * dr * dr + z * z * dz * dz;
	scale = dr * dz / rr2;

	// Set values.
	aa[offset + 0] = ((S1_4_0) * r) * scale;
	aa[offset + 1] = ((S1_4_1) * r) * scale;
	aa[offset + 2] = ((S1_4_2) * r) * scale;
	aa[offset + 3] = ((S1_4_3) * r) * scale;
	aa[offset + 4] = ((SO1_4_0) * z) * scale;
	aa[offset + 5] = ((SO1_4_1) * z) * scale;
	aa[offset + 6] = ((SO1_4_2) * z) * scale;
	aa[offset + 7] = ((S1_4_4) * r + (SO1_4_3) * z) * scale;
	aa[offset + 8] = ((SO1_4_4) * z) * scale;

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
    DEPRECATED: OLD BOUNDARY CONDITION */