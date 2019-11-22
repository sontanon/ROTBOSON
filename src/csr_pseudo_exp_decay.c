#include "tools.h"
#include "omega_calc.h"

// Pseudo-Exponential decay comes from approximating the radial derivative assuming no 
// angular dependence as in the Robin case (see csr_pseudo_robin.c source file).
//
// The main idea of these subroutines is to implement the following boundary condition:
// We have that the scalar field phi = r**l * psi goes as 
//                           l
// lim          phi = A * sin(theta) * exp(-chi * rr) / rr  .
//    rr -> inf
// 
// Where A is a constant and chi = sqrt(m**2 - w**2). This can easily lead to that
//                                           l+1
// lim          psi = A * exp(-chi * rr) / rr  .
//    rr -> inf
//
// This implies that psi * exp(+chi * rr) decays as an l+1 Robin quantity (see Robin
// subroutine). For example, by taking equation (1) in the aforementioned source file,
// we have:
//                                                                           -(l+2)
// f1 = rr * d  (exp(chi * rr) * psi *) + (l + 1) * exp(chi * rr) * psi = O(rr   )  .
//            rr                                                              inf
//                                                                        -(l+2)
//    = exp(chi * rr) * (rr * d   psi + (rr * chi + (l + 1)) * psi) =  O(rr   )   .
//                             rr                                          inf
//
// As in pseudo-Robin, we can now approximate the radial derivative by taking a derivative
// along a single direction (r, z, or diagonal). Notice that there is an overall factor
// of exp(chi * rr) multiplying the above equation. Since this factor can explode very
// sensitively, we choose to eliminate it by multiplying all equations by exp(-chi * rr). 
// However, before this is done we must calculate the Jacobian using the correct equation
// above. In particular de omega derivative is:
// 
// (d f1 / d w) = (d chi / d w) * exp(chi * rr) * (rr * (f1 * exp(-chi *rr)) + rr * psi) .
// 
// For completeness we will also expand equation (2).
//                         2   2                                                   2    2
// f2 = exp(chi * rr) * (rr * d   psi + 2 * (chi * rr + l + 2) * rr * d  psi + (chi * rr + 2 * (l + 2) * chi * rr + (l + 1) * (l + 2)) * psi) .
//                             rr                                      rr
//
// (d f2 / d w) = (d chi / d w) * exp(chi * rr) * (rr * (f2 * exp(-chi * rr)) + rr * (2 * rr * d  psi + (2 * chi * rr + 2 * (l + 2)) * psi))  .
//                                                                                              rr

// Pseudo-Exponential decay along z direction.
void csr_z_pseudo_exp_decay_2nd(
	double *a, 			// CSR matrix values.
	MKL_INT *ia, 			// CSR matrix row beginnings.
	MKL_INT *ja,			// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,		// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 		// R integer coordinate.
	const MKL_INT j, 		// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT bound_error,	// Whether to use equation 1, 2, or 3 as above. 
	const double *u, 	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l			// Scalar field rotation number.
	)
{
	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Coordinates and angular factors.
	double r, z, r2, z2, rr, rr2, sec_theta, sec_theta2;

	// Derivatives.
	double u0, D1z_u, D2z_u;

	// Equations constants.
	double k0, k1, k2;

	// RHS and scale factor.
	double f = 1.0;
	// If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
	double scale = 1.0;
	
	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	r2 = r * r * (dr / dz) * (dr / dz);
	z2 = z * z;
	rr = sqrt(r2 + z2);

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = dz * sqrt(m2 - w2);

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;
	
	// Select boundary type.
	switch (bound_error)
	{
		//               -(l + 1)
		// Dirichlet O(rr   )     = exp(chi * rr) * psi.
		//               inf
		case 0:
			// Calculate RHS without exponential.
			u0 = u[k + IDX(i, j)];
			f = u0;

			// Set values.
			a[offset + 0] = (1.0) * scale; // CONSTANT!
			a[offset + 1] = scale * (rr * f) * (-dz * w / sqrt(m2 - w2)) * dw_du(v, m);

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);
			ja[offset + 1] = BASE + w_idx;

			break;
		//                       -(l + 2)
		// Exponential Decay O(rr   )     = exp(chi * rr) * ((rr / z) * rr * d  psi + (rr * chi + (l + 1)) * psi).
		//                       inf                                          z
		case 1:
			// Coordinates.
			sec_theta = (rr / z);

			// Derivatives.
			u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];

			// Equation constants.
			k0 = rr * sec_theta;
			k1 = (l + 1.0) + rr * chi;

			// Calculate RHS.
			f = (k0 * D1z_u + k1 * u0);

			// Set values.
			a[offset + 0] = ((+0.5) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((-2.0) * k0) * scale; // CONSTANT!
			a[offset + 2] = ((+1.5) * k0 + k1) * scale;
			a[offset + 3] = scale * ((u0 + f) * rr * (-dz * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j - 2);
			ja[offset + 1] = BASE + k + IDX(i, j - 1);
			ja[offset + 2] = BASE + k + IDX(i, j);
			ja[offset + 3] = BASE + w_idx;

			break;
		//                       -(l+3)                     2           2    2                                                             2    2
		// Exponential Decay O(rr    )  = exp(chi * rr) * (rr * (rr / z)  * d  psi + 2 * (chi * rr + l + 2) * rr * (rr / z) * d  psi + (chi * rr + 2 * (l + 2) * chi * rr + (l + 1) * (l + 2)) * psi) .
		//                       inf                                         z                                                 z
		case 2:
		case 3:
			// Coordinates.
			rr2 = rr * rr;
			sec_theta = (rr / z);
			sec_theta2 = sec_theta * sec_theta;

			// Derivatives.
			u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];
            D2z_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i, j - 1)] + (+4.0) * u[k + IDX(i, j - 2)] + (-1.0) * u[k + IDX(i, j - 3)];

			// Equation constants.
			k0 = rr2 * sec_theta2;
			k1 = 2.0 * (l + 2.0 * rr * chi) * rr * sec_theta;
			k2 = ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0));

			// Calculate RHS.
			f = (k0 * D2z_u + k1 * D1z_u + k2 * u0);

			// Set values.
			a[offset + 0] = ((-1.0) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((+4.0) * k0 + (+0.5) * k1) * scale; // CONSTANT!
			a[offset + 2] = ((-5.0) * k0 + (-2.0) * k1) * scale; // CONSTANT!
			a[offset + 3] = ((+2.0) * k0 + (+1.5) * k1 + k2) * scale;
			a[offset + 4] = scale * ((2.0 * (rr * sec_theta * D1z_u + (chi * rr + l + 2.0) * u0) + f) * rr * (-dz * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j - 3);
			ja[offset + 1] = BASE + k + IDX(i, j - 2);
			ja[offset + 2] = BASE + k + IDX(i, j - 1);
			ja[offset + 3] = BASE + k + IDX(i, j    );
			ja[offset + 4] = BASE + w_idx;

			break;
	}

	// All done.
	return;
}

// Pseudo-Exponential decay along r direction.
void csr_r_pseudo_exp_decay_2nd(
	double *a, 			// CSR matrix values.
	MKL_INT *ia, 			// CSR matrix row beginnings.
	MKL_INT *ja,			// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,		// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 		// R integer coordinate.
	const MKL_INT j, 		// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT bound_error,	// Whether to use equation 1, 2, or 3 as above. 
	const double *u, 	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l			// Scalar field rotation number.
	)
{
	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Coordinates and angular factors.
	double r, z, r2, z2, rr, rr2, csc_theta, csc_theta2;

	// Derivatives.
	double u0, D1r_u, D2r_u;

	// Equations constants.
	double k0, k1, k2;

	// RHS and scale factor.
	double f = 1.0;
	// If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
	double scale = 1.0;
	
	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	r2 = r * r;
	z2 = z * z * (dz / dr) * (dz/  dr);
	rr = sqrt(r2 + z2);

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = dr * sqrt(m2 - w2);

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;
	
	// Select boundary type.
	switch (bound_error)
	{
		//               -(l + 1)
		// Dirichlet O(rr   )     = exp(chi * rr) * psi.
		//               inf
		case 0:
			// Calculate RHS without exponential.
			u0 = u[k + IDX(i, j)];
			f = u0;

			// Set values.
			a[offset + 0] = (1.0) * scale; // CONSTANT!
			a[offset + 1] = scale * (rr * f) * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m);

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);
			ja[offset + 1] = BASE + w_idx;

			break;
		//                       -(l + 2)
		// Exponential Decay O(rr   )     = exp(chi * rr) * ((rr / r) * rr * d  psi + (rr * chi + (l + 1)) * psi).
		//                       inf                                          r
		case 1:
			// Coordinates.
			csc_theta = (rr / r);

			// Derivatives.
			u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];

			// Equation constants.
			k0 = rr * csc_theta;
			k1 = (l + 1.0) + rr * chi;

			// Calculate RHS.
			f = (k0 * D1r_u + k1 * u0);

			// Set values.
			a[offset + 0] = ((+0.5) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((-2.0) * k0) * scale; // CONSTANT!
			a[offset + 2] = ((+1.5) * k0 + k1) * scale;
			a[offset + 3] = scale * ((u0 + f) * rr * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j);
			ja[offset + 1] = BASE + k + IDX(i - 1, j);
			ja[offset + 2] = BASE + k + IDX(i    , j);
			ja[offset + 3] = BASE + w_idx;

			break;
		//                       -(l+3)                     2           2    2                                                             2    2
		// Exponential Decay O(rr    )  = exp(chi * rr) * (rr * (rr / r)  * d  psi + 2 * (chi * rr + l + 2) * rr * (rr / r) * d  psi + (chi * rr + 2 * (l + 2) * chi * rr + (l + 1) * (l + 2)) * psi) .
		//                       inf                                         r                                                 r
		case 2:
		case 3:
			// Coordinates.
			rr2 = rr * rr;
			csc_theta = (rr / r);
			csc_theta2 = csc_theta * csc_theta;

			// Derivatives.
			u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];
            D2r_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j)] + (+4.0) * u[k + IDX(i - 2, j)] + (-1.0) * u[k + IDX(i - 3, j)];

			// Equation constants.
			k0 = rr2 * csc_theta2;
			k1 = 2.0 * (l + 2.0 * rr * chi) * rr * csc_theta;
			k2 = ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0));

			// Calculate RHS.
			f = (k0 * D2r_u + k1 * D1r_u + k2 * u0);

			// Set values.
			a[offset + 0] = ((-1.0) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((+4.0) * k0 + (+0.5) * k1) * scale; // CONSTANT!
			a[offset + 2] = ((-5.0) * k0 + (-2.0) * k1) * scale; // CONSTNAT!
			a[offset + 3] = ((+2.0) * k0 + (+1.5) * k1 + k2) * scale;
			a[offset + 4] = scale * ((2.0 * (rr * csc_theta * D1r_u + (chi * rr + l + 2.0) * u0) + f) * rr * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 3, j);
			ja[offset + 1] = BASE + k + IDX(i - 2, j);
			ja[offset + 2] = BASE + k + IDX(i - 1, j);
			ja[offset + 3] = BASE + k + IDX(i    , j);
			ja[offset + 4] = BASE + w_idx;

			break;
	}

	// All done.
	return;
}

// Pseudo-Exponential decay along corner direction.
void csr_corner_pseudo_exp_decay_2nd(
	double *a, 			// CSR matrix values.
	MKL_INT *ia, 			// CSR matrix row beginnings.
	MKL_INT *ja,			// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,		// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 		// R integer coordinate.
	const MKL_INT j, 		// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT bound_error,	// Whether to use equation 1, 2, or 3 as above. 
	const double *u, 	// Solution u.
	const MKL_INT w_idx,	// Omega index.
	const double m,		// Scalar field mass.
	const MKL_INT l			// Scalar field rotation number.
	)
{
	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Coordinates and angular factors.
	double r, z, r2, z2, rr, rr2, dif_theta, dif_theta2;

	// Derivatives.
	double u0, D1d_u, D2d_u;

	// Equations constants.
	double k0, k1, k2;

	// RHS and scale factor.
	double f = 1.0;
	// If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
	double scale = 1.0;
	
	// Coordinates.
	r = (double)i - 0.5;
	z = (double)j - 0.5;
	r2 = r * r;
	z2 = z * z * (dz / dr) * (dz/  dr);
	rr = sqrt(r2 + z2);

	// Omega.
	double v = u[w_idx];
	double w = omega_calc(v, m);
	double w2 = w * w;
	double m2 = m * m;
	double chi = dr * sqrt(m2 - w2);

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;
	
	// Select boundary type.
	switch (bound_error)
	{
		//               -(l + 1)
		// Dirichlet O(rr   )     = exp(chi * rr) * psi.
		//               inf
		case 0:
			// Calculate RHS without exponential.
			u0 = u[k + IDX(i, j)];
			f = u0;

			// Set values.
			a[offset + 0] = (1.0) * scale; // CONSTANT!
			a[offset + 1] = scale * (rr * f) * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m);

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);
			ja[offset + 1] = BASE + w_idx;

			break;
		//                       -(l + 2)
		// Exponential Decay O(rr   )     = exp(chi * rr) * ((rr / r) * rr * d  psi + (rr * chi + (l + 1)) * psi).
		//                       inf                                          r
		case 1:
			// Coordinates.
			dif_theta = 1.0 / (cos(atan((r / z) * (dr / dz)) - atan(dr / dz)) * sqrt(1.0 + (dz / dr) * (dz / dr)));

			// Derivatives: rescale by diagonal step.
			u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];

			// Equation constants.
			k0 = rr * dif_theta;
			k1 = (l + 1.0) + rr * chi;

			// Calculate RHS.
			f = (k0 * D1d_u + k1 * u0);

			// Set values.
			a[offset + 0] = ((+0.5) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((-2.0) * k0) * scale; // CONSTANT!
			a[offset + 2] = ((+1.5) * k0 + k1) * scale;
			a[offset + 3] = scale * ((u0 + f) * rr * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j - 2);
			ja[offset + 1] = BASE + k + IDX(i - 1, j - 1);
			ja[offset + 2] = BASE + k + IDX(i    , j    );
			ja[offset + 3] = BASE + w_idx;

			break;
		//                       -(l+3)                     2           2    2                                                             2    2
		// Exponential Decay O(rr    )  = exp(chi * rr) * (rr * (rr / r)  * d  psi + 2 * (chi * rr + l + 2) * rr * (rr / r) * d  psi + (chi * rr + 2 * (l + 2) * chi * rr + (l + 1) * (l + 2)) * psi) .
		//                       inf                                         r                                                 r
		case 2:
		case 3:
			// Coordinates.
			rr2 = rr * rr;
			dif_theta = 1.0 / (cos(atan((r / z) * (dr / dz)) - atan(dr / dz)) * sqrt(1.0 + (dz / dr) * (dz / dr)));
			dif_theta2 = dif_theta * dif_theta;

			// Derivatives.
			u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];
            D2d_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j - 1)] + (+4.0) * u[k + IDX(i - 2, j - 2)] + (-1.0) * u[k + IDX(i - 3, j - 3)];

			// Equation constants.
			k0 = rr2 * dif_theta2;
			k1 = 2.0 * (l + 2.0 * rr * chi) * rr * dif_theta;
			k2 = ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0));

			// Calculate RHS.
			f = (k0 * D2d_u + k1 * D1d_u + k2 * u0);

			// Set values.
			a[offset + 0] = ((-1.0) * k0) * scale; // CONSTANT!
			a[offset + 1] = ((+4.0) * k0 + (+0.5) * k1) * scale; // CONSTANT!
			a[offset + 2] = ((-5.0) * k0 + (-2.0) * k1) * scale; // CONSTANT!
			a[offset + 3] = ((+2.0) * k0 + (+1.5) * k1 + k2) * scale; 
			a[offset + 4] = scale * ((2.0 * (rr * dif_theta * D1d_u + (chi * rr + l + 2.0) * u0) + f) * rr * (-dr * w / sqrt(m2 - w2)) * dw_du(v, m));

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 3, j - 3);
			ja[offset + 1] = BASE + k + IDX(i - 2, j - 2);
			ja[offset + 2] = BASE + k + IDX(i - 1, j - 1);
			ja[offset + 3] = BASE + k + IDX(i    , j    );
			ja[offset + 4] = BASE + w_idx;

			break;
	}

	// All done.
	return;
}
