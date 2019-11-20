#include "tools.h"

// Pseudo-Robin comes from approximating the radial derivative assuming no angular dependence.
// Normally, we would have
//
// d   = (d r / d rr) d  + (d z / d rr) d  = (r / rr) d  + (z / rr) d  .
//  rr                 r                 z             r             z
//
// However, we now approximate that there is no angular dependence, so that
//                                            -1
// d   = (d rr / d r) d   => d   = (d rr / d r)  d  = (rr / r) d  .
//  r                  rr     rr                  r             r
//
// This way we can generate the Robin conditions with a derivative in only one direction.
// Recall that Robin approximates that a function u(rr) goes as rr -> inf as
//                       n        (n+1)     (n+2)
// u(rr) = u    + u  / rr  + u / rr + u  / rr + ...
//          inf    1          2        3
//
// We can thus form equations or boundary conditions that cancel more and more terms.
//
//                                -(n+1)
// 1. rr d  u + n (u - u   ) = O(rr   ),
//        rr            inf        inf
// 
//      2  2                                                   -(n+2)
// 2. rr  d  u + 2 (n + 1) rr d  u + n (n + 1) (u - u   ) = O(rr   ),
//         rr                  rr                    inf        inf
//
//      3  3                 2  2                                                                   -(n+3)
// 3. rr  d  u + 3 (n + 2) rr  d  u + 3 (n + 1) (n + 2) rr d  u + n (n + 1) (n + 2) (u - u   ) = O(rr   ).
//         rr                   rr                          rr                            inf        inf
// 
// For example, for the first condition, we have:
//
// (rr / r) rr d u + n (u - u   ) = 0.
//              r            inf

// Pseudo-Robin along z direction.
void csr_z_pseudo_robin_2nd(
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
	const MKL_INT n, 		// Robin decay type (see above).
	const MKL_INT bound_error)	// Whether to use equation 1, 2, or 3 as above.
{
	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z, r2, z2, rr, rr2, rr3, sec_theta, sec_theta2, sec_theta3;

	// Equations constants.
	double k0, k1, k2, k3;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			a[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = rr (rr / z) d u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf                 z    inf          inf
		case 1:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r * (dr / dz) * (dr / dz);
			z2 = z * z;
			rr = sqrt(r2 + z2);
			sec_theta = (rr / z);

			// Equation constants.
			k0 = rr * sec_theta;
			k1 = n;

			// Set values.
			a[offset + 0] = (+0.5) * k0;
			a[offset + 1] = (-2.0) * k0;
			a[offset + 2] = (+1.5) * k0 + k1;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j - 2);
			ja[offset + 1] = BASE + k + IDX(i, j - 1);
			ja[offset + 2] = BASE + k + IDX(i, j    );

			break;
		//               -(n+2)   2  2                                         2       2  2
		// Robin with O(rr   ): rr  d  u + 2 (n + 1) rr d  u + n (n + 1) u = rr (rr / z) d  u + 2 (n + 1) rr (rr / z) d  u + n (n + 1) u = 0.
		//                inf        rr                  rr                               z                            z
		case 2:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r * (dr / dz) * (dr / dz);
			z2 = z * z;
			rr = sqrt(r2 + z2);
			rr2 = rr * rr;
			sec_theta = (rr / z);
			sec_theta2 = sec_theta * sec_theta;

			// Equation constants.
			k0 = rr2 * sec_theta2;
			k1 = 2.0 * (n + 1.0) * rr * sec_theta;
			k2 = n * (n + 1.0);
	
			// Set values.
			a[offset + 0] = (-1.0) * k0;
			a[offset + 1] = (+4.0) * k0 + (+0.5) * k1;
			a[offset + 2] = (-5.0) * k0 + (-2.0) * k1;
			a[offset + 3] = (+2.0) * k0 + (+1.5) * k1 + k2;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j - 3);
			ja[offset + 1] = BASE + k + IDX(i, j - 2);
			ja[offset + 2] = BASE + k + IDX(i, j - 1);
			ja[offset + 3] = BASE + k + IDX(i, j    );

			break;
		//               -(n+3)   3       3  3                 2       2  2       
		// Robin with O(rr   ): rr (rr / z) d  u + 3 (n + 2) rr (rr / z) d  u + 3 (n + 1) (n + 2) rr (rr / z) d  u + n (n + 1) (n + 2) u = 0.
		//                inf                z                            z                                    z
		case 3:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r * (dr / dz) * (dr / dz);
			z2 = z * z;
			rr = sqrt(r2 + z2);
			rr2 = rr * rr;
			rr3 = rr2 * rr;
			sec_theta = (rr / z);
			sec_theta2 = sec_theta * sec_theta;
			sec_theta3 = sec_theta2 * sec_theta;

			// Equation constants.
			k0 = rr3 * sec_theta3;
			k1 = 3.0 * (n + 2.0) * rr2 * sec_theta2;
			k2 = 3.0 * (n + 1.0) * (n + 2.0) * rr * sec_theta;
			k3 = n * (n + 1.0) * (n + 2.0);
	
			// Set values.
			a[offset + 0] = ( +1.5) * k0;
			a[offset + 1] = ( -7.0) * k0 + (-1.0) * k1;
			a[offset + 2] = (+12.0) * k0 + (+4.0) * k1 + (+0.5) * k2;
			a[offset + 3] = ( -9.0) * k0 + (-5.0) * k1 + (-2.0) * k2;
			a[offset + 4] = ( +2.5) * k0 + (+2.0) * k1 + (+1.5) * k2 + k3;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j - 4);
			ja[offset + 1] = BASE + k + IDX(i, j - 3);
			ja[offset + 2] = BASE + k + IDX(i, j - 2);
			ja[offset + 3] = BASE + k + IDX(i, j - 1);
			ja[offset + 4] = BASE + k + IDX(i, j    );

			break;
	}

	// All done.
	return;
}

// Robin along r direction.
void csr_r_pseudo_robin_2nd(
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
	const MKL_INT n, 		// Robin decay type (see above).
	const MKL_INT bound_error)	// Whether to use equation 1, 2, or 3 as above
{
	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z, r2, z2, rr, rr2, rr3, csc_theta, csc_theta2, csc_theta3;

	// Equations constants.
	double k0, k1, k2, k3;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			a[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = rr (rr / r) d u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf                 r    inf          inf
		case 1:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2);
			csc_theta = (rr / r);

			// Equation constants.
			k0 = rr * csc_theta;
			k1 = n;

			// Set values.
			a[offset + 0] = (+0.5) * k0;
			a[offset + 1] = (-2.0) * k0;
			a[offset + 2] = (+1.5) * k0 + k1;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j);
			ja[offset + 1] = BASE + k + IDX(i - 1, j);
			ja[offset + 2] = BASE + k + IDX(i    , j);

			break;
		//               -(n+2)   2  2                                         2       2  2
		// Robin with O(rr   ): rr  d  u + 2 (n + 1) rr d  u + n (n + 1) u = rr (rr / r) d  u + 2 (n + 1) rr (rr / r) d  u + n (n + 1) u = 0.
		//                inf        rr                  rr                               r                            r
		case 2:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2);
			rr2 = rr * rr;
			csc_theta = (rr / r);
			csc_theta2 = csc_theta * csc_theta;

			// Equation constants.
			k0 = rr2 * csc_theta2;
			k1 = 2.0 * (n + 1.0) * rr * csc_theta;
			k2 = n * (n + 1.0);
	
			// Set values.
			a[offset + 0] = (-1.0) * k0;
			a[offset + 1] = (+4.0) * k0 + (+0.5) * k1;
			a[offset + 2] = (-5.0) * k0 + (-2.0) * k1;
			a[offset + 3] = (+2.0) * k0 + (+1.5) * k1 + k2;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 3, j);
			ja[offset + 1] = BASE + k + IDX(i - 2, j);
			ja[offset + 2] = BASE + k + IDX(i - 1, j);
			ja[offset + 3] = BASE + k + IDX(i    , j);

			break;
		//               -(n+3)   3       3  3                 2       2  2       
		// Robin with O(rr   ): rr (rr / r) d  u + 3 (n + 2) rr (rr / r) d  u + 3 (n + 1) (n + 2) rr (rr / r) d  u + n (n + 1) (n + 2) u = 0.
		//                inf                r                            r                                    r
		case 3:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2);
			rr2 = rr * rr;
			rr3 = rr2 * rr;
			csc_theta = (rr / r);
			csc_theta2 = csc_theta * csc_theta;
			csc_theta3 = csc_theta2 * csc_theta;

			// Equation constants.
			k0 = rr3 * csc_theta3;
			k1 = 3.0 * (n + 2.0) * rr2 * csc_theta2;
			k2 = 3.0 * (n + 1.0) * (n + 2.0) * rr * csc_theta;
			k3 = n * (n + 1.0) * (n + 2.0);
	
			// Set values.
			a[offset + 0] = ( +1.5) * k0;
			a[offset + 1] = ( -7.0) * k0 + (-1.0) * k1;
			a[offset + 2] = (+12.0) * k0 + (+4.0) * k1 + (+0.5) * k2;
			a[offset + 3] = ( -9.0) * k0 + (-5.0) * k1 + (-2.0) * k2;
			a[offset + 4] = ( +2.5) * k0 + (+2.0) * k1 + (+1.5) * k2 + k3;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 4, j);
			ja[offset + 1] = BASE + k + IDX(i - 3, j);
			ja[offset + 2] = BASE + k + IDX(i - 2, j);
			ja[offset + 3] = BASE + k + IDX(i - 1, j);
			ja[offset + 4] = BASE + k + IDX(i    , j);

			break;
	}

	// All done.
	return;

}

// Robin along corner.
// This calculation along the corner is a bit more subtle than above. This is because the diagonal direction
// is determined by the angle theta' = atan(dr / dz). This direction may not coincide with the rr direction.
// However, we can easily see that the diagonal direction is given by theta - theta', therefore the multiplication
// analogous to 1/cos(theta) in the z subroutine is 1/cos(theta - theta').
// Notice also that the spatial step in the diagonal direction is sqrt(dr**2 + dz**2).
// When NrTotal = NzTotal and dr = dz, the spatial step is sqrt(2) * dr and the diagonal direction coincides
// with the rr direction and 1/cos(theta - theta') = 1.
void csr_corner_pseudo_robin_2nd(
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
	const MKL_INT n, 		// Robin decay type (see above).
	const MKL_INT bound_error)	// Whether to use equation 1, 2, or 3 as above
{	// Grid offset.
	MKL_INT k = (g_num - 1) * dim;

	// Normalized coordinate values, i.e. dr and dz have been factored and canceled.
	double r, z, r2, z2, rr, rr2, rr3, dif_theta, dif_theta2, dif_theta3;

	// Equations constants.
	double k0, k1, k2, k3;

	// Row starts at offset.
	ia[k + IDX(i, j)] = BASE + offset;

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			a[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = rr (rr / z) d u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf                 z    inf          inf
		case 1:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2) / sqrt(1.0 + (dz / dr) * (dz / dr));
			dif_theta = 1.0 / cos(atan((r / z) * (dr / dz)) - atan(dr / dz));

			// Equation constants.
			k0 = rr * dif_theta;
			k1 = n;

			// Set values.
			a[offset + 0] = (+0.5) * k0;
			a[offset + 1] = (-2.0) * k0;
			a[offset + 2] = (+1.5) * k0 + k1; 

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j - 2);
			ja[offset + 1] = BASE + k + IDX(i - 1, j - 1);
			ja[offset + 2] = BASE + k + IDX(i    , j    );

			break;
		//               -(n+2)   2  2                                         2       2  2
		// Robin with O(rr   ): rr  d  u + 2 (n + 1) rr d  u + n (n + 1) u = rr (rr / z) d  u + 2 (n + 1) rr (rr / z) d  u + n (n + 1) u = 0.
		//                inf        rr                  rr                               z                            z
		case 2:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2) / sqrt(1.0 + (dz / dr) * (dz / dr));
			rr2 = rr * rr;
			dif_theta = 1.0 / cos(atan((r / z) * (dr / dz)) - atan(dr / dz));
			dif_theta2 = dif_theta * dif_theta;

			// Equation constants.
			k0 = rr2 * dif_theta2;
			k1 = 2.0 * (n + 1.0) * rr * dif_theta;
			k2 = n * (n + 1.0);
	
			// Set values.
			a[offset + 0] = (-1.0) * k0;
			a[offset + 1] = (+4.0) * k0 + (+0.5) * k1;
			a[offset + 2] = (-5.0) * k0 + (-2.0) * k1;
			a[offset + 3] = (+2.0) * k0 + (+1.5) * k1 + k2;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 3, j - 3);
			ja[offset + 1] = BASE + k + IDX(i - 2, j - 2);
			ja[offset + 2] = BASE + k + IDX(i - 1, j - 1);
			ja[offset + 3] = BASE + k + IDX(i    , j    );

			break;
		//               -(n+3)   3       3  3                 2       2  2       
		// Robin with O(rr   ): rr (rr / z) d  u + 3 (n + 2) rr (rr / z) d  u + 3 (n + 1) (n + 2) rr (rr / z) d  u + n (n + 1) (n + 2) u = 0.
		//                inf                z                            z                                    z
		case 3:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			r2 = r * r;
			z2 = z * z * (dz / dr) * (dz / dr);
			rr = sqrt(r2 + z2) / sqrt(1.0 + (dz / dr) * (dz / dr));
			rr2 = rr * rr;
			rr3 = rr2 * rr;
			dif_theta = 1.0 / cos(atan((r / z) * (dr / dz)) - atan(dr / dz));
			dif_theta2 = dif_theta * dif_theta;
			dif_theta3 = dif_theta2 * dif_theta;

			// Equation constants.
			k0 = rr3 * dif_theta3;
			k1 = 3.0 * (n + 2.0) * rr2 * dif_theta2;
			k2 = 3.0 * (n + 1.0) * (n + 2.0) * rr * dif_theta;
			k3 = n * (n + 1.0) * (n + 2.0);
	
			// Set values.
			a[offset + 0] = ( +1.5) * k0;
			a[offset + 1] = ( -7.0) * k0 + (-1.0) * k1;
			a[offset + 2] = (+12.0) * k0 + (+4.0) * k1 + (+0.5) * k2;
			a[offset + 3] = ( -9.0) * k0 + (-5.0) * k1 + (-2.0) * k2;
			a[offset + 4] = ( +2.5) * k0 + (+2.0) * k1 + (+1.5) * k2 + k3;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 4, j - 4);
			ja[offset + 1] = BASE + k + IDX(i - 3, j - 3);
			ja[offset + 2] = BASE + k + IDX(i - 2, j - 2);
			ja[offset + 3] = BASE + k + IDX(i - 1, j - 1);
			ja[offset + 4] = BASE + k + IDX(i    , j    );

			break;
	}

	// All done.
	return;
}
