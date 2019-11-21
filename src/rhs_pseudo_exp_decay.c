#include "tools.h"
#include "omega_calc.h"

// These subroutines will calculate the right hand sides of the 
// discretization appropriate to the pseudo-exponential decay boundary 
// conditions detailed in the source file csr_pseudo_exp_decay.c

// Along z direction.
void rhs_z_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l         // Scalar field rotation number.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, sec_theta, sec_theta2;

    // Derivatives.
    double u0, D1z_u, D2z_u;

    // Rescale factor.
    // If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
    double rescale = 1.0;

    // Omega.
    double v, w, w2, m2, chi;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = u0 * scale * rescale;
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r * (dr / dz) * (dr / dz);
            z2 = z * z;
            rr = sqrt(r2 + z2);
            sec_theta = (rr / z);

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dz * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr * sec_theta * D1z_u + (rr * chi + l + 1.0) * u0);
            break;

        case 2:
        case 3:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r * (dr / dz) * (dr / dz);
            z2 = z * z;
            rr = sqrt(r2 + z2);
            rr2 = rr * rr;
            sec_theta = (rr / z);
            sec_theta2 = sec_theta * sec_theta;

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dz * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];
            D2z_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i, j - 1)] + (+4.0) * u[k + IDX(i, j - 2)] + (-1.0) * u[k + IDX(i, j - 3)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr2 * sec_theta2 * D2z_u
                + 2.0 * (rr * chi + l + 2.0) * rr * sec_theta * D1z_u
                + ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0)) * u0);
            break;
    }

    // All done.
    return;
}

// Along r direction.
void rhs_r_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l         // Scalar field rotation number.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, csc_theta, csc_theta2;

    // Derivatives.
    double u0, D1r_u, D2r_u;

    // Rescale factor.
    // If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
    double rescale = 1.0;

    // Omega.
    double v, w, w2, m2, chi;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = u0 * scale * rescale;
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2);
            csc_theta = (rr / r);

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dr * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr * csc_theta * D1r_u + (rr * chi + l + 1.0) * u0);
            break;

        case 2:
        case 3:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2);
            rr2 = rr * rr;
            csc_theta = (rr / r);
            csc_theta2 = csc_theta * csc_theta;

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dr * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];
            D2r_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j)] + (+4.0) * u[k + IDX(i - 2, j)] + (-1.0) * u[k + IDX(i - 3, j)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr2 * csc_theta2 * D2r_u
                + 2.0 * (rr * chi + l + 2.0) * rr * csc_theta * D1r_u
                + ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0)) * u0);
            break;
    }

    // All done.
    return;
}

// Along corner direction.
void rhs_corner_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l         // Scalar field rotation number.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, dif_theta, dif_theta2;

    // Derivatives.
    double u0, D1d_u, D2d_u;

    // Rescale factor.
    // If we wrote down the "true" Jacobian derivatives, scale would be exp(-chi * rr).
    double rescale = 1.0;

    // Omega.
    double v, w, w2, m2, chi;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = u0 * scale * rescale;
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2);
			dif_theta = 1.0 / (cos(atan((r / z) * (dr / dz)) - atan(dr / dz)) * sqrt(1.0 + (dz / dr) * (dz / dr)));

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dr * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr * dif_theta * D1d_u + (rr * chi + l + 1.0) * u0);
            break;

        case 2:
        case 3:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2);
            rr2 = rr * rr;
			dif_theta = 1.0 / (cos(atan((r / z) * (dr / dz)) - atan(dr / dz)) * sqrt(1.0 + (dz / dr) * (dz / dr)));
			dif_theta2 = dif_theta * dif_theta;

            // Omega.
            v = u[w_idx];
            w = omega_calc(v, m);
            w2 = w * w;
            m2 = m * m;
            chi = dr * sqrt(m2 - w2);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];
            D2d_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j - 1)] + (+4.0) * u[k + IDX(i - 2, j - 2)] + (-1.0) * u[k + IDX(i - 3, j - 3)];

            // Calculate RHS.
            f[k + IDX(i, j)] = scale * rescale * (rr2 * dif_theta2 * D2d_u
                + 2.0 * (rr * chi + l + 2.0) * rr * dif_theta * D1d_u
                + ((chi * rr) * (chi * rr) + 2.0 * (l + 2.0) * (chi * rr) + (l + 1.0) * (l + 2.0)) * u0);
            break;
    }

    // All done.
    return;
}