#include "tools.h"

// These subroutines will calculate the right hand sides of the 
// discretization appropriate to the pseudo-Robin boundary conditions
// detailed in the source file csr_pseudo_robin.c

// Along z direction.
void rhs_z_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale      // Multiply equation by an overall factor.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, rr3, sec_theta, sec_theta2, sec_theta3;

    // Derivatives.
    double u0, D1z_u, D2z_u, D3z_u;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = scale * (u0 - u_inf);
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r * (dr / dz) * (dr / dz);
            z2 = z * z;
            rr = sqrt(r2 + z2);
            sec_theta = (rr / z);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr * sec_theta * D1z_u + n * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];
            D2z_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i, j - 1)] + (+4.0) * u[k + IDX(i, j - 2)] + (-1.0) * u[k + IDX(i, j - 3)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr2 * sec_theta2 * D2z_u + 2.0 * (n + 1.0) * rr * sec_theta * D1z_u + n * (n + 1.0) * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1z_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i, j - 1)] + (+0.5) * u[k + IDX(i, j - 2)];
            D2z_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i, j - 1)] + (+4.0) * u[k + IDX(i, j - 2)] + (-1.0) * u[k + IDX(i, j - 3)];
            D3z_u = (+2.5) * u0 + (-9.0) * u[k + IDX(i, j - 1)] + (+12.0) * u[k + IDX(i, j - 2)] + (-7.0) * u[k + IDX(i, j - 3)] + (+1.5) * u[k + IDX(i, j - 4)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr3 * sec_theta3 * D3z_u + 3.0 * (n + 2.0) * rr2 * sec_theta2 * D2z_u + 3.0 * (n + 1.0) * (n + 2.0) * rr * sec_theta * D1z_u + n * (n + 1.0) * (n + 2.0) * (u0 - u_inf));
            break;
    }

    // All done.
    return;
}

// Along r direction.
void rhs_r_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale      // Multiply equation by an overall factor.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, rr3, csc_theta, csc_theta2, csc_theta3;

    // Derivatives.
    double u0, D1r_u, D2r_u, D3r_u;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = scale * (u0 - u_inf);
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2);
            csc_theta = (rr / r);

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr * csc_theta * D1r_u + n * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];
            D2r_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j)] + (+4.0) * u[k + IDX(i - 2, j)] + (-1.0) * u[k + IDX(i - 3, j)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr2 * csc_theta2 * D2r_u + 2.0 * (n + 1.0) * rr * csc_theta * D1r_u + n * (n + 1.0) * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1r_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j)] + (+0.5) * u[k + IDX(i - 2, j)];
            D2r_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j)] + (+4.0) * u[k + IDX(i - 2, j)] + (-1.0) * u[k + IDX(i - 3, j)];
            D3r_u = (+2.5) * u0 + (-9.0) * u[k + IDX(i - 1, j)] + (+12.0) * u[k + IDX(i - 2, j)] + (-7.0) * u[k + IDX(i - 3, j)] + (+1.5) * u[k + IDX(i - 4, j)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr3 * csc_theta3 * D3r_u + 3.0 * (n + 2.0) * rr2 * csc_theta2 * D2r_u + 3.0 * (n + 1.0) * (n + 2.0) * rr * csc_theta * D1r_u + n * (n + 1.0) * (n + 2.0) * (u0 - u_inf));
            break;
    }

    // All done.
    return;
}

// Along corner direction.
void rhs_corner_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale      // Multiply equation by an overall factor.
    )
{
    // Grid integer offset.
    MKL_INT k = (g_num - 1) * dim;

    // Coordinates and angular factors.
    double r, z, r2, z2, rr, rr2, rr3, dif_theta, dif_theta2, dif_theta3;

    // Derivatives.
    double u0, D1d_u, D2d_u, D3d_u;

    // Select boundary type.
    switch (bound_error)
    {
        case 0:
            u0 = u[k + IDX(i, j)];

            f[k + IDX(i, j)] = scale * (u0 - u_inf);
            break;

        case 1:
            // Coordinates.
            r = (double)i - 0.5;
            z = (double)j - 0.5;
            r2 = r * r;
            z2 = z * z * (dz / dr) * (dz / dr);
            rr = sqrt(r2 + z2) / sqrt(1.0 + (dz / dr) * (dz / dr));
			dif_theta = 1.0 / cos(atan((r / z) * (dr / dz)) - atan(dr / dz));

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr * dif_theta * D1d_u + n * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];
            D2d_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j - 1)] + (+4.0) * u[k + IDX(i - 2, j - 2)] + (-1.0) * u[k + IDX(i - 3, j - 3)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr2 * dif_theta2 * D2d_u + 2.0 * (n + 1.0) * rr * dif_theta * D1d_u + n * (n + 1.0) * (u0 - u_inf));
            break;

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

            // Derivatives.
            u0 = u[k + IDX(i, j)];
            D1d_u = (+1.5) * u0 + (-2.0) * u[k + IDX(i - 1, j - 1)] + (+0.5) * u[k + IDX(i - 2, j - 2)];
            D2d_u = (+2.0) * u0 + (-5.0) * u[k + IDX(i - 1, j - 1)] + (+4.0) * u[k + IDX(i - 2, j - 2)] + (-1.0) * u[k + IDX(i - 3, j - 3)];
            D3d_u = (+2.5) * u0 + (-9.0) * u[k + IDX(i - 1, j - 1)] + (+12.0) * u[k + IDX(i - 2, j - 2)] + (-7.0) * u[k + IDX(i - 3, j - 3)] + (+1.5) * u[k + IDX(i - 4, j - 4)];

            // RHS.
            f[k + IDX(i, j)] = scale * (rr3 * dif_theta3 * D3d_u + 3.0 * (n + 2.0) * rr2 * dif_theta2 * D2d_u + 3.0 * (n + 1.0) * (n + 2.0) * rr * dif_theta * D1d_u + n * (n + 1.0) * (n + 2.0) * (u0 - u_inf));
            break;
    }

    // All done.
    return;
}