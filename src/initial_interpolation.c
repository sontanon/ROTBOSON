#include "tools.h"
#include "derivatives.h"
#include "cart_to_pol.h"
#include "bicubic_interpolation.h"
#include "analysis.h"

// We are reading from an initial file, u.asc with grid steps dr_0, dz_0, and dimensions
// NrInterior_0, NzInterior_0, and ghost_0.
// The grid from which we are reading will be denoted by the "0" subscript.
// Final grid is denoted by "1" subscript.
void initial_interpolator(double *u_1,
	const double *u_0,
	const MKL_INT NrInterior_0,
	const MKL_INT NzInterior_0,
	const MKL_INT ghost_0,
	const MKL_INT order_0,
	const double dr_0, 
	const double dz_0,
	const MKL_INT NrInterior_1,
	const MKL_INT NzInterior_1,
	const MKL_INT ghost_1,
	const MKL_INT order_1,
	const double dr_1, 
	const double dz_1,
	const double w,
	const double m,
	const MKL_INT l)
{
	// Grid 0.
	MKL_INT NrTotal_0 = NrInterior_0 + 2 * ghost_0;
	MKL_INT NzTotal_0 = NzInterior_0 + 2 * ghost_0;
	MKL_INT dim_0 = NrTotal_0 * NzTotal_0;

	// Grid 1.
	MKL_INT NrTotal_1 = NrInterior_1 + 2 * ghost_1;
	MKL_INT NzTotal_1 = NzInterior_1 + 2 * ghost_1;
	MKL_INT dim_1 = NrTotal_1 * NzTotal_1;

	// Allocate memory for read buffer.
	double *Dr_u_0  = (double *)SAFE_MALLOC(dim_0 * sizeof(double));
	double *Dz_u_0  = (double *)SAFE_MALLOC(dim_0 * sizeof(double));
	double *Drz_u_0 = (double *)SAFE_MALLOC(dim_0 * sizeof(double));

	// Differentiate at the 0 level.
	ex_diff1r (Dr_u_0             , u_0            , 1   , dr_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff1z (Dz_u_0             , u_0            , 1   , dz_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff2rz(Drz_u_0            , u_0            , 1, 1, dr_0, dz_0, NrTotal_0, NzTotal_0, ghost_0, order_0);	
	ex_diff1r (Dr_u_0  +     dim_0, u_0 +     dim_0, 1   , dr_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff1z (Dz_u_0  +     dim_0, u_0 +     dim_0, 1   , dz_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff2rz(Drz_u_0 +     dim_0, u_0 +     dim_0, 1, 1, dr_0, dz_0, NrTotal_0, NzTotal_0, ghost_0, order_0);	
	ex_diff1r (Dr_u_0  + 2 * dim_0, u_0 + 2 * dim_0, 1   , dr_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff1z (Dz_u_0  + 2 * dim_0, u_0 + 2 * dim_0, 1   , dz_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff2rz(Drz_u_0 + 2 * dim_0, u_0 + 2 * dim_0, 1, 1, dr_0, dz_0, NrTotal_0, NzTotal_0, ghost_0, order_0);	
	ex_diff1r (Dr_u_0  + 3 * dim_0, u_0 + 3 * dim_0, 1   , dr_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff1z (Dz_u_0  + 3 * dim_0, u_0 + 3 * dim_0, 1   , dz_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff2rz(Drz_u_0 + 3 * dim_0, u_0 + 3 * dim_0, 1, 1, dr_0, dz_0, NrTotal_0, NzTotal_0, ghost_0, order_0);	
	ex_diff1r (Dr_u_0  + 4 * dim_0, u_0 + 4 * dim_0, 1   , dr_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff1z (Dz_u_0  + 4 * dim_0, u_0 + 4 * dim_0, 1   , dz_0      , NrTotal_0, NzTotal_0, ghost_0, order_0);
	ex_diff2rz(Drz_u_0 + 4 * dim_0, u_0 + 4 * dim_0, 1, 1, dr_0, dz_0, NrTotal_0, NzTotal_0, ghost_0, order_0);	

	// The 0 grid extends up to dr_0 * (NrInterior_0 + ghost_0 - 0.5).
	// We first want to know if the 0 grid covers in its entirety the 1 grid or not.
	double r_inf_0 = dr_0 * (NrInterior_0 + ghost_0 - 0.5);
	double z_inf_0 = dz_0 * (NzInterior_0 + ghost_0 - 0.5);

	//double r_inf_1 = dr_1 * (NrInterior_1 + ghost_1 - 0.5);
	//double z_inf_1 = dz_1 * (NzInterior_1 + ghost_1 - 0.5);
	
	// If r_inf_0 > r_inf_1 we can interpolate all along the r direction.
	// If r_inf_0 = r_inf_1 we can interpolate up to the previous point and calculate the last one via BC.
	// If r_inf_0 < r_inf_1 we can interpolate up to a set point and extrapolate the remaining via BC.
	// In summary, we must obtain the grid 1 point up to where we can interpolate.
	MKL_INT i_inf_1 = MIN(NrTotal_1 - 2, (MKL_INT)floor(r_inf_0 / dr_1 + ghost_1 - 0.5));
	MKL_INT j_inf_1 = MIN(NzTotal_1 - 2, (MKL_INT)floor(z_inf_0 / dz_1 + ghost_1 - 0.5));

	// Other variables.
	double f_i_0, di;
	double f_j_0, dj;

	// Loop counters.
	MKL_INT i_1, i_0;
	MKL_INT j_1, j_0;

	// Now loop over grid elements.
	#pragma omp parallel for schedule(dynamic, 1) private(i_1, j_1, f_i_0, i_0, di, f_j_0, j_0, dj) shared(u_1)
	for (i_1 = ghost_1; i_1 < i_inf_1 + 1; ++i_1)
	{
		// 0 grid coordinates.
		f_i_0 = (dr_1 / dr_0) * (i_1 - ghost_1 + 0.5) + ghost_0 - 0.5;
		i_0   = (MKL_INT)floor(f_i_0);
		di    = f_i_0 - (double)i_0;

		for (j_1 = ghost_1; j_1 < j_inf_1 + 1; ++j_1)
		{
			// 0 grid coordinates.
			f_j_0 = (dz_1 / dz_0) * (j_1 - ghost_1 + 0.5) + ghost_0 - 0.5;
			j_0   = (MKL_INT)floor(f_j_0);
			dj    = f_j_0 - (double)j_0;

			// Use bicubic interpolation.
			u_1[            i_1 * NzTotal_1 + j_1] = bicubic(i_0, j_0, di, dj, u_0            , Dr_u_0            , Dz_u_0            , Drz_u_0            , dr_0, dz_0, NrTotal_0, NzTotal_0);
			u_1[    dim_1 + i_1 * NzTotal_1 + j_1] = bicubic(i_0, j_0, di, dj, u_0 +     dim_1, Dr_u_0 +     dim_1, Dz_u_0 +     dim_1, Drz_u_0 +     dim_1, dr_0, dz_0, NrTotal_0, NzTotal_0);
			u_1[2 * dim_1 + i_1 * NzTotal_1 + j_1] = bicubic(i_0, j_0, di, dj, u_0 + 2 * dim_1, Dr_u_0 + 2 * dim_1, Dz_u_0 + 2 * dim_1, Drz_u_0 + 2 * dim_1, dr_0, dz_0, NrTotal_0, NzTotal_0);
			u_1[3 * dim_1 + i_1 * NzTotal_1 + j_1] = bicubic(i_0, j_0, di, dj, u_0 + 3 * dim_1, Dr_u_0 + 3 * dim_1, Dz_u_0 + 3 * dim_1, Drz_u_0 + 3 * dim_1, dr_0, dz_0, NrTotal_0, NzTotal_0);
			u_1[4 * dim_1 + i_1 * NzTotal_1 + j_1] = bicubic(i_0, j_0, di, dj, u_0 + 4 * dim_1, Dr_u_0 + 4 * dim_1, Dz_u_0 + 4 * dim_1, Drz_u_0 + 4 * dim_1, dr_0, dz_0, NrTotal_0, NzTotal_0);
		}
	}

	// Coordinate grids 0.
	double *r_0 = (double *)SAFE_MALLOC(dim_0 * sizeof(double));
	double *z_0 = (double *)SAFE_MALLOC(dim_0 * sizeof(double));

	// Fill coordinate grids.
	double aux_r;
	#pragma omp parallel shared(r_0, z_0) private(i_0, j_0, aux_r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i_0 = 0; i_0 < NrTotal_0; ++i_0)
		{
			// Calculate rho value.
			aux_r = ((double)(i_0 - ghost_0) + 0.5) * dr_0;
			// Loop over z points.
			for (j_0 = 0; j_0 < NzTotal_0; ++j_0)
			{
				r_0[i_0 * NzTotal_0 + j_0] = aux_r;
				z_0[i_0 * NzTotal_0 + j_0] = ((double)(j_0 - ghost_0) + 0.5) * dz_0;
			}
		}
	}
	
	// Spherical analysis at level 0.
	MKL_INT NrrTotal_0, NthTotal_0, p_dim_0;
	double drr_0, dth_0, rr_inf_0;
	double *i_rr_0 = NULL;
	double *i_th_0 = NULL;
	double *i_u_0  = NULL;
	double M_0, J_0, GRV2_0, GRV3_0;

	// Interpolate to polar coordinates.
	ex_cart_to_pol(&i_u_0, &i_rr_0, &i_th_0, r_0, z_0, u_0, Dr_u_0, Dz_u_0, Drz_u_0, 5, dr_0, dz_0, NrInterior_0, NzInterior_0, ghost_0, &NrrTotal_0, &NthTotal_0, &p_dim_0, &drr_0, &dth_0, &rr_inf_0);

	// Extract global quantities.
	ex_analysis(0, &M_0, &J_0, &GRV2_0, &GRV3_0, i_u_0, i_rr_0, i_th_0, w, m, l, ghost_0, order_0, NrrTotal_0, NthTotal_0, p_dim_0, drr_0, dth_0, rr_inf_0);

	// Now extrapolate beyond interpolation limits.
	double rr_1;

	#pragma omp parallel for schedule(dynamic, 1) private(i_1, j_1, rr_1) shared(u_1)
	for (i_1 = i_inf_1 + 1; i_1 < NrTotal_1; ++i_1)
	{
		for (j_1 = j_inf_1 + 1; j_1 < NzTotal_1; ++j_1)
		{
			rr_1 = sqrt(pow(dr_1 * (i_1 - ghost_1 + 0.5), 2) + pow(dz_1 * (j_1 - ghost_1 + 0.5), 2));
			u_1[0 * dim_1 + i_1 * NzTotal_1 + j_1] = log(1.0 - M_0 / rr_1);
			u_1[1 * dim_1 + i_1 * NzTotal_1 + j_1] = - 2.0 * J_0 / (rr_1 * rr_1 * rr_1); 
			u_1[2 * dim_1 + i_1 * NzTotal_1 + j_1] = log(1.0 + M_0 / rr_1);
			u_1[3 * dim_1 + i_1 * NzTotal_1 + j_1] = log(1.0 + M_0 / rr_1);
			u_1[4 * dim_1 + i_1 * NzTotal_1 + j_1] = 0.0;
		}
	}	

	// Free memory.
	SAFE_FREE(Dr_u_0);
	SAFE_FREE(Dz_u_0);
	SAFE_FREE(Drz_u_0);

	SAFE_FREE(r_0);
	SAFE_FREE(z_0);

	SAFE_FREE(i_rr_0);
	SAFE_FREE(i_th_0);
	SAFE_FREE(i_u_0);

	// All done!
	return ;
}