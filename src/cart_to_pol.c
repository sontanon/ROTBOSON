#include "tools.h"
#include "bicubic_interpolation.h"

// Interpolate grid variables from cartesian to polar.
// This grid will be interpolated via bicubic interpolation
// where derivatives are calculated using a centered stencil.
//
// Since we will later integrate using Simpson's rule, we want
// a grid that is uneven in both extensions.
void ex_cart_to_pol(
	double **i_u,
	double **i_rr,
	double **i_th,
	const double *r,
	const double *z,
	const double *u,
	const double *Dr_u,
	const double *Dz_u,
	const double *Drz_u,
	const MKL_INT g_num,
	const double dr,
	const double dz,
	const MKL_INT NrInterior, 
	const MKL_INT NzInterior,
	const MKL_INT ghost,
	MKL_INT *p_NrrTotal,
	MKL_INT *p_NthTotal,
	MKL_INT *p_p_dim,
	double *p_drr,
	double *p_dth,
	double *p_rr_inf
)
{
	MKL_INT NrTotal = NrInterior + 2 * ghost;
	MKL_INT NzTotal = NzInterior + 2 * ghost;
	MKL_INT dim = NrTotal * NzTotal;

	// Auxiliary variables.
	// Loop counters.
	MKL_INT i, j, k;
	// Doubles for coordinates.
	double aux_rr, aux_th, aux_r, aux_z, aux_u;
	// Floating point coordiantes.
	double fi, fj, di, dj;
	// Interpolation anchors.
	MKL_INT i0, j0;

	// Spherical parameters.
	MKL_INT NrrTotal, NthTotal, p_dim;
	double drr, dth, rr_inf;

	// Establish polar grid sizes and dimensions.
	// Maximum rr extension.
	*p_rr_inf = rr_inf = MAX(dr * (NrInterior + ghost), dz * (NzInterior + ghost));
	// Radial step size.
	*p_drr = drr = MIN(dr, dz);
	// Number of rr points.
	NrrTotal = (MKL_INT)floor(rr_inf / drr);
	// Assert that NrrTotal is uneven.
	*p_NrrTotal = NrrTotal = (NrrTotal % 2) ? NrrTotal : NrrTotal - 1;
	// Number of th points.
	NthTotal = MAX(NrInterior, NzInterior);
	// Assert that NthTotal is uneven.
	*p_NthTotal = NthTotal = (NthTotal % 2) ? NthTotal : NthTotal - 1;
	// Angular step size.
	*p_dth = dth = 0.5 * M_PI / ((double)NthTotal - 1);

	// Polar grid dimension.
	*p_p_dim = p_dim = NrrTotal * NthTotal;

	// Allocate memory for polar grids. Notice that we add extra point accounting for the origin.
	*i_rr = (double *)SAFE_MALLOC(p_dim * sizeof(double));
	*i_th = (double *)SAFE_MALLOC(p_dim * sizeof(double));
	// Main grid contains five variables and for each we also interpolate the value at the origin.
	*i_u = (double *)SAFE_MALLOC(g_num * p_dim * sizeof(double));

	// Pass by reference.
	double *p_rr = *i_rr;
	double *p_th = *i_th;
	double *p_u = *i_u;

	printf("*** CARTESIAN TO POLAR INTERPOLATOR\n");
	printf("*** \n");
	printf("*** Parameters are: \n");
	printf("*** rr_inf = %lf\t drr = %lf\t dth = %lf\t NrrTotal = %lld\t NthTotal = %lld\t p_dim = %lld\n", rr_inf, drr, dth, NrrTotal, NthTotal, p_dim);
	printf("*** Doing bicubic interpolation...\n");

	// Fill coordinate grids.
	#pragma omp parallel for schedule(dynamic, 1) private(i, j, aux_rr) shared(p_rr, p_th)
	for (i = 0; i < NrrTotal; ++i)
	{
		// Radial value.
		aux_rr = i * drr;
		// Loop over angular variables.
		for (j = 0; j < NthTotal; ++j)
		{
			p_rr[P_IDX(i, j)] = aux_rr;
			p_th[P_IDX(i, j)] = j * dth;
		}
	}
	//printf("*** Filled coordinate rr, th grids.\n");

	// First deal with the origin, rr = 0.
	// Loop over g_num of variables.
	for (k = 0; k < g_num; ++k)
	{
		// Calculate center value.
		aux_u = bicubic(ghost - 1, ghost - 1, 0.5, 0.5, u + k * dim, Dr_u + k * dim, Dz_u + k * dim, Drz_u + k * dim, dr, dz, NrTotal, NzTotal);
		printf("*** i_u[%lld](0) = %lf\n", k, aux_u);
		// Fill in to trivial angular array.
		#pragma omp parallel for schedule(dynamic, 1) private(j) shared(p_u)
		for (j = 0; j < NthTotal; ++j)
		{
			p_u[k * p_dim + j] = aux_u; 
		}
	}
	//printf("*** Filled values at origin.\n");

	// Now loop over other rr values.
	#pragma omp parallel for schedule(dynamic, 1) private(aux_r, aux_z, aux_rr, aux_th,\
		fi, fj, i0, j0, di, dj, i, j, k) shared(p_u)
	for (i = 1; i < NrrTotal; ++i)
	{
		// Radial value.
		aux_rr = p_rr[P_IDX(i, 0)];

		// Loop over th values.
		for (j = 0; j < NthTotal; ++j)
		{
			// Theta coordinate.
			aux_th = p_th[P_IDX(i, j)];

			// For each values of rr, th we first calculate the r, z coordinates.
			aux_r = aux_rr * sin(aux_th);
			aux_z = aux_rr * cos(aux_th);

			// Now we must find where these coordinates are.
			// This is done in floating-point form.
			fi = aux_r / dr - 0.5 + ghost;
			fj = aux_z / dz - 0.5 + ghost;

			// We need the floor and ceiling integers.
			i0 = (MKL_INT)floor(fi);
			j0 = (MKL_INT)floor(fj);

			// Get normalized separation from floor integer.
			di = fi - (double)i0;
			dj = fj - (double)j0;

			// With these coordinates we can fetch the values via bicubic interpolation.
			for (k = 0; k < g_num; ++k)
			{
				p_u[k * p_dim + P_IDX(i, j)] = bicubic(i0, j0, di, dj, u + k * dim, Dr_u + k * dim, Dz_u + k * dim, Drz_u + k * dim, dr, dz, NrTotal, NzTotal);
			}
		}
	}
	printf("*** Finished all interpolation!\n");
	printf("***\n");

	// All done.
	return;
}

