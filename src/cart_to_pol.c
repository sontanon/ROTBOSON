#include "param.h"
#include "tools.h"

// Use simple matrix multiplication or full CSR fast matrix multiplication.
#undef SIMPLE

// Polar indexing macro.
#define P_IDX(i, j) ((i) * NthTotal + (j))

// Interpolate grid variables from cartesian to polar.
// This grid will be interpolated via bicubic interpolation
// where derivatives are calculated using a centered stencil.
//
// Since we will later integrate using Simpson's rule, we want
// a grid that is even in both extensions.
void cart_to_pol(
	double *i_u,
	double *rr,
	double *th,
	const double *r,
	const double *z,
	const double *u,
	const double *Dr_u,
	const double *Dz_u,
	const double *Drz_u,
	const MKL_INT g_num
)
{
	// Auxiliary variables.
	// Loop counters.
	MKL_INT i, j, k;
	// Doubles for coordinates.
	double aux_rr, aux_th, aux_r, aux_z, aux_u;
	// Floating point coordiantes.
	double fi, fj, di, dj;
	// Interpolation anchors.
	MKL_INT i0, j0;

	// Establish polar grid sizes and dimensions.
	// Maximum rr extension.
	rr_inf = MAX(dr * NrInterior, dz * NzInterior);
	// Radial step size.
	drr = MIN(dr, dz);
	// Number of rr points.
	NrrTotal = (MKL_INT)floor(rr_inf / drr);
	// Assert that NrrTotal is even.
	NrrTotal = (NrrTotal % 2 == 0) ? NrrTotal : NrrTotal - 1;
	// Number of th points.
	NthTotal = MAX(NrInterior, NzInterior);
	// Assert that NthTotal is even.
	NthTotal = (NthTotal % 2 == 0) ? NthTotal : NthTotal - 1;
	// Angular step size.
	dth = 0.5 * M_PI / ((double)NthTotal - 1);

	// Polar grid dimension.
	p_dim = NrrTotal * NthTotal;

	// Allocate memory for polar grids. Notice that we add extra point accounting for the origin.
	rr = (double *)SAFE_MALLOC(p_dim * sizeof(double));
	th = (double *)SAFE_MALLOC(p_dim * sizeof(double));
	// Main grid contains five variables and for each we also interpolate the value at the origin.
	i_u = (double *)SAFE_MALLOC(5 * p_dim * sizeof(double));

	// Fill coordinate grids.
	#pragma omp parallel for schedule(dynamic, 1) private(i, j, aux_rr) shared(rr, th)
	for (i = 0; i < NrrTotal; ++i)
	{
		// Radial value.
		aux_rr = i * drr;
		// Loop over angular variables.
		for (j = 0; j < NthTotal; ++j)
		{
			rr[P_IDX(i, j)] = aux_rr;
			th[P_IDX(i, j)] = j * dth;
		}
	}

	// First deal with the origin, rr = 0.
	// Loop over g_num of variables.
	for (k = 0; k < g_num; ++k)
	{
		// Calculate center value.
		aux_u = bicubic(ghost - 1, ghost - 1, 0.5, 0.5, u + k * dim, Dr_u + k * dim, Dz_u + k * dim, Drz_u + k * dim);
		// Fill in to trivial angular array.
		#pragma omp parallel for schedule(dynamic, 1) private(j) shared(i_u)
		for (j = 0; j < NthTotal; ++j)
		{
			i_u[k * p_dim + j] = aux_u; 
		}
	}

	// Now loop over other rr values.
	#pragma omp parallel for schedule(dynamic, 1) private(aux_r, aux_z, aux_rr, aux_th,\
		fi, fj, i0, j0, di, dj, i, j, k) shared(i_u)
	for (i = 1; i < NrrTotal; ++i)
	{
		// Radial value.
		aux_rr = rr[P_IDX(i, 0)];

		// Loop over th values.
		for (j = 0; j < NthTotal; ++j)
		{
			// Theta coordinate.
			aux_th = th[P_IDX(i, j)];

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
				i_u[k * p_dim + P_IDX(i, j)] = bicubic(i0, j0, di, dj, u + k * dim, Dr_u + k * dim, Dz_u + k * dim, Drz_u + k * dim);
			}
		}
	}

	// All done.
	return;
}

// Bicubic wrapper.
double bicubic(const MKL_INT i1, const MKL_INT j1, const double di, const double dj, const double *u, const double *Dr_u, const double *Dz_u, const double *Drz_u)
{
	return bicubic_csr_interpolator(u[IDX(i1, j1)], u[IDX(i1, j1 + 1)], u[IDX(i1 + 1, j1)], u[IDX(i1 + 1, j1 + 1)],
		dr * Dr_u[IDX(i1, j1)], dr * Dr_u[IDX(i1, j1 + 1)], dr * Dr_u[IDX(i1 + 1, j1)], dr * Dr_u[IDX(i1 + 1, j1 + 1)],
		dz * Dz_u[IDX(i1, j1)], dz * Dz_u[IDX(i1, j1 + 1)], dz * Dz_u[IDX(i1 + 1, j1)], dz * Dz_u[IDX(i1 + 1, j1 + 1)],
		dr * dz * Drz_u[IDX(i1, j1)], dr * dz * Drz_u[IDX(i1, j1 + 1)], dr * dz * Drz_u[IDX(i1 + 1, j1)], dr * dz * Drz_u[IDX(i1 + 1, j1 + 1)],
		di, dj);
}

// Bicubic interpolation using CSR matrix multiplication.
// Base index for CSR multiplication is one.
double bicubic_csr_interpolator(const double f00,
		 const double f01,
		 const double f10,
		 const double f11,
		 const double Dx_f00,
		 const double Dx_f01,
		 const double Dx_f10,
		 const double Dx_f11,
		 const double Dy_f00,
		 const double Dy_f01,
		 const double Dy_f10,
		 const double Dy_f11,
		 const double Dxy_f00,
		 const double Dxy_f01,
		 const double Dxy_f10,
		 const double Dxy_f11,
		 const double dx,
		 const double dy)
{
#ifdef SIMPLE
	double p = 0.0;
	MKL_INT i, j, k, l;

	const double M[4][4] = {{ 1.0,  0.0,  0.0,  0.0},
				{ 0.0,  0.0,  1.0,  0.0},
				{-3.0,  3.0, -2.0, -1.0},
				{ 2.0, -2.0,  1.0,  1.0}};

	double F[4][4] = {{   f00,    f01,  Dy_f00,  Dy_f01},
			  {   f10,    f11,  Dy_f10,  Dy_f11},
			  {Dx_f00, Dx_f01, Dxy_f00, Dxy_f01},
			  {Dx_f10, Dx_f11, Dxy_f10, Dxy_f11}};

	double x[4] = {1.0, dx, dx * dx, dx * dx * dx};
	double y[4] = {1.0, dy, dy * dy, dy * dy * dy};

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			for (k = 0; k < 4; k++)
			{
				for (l = 0; l < 4; l++)
				{
					p += x[i] * M[i][j] * F[j][k] * M[l][k] * y[l];
				}
			}
		}
	}

	return p;
#else
	// Matrix system properties.
	const MKL_INT NNZ = 100;
	const MKL_INT NROWS = 16;
	const MKL_INT NCOLS = 16;

	// Matrix multiplication type.
	const char T = 'N';

	// Nonzero elements of the matrix.
	const double  A[NNZ] = { 1.,                                                         //  1 
		                           1.,                                               //  1
			  -3., 3.,        -2.,-1.,                                           //  4
			   2.,-2.,         1., 1.,                                           //  4
	                                                   1.,                               //  1
	                                                                   1.,               //  1
				                          -3., 3.,        -2.,-1.,           //  4
							   2.,-2.,         1., 1.,           //  4
			  -3.,     3.,			  -2.,    -1.,                       //  4
			                  -3.,     3.,                    -2.,    -1.,       //  4
			   9.,-9.,-9., 9., 6., 3.,-6.,-3., 6.,-6., 3.,-3., 4., 2., 2., 1.,   // 16
			  -6., 6., 6.,-6.,-3.,-3., 3., 3.,-4., 4.,-2., 2.,-2.,-2.,-1.,-1.,   // 16
			   2.,    -2.,                     1.,     1.,                       //  4
			                   2.,    -2.,                     1.,     1.,       //  4
			  -6., 6., 6.,-6.,-4.,-2., 4., 2.,-3., 3.,-3., 3.,-2.,-1.,-2.,-1.,   // 16
			   4.,-4.,-4., 4., 2., 2.,-2.,-2., 2.,-2., 2.,-2., 1., 1., 1., 1. }; // 16

	// Column indices.
	const MKL_INT jA[NNZ] = {  1,                                                            //  1
		                            5,                                               //  2
			    1,  2,          5,  6,                                           //  3
			    1,  2,          5,  6,                                           //  7
	                                                    9,                               // 11
	                                                                   13,               // 12
				                            9, 10,         13, 14,           // 13
							    9, 10,         13, 14,           // 17
			    1,      3,			    9,     11,                       // 21
			                    5,      7,                     13,     15,       // 25
	                    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,   // 29
	                    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,   // 45
			    1,      3,                      9,     11,                       // 61
			                    5,      7,                     13,     15,       // 65
	                    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,   // 69
	                    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16 }; // 85

	// Row starts.
	const MKL_INT iA[NROWS + 1] = { 1, 2, 3, 7, 11, 12, 13, 17, 21, 25, 29, 45, 61, 65, 69, 85, 101 };

	// Values of the function and its derivatives.
	double x[NCOLS] = { f00, f10, f01, f11, Dx_f00, Dx_f10, Dx_f01, Dx_f11, Dy_f00, Dy_f10, Dy_f01, Dy_f11, Dxy_f00, Dxy_f10, Dxy_f01, Dxy_f11 };

	// Coefficients.
	double a[NROWS] = { 0.0 };

	// Perform CSR matrix multiplication.
	mkl_dcsrgemv(&T, &NROWS, A, iA, jA, x, a);

	// Now do dot product using step sizes obtained as follows:
	double d00 = 1.0;
	double d10 = dx;
	double d01 = dy;
	double d11 = d10 * d01;
	double d20 = dx * d10;
	double d21 = dx * d11;
	double d02 = dy * d01;
	double d12 = dy * d11;
	double d22 = d20 * d02;
	double d30 = dx * d20;
	double d31 = dx * d21;
	double d32 = dx * d22;
	double d03 = dy * d02;
	double d13 = dy * d12;
	double d23 = dy * d22;
	double d33 = d30 * d03;

	// Step sizes vector.
	double d[NROWS] = { 1.0, d10, d20, d30, d01, d11, d21, d31, d02, d12, d22, d32, d03, d13, d23, d33 };

	// Result.
	double p = 0.0;

	// Do dot product.
	p = cblas_ddot(NROWS, a, 1, d, 1);

	// Return interpolated value.
	return p;
#endif
}