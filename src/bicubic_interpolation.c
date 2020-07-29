#include "tools.h"

// Use simple matrix multiplication or full CSR fast matrix multiplication.
#undef SIMPLE

// Matrix parameters.
#define D_NNZ   100
#define D_NROWS  16
#define D_NCOLS  16

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
	//const MKL_INT NNZ = 100;
	const MKL_INT NROWS = 16;
	//const MKL_INT NCOLS = 16;

	// Matrix multiplication type.
	const char T = 'N';

	// Nonzero elements of the matrix.
	const double  A[D_NNZ] = { 1.,                                                         //  1 
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
	const MKL_INT jA[D_NNZ] = {  1,                                                            //  1
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
	const MKL_INT iA[D_NROWS + 1] = { 1, 2, 3, 7, 11, 12, 13, 17, 21, 25, 29, 45, 61, 65, 69, 85, 101 };

	// Values of the function and its derivatives.
	double x[D_NCOLS] = { f00, f10, f01, f11, Dx_f00, Dx_f10, Dx_f01, Dx_f11, Dy_f00, Dy_f10, Dy_f01, Dy_f11, Dxy_f00, Dxy_f10, Dxy_f01, Dxy_f11 };

	// Coefficients.
	double a[D_NCOLS] = { 0.0 };

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
	double d[D_NCOLS] = { d00, d10, d20, d30, d01, d11, d21, d31, d02, d12, d22, d32, d03, d13, d23, d33 };

	// Result.
	double p = 0.0;

	// Do dot product.
	p = cblas_ddot(D_NROWS, a, 1, d, 1);

	// Return interpolated value.
	return p;
#endif
}

// Bicubic wrapper.
double bicubic(const MKL_INT i1, const MKL_INT j1, const double di, const double dj, double *u, double *Dr_u, double *Dz_u, double *Drz_u, const double dr, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal)
{
	return bicubic_csr_interpolator(u[IDX(i1, j1)], u[IDX(i1, j1 + 1)], u[IDX(i1 + 1, j1)], u[IDX(i1 + 1, j1 + 1)],
		dr * Dr_u[IDX(i1, j1)], dr * Dr_u[IDX(i1, j1 + 1)], dr * Dr_u[IDX(i1 + 1, j1)], dr * Dr_u[IDX(i1 + 1, j1 + 1)],
		dz * Dz_u[IDX(i1, j1)], dz * Dz_u[IDX(i1, j1 + 1)], dz * Dz_u[IDX(i1 + 1, j1)], dz * Dz_u[IDX(i1 + 1, j1 + 1)],
		dr * dz * Drz_u[IDX(i1, j1)], dr * dz * Drz_u[IDX(i1, j1 + 1)], dr * dz * Drz_u[IDX(i1 + 1, j1)], dr * dz * Drz_u[IDX(i1 + 1, j1 + 1)],
		di, dj);
}
