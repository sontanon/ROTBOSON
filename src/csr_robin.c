#include "tools.h"

#define D10 (-0.5)
#define D11 (+0.0)
#define D12 (+0.5)

#define S10 (+0.5)
#define S11 (-2.0)
#define S12 (+1.5)

// Robin comes from approximating the function u as only a function of the radius rr:
//                       n
// u(rr) = u    + u  / rr
//          inf    1
//
// Where u_inf is the value of u at spatial infinity, u_1 is a constant, and n is an integer.
// Notices that we could add further terms to this approximation, i.e., terms rr**(-n-1) but
// in practice this has not worked for some unkown reason.
// Using this equation we can get an equation for the boundary term:
//                                                               -(n+1)
// rr d   u + n (u - u   ) = r d  u + z d  u + n (u - u   ) = O(rr   ) .
//     rr             inf       r        z             inf        inf
//
// When calcuting the r,z derivatives it may be necessary to use one-sided stencils.

// Robin along z direction.
void z_robin_2nd_order
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

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			rr2 = r * r * dr * dr + z * z * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((D10) * r) * scale;
			aa[offset + 1] = ((S10) * z) * scale;
			aa[offset + 2] = ((S11) * z) * scale;
			aa[offset + 3] = ((S12) * z + (double)n) * scale;
			aa[offset + 4] = ((D12) * r) * scale;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 1, j);
			ja[offset + 1] = BASE + k + IDX(i, j - 2);
			ja[offset + 2] = BASE + k + IDX(i, j - 1);
			ja[offset + 3] = BASE + k + IDX(i, j    );
			ja[offset + 4] = BASE + k + IDX(i + 1, j);

			break;
	}

	// All done.
	return;
}

// Robin along r direction.
void r_robin_2nd_order
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

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			r = (double)i - 0.5;
			z = (double)j - 0.5;
			rr2 = r * r * dr * dr + z * z * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((S10) * r) * scale;
			aa[offset + 1] = ((S11) * r) * scale;
			aa[offset + 2] = ((D10) * z) * scale;
			aa[offset + 3] = ((S12) * r + (double)n) * scale;
			aa[offset + 4] = ((D12) * z) * scale;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j);
			ja[offset + 1] = BASE + k + IDX(i - 1, j);
			ja[offset + 2] = BASE + k + IDX(i, j - 1);
			ja[offset + 3] = BASE + k + IDX(i, j    );
			ja[offset + 4] = BASE + k + IDX(i, j + 1);

			break;
	}

	// All done.
	return;
}

// Robin along corner direction.
void corner_robin_2nd_order
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

	// Select boundary condition type by error.
	switch (bound_error)
	{
		//                    -n
		// Dirichlet with O(rr   ): u(rr   ) = 0.
		//                    inf       inf
		case 0:
			// Set values.
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
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
			aa[offset + 4] = ((double)n + (S12) * (r + z)) * scale;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i - 2, j);
			ja[offset + 1] = BASE + k + IDX(i - 1, j);
			ja[offset + 2] = BASE + k + IDX(i, j - 2);
			ja[offset + 3] = BASE + k + IDX(i, j - 1);
			ja[offset + 4] = BASE + k + IDX(i, j);

			break;
	}

	// All done.
	return;
}

#define D1_4_0 (+1.0 / 12.0)
#define D1_4_1 (-2.0 / 3.0)
#define D1_4_2 (+0.0)
#define D1_4_3 (+2.0 / 3.0)
#define D1_4_4 (-1.0 / 12.0)

#define S1_4_0 (+0.25)
#define S1_4_1 (-4.0 / 3.0)
#define S1_4_2 (+3.0)
#define S1_4_3 (-4.0)
#define S1_4_4 (25.0 / 12.0)

#define SO1_4_0 (-1.0 / 12.0)
#define SO1_4_1 (+0.5)
#define SO1_4_2 (-1.5)
#define SO1_4_3 (+5.0 / 6.0)
#define SO1_4_4 (+0.25)

// Robin along z direction.
void z_robin_4th_order
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
	double ri, zi;
	double rr2;
	double scale;

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
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			ri = (double)i - 1.5;
			zi = (double)j - 1.5;
			rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((D1_4_0) * ri) * scale;
			aa[offset + 1] = ((D1_4_1) * ri) * scale;
			aa[offset + 2] = ((S1_4_0) * zi) * scale;
			aa[offset + 3] = ((S1_4_1) * zi) * scale;
			aa[offset + 4] = ((S1_4_2) * zi) * scale;
			aa[offset + 5] = ((S1_4_3) * zi) * scale;
			aa[offset + 6] = ((double)n + (S1_4_4) * zi) * scale;
			aa[offset + 7] = ((D1_4_3) * ri) * scale;
			aa[offset + 8] = ((D1_4_4) * ri) * scale;

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

			break;
	}

	// All done.
	return;
}

// Robin along r direction.
void r_robin_4th_order
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
	double ri, zi;
	double rr2;
	double scale;

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
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			ri = (double)i - 1.5;
			zi = (double)j - 1.5;
			rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((S1_4_0) * ri) * scale;
			aa[offset + 1] = ((S1_4_1) * ri) * scale;
			aa[offset + 2] = ((S1_4_2) * ri) * scale;
			aa[offset + 3] = ((S1_4_3) * ri) * scale;
			aa[offset + 4] = ((D1_4_0) * zi) * scale;
			aa[offset + 5] = ((D1_4_1) * zi) * scale;
			aa[offset + 6] = ((double)n + (S1_4_4) * ri) * scale;
			aa[offset + 7] = ((D1_4_3) * zi) * scale;
			aa[offset + 8] = ((D1_4_4) * zi) * scale;

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

			break;
	}

	// All done.
	return;
}

// Robin along corner direction.
void corner_robin_4th_order
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
	double ri, zi;
	double rr2;
	double scale;

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
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			ri = (double)i - 1.5;
			zi = (double)j - 1.5;
			rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((S1_4_0) * ri) * scale;
			aa[offset + 1] = ((S1_4_1) * ri) * scale;
			aa[offset + 2] = ((S1_4_2) * ri) * scale;
			aa[offset + 3] = ((S1_4_3) * ri) * scale;
			aa[offset + 4] = ((S1_4_0) * zi) * scale;
			aa[offset + 5] = ((S1_4_1) * zi) * scale;
			aa[offset + 6] = ((S1_4_2) * zi) * scale;
			aa[offset + 7] = ((S1_4_3) * zi) * scale;
			aa[offset + 8] = ((double)n + (S1_4_4) * (ri + zi)) * scale;

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

			break;
	}

	// All done.
	return;
}

// Robin along z but with semi-onesided r derivative.
void z_so_robin_4th_order
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
	double ri, zi;
	double rr2;
	double scale;

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
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			ri = (double)i - 1.5;
			zi = (double)j - 1.5;
			rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((SO1_4_0) * ri) * scale;
			aa[offset + 1] = ((SO1_4_1) * ri) * scale;
			aa[offset + 2] = ((SO1_4_2) * ri) * scale;
			aa[offset + 3] = ((S1_4_0) * zi) * scale;
			aa[offset + 4] = ((S1_4_1) * zi) * scale;
			aa[offset + 5] = ((S1_4_2) * zi) * scale;
			aa[offset + 6] = ((S1_4_3) * zi) * scale;
			aa[offset + 7] = ((double)n + (S1_4_4) * zi + (SO1_4_3) * ri) * scale;
			aa[offset + 8] = ((SO1_4_4) * ri) * scale;

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

			break;
	}

	// All done.
	return;
}

// Robin along r but with semi-onesided z derivative.
void r_so_robin_4th_order
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
	double ri, zi;
	double rr2;
	double scale;

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
			aa[offset + 0] = 1.0;

			// Column indices.
			ja[offset + 0] = BASE + k + IDX(i, j);

			break;

		//               -(n+1)
		// Robin with O(rr   ): rr d  u(rr   ) + n u(rr   ) = 0.
		//                inf       rr    inf          inf 
		case 1:
		default:
			// Coordinates.
			ri = (double)i - 1.5;
			zi = (double)j - 1.5;
			rr2 = ri * ri * dr * dr + zi * zi * dz * dz;
			scale = dr * dz / rr2;

			// Set values.
			aa[offset + 0] = ((S1_4_0) * ri) * scale;
			aa[offset + 1] = ((S1_4_1) * ri) * scale;
			aa[offset + 2] = ((S1_4_2) * ri) * scale;
			aa[offset + 3] = ((S1_4_3) * ri) * scale;
			aa[offset + 4] = ((SO1_4_0) * zi) * scale;
			aa[offset + 5] = ((SO1_4_1) * zi) * scale;
			aa[offset + 6] = ((SO1_4_2) * zi) * scale;
			aa[offset + 7] = ((double)n + (S1_4_4) * ri + (SO1_4_3) * zi) * scale;
			aa[offset + 8] = ((SO1_4_4) * zi) * scale;

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
			
			break;
	}

	// All done.
	return;
}