// Unit tests for the finite-difference derivative operators (src/derivatives.c).
//
// Strategy: apply each operator to a delta (unit impulse) at a deep interior
// point and read back the stencil weights; compare against Fornberg-generated
// reference weights. Then check the convergence order on a smooth function.
// The reference (tests/fornberg.c) is independent of the production code.
#include "tools.h"        // MKL_INT
#include "derivatives.h"  // ex_diff1, ex_diff2, ex_diff3, ex_diff*r/z/rz
#include "fornberg.h"
#include "test.h"

#define EVEN 1
#define ODD -1

// ---------------------------------------------------------------------------
// 1D operators: ex_diff1 / ex_diff2 / ex_diff3 (4th order only).
// ---------------------------------------------------------------------------

typedef void (*op1d_t)(double *, double *, const MKL_INT, const double, const MKL_INT, const MKL_INT, const MKL_INT);

// Apply fn to a delta at k0 and return the interior stencil weights
// w[-half..+half] (unit spacing, derivative order factored out), so
//   (fn f)[k0 + s] = sum_s w[s] * f[k0 + s].
static void extract_1d(op1d_t fn, MKL_INT half, MKL_INT deriv, double *w)
{
	const MKL_INT dim = 64;
	const MKL_INT ghost = 2; // 4th-order operators use 2 ghost zones
	const MKL_INT k0 = 32;
	const double h = 0.25;
	double *u = (double *)calloc((size_t)dim, sizeof(double));
	double *du = (double *)calloc((size_t)dim, sizeof(double));
	double scale = 1.0;
	MKL_INT d;

	u[k0] = 1.0;
	fn(du, u, EVEN, h, dim, ghost, 4);

	for (d = 0; d < deriv; ++d)
		scale *= h;
	// du[k0 - d] is the coefficient that multiplies u[k0 + d].
	for (d = -half; d <= half; ++d)
		w[d + half] = du[k0 - d] * scale;

	free(u);
	free(du);
}

// Compare the extracted stencil of a 1D operator to Fornberg weights for the
// given derivative order on a (2*half+1)-point uniform grid.
static void check_1d(const char *name, op1d_t fn, MKL_INT half, MKL_INT deriv)
{
	double w[16];
	double x[16];
	double c[16 * 4];
	double tol = 1e-12;
	int s;

	extract_1d(fn, half, deriv, w);

	uniform_nodes(x, (int)half, 1.0);
	fornberg_weights(x, 2 * half + 1, 0.0, deriv, c);

	for (s = -half; s <= half; ++s)
		CHECK_NEAR(w[s + half], c[(s + half) * (deriv + 1) + deriv], tol);

	(void)name;
}

static void test_1d_weights(void)
{
	// 4th-order first derivative, 5-point stencil: [1/12, -2/3, 0, 2/3, -1/12].
	check_1d("ex_diff1", ex_diff1, 2, 1);
	// 4th-order second derivative, 5-point stencil: [-1/12, 4/3, -5/2, 4/3, -1/12].
	check_1d("ex_diff2", ex_diff2, 2, 2);
	// 4th-order third derivative, 7-point stencil.
	check_1d("ex_diff3", ex_diff3, 3, 3);
}

// ---------------------------------------------------------------------------
// 2D operators: interior stencil along r (or z) must match the 1D weights.
// ---------------------------------------------------------------------------

typedef void (*op2d_t)(double *, double *, const MKL_INT, const double, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT);

// Extract the stencil along the r direction (deriv in {1,2}) at a deep interior
// point. Returns weights w[-half..+half].
static void extract_2d_r(op2d_t fn, MKL_INT order, MKL_INT half, MKL_INT deriv, double *w)
{
	const MKL_INT ghost = (order == 2) ? 1 : 2;
	const MKL_INT NrTotal = 48, NzTotal = 48;
	const MKL_INT i0 = 24, j0 = 24;
	const double h = 0.25;
	double *u = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double *du = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double scale = 1.0;
	MKL_INT d;

	u[i0 * NzTotal + j0] = 1.0;
	fn(du, u, EVEN, h, NrTotal, NzTotal, ghost, order);

	for (d = 0; d < deriv; ++d)
		scale *= h;
	for (d = -half; d <= half; ++d)
		w[d + half] = du[(i0 - d) * NzTotal + j0] * scale;

	free(u);
	free(du);
}

static void check_2d(const char *name, op2d_t fn, MKL_INT order, MKL_INT half, MKL_INT deriv)
{
	double w[16];
	double x[16];
	double c[16 * 4];
	double tol = 1e-12;
	int s;

	extract_2d_r(fn, order, half, deriv, w);

	uniform_nodes(x, (int)half, 1.0);
	fornberg_weights(x, 2 * half + 1, 0.0, deriv, c);

	for (s = -half; s <= half; ++s)
		CHECK_NEAR(w[s + half], c[(s + half) * (deriv + 1) + deriv], tol);

	(void)name;
}

static void test_2d_weights(void)
{
	check_2d("ex_diff1r o2", ex_diff1r, 2, 1, 1);
	check_2d("ex_diff1r o4", ex_diff1r, 4, 2, 1);
	check_2d("ex_diff1r o6", ex_diff1r, 6, 3, 1);
	check_2d("ex_diff2r o2", ex_diff2r, 2, 1, 2);
	check_2d("ex_diff2r o4", ex_diff2r, 4, 2, 2);
}

// ---------------------------------------------------------------------------
// Mixed derivative ex_diff2rz: interior stencil is the outer product of the
// 1D first-derivative stencil with itself.
// ---------------------------------------------------------------------------

static void test_2drz_weights(void)
{
	const MKL_INT order = 2;
	const MKL_INT ghost = 1;
	const MKL_INT half = 1;
	const MKL_INT NrTotal = 48, NzTotal = 48;
	const MKL_INT i0 = 24, j0 = 24;
	const double h = 0.25;
	double *u = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double *du = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double x[16];
	double c[16 * 4];
	int sr, sz;

	// Reference: 1D first-derivative weights on [-1, 0, 1].
	uniform_nodes(x, (int)half, 1.0);
	fornberg_weights(x, 2 * half + 1, 0.0, 1, c);

	u[i0 * NzTotal + j0] = 1.0;
	ex_diff2rz(du, u, EVEN, EVEN, h, h, NrTotal, NzTotal, ghost, order);

	for (sr = -half; sr <= half; ++sr)
		for (sz = -half; sz <= half; ++sz)
		{
			double ref = c[(sr + half) * 2 + 1] * c[(sz + half) * 2 + 1];
			double got = du[(i0 - sr) * NzTotal + (j0 - sz)] * h * h;
			CHECK_NEAR(got, ref, 1e-12);
		}

	free(u);
	free(du);
}

// ---------------------------------------------------------------------------
// Convergence order: error on a smooth function must drop like h^order.
// ---------------------------------------------------------------------------

static double interior_error_1d(op1d_t fn, MKL_INT deriv, double h)
{
	const MKL_INT dim = 256;
	const MKL_INT ghost = 2;
	const MKL_INT order = 4;
	const double k = 1.7; // wavenumber
	double *u = (double *)malloc((size_t)dim * sizeof(double));
	double *du = (double *)malloc((size_t)dim * sizeof(double));
	double maxe = 0.0;
	MKL_INT i;

	for (i = 0; i < dim; ++i)
		u[i] = sin(k * ((double)(i - ghost) * h));
	fn(du, u, EVEN, h, dim, ghost, order);

	// Only deep interior points, far from the one-sided boundary stencils.
	for (i = ghost + 4; i < dim - 4; ++i)
	{
		double x = (double)(i - ghost) * h;
		double exact = 0.0;
		if (deriv == 1)
			exact = k * cos(k * x);
		else if (deriv == 2)
			exact = -k * k * sin(k * x);
		else if (deriv == 3)
			exact = -k * k * k * cos(k * x);
		double e = fabs(du[i] - exact);
		if (e > maxe)
			maxe = e;
	}

	free(u);
	free(du);
	return maxe;
}

static void test_convergence(void)
{
	// 4th-order operators: halving h should reduce the error by ~2^4 = 16.
	double e1 = interior_error_1d(ex_diff1, 1, 0.1);
	double e2 = interior_error_1d(ex_diff1, 1, 0.05);
	double ratio = e1 / e2;
	CHECK(ratio > 12.0 && ratio < 20.0);

	e1 = interior_error_1d(ex_diff2, 2, 0.1);
	e2 = interior_error_1d(ex_diff2, 2, 0.05);
	ratio = e1 / e2;
	CHECK(ratio > 12.0 && ratio < 20.0);

	e1 = interior_error_1d(ex_diff3, 3, 0.1);
	e2 = interior_error_1d(ex_diff3, 3, 0.05);
	ratio = e1 / e2;
	CHECK(ratio > 12.0 && ratio < 20.0);
}

// ---------------------------------------------------------------------------
// One-sided boundary stencils (the hand-derived edge coefficients).
// Fornberg also works on one-sided node sets, so these can be checked against
// an independent reference instead of trusting the coefficients by hand.
// ---------------------------------------------------------------------------

static void check_boundary_r(op2d_t fn, MKL_INT order, MKL_INT deriv, MKL_INT npts)
{
	MKL_INT ghost = (order == 2) ? 1 : 2;
	MKL_INT NrTotal = 40, NzTotal = 40, j0 = 20;
	double h = 0.25;
	double *u = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double *du = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double x[8], c[8 * 4], w[8];
	MKL_INT s, d;
	double scale = 1.0;

	for (d = 0; d < deriv; ++d)
		scale *= h;

	// Reference: Fornberg on one-sided nodes x[k] = -k*h (k = 0..npts-1) at z = 0.
	for (s = 0; s < npts; ++s)
		x[s] = -(double)s;
	fornberg_weights(x, npts, 0.0, deriv, c);

	// Extract the operator's weights: delta at node NrTotal-1-s, read du[NrTotal-1].
	for (s = 0; s < npts; ++s)
	{
		memset(u, 0, (size_t)NrTotal * NzTotal * sizeof(double));
		u[(NrTotal - 1 - s) * NzTotal + j0] = 1.0;
		fn(du, u, EVEN, h, NrTotal, NzTotal, ghost, order);
		w[s] = du[(NrTotal - 1) * NzTotal + j0] * scale;
	}

	for (s = 0; s < npts; ++s)
		CHECK_NEAR(w[s], c[s * (deriv + 1) + deriv], 1e-12);

	free(u);
	free(du);
}

static void test_boundary_weights(void)
{
	// 1st derivative, 2nd order, one-sided: (3,-4,1)/(2h).
	check_boundary_r(ex_diff1r, 2, 1, 3);
	// 1st derivative, 4th order, one-sided: (25,-48,36,-16,3)/(12h).
	check_boundary_r(ex_diff1r, 4, 1, 5);
	// 2nd derivative, 2nd order, one-sided: (2,-5,4,-1)/h^2.
	check_boundary_r(ex_diff2r, 2, 2, 4);
	// 2nd derivative, 4th order, one-sided: (45,-154,214,-156,61,-10)/(12h^2).
	check_boundary_r(ex_diff2r, 4, 2, 6);
}

// ---------------------------------------------------------------------------
// Symmetry / ghost-zone reflection: for an even function the derivative must
// be odd, so du[ghost-1-i] == -du[ghost+i] (sym = EVEN = +1). This validates
// the reflection loop, which is the other hand-wired non-trivial part.
// ---------------------------------------------------------------------------

static void test_symmetry(void)
{
	const MKL_INT order = 4, ghost = 2;
	const MKL_INT NrTotal = 48, NzTotal = 48;
	const double h = 0.25;
	const double k = 1.3;
	MKL_INT i, j, d;
	double *u = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));
	double *du = (double *)calloc((size_t)NrTotal * NzTotal, sizeof(double));

	// Even function about the axis (r = (i-ghost+0.5)*h is odd about i = ghost-0.5).
	for (i = 0; i < NrTotal; ++i)
	{
		double r = ((double)(i - ghost) + 0.5) * h;
		for (j = 0; j < NzTotal; ++j)
			u[i * NzTotal + j] = cos(k * r);
	}

	ex_diff1r(du, u, EVEN, h, NrTotal, NzTotal, ghost, order);

	// Interior derivative is -k sin(k r); ghost zone must be the odd reflection.
	for (j = 0; j < NzTotal; ++j)
	{
		for (d = 0; d < ghost; ++d)
		{
			double ghost_val = du[(ghost - 1 - d) * NzTotal + j];
			double refl_val = du[(ghost + d) * NzTotal + j];
			CHECK_NEAR(ghost_val, -refl_val, 1e-12);
		}
	}

	free(u);
	free(du);
}

int main(void)
{
	test_1d_weights();
	test_2d_weights();
	test_2drz_weights();
	test_convergence();
	test_boundary_weights();
	test_symmetry();
	return rb_test_summary();
}
