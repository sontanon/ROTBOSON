// Sanity test: the residual of the flat Minkowski vacuum (all fields zero,
// no scalar field) must be ~0. This exercises the full rhs assembly (interior
// + boundary + omega constraint) without needing a symbolic reference.
#include "tools.h"
#include "context.h"
#include "rhs.h"
#include "test.h"

int main(void)
{
	rb_context ctx;
	rb_context_init(&ctx);

	// Small grid, 4th order.
	ctx.dr = ctx.dz = 0.5;
	ctx.NrInterior = 20;
	ctx.NzInterior = 20;
	ctx.order = 4;
	ctx.ghost = 2;
	ctx.NrTotal = ctx.NrInterior + 2 * ctx.ghost;
	ctx.NzTotal = ctx.NzInterior + 2 * ctx.ghost;
	ctx.dim = ctx.NrTotal * ctx.NzTotal;
	ctx.w_idx = GNUM * ctx.dim;
	ctx.l = 1;
	ctx.m = 1.0;
	ctx.M_KOMAR = 0.0;
	ctx.J_KOMAR = 0.0;

	// Derivative / auxiliary buffers.
	ctx.Dr_u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	ctx.Dz_u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	ctx.Drr_u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	ctx.Dzz_u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	ctx.Drz_u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	ctx.u_aux = (double *)calloc((size_t)2 * ctx.dim, sizeof(double));
	ctx.Dr_u_aux = (double *)calloc((size_t)2 * ctx.dim, sizeof(double));

	double *u = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));
	double *f = (double *)calloc((size_t)GNUM * ctx.dim + 1, sizeof(double));

	// Flat Minkowski: log_alpha = beta = log_h = log_a = psi = lambda = 0.
	rhs(&ctx, f, u);

	// The residual of the vacuum should be ~0. Report the max (and where) so a
	// failure is diagnostic rather than opaque.
	double maxres = 0.0;
	MKL_INT argmax = 0;
	MKL_INT i;
	for (i = 0; i < GNUM * ctx.dim + 1; ++i)
	{
		double a = fabs(f[i]);
		if (a > maxres)
		{
			maxres = a;
			argmax = i;
		}
	}
	printf("  max |residual| over flat vacuum = %.3e (index %lld)\n", maxres, argmax);
	CHECK(maxres < 1e-12);

	free(u);
	free(f);
	free(ctx.Dr_u);
	free(ctx.Dz_u);
	free(ctx.Drr_u);
	free(ctx.Dzz_u);
	free(ctx.Drz_u);
	free(ctx.u_aux);
	free(ctx.Dr_u_aux);

	return rb_test_summary();
}
