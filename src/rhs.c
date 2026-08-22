#include "tools.h"
#include "context.h"

#include "derivatives.h"
#include "omega_calc.h"

#include "rhs_vars.h"
#include "cart_to_pol.h"
#include "analysis.h"

// All functions are even about the equator and the axis.
#define EVEN 1

#undef DERIVATIVE_DEBUG

#define REGULARIZATION_AUXILIARIES

void rhs(rb_context *ctx, double *f, double *u)
{
	// Local alias for the IDX macro, which indexes by row-major stride NzTotal.
	const MKL_INT NzTotal = ctx->NzTotal;

	// Omega.
	double w = omega_calc(u[ctx->w_idx], ctx->m);

	// Loop counters.
	MKL_INT i = 0;
	MKL_INT j = 0;
	MKL_INT k = 0;

	// Axis coordinate.
	double r = 0.0;

	// Calculate derivatives.
	for (k = 0; k < GNUM; ++k)
	{
		ex_diff1r(ctx->Dr_u  + k * ctx->dim, u + k * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order);
		ex_diff2r(ctx->Drr_u + k * ctx->dim, u + k * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order);
		ex_diff1z(ctx->Dz_u  + k * ctx->dim, u + k * ctx->dim, EVEN, ctx->dz, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order);
		ex_diff2z(ctx->Dzz_u + k * ctx->dim, u + k * ctx->dim, EVEN, ctx->dz, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order);
		/* Mixed derivatives are only used for interpolation */
		ex_diff2rz(ctx->Drz_u + k * ctx->dim , u + k * ctx->dim, EVEN, EVEN, ctx->dr, ctx->dz, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order);
	}

	// Regularization Auxiliaries.
#ifdef REGULARIZATION_AUXILIARIES
	// First calculate derivatives: Dr(log(alpha)) and Dr(log(h)).
	// Notive that derivatives are calculate to greater order.
	// For 4th order, they are calculated at 6th order.
	ex_diff1r(ctx->u_aux + 0 * ctx->dim, u + 0 * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order + 2);
	ex_diff1r(ctx->u_aux + 1 * ctx->dim, u + 2 * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order + 2);
	// Rescale.
	#pragma omp parallel private(i, j, r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < ctx->NrTotal; ++i)
		{
			r = ((double)(i - ctx->ghost) + 0.5) * ctx->dr;
			for (j = 0; j < ctx->NzTotal; ++j)
			{
				// u6 = (Dr(alpha) / r) = alpha * (Dr(log(alpha)) / r)
				ctx->u_aux[0 * ctx->dim + IDX(i, j)] *= exp(u[0 * ctx->dim + IDX(i, j)]) / r;
				// u7 = (Dr(H) / r) = 2.0 * H * (Dr(log(h)) / r)
				ctx->u_aux[1 * ctx->dim + IDX(i, j)] *= 2.0 * exp(2.0 * u[2 * ctx->dim + IDX(i, j)]) / r;
			}
		}
	}
	// Now calculate auxiliary derivatives. Similarily, to greater order.
	ex_diff1r(ctx->Dr_u_aux + 0 * ctx->dim, ctx->u_aux + 0 * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order + 2);
	ex_diff1r(ctx->Dr_u_aux + 1 * ctx->dim, ctx->u_aux + 1 * ctx->dim, EVEN, ctx->dr, ctx->NrTotal, ctx->NzTotal, ctx->ghost, ctx->order + 2);
#else
	#pragma omp parallel private(i, j, r)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = 0; i < ctx->NrTotal; ++i)
		{
			r = ((double)(i - ctx->ghost) + 0.5) * ctx->dr;
			for (j = 0; j < ctx->NzTotal; ++j)
			{
				// Dr(alpha) / r.
				ctx->u_aux[0 * ctx->dim + IDX(i, j)] = exp(u[0 * ctx->dim + IDX(i, j)]) * (ctx->Dr_u[0 * ctx->dim + IDX(i, j)] / r);
				// Dr(H) / r.
				ctx->u_aux[1 * ctx->dim + IDX(i, j)] = 2.0 * exp(2.0 * u[2 * ctx->dim + IDX(i, j)]) * (ctx->Dr_u[2 * ctx->dim + IDX(i, j)] / r);
				// Dr(Dr(alpha) / r) = (Drr(alpha) - Dr(alpha) / r) / r = alpha * ((Drr(log(alpha)) - Dr(log(alpha)) / r) + Dr(log(alpha))**2) / r.
				ctx->Dr_u_aux[0 * ctx->dim + IDX(i, j)] = exp(u[0 * ctx->dim + IDX(i, j)]) * ((ctx->Drr_u[0 * ctx->dim + IDX(i, j)] - ctx->Dr_u[0 * ctx->dim + IDX(i, j)] / r) + ctx->Dr_u[0 * ctx->dim + IDX(i, j)] * ctx->Dr_u[0 * ctx->dim + IDX(i, j)]) / r;
				// Dr(Dr(H) / r) = 2.0 * H * ((Drr(log(h)) - Dr(log(h)) / r) + 2.0 * Dr(log(h))**2) / r.
				ctx->Dr_u_aux[1 * ctx->dim + IDX(i, j)] = 2.0 * exp(2.0 * u[2 * ctx->dim + IDX(i, j)]) * ((ctx->Drr_u[2 * ctx->dim + IDX(i, j)] - ctx->Dr_u[2 * ctx->dim + IDX(i, j)] / r) + 2.0 * ctx->Dr_u[2 * ctx->dim + IDX(i, j)] * ctx->Dr_u[2 * ctx->dim + IDX(i, j)]) / r;
			}
		}
	}
#endif

#ifdef DERIVATIVE_DEBUG
	write_single_file_2d(ctx->Dr_u, "Dr_log_alpha.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dr_u + ctx->dim, "Dr_beta.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dr_u + 2 * ctx->dim, "Dr_log_h.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dr_u + 3 * ctx->dim, "Dr_log_a.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dr_u + 4 * ctx->dim, "Dr_psi.asc", ctx->NrTotal, ctx->NzTotal);

	write_single_file_2d(ctx->Dz_u, "Dz_log_alpha.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dz_u + ctx->dim, "Dz_beta.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dz_u + 2 * ctx->dim, "Dz_log_h.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dz_u + 3 * ctx->dim, "Dz_log_a.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dz_u + 4 * ctx->dim, "Dz_psi.asc", ctx->NrTotal, ctx->NzTotal);

	write_single_file_2d(ctx->Drr_u, "Drr_log_alpha.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Drr_u + ctx->dim, "Drr_beta.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Drr_u + 2 * ctx->dim, "Drr_log_h.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Drr_u + 3 * ctx->dim, "Drr_log_a.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Drr_u + 4 * ctx->dim, "Drr_psi.asc", ctx->NrTotal, ctx->NzTotal);

	write_single_file_2d(ctx->Dzz_u, "Dzz_log_alpha.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dzz_u + ctx->dim, "Dzz_beta.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dzz_u + 2 * ctx->dim, "Dzz_log_h.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dzz_u + 3 * ctx->dim, "Dzz_log_a.asc", ctx->NrTotal, ctx->NzTotal);
	write_single_file_2d(ctx->Dzz_u + 4 * ctx->dim, "Dzz_psi.asc", ctx->NrTotal, ctx->NzTotal);
#endif

	// Lower-left corner with parity.
	for (i = 0; i < ctx->ghost; ++i)
	{
		for (j = 0; j < ctx->ghost; ++j)
		{
			for (k = 0; k < GNUM; ++k)
			{
				f[k * ctx->dim + IDX(i, j)] = 0.0;
			}
		}
	}

	// Parity on r axis.
	for (i = 0; i < ctx->ghost; ++i)
	{
		#pragma omp parallel shared(f) private(j, k)
		{
			#pragma omp for schedule(dynamic, 1)
			for (j = ctx->ghost; j < ctx->NzTotal; ++j)
			{
				for (k = 0; k < GNUM; ++k)
				{
					f[k * ctx->dim + IDX(i, j)] = 0.0;
				}
			}
		}
	}

	// Parity on z axis.
	for (j = 0; j < ctx->ghost; ++j)
	{
		#pragma omp parallel shared(f) private(i, k)
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ctx->ghost; i < ctx->NrTotal; ++i)
			{
				for (k = 0; k < GNUM; ++k)
				{
					f[k * ctx->dim + IDX(i, j)] = 0.0;
				}
			}
		}
	}
	// Main interior points.
	#pragma omp parallel shared(f) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ctx->ghost; i < ctx->NrTotal - 1; ++i)
		{
			for (j = ctx->ghost; j < ctx->NzTotal - 1; ++j)
			{
				rhs_vars(f, u, ctx->Dr_u, ctx->Dz_u, ctx->Drr_u, ctx->Dzz_u, ctx->NrTotal, ctx->NzTotal, ctx->dim, ctx->ghost, i, j, ctx->dr, ctx->dz, ctx->l, ctx->m, w, -1.0, ctx->u_aux, ctx->Dr_u_aux);
			}
		}
	}

	// Before doing boundary, analysis must be done.
	/*
	ex_cart_to_pol(&i_u, &i_rr, &i_th, NULL, NULL, u, Dr_u, Dz_u, Drz_u, GNUM, dr, dz, NrInterior, NzInterior, ghost, &NrrTotal, &NthTotal, &p_dim, &drr, &dth, &rr_inf);
	ex_analysis(0, &M_KOMAR, &J_KOMAR, &GRV2, &GRV3, i_u, i_rr, i_th, w, m, l, ghost, order, NrrTotal, NthTotal, p_dim, drr, dth, rr_inf);

	//printf("Hi there, friend! M_KOMAR = %.5E, J_KOMAR = %.5E.\n", M_KOMAR, J_KOMAR);

	SAFE_FREE(i_u);
	SAFE_FREE(i_rr);
	SAFE_FREE(i_th);
	*/

	// Z boundary condition.
	j = ctx->NzTotal - 1;
	#pragma omp parallel shared(f) private(i)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ctx->ghost; i < ctx->NrTotal - 1; ++i)
		{
			rhs_bdry(f, u, ctx->Dr_u, ctx->Dz_u, ctx->NrTotal, ctx->NzTotal, ctx->dim, ctx->ghost, i, j, ctx->dr, ctx->dz, ctx->l, ctx->m, w, ctx->M_KOMAR, ctx->J_KOMAR, -1.0);
		}
	}

	/*
	// Lower-right corner with parity.
	for (i = ghost + NzInterior; i < NzTotal; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			f[IDX(i, j)] = f[dim + IDX(i, j)] = f[2 * dim + IDX(i, j)] = f[3 * dim + IDX(i, j)] = f[4 * dim + IDX(i, j)] = 0.0;
		}
	}
	*/

	// R boundary condition.
	i = ctx->NrTotal - 1;
	#pragma omp parallel shared(f) private(j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (j = ctx->ghost; j < ctx->NzTotal - 1; ++j)
		{
			rhs_bdry(f, u, ctx->Dr_u, ctx->Dz_u, ctx->NrTotal, ctx->NzTotal, ctx->dim, ctx->ghost, i, j, ctx->dr, ctx->dz, ctx->l, ctx->m, w, ctx->M_KOMAR, ctx->J_KOMAR, -1.0);
		}
	}

	// Top-right corner boundary condition.
	i = ctx->NrTotal - 1;
	j = ctx->NzTotal - 1;
	rhs_bdry(f, u, ctx->Dr_u, ctx->Dz_u, ctx->NrTotal, ctx->NzTotal, ctx->dim, ctx->ghost, i, j, ctx->dr, ctx->dz, ctx->l, ctx->m, w, ctx->M_KOMAR, ctx->J_KOMAR, -1.0);

	// Omega constraint.
	f[ctx->w_idx] = 0.0;

	// All done. 
	return;
}