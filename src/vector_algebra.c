#include "tools.h"
#include "context.h"

#define GRID_VARIABLE_START 1
#define GRID_VARIABLE_END 6

double norm2(const rb_context *ctx, double *x)
{
    return cblas_dnrm2(ctx->dim, x, 1) / sqrt(ctx->dim);
}

double dot(const rb_context *ctx, double *x, double *y)
{
    return cblas_ddot(ctx->dim, x, 1, y, 1) / ctx->dim;
}

double norm2_interior(const rb_context *ctx, double *x)
{
    double sum = cblas_ddot(ctx->NzTotal * ctx->NrInterior, x + ctx->ghost * ctx->NzTotal, 1,
                            x + ctx->ghost * ctx->NzTotal, 1);

    MKL_INT k = 0;

    for (k = 0; k < ctx->ghost; ++k)
    {
        sum -= cblas_ddot(ctx->NrInterior, x + ctx->ghost * ctx->NzTotal + k, ctx->NzTotal,
                          x + ctx->ghost * ctx->NzTotal + k, ctx->NzTotal);
        sum -= cblas_ddot(ctx->NrInterior, x + (ctx->ghost + 1) * ctx->NzTotal - ctx->ghost + k,
                          ctx->NzTotal, x + (ctx->ghost + 1) * ctx->NzTotal - ctx->ghost + k,
                          ctx->NzTotal);
    }

    return sqrt(sum) / sqrt(ctx->NrInterior * ctx->NzInterior);
}

double dot_interior(const rb_context *ctx, double *x, double *y)
{
    double sum = cblas_ddot(ctx->NzTotal * ctx->NrInterior, x + ctx->ghost * ctx->NzTotal, 1,
                            y + ctx->ghost * ctx->NzTotal, 1);

    MKL_INT k = 0;

    for (k = 0; k < ctx->ghost; ++k)
    {
        sum -= cblas_ddot(ctx->NrInterior, x + ctx->ghost * ctx->NzTotal + k, ctx->NzTotal,
                          y + ctx->ghost * ctx->NzTotal + k, ctx->NzTotal);
        sum -= cblas_ddot(ctx->NrInterior, x + (ctx->ghost + 1) * ctx->NzTotal - ctx->ghost + k,
                          ctx->NzTotal, y + (ctx->ghost + 1) * ctx->NzTotal - ctx->ghost + k,
                          ctx->NzTotal);
    }

    return sum / (ctx->NrInterior * ctx->NzInterior);
}

double dot_interior_all_variables(const rb_context *ctx, double *x, double *y)
{
    double sum = 0.0;
    MKL_INT k = 0;

    // Add all dot products.
    for (k = GRID_VARIABLE_START; k < GRID_VARIABLE_END; ++k)
    {
        sum += dot_interior(ctx, x + k * ctx->dim, y + k * ctx->dim);
    }

    // Rescale.
    return sum / ((double)(GRID_VARIABLE_END - GRID_VARIABLE_START));
}

double norm2_interior_all_variables(const rb_context *ctx, double *x)
{
    double sum = dot_interior_all_variables(ctx, x, x);

    return sqrt(sum);
}

double dot_all_variables(const rb_context *ctx, double *x, double *y)
{
    double sum = 0.0;
    MKL_INT k = 0;

    // Add all dot products.
    for (k = GRID_VARIABLE_START; k < GRID_VARIABLE_END; ++k)
    {
        sum += dot(ctx, x + k * ctx->dim, y + k * ctx->dim);
    }

    // Rescale.
    return sum / ((double)(GRID_VARIABLE_END - GRID_VARIABLE_START));
}

double norm2_all_variables(const rb_context *ctx, double *x)
{
    double sum = dot_all_variables(ctx, x, x);

    return sqrt(sum);
}