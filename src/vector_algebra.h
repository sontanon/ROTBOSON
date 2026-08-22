#include "context.h"

double dot(const rb_context *ctx, double *x, double *y);
double norm2(const rb_context *ctx, double *x);

double dot_interior(const rb_context *ctx, double *x, double *y);
double norm2_interior(const rb_context *ctx, double *x);

double dot_interior_all_variables(const rb_context *ctx, double *x, double *y);
double norm2_interior_all_variables(const rb_context *ctx, double *x);

double dot_all_variables(const rb_context *ctx, double *x, double *y);
double norm2_all_variables(const rb_context *ctx, double *x);
