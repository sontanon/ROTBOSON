#include "context.h"

MKL_INT nnz_jacobian(const rb_context *ctx);
void csr_gen_jacobian(rb_context *ctx, csr_matrix A, double *u, const MKL_INT print);
