#include "context.h"

MKL_INT
nleq_err_qnerr(rb_context *ctx, // INPUT: Runtime context.
               MKL_INT *err_code, // OUTPUT: Pointer to integer containing error code.
               double **u, // IN-OUTPUT: Pointer to array of solution vectors.
                           //            First entry contains initial guess.
               double **f, // IN-OUTPUT: Pointer to array of RHS's.
                           //            First entry contains initial RHS.
               double **du, // OUTPUT: Pointer to array of updates.
               double **du_bar, // OUTPUT: Pointer to array of updates (bar).
               double *norm_du, // OUTPUT: Pointer to array of update norms.
               double *norm_du_bar, // OUTPUT: Pointer to array of update (bar) norms.
               double *Theta, // OUTPUT: Pointer to array of monitoring quantity.
               double *alpha, // OUTPUT: Pointer to array of alpha's.
               csr_matrix *J, // INPUT: Pointer to Jacobian matrix type.
               const double epsilon, // INPUT: Exit tolerance.
               const MKL_INT max_newton_iterations, // INPUT: Maximum number of Newton iterations.
               rb_rhs_fn RHS_CALC, // INPUT: RHS calculation subroutine.
               rb_jacobian_fn JACOBIAN_CALC, // INPUT: Jacobian calculation subroutine.
               rb_norm_fn NORM, // INPUT: Norm calculation subroutine.
               rb_dot_fn DOT, // INPUT: Dot product calculation subroutine.
               rb_linear_solve_fn LINEAR_SOLVE_1, rb_linear_solve_fn LINEAR_SOLVE_2);
