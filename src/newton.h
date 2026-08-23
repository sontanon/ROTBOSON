#include "context.h"

MKL_INT newton(rb_context *ctx, // INPUT: Runtime context (grid/field/solver params).
               MKL_INT *err_code, // OUTPUT: Pointer to integer containing error code.
               double **u, // IN-OUTPUT: Pointer to array of solution vectors.
                           //            First entry contains initial guess.
               double **f, // IN-OUTPUT: Pointer to array of RHS's.
                           //            First entry contains initial RHS.
               double *lambda, // IN-OUTPUT: Pointer to array of damping factors.
                               //            First entry contains initial damping factor.
               double **du, // OUTPUT: Pointer to array of updates.
               double *norm_du, // OUTPUT: Pointer to array of update norms.
               double *Theta, // OUTPUT: Pointer to array of monitoring quantity.
               csr_matrix *J, // INPUT: Pointer to jacobian matrix type.
               const double epsilon, // INPUT: Exit tolerance.
               const MKL_INT max_newton_iterations, // INPUT: Maximum number of Newton iterations.
               rb_rhs_fn RHS_CALC, // INPUT: RHS calculation subroutine.
               rb_jacobian_fn JACOBIAN_CALC, // INPUT: Jacobian calculation subroutine.
               rb_norm_fn NORM, // INPUT: Norm calculation subroutine.
               rb_linear_solve_fn LINEAR_SOLVE_1 // INPUT: Linear solver subroutine.
);
