#include "tools.h"
#include "output.h"

// ex_analysis / ex_phi_analysis take a solution_writer (or NULL to skip file
// output) instead of the old "print" flag, so analysis results flow through
// the same I/O abstraction as every other field. `sw` is named to avoid
// clashing with the scalar-field frequency `w` in ex_analysis.

void ex_analysis(solution_writer *sw, double *M, double *J, double *GRV2, double *GRV3,
                 double *sph_u, double *sph_rr, double *sph_th, const double w, const double m,
                 const MKL_INT l, const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal,
                 const MKL_INT NthTotal, const MKL_INT p_dim, const double drr, const double dth,
                 const double rr_inf);

void ex_phi_analysis(solution_writer *sw, double *phi_max, double *rr_phi_max, MKL_INT *k_rr_max,
                     double *sph_u, double *sph_rr, double *sph_th, const MKL_INT l,
                     const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal,
                     const MKL_INT NthTotal, const MKL_INT p_dim, const double drr,
                     const double dth, const double rr_inf);
