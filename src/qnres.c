#include "tools.h"

MKL_INT nleq_res_qnres(MKL_INT 	*err_code,
		double 	**u,
		double 	**f,
		double	**du,
		double 	*norm_f,
		double	*Theta,
		double 	*kappa,
		csr_matrix *J,
		double	(*NORM)(const double *)	,			// INPUT: Norm calculation subroutine.
		double	(*DOT)(const double *, const double *)	,	// INPUT: Dot product calculation subroutine.
		void 	(*LINEAR_SOLVE_1)(double *, csr_matrix *, double *),	
		void 	(*LINEAR_SOLVE_2)(double *, csr_matrix *, double *)

		)
{
	/* Iteration counters. */
	MKL_INT l = 0, i = 0;

	/* Get matrix dimension. */
	MKL_INT dim = J->nrows;

	// Intial RHS is stored in f[0].
	// Then calculated sigma0 = ||f[0]||**2.
	double sigma0 = DOR(f[l], f[l]);

	// Solve linear

}