#define analysis(sph_u, sph_rr, sph_th, w) ex_analysis(1, &M_KOMAR, &J_KOMAR, &GRV2, &GRV3, (sph_u), (sph_rr), (sph_th), (w), m, l, ghost, order, NrrTotal, NthTotal, p_dim, drr, dth, rr_inf)

void ex_analysis(const MKL_INT print, double *M, double *J, double *GRV2, double *GRV3,
	double *sph_u, double *sph_rr, double *sph_th, 
	const double w, const double m, const MKL_INT l,
	const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT p_dim, const double drr, const double dth, const double rr_inf);

void ex_phi_analysis(const MKL_INT print, 
	double *phi_max, double *rr_phi_max, MKL_INT *k_rr_max,
	double *sph_u, double *sph_rr, double *sph_th, 
	const MKL_INT l,
	const MKL_INT ghost, const MKL_INT order, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT p_dim, const double drr, const double dth, const double rr_inf);