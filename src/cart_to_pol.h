#define cart_to_pol(i_u, i_rr, i_th, r, z, u, Dr_u, Dz_u, Drz_u, g_num) ex_cart_to_pol((i_u), (i_rr), (i_th), (r), (z), (u), (Dr_u), (Dz_u), (Drz_u), (g_num), dr, dz, NrInterior, NzInterior, ghost, &NrrTotal, &NthTotal, &p_dim, &drr, &dth, &rr_inf)

void ex_cart_to_pol(
	double **i_u,
	double **i_rr,
	double **i_th,
	double *r,
	double *z,
	double *u,
	double *Dr_u,
	double *Dz_u,
	double *Drz_u,
	const MKL_INT g_num,
	const double dr,
	const double dz,
	const MKL_INT NrInterior, 
	const MKL_INT NzInterior,
	const MKL_INT ghost,
	MKL_INT *p_NrrTotal,
	MKL_INT *p_NthTotal,
	MKL_INT *p_p_dim,
	double *p_drr,
	double *p_dth,
	double *p_rr_inf);