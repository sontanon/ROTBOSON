void z_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT z_sym);

void r_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT r_sym);

void corner_symmetry(double *a, MKL_INT *ia, MKL_INT *ja, const MKL_INT offset,
	const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
	const MKL_INT g_num, const MKL_INT i, const MKL_INT j, const MKL_INT r_sym, const MKL_INT z_sym);
