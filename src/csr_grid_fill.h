void csr_grid_fill_2nd(
	csr_matrix A,
	const MKL_INT NrInterior, 
	const MKL_INT NzInterior, 
	const double dr, 
	const double dz,
	const double *u,
	const MKL_INT l,
	const double m,
	const MKL_INT r_sym[5],
	const MKL_INT z_sym[5],
	const MKL_INT bound_order[5],
	const MKL_INT nnz[5],
	const MKL_INT p_center[5],
	const MKL_INT p_bound[5],
	void (*j_cc)(double *, MKL_INT *, MKL_INT *,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT,
		const double, const double, const MKL_INT, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const double, const double, const double, const double, const double,
		const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT, const MKL_INT)
);