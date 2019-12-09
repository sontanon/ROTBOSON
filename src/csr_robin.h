void z_robin_2nd_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void r_robin_2nd_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void corner_robin_2nd_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void z_robin_4th_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void r_robin_4th_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void corner_robin_4th_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void z_so_robin_4th_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

void r_so_robin_4th_order(
	double *aa, 		// CSR matrix values.
	MKL_INT *ia, 		// CSR matrix row beginnings.
	MKL_INT *ja,		// CSR matrix column indices.
	const MKL_INT offset, 	// Number of elements previously filled into CSR a array.
	const MKL_INT NrTotal, 	// R total dimension.
	const MKL_INT NzTotal, 	// Z total dimension.
	const MKL_INT dim,	// Grid function total dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// R integer coordinate.
	const MKL_INT j, 	// Z integer coordinate.
	const double dr, 	// R spatial step.
	const double dz,	// Z spatial step.
	const MKL_INT n, 	// Robin rr power decay type.
	const MKL_INT bound_error);	// Whether to use Dirichlet (0) or Robin (1).

