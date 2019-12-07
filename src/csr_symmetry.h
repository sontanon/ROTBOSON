void z_symmetry(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings.
	MKL_INT *ja, 		// CSR array for columns.
	const MKL_INT offset,	// Number of elements filled before.
	const MKL_INT NrTotal, 	// Grid total dimension in r. 
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const MKL_INT ghost,	// Number of parity ghost zones.
	const MKL_INT z_sym);	// Equator parity.

void r_symmetry(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings.
	MKL_INT *ja, 		// CSR array for columns.
	const MKL_INT offset,	// Number of elements filled before.
	const MKL_INT NrTotal, 	// Grid total dimension in r. 
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const MKL_INT ghost,	// Number of parity ghost zones.
	const MKL_INT r_sym);	// Axis parity.

void corner_symmetry(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings.
	MKL_INT *ja, 		// CSR array for columns.
	const MKL_INT offset,	// Number of elements filled before.
	const MKL_INT NrTotal, 	// Grid total dimension in r. 
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT g_num, 	// Grid number.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const MKL_INT ghost,	// Number of parity ghost zones.
	const MKL_INT r_sym, 	// Axis parity.
	const MKL_INT z_sym);	// Equator parity.