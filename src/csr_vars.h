void jacobian_2nd_order_variable_omega_cc(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For cc stencil, each grid function has 5 variables.
	const double u101, const double u110, const double u111, const double u112, const double u121,
	const double u201, const double u210, const double u211, const double u212, const double u221,
	const double u301, const double u310, const double u311, const double u312, const double u321,
	const double u401, const double u410, const double u411, const double u412, const double u421,
	const double u501, const double u510, const double u511, const double u512, const double u521,
	const double u601, const double u610, const double u611, const double u612, const double u621,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5.
	const MKL_INT offset6);	// Number of elements filled before filling function 5.

void jacobian_4th_order_variable_omega_cc(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For cc stencil, each grid function has 9 variables.
	const double u102, const double u112, const double u120, const double u121, const double u122, const double u123, const double u124, const double u132, const double u142,
	const double u202, const double u212, const double u220, const double u221, const double u222, const double u223, const double u224, const double u232, const double u242,
	const double u302, const double u312, const double u320, const double u321, const double u322, const double u323, const double u324, const double u332, const double u342,
	const double u402, const double u412, const double u420, const double u421, const double u422, const double u423, const double u424, const double u432, const double u442,
	const double u502, const double u512, const double u520, const double u521, const double u522, const double u523, const double u524, const double u532, const double u542,
	const double u602, const double u612, const double u620, const double u621, const double u622, const double u623, const double u624, const double u632, const double u642,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5.
	const MKL_INT offset6);	// Number of elements filled before filling function 5.

void jacobian_4th_order_variable_omega_cs(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For cs stencil, each grid function has 10 variables.
	const double u104, const double u114, const double u120, const double u121, const double u122, const double u123, const double u124, const double u125, const double u134, const double u144,
	const double u204, const double u214, const double u220, const double u221, const double u222, const double u223, const double u224, const double u225, const double u234, const double u244,
	const double u304, const double u314, const double u320, const double u321, const double u322, const double u323, const double u324, const double u325, const double u334, const double u344,
	const double u404, const double u414, const double u420, const double u421, const double u422, const double u423, const double u424, const double u425, const double u434, const double u444,
	const double u504, const double u514, const double u520, const double u521, const double u522, const double u523, const double u524, const double u525, const double u534, const double u544,
	const double u604, const double u614, const double u620, const double u621, const double u622, const double u623, const double u624, const double u625, const double u634, const double u644,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5
	const MKL_INT offset6);	// Number of elements filled before filling function 5

void jacobian_4th_order_variable_omega_sc(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For sc stencil, each grid function has 10 variables.
	const double u102, const double u112, const double u122, const double u132, const double u140, const double u141, const double u142, const double u143, const double u144, const double u152,
	const double u202, const double u212, const double u222, const double u232, const double u240, const double u241, const double u242, const double u243, const double u244, const double u252,
	const double u302, const double u312, const double u322, const double u332, const double u340, const double u341, const double u342, const double u343, const double u344, const double u352,
	const double u402, const double u412, const double u422, const double u432, const double u440, const double u441, const double u442, const double u443, const double u444, const double u452,
	const double u502, const double u512, const double u522, const double u532, const double u540, const double u541, const double u542, const double u543, const double u544, const double u552,
	const double u602, const double u612, const double u622, const double u632, const double u640, const double u641, const double u642, const double u643, const double u644, const double u652,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5, 	// Number of elements filled before filling function 5
	const MKL_INT offset6);	// Number of elements filled before filling function 5

void jacobian_4th_order_variable_omega_ss(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For ss stencil, each grid function has 11 variables.
	const double u104, const double u114, const double u124, const double u134, const double u140, const double u141, const double u142, const double u143, const double u144, const double u145, const double u154,
	const double u204, const double u214, const double u224, const double u234, const double u240, const double u241, const double u242, const double u243, const double u244, const double u245, const double u254,
	const double u304, const double u314, const double u324, const double u334, const double u340, const double u341, const double u342, const double u343, const double u344, const double u345, const double u354,
	const double u404, const double u414, const double u424, const double u434, const double u440, const double u441, const double u442, const double u443, const double u444, const double u445, const double u454,
	const double u504, const double u514, const double u524, const double u534, const double u540, const double u541, const double u542, const double u543, const double u544, const double u545, const double u554,
	const double u604, const double u614, const double u624, const double u634, const double u640, const double u641, const double u642, const double u643, const double u644, const double u645, const double u654,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5, 	// Number of elements filled before filling function 5
	const MKL_INT offset6);	// Number of elements filled before filling function 5