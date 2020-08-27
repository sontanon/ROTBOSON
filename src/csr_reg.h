void c_reg_2nd_order(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM);

void s_reg_2nd_order(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM);

void c_reg_4th_order(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM);

void s_reg_4th_order(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM);

void o_reg_4th_order(
	double *aa,
	MKL_INT *ia,
	MKL_INT *ja,
	const MKL_INT offset,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim, 
	const MKL_INT U_GNUM,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const double v,
	const MKL_INT V_GNUM);