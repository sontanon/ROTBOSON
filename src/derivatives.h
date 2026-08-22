void ex_diff1r(double *dvar, const double *var, const MKL_INT symr, const double dr, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff1z(double *dvar, const double *var, const MKL_INT symz, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2r(double *dvar, const double *var, const MKL_INT symr, const double dr, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2z(double *dvar, const double *var, const MKL_INT symz, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2rz(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz, const double dr, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);

void ex_diff1th(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz, const double dth, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff1rr(double *dvar, const double *var, const MKL_INT symrr, const double drr, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT ghost, const MKL_INT order);


void ex_diff1(double *du, const double *u, const MKL_INT sym, const double h, const MKL_INT dim, const MKL_INT ghost, const MKL_INT order);
void ex_diff2(double *du, const double *u, const MKL_INT sym, const double h, const MKL_INT dim, const MKL_INT ghost, const MKL_INT order);
void ex_diff3(double *du, const double *u, const MKL_INT sym, const double h, const MKL_INT dim, const MKL_INT ghost, const MKL_INT order);