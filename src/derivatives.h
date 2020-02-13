void diff1r(double *Dr_u, const double *u, const MKL_INT r_sym);
void diff1z(double *Dz_u, const double *u, const MKL_INT z_sym);
void diff2r(double *Drr_u, const double *u, const MKL_INT r_sym);
void diff2z(double *Dzz_u, const double *u, const MKL_INT z_sym);
void diff2rz(double *Drz_u, const double *u, const MKL_INT r_sym, const MKL_INT z_sym);
void diff1rr(double *dvar, const double *var, const MKL_INT symrr);