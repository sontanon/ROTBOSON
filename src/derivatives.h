#define diff1r(Dr_u, u, r_sym) ex_diff1r((Dr_u), (u), (r_sym), dr, NrTotal, NzTotal, ghost, order)
#define diff1z(Dz_u, u, z_sym) ex_diff1z((Dz_u), (u), (z_sym), dz, NrTotal, NzTotal, ghost, order)
#define diff2r(Drr_u, u, r_sym) ex_diff2r((Drr_u), (u), (r_sym), dr, NrTotal, NzTotal, ghost, order)
#define diff2z(Dzz_u, u, z_sym) ex_diff2z((Dzz_u), (u), (z_sym), dz, NrTotal, NzTotal, ghost, order)
#define diff2rz(Drz_u, u, r_sym, z_sym) ex_diff2rz((Drz_u), (u), (r_sym), (z_sym), dr, dz, NrTotal, NzTotal, ghost, order)
#define diff1th(D_th_u, u, r_sym, z_sym) ex_diff1th((D_th_u), (u), (r_sym), (z_sym), dth, NrrTotal, NthTotal, ghost, order)
#define diff1rr(D_rr_u, u, rr_sym) ex_diff1rr((D_rr_u), (u), (rr_sym), drr, NrrTotal, NthTotal, ghost, order)

void ex_diff1r(double *dvar, const double *var, const MKL_INT symr, const double dr, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff1z(double *dvar, const double *var, const MKL_INT symz, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2r(double *dvar, const double *var, const MKL_INT symr, const double dr, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2z(double *dvar, const double *var, const MKL_INT symz, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff2rz(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz, const double dr, const double dz, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const MKL_INT order);

void ex_diff1th(double *dvar, const double *var, const MKL_INT symr, const MKL_INT symz, const double dth, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT ghost, const MKL_INT order);
void ex_diff1rr(double *dvar, const double *var, const MKL_INT symrr, const double drr, const MKL_INT NrrTotal, const MKL_INT NthTotal, const MKL_INT ghost, const MKL_INT order);