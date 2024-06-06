// Regularization coupling.
#undef REGULARIZATION_COUPLING

#ifdef REGULARIZATION_COUPLING
#define REG_MU 0.5

// GRID
#ifdef MAIN_FILE
double solver_dr;
MKL_INT solver_NrTotal;
MKL_INT solver_NzTotal;
MKL_INT solver_ghost;
#else
extern double solver_dr;
extern MKL_INT solver_NrTotal;
extern MKL_INT solver_NzTotal;
extern MKL_INT solver_ghost;
#endif
#endif