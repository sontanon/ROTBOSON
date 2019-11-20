void rhs_z_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l);       // Scalar field rotation number.

void rhs_r_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l);       // Scalar field rotation number.

void rhs_corner_pseudo_exp_decay_2nd(
    double *f,              // Array containing RHS.
    const double *u,        // Array contianing solution.
    const MKL_INT NrTotal,      // R total dimension.
    const MKL_INT NzTotal,      // Z total dimension.
    const MKL_INT dim,          // Grid function total dimension: dim = NrTotal * NzTotal.
    const MKL_INT g_num,        // Grid number.
    const MKL_INT i,            // R integer coordinate.
    const MKL_INT j,            // Z integer coordinate.
    const double dr,        // R spatial step.
    const double dz,        // Z spatial step.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale,     // Multiply equation by an overall factor.
    const MKL_INT w_idx,    // Omega index.
    const double m,         // Scalar field mass.
    const MKL_INT l);       // Scalar field rotation number.