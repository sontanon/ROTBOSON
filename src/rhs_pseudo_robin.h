
void rhs_z_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale);    // Multiply equation by an overall factor.

void rhs_r_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale);    // Multiply equation by an overall factor.

void rhs_corner_pseudo_robin_2nd(
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
    const MKL_INT n,            // Robin decay type.
    const MKL_INT bound_error,  // What type of Robin condition to use.
    const double u_inf,     // Function value at infinity.
    const double scale);    // Multiply equation by an overall factor.