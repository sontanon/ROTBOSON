void omega_constraint(
  double *a,          // CSR matrix values.
  MKL_INT *ia,            // CSR matrix row beginnings.
  MKL_INT *ja,            // CSR matrix column indices.
  const MKL_INT offset,   // Number of elements previously filled into CSR arrays.
  const MKL_INT NrTotal,  // R total dimension.
  const MKL_INT NzTotal,  // Z total dimension.
  const MKL_INT dim,      // Grid function total dimension: dim = NrTotal * NzTotal.
  const MKL_INT g_num,    // Grid number to hold fixed.
  const MKL_INT w_idx,    // Omega index value.
  const MKL_INT i,        // R integer coordinate.
  const MKL_INT j         // Z integer coordinate.
);