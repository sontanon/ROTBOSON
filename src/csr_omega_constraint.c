#include "tools.h"

// Fill-in omega constraint.
// This constraint fixed the field value at integer coordinates i, j.
// For example, to hold the grid function u5 fixed at coordinates i, j,
// one would set g_num = 5 and input i, j as parameters below.
// To hold omega fixed, g_num = w_idx, and i = j = 0.
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
  const MKL_INT fixedPhi, // Indicates if omega is fixed.
  const MKL_INT i,        // R integer coordinate.
  const MKL_INT j         // Z integer coordinate.
)
{
  // This constrains g_num field to have a fixed value at IDX(i,j).
  // This row corresponds to the omega variable.
  ia[w_idx] = BASE + offset;
  // Set value to one, thus forcing the solution to be zero.
  a[offset] = 1.0;
  // Column index fixed the value.
  if (fixedPhi)
  {
    ja[offset] = BASE + (g_num - 1) * dim + IDX(i, j);
  }
  else
  {
    ja[offset] = BASE + w_idx;
  }
  

  return;
}
