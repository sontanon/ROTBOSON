//! faer sparse-LU backend — **transitional, parity sizes only** (ADR 0001).
//!
//! Direct sparse solves exist in RBS-NG for two purposes: the oracle parity
//! gates (N <= 128 seed solves) and unit tests of the Newton machinery.
//! They must never appear in a scaling path; the production solver is
//! geometric multigrid + Krylov ([`crate::mg_krylov`]).

use crate::{csr::CsrMatrix, LinearSolver, SolveError};
use faer::linalg::solvers::Solve;
use faer::sparse::{linalg::solvers::Lu, SparseColMat, SymbolicSparseColMat};

/// Sparse LU with partial (row) pivoting over faer.
#[derive(Debug, Default)]
pub struct FaerSparseLu {
    lu: Option<Lu<usize, f64>>,
    n: usize,
}

impl FaerSparseLu {
    /// An empty (unfactored) backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// `faer`'s CSC arrays of a matrix: (`col_ptrs`, `row_idx`, `vals`).
type CscArrays = (Vec<usize>, Vec<usize>, Vec<f64>);

/// Convert a validated CSR matrix to faer's CSC arrays (column pointers,
/// row indices sorted within columns, values).
fn csr_to_csc(a: &CsrMatrix) -> Result<CscArrays, SolveError> {
    a.validate()?;
    let nnz = a.nnz();
    let mut col_counts = vec![0usize; a.ncols];
    for &c in &a.col_inds {
        col_counts[c] += 1;
    }
    let mut col_ptrs = Vec::with_capacity(a.ncols + 1);
    col_ptrs.push(0);
    for &c in &col_counts {
        col_ptrs.push(col_ptrs[col_ptrs.len() - 1] + c);
    }
    let mut next = col_ptrs.clone();
    let mut row_idx = vec![0usize; nnz];
    let mut vals = vec![0.0; nnz];
    for r in 0..a.nrows {
        for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
            let c = a.col_inds[k];
            let dest = next[c];
            row_idx[dest] = r;
            vals[dest] = a.vals[k];
            next[c] += 1;
        }
    }
    Ok((col_ptrs, row_idx, vals))
}

impl LinearSolver for FaerSparseLu {
    fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError> {
        let (col_ptrs, row_idx, vals) = csr_to_csc(a)?;
        let symbolic =
            SymbolicSparseColMat::<usize>::new_checked(a.nrows, a.ncols, col_ptrs, None, row_idx);
        let mat = SparseColMat::<usize, f64>::new(symbolic, vals);
        match mat.sp_lu() {
            Ok(lu) => {
                self.lu = Some(lu);
                self.n = a.nrows;
                Ok(())
            }
            Err(faer::sparse::linalg::LuError::SymbolicSingular { .. }) => {
                Err(SolveError::Singular)
            }
            Err(_) => Err(SolveError::InvalidMatrix("faer LU failure")),
        }
    }

    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        let Some(lu) = &self.lu else {
            return Err(SolveError::NotFactorized);
        };
        if b.len() != self.n || x.len() != self.n {
            return Err(SolveError::DimensionMismatch {
                expected: self.n,
                got: b.len().min(x.len()),
            });
        }
        let mut rhs = faer::Col::<f64>::from_fn(self.n, |i| b[i]);
        lu.solve_in_place(&mut rhs);
        for (i, out) in x.iter_mut().enumerate() {
            *out = rhs[i];
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "faer-sparse-lu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_matrix() -> CsrMatrix {
        // [[4, 1, 0], [1, 5, 2], [0, 2, 6]]
        CsrMatrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0],
        )
    }

    #[test]
    fn solves_to_working_precision() {
        let a = demo_matrix();
        let b = [7.0, 8.0, 8.0];
        let mut solver = FaerSparseLu::new();
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        let ax = a.matvec(&x);
        for (i, r) in ax.iter().enumerate() {
            assert!((r - b[i]).abs() < 1e-12, "row {i}: {r} vs {}", b[i]);
        }
    }

    #[test]
    fn stale_factor_reuse_across_right_hand_sides() {
        // The campaign pattern near folds: factor once, solve many.
        let a = demo_matrix();
        let mut solver = FaerSparseLu::new();
        solver.factor(&a).expect("factor");
        for t in 0..3_usize {
            let base = u32::try_from(t).expect("t fits u32");
            let b: Vec<f64> = (0..3_u32).map(|i| f64::from(i + base)).collect();
            let x = solver.solve(&b).expect("solve");
            let ax = a.matvec(&x);
            for (i, r) in ax.iter().enumerate() {
                assert!((r - b[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn singular_matrix_reports_singularity() {
        // Both rows only reach column 0: structurally rank-deficient.
        let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0]);
        let mut solver = FaerSparseLu::new();
        assert_eq!(solver.factor(&a), Err(SolveError::Singular));
    }

    #[test]
    fn solve_before_factor_is_an_error() {
        let mut solver = FaerSparseLu::new();
        assert_eq!(
            solver.solve_into(&[1.0, 2.0], &mut [0.0; 2]),
            Err(SolveError::NotFactorized)
        );
    }

    #[test]
    fn invalid_structure_is_rejected() {
        let bad = CsrMatrix {
            nrows: 1,
            ncols: 2,
            row_ptrs: vec![0, 1],
            col_inds: vec![9],
            vals: vec![1.0],
        };
        let mut solver = FaerSparseLu::new();
        assert!(matches!(
            solver.factor(&bad),
            Err(SolveError::InvalidMatrix(_))
        ));
    }
}
