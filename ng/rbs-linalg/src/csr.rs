//! The CSR matrix contract type, mirroring the C oracle's `csr_matrix`.

/// Compressed sparse row matrix, 0-based: row `r`'s entries live at
/// `row_ptrs[r]..row_ptrs[r+1]` in `col_inds`/`vals`, sorted by column
/// within each row.
///
/// This is the C `csr_matrix` layout (`src/tools.h`: `ia`/`ja`/`a` arrays,
/// 1-based there) with the index base dropped. Fields are public so the
/// assembly can fill them directly like the C `csr_gen_jacobian` fills
/// `ia`/`ja`/`a`; [`CsrMatrix::validate`] checks the contract and
/// [`crate::faer_lu::FaerSparseLu`] refuses invalid matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Row count.
    pub nrows: usize,
    /// Column count.
    pub ncols: usize,
    /// Row pointers, length `nrows + 1`, `row_ptrs[0] == 0`.
    pub row_ptrs: Vec<usize>,
    /// Column indices per row, sorted ascending within each row.
    pub col_inds: Vec<usize>,
    /// Values, parallel to `col_inds`.
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    /// A validated CSR matrix.
    ///
    /// # Panics
    ///
    /// Panics if the structure violates the CSR contract (see
    /// [`CsrMatrix::validate`]).
    #[must_use]
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<usize>,
        col_inds: Vec<usize>,
        vals: Vec<f64>,
    ) -> Self {
        let m = Self {
            nrows,
            ncols,
            row_ptrs,
            col_inds,
            vals,
        };
        if let Err(why) = m.validate() {
            panic!("invalid CSR matrix: {why}");
        }
        m
    }

    /// The identity matrix (for tests and preconditioner scaffolding).
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            nrows: n,
            ncols: n,
            row_ptrs: (0..=n).collect(),
            col_inds: (0..n).collect(),
            vals: vec![1.0; n],
        }
    }

    /// Number of stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.col_inds.len()
    }

    /// Check the CSR contract: row pointers monotone and consistent,
    /// indices in range, columns sorted within each row, values parallel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SolveError::InvalidMatrix`] with the violated
    /// invariant when the structure is not a valid CSR matrix.
    pub fn validate(&self) -> Result<(), crate::SolveError> {
        use crate::SolveError;
        if self.row_ptrs.len() != self.nrows + 1 {
            return Err(SolveError::InvalidMatrix(
                "row_ptrs must have nrows+1 entries",
            ));
        }
        if self.row_ptrs.first() != Some(&0) {
            return Err(SolveError::InvalidMatrix("row_ptrs[0] must be 0"));
        }
        if self.col_inds.len() != self.vals.len() {
            return Err(SolveError::InvalidMatrix(
                "col_inds and vals must be parallel",
            ));
        }
        if self.row_ptrs[self.nrows] > self.col_inds.len() {
            return Err(SolveError::InvalidMatrix("row_ptrs[nrows] exceeds nnz"));
        }
        for row in 0..self.nrows {
            let (lo, hi) = (self.row_ptrs[row], self.row_ptrs[row + 1]);
            if lo > hi {
                return Err(SolveError::InvalidMatrix("row_ptrs must be monotone"));
            }
            for k in lo..hi {
                if self.col_inds[k] >= self.ncols {
                    return Err(SolveError::InvalidMatrix("column index out of range"));
                }
                if k > lo && self.col_inds[k - 1] >= self.col_inds[k] {
                    return Err(SolveError::InvalidMatrix("columns must be sorted per row"));
                }
            }
        }
        Ok(())
    }

    /// Dense mat-vec product `A x` (diagnostics and tests; not a hot path).
    ///
    /// # Panics
    ///
    /// Panics if `x` is shorter than `ncols`.
    #[must_use]
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert!(x.len() >= self.ncols, "x too short for matvec");
        (0..self.nrows)
            .map(|r| {
                (self.row_ptrs[r]..self.row_ptrs[r + 1])
                    .map(|k| self.vals[k] * x[self.col_inds[k]])
                    .sum()
            })
            .collect()
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
    fn matvec_matches_dense_reference() {
        let a = demo_matrix();
        let x = [1.0, 0.5, -1.0];
        let y = a.matvec(&x);
        assert_eq!(y, vec![4.5, 1.5, -5.0]);
    }

    #[test]
    fn validate_rejects_unsorted_rows() {
        let bad = CsrMatrix {
            nrows: 2,
            ncols: 2,
            row_ptrs: vec![0, 2, 4],
            col_inds: vec![1, 0, 0, 1],
            vals: vec![1.0; 4],
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_rejects_dangling_indices() {
        let bad = CsrMatrix {
            nrows: 1,
            ncols: 2,
            row_ptrs: vec![0, 1],
            col_inds: vec![7],
            vals: vec![1.0],
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn validate_rejects_short_row_ptrs() {
        let bad = CsrMatrix {
            nrows: 2,
            ncols: 2,
            row_ptrs: vec![0],
            col_inds: vec![],
            vals: vec![],
        };
        assert!(bad.validate().is_err());
    }
}
