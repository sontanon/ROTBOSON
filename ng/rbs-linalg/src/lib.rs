//! rbs-linalg: sparse linear solver backends behind one trait.
//!
//! The Newton loop (rbs-elliptic) codes against [`LinearSolver`]; this crate
//! provides the checked-in CSR matrix type and the backends: faer sparse LU
//! (transitional, parity sizes only), the field-blocked Newton–Krylov
//! backend ([`BlockFieldKrylov`], the SAN-67 N1 candidate), and the
//! MG-Krylov placeholder for the far production path (ADR 0001).

pub mod block_field;
pub mod csr;
pub mod faer_lu;
pub mod mg_krylov;

pub use block_field::{BlockFieldKrylov, BlockKrylovConfig, KrylovStats, Preconditioner};

pub use csr::CsrMatrix;

use core::fmt;

/// Error reported by linear solver backends.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    /// `solve_into` was called before a successful [`LinearSolver::factor`].
    NotFactorized,
    /// The factorization found the matrix singular to working precision.
    Singular,
    /// The matrix structure violates the CSR contract (dangling index,
    /// unsorted row, or a row-pointer/count mismatch).
    InvalidMatrix(&'static str),
    /// A right-hand side or output slice has the wrong length.
    DimensionMismatch {
        /// Length the backend expected (`nrows` of the factored matrix).
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// The backend is a placeholder and not implemented yet.
    NotImplemented(&'static str),
    /// The Krylov loop missed its tolerance within the iteration
    /// budget (strict mode; see
    /// [`BlockFieldKrylov::last_stats`]).
    NotConverged {
        /// Krylov iterations spent.
        iterations: usize,
        /// The true relative residual `‖b − Ax‖ / ‖b‖` reached.
        residual: f64,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFactorized => write!(f, "solve before factor"),
            Self::Singular => write!(f, "matrix singular to working precision"),
            Self::InvalidMatrix(why) => write!(f, "invalid CSR matrix: {why}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NotImplemented(what) => write!(f, "not implemented: {what}"),
            Self::NotConverged {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Krylov loop not converged: {residual:.3e} after {iterations} iterations"
                )
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// Sparse linear backend contract for the Newton loop.
///
/// Backends are stateful: [`LinearSolver::factor`] analyzes and factors a
/// matrix (or builds a preconditioner), and [`LinearSolver::solve_into`]
/// may then be called repeatedly for different right-hand sides against the
/// same factorization — the stale-factor reuse the C campaign leans on near
/// folds (`iparm[3]=61` CGS in `docs/profiling.md`).
pub trait LinearSolver {
    /// Analyze and factor `a` (or set up the preconditioner).
    ///
    /// Replaces any previous factorization; the matrix is consumed by
    /// value (borrowed) so the caller may update entries in place and
    /// refactor, mirroring the C `csr_gen_jacobian` + PARDISO phase flow.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidMatrix`] on a violated CSR contract,
    /// [`SolveError::Singular`] when the factorization finds the matrix
    /// (structurally) singular.
    fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError>;

    /// Solve `A x = b` against the current factorization, writing `x`.
    ///
    /// `b` and `x` must each have length `nrows` of the factored matrix;
    /// otherwise [`SolveError::DimensionMismatch`] is returned.
    ///
    /// # Errors
    ///
    /// [`SolveError::NotFactorized`] before the first successful
    /// [`LinearSolver::factor`]; [`SolveError::DimensionMismatch`] on a
    /// wrong slice length; backend-specific failures otherwise.
    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError>;

    /// Backend name, for campaign logs and diagnostics.
    fn name(&self) -> &'static str;

    /// Convenience wrapper: allocate the solution and solve `A x = b`.
    ///
    /// # Errors
    ///
    /// Whatever [`LinearSolver::solve_into`] reports.
    fn solve(&mut self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut x = vec![0.0; b.len()];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial backend exercising the trait contract end to end.
    struct Identity;

    impl LinearSolver for Identity {
        fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError> {
            (a.nrows == a.ncols)
                .then_some(())
                .ok_or(SolveError::InvalidMatrix("square required"))
        }

        fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            if b.len() != x.len() {
                return Err(SolveError::DimensionMismatch {
                    expected: b.len(),
                    got: x.len(),
                });
            }
            x.copy_from_slice(b);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "identity"
        }
    }

    #[test]
    fn trait_contract_smoke() {
        let a = CsrMatrix::identity(3);
        let mut s = Identity;
        assert_eq!(s.name(), "identity");
        s.factor(&a).expect("factor");
        let x = s.solve(&[1.0, -2.0, 3.0]).expect("solve");
        assert_eq!(x, vec![1.0, -2.0, 3.0]);
        let mut out = [0.0; 2];
        assert_eq!(
            s.solve_into(&[1.0, 2.0, 3.0], &mut out),
            Err(SolveError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn error_display_is_stable() {
        assert_eq!(
            SolveError::DimensionMismatch {
                expected: 4,
                got: 3
            }
            .to_string(),
            "dimension mismatch: expected 4, got 3"
        );
        assert_eq!(SolveError::NotFactorized.to_string(), "solve before factor");
    }
}
