//! The global Newton loop with the frequency eigenvalue — the SAN-62/63
//! port target.
//!
//! C cross-references: `src/newton.c` (the driver), `src/nleq_err.c` /
//! `src/nleq_res.c` (residual-norm bookkeeping), `src/qnerr.c` / `src/qnres.c`
//! (the quasi-Newton error relaxation), and the omega constraint row
//! (`src/csr_omega_constraint.c`) that closes the system: six PDEs plus the
//! normalization condition make the global Newton system square in
//! `(u1..u6, xi)`.

/// Outcome of one Newton solve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NewtonStatus {
    /// Converged: the scaled residual norm dropped below the tolerance.
    Converged {
        /// Newton iterations spent.
        iterations: usize,
        /// Final scaled residual norm.
        residual_norm: f64,
    },
    /// The iteration diverged (norm blow-up or step rejection collapse).
    Diverged,
    /// The iteration budget ran out.
    MaxIterations,
}

/// Solve the elliptic system for `(u1..u6, xi)` with `solver` as the
/// linear backend.
///
/// # TODO (SAN-62/63)
///
/// Port the C loop: assemble `F(u, xi)` and `J(u, xi)`, solve
/// `J dx = -F`, relax with the QNERR rule near folds, and stop on the
/// scaled residual norm. The linear backend is whatever implements
/// [`rbs_linalg::LinearSolver`] — faer for the parity gates, MG-Krylov for
/// production (ADR 0001).
#[allow(clippy::missing_errors_doc)]
pub fn solve<S: rbs_linalg::LinearSolver>(
    _solver: &mut S,
    _grid: &rbs_core::grid::Grid2d,
    _fields: &mut rbs_core::fields::FieldSet,
    _params: &crate::assembly::ProblemParams,
) -> NewtonStatus {
    unimplemented!("SAN-62/63: global Newton port (newton.c/qnerr.c)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_is_documented_shape() {
        let s = NewtonStatus::Converged {
            iterations: 4,
            residual_norm: 1e-14,
        };
        assert_eq!(
            s,
            NewtonStatus::Converged {
                iterations: 4,
                residual_norm: 1e-14
            }
        );
        assert_ne!(s, NewtonStatus::Diverged);
    }
}
