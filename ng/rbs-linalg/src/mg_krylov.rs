//! Geometric multigrid + Krylov backend — the **production path**
//! placeholder (ADR 0001 decision 2).
//!
//! Design sketch for the implementing stage:
//!
//! * grid ladder: uniform (r, z) -> stretched radial -> compactified /
//!   multiblock (`ng/docs/architecture.md`), so the hierarchy is geometric,
//!   not algebraic;
//! * per-field block preconditioner over the 6-field Jacobian structure as
//!   the intermediate step before full MG;
//! * FGMRES-class outer iteration with the MG cycle as flexible
//!   preconditioner, matching the C campaign's stale-factor robustness
//!   near folds.
//!
//! **Never rely on global sparse LU scaling** — see ADR 0001.

use crate::{csr::CsrMatrix, LinearSolver, SolveError};

/// Placeholder geometric-multigrid + Krylov backend.
///
/// Implements the [`LinearSolver`] contract so the Newton loop can already
/// be written against the production interface; every operation reports
/// [`SolveError::NotImplemented`] until the backend lands.
#[derive(Debug, Default)]
pub struct MgKrylov {
    _reserved: (),
}

impl MgKrylov {
    /// An empty (not-yet-implemented) backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl LinearSolver for MgKrylov {
    fn factor(&mut self, _a: &CsrMatrix) -> Result<(), SolveError> {
        Err(SolveError::NotImplemented(
            "SAN-6x: geometric multigrid + FGMRES backend",
        ))
    }

    fn solve_into(&mut self, _b: &[f64], _x: &mut [f64]) -> Result<(), SolveError> {
        Err(SolveError::NotImplemented(
            "SAN-6x: geometric multigrid + FGMRES backend",
        ))
    }

    fn name(&self) -> &'static str {
        "mg-krylov (placeholder)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_is_honest() {
        let mut s = MgKrylov::new();
        assert_eq!(s.name(), "mg-krylov (placeholder)");
        assert_eq!(
            s.factor(&CsrMatrix::identity(2)),
            Err(SolveError::NotImplemented(
                "SAN-6x: geometric multigrid + FGMRES backend",
            ))
        );
        assert_eq!(
            s.solve_into(&[1.0, 2.0], &mut [0.0; 2]),
            Err(SolveError::NotImplemented(
                "SAN-6x: geometric multigrid + FGMRES backend",
            ))
        );
    }
}
