//! Geometric multigrid + Krylov backend: a **parked** placeholder.
//!
//! ADR 0002 withdrew multigrid as the production path; `BlockFieldKrylov`
//! is the backend. Multigrid comes back only if grids beyond N=512 are
//! needed, as a per-block replacement for the block LU factors.
//!
//! **Never rely on global sparse LU scaling.**

use crate::{csr::CsrMatrix, LinearSolver, SolveError};

/// Placeholder geometric-multigrid + Krylov backend.
///
/// Implements the [`LinearSolver`] contract; every operation reports
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
