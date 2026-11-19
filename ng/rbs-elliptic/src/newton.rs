//! The global Newton loop with the frequency eigenvalue — the SAN-62
//! slice port.
//!
//! C cross-references: `src/newton.c` (the classic error-based driver
//! whose normalized convergence test this port mirrors:
//! `||du|| < epsilon * ||u||` with the `norm2_all_variables` norm),
//! and `src/nleq_err.c` for the damping semantics (the trial iterate
//! `u + lambda * du`, the monitoring Theta, the `lambda_min`
//! regularity floor). The full NLEQ-ERR/QNERR relaxation port is
//! SAN-52; this slice uses a backtracking line search on the same
//! iterates, which reaches the same fixed point for the weak-field
//! seed.
//!
//! Two deliberate divergences from the C, both documented:
//!
//! * the linear solve is `J du = f` with the C's sign convention (the
//!   Jacobian is the rescale-free derivative, `f` already carries
//!   `rescale = -1`) — mirroring `LINEAR_SOLVE_1(du[k], J, f[k])`;
//! * the trial-step acceptance is a residual-decrease test on the
//!   FULL six-field residual norm. The C's NLEQ-ERR accepts on
//!   `Theta = ||du_bar|| / ||du||` (a second linear solve, not a
//!   residual norm); the basic backtracking of this slice needs a
//!   merit function that sees every field, because the C's monitoring
//!   norm [`norm2_all_variables`] is log-alpha-blind and would accept
//!   steps that trash the lapse.

use crate::assembly::{
    assemble_jacobian_values, assemble_residual, JacobianPattern, ProblemParams, Workspace,
};
use crate::initial::SolveState;
use rbs_core::grid::Grid2d;
use rbs_linalg::LinearSolver;

/// Number of grid functions (`GNUM`).
const GNUM: usize = crate::generated::GNUM;

/// Outcome of one Newton solve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NewtonStatus {
    /// Converged: the normalized update norm dropped below the
    /// tolerance (`CONVERGED A` in the C).
    Converged {
        /// Newton iterations spent.
        iterations: usize,
        /// Final normalized update norm `||du|| / ||u||`.
        residual_norm: f64,
    },
    /// The line search hit the regularity floor (`lambda_min`) without
    /// a residual-decreasing step (`ERROR_CODE_FAILED_REGULARITY_TEST`).
    Diverged,
    /// The iteration budget ran out (`ERROR_CODE_EXCEEDED_MAX_ITERATIONS`).
    MaxIterations,
}

/// Solver settings (the C config's `epsilon`, `maxNewtonIter`,
/// `lambdaMin`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonConfig {
    /// Exit tolerance on the normalized update norm.
    pub epsilon: f64,
    /// Maximum Newton iterations.
    pub max_iterations: usize,
    /// Regularity floor for the damping factor.
    pub lambda_min: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        // The smoke specs' solver block: epsilon = 1e-8,
        // maxNewtonIter = 20, lambdaMin = 1e-5.
        Self {
            epsilon: 1e-8,
            max_iterations: 20,
            lambda_min: 1e-5,
        }
    }
}

/// One accepted iteration's monitoring record (the C's table row).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonLog {
    /// Iteration index `k`.
    pub iter: usize,
    /// Normalized update norm `||du[k]|| / ||u[k]||`.
    pub norm_du: f64,
    /// Accepted damping factor.
    pub lambda: f64,
    /// Residual-norm ratio `||F(u + lambda du)|| / ||F(u)||`.
    pub theta: f64,
}

/// The C `norm2_all_variables` (`src/vector_algebra.c`): the RMS over
/// fields 1..=5 (beta, log h, log a, psi, lambda — the C loop is
/// `k = GRID_VARIABLE_START..GRID_VARIABLE_END` with 1-based fields,
/// so log alpha and the omega slot are excluded), averaged over the
/// five fields.
#[must_use]
pub fn norm2_all_variables(grid: &Grid2d, flat: &[f64]) -> f64 {
    let dim = grid.dim();
    let mut sum = 0.0;
    for k in 1..GNUM {
        let field = &flat[k * dim..(k + 1) * dim];
        sum += field.iter().map(|v| v * v).sum::<f64>() / dim as f64;
    }
    (sum / 5.0).sqrt()
}

/// The full-state residual RMS over all six fields (the line-search
/// merit function; see the module docs for why it differs from the
/// C's monitoring norm).
#[must_use]
fn residual_rms_all(grid: &Grid2d, f: &[f64]) -> f64 {
    let dim = grid.dim();
    let mut sum = 0.0;
    for v in &f[..GNUM * dim] {
        sum += v * v;
    }
    (sum / (GNUM * dim) as f64).sqrt()
}

/// Solve the elliptic system with `solver` as the linear backend,
/// mutating `state` in place and returning the status plus the
/// iteration log.
///
/// # Errors
///
/// Propagates the backend's [`rbs_linalg::SolveError`] (singular
/// Jacobian, factorization failure).
///
/// # Panics
///
/// Panics on dimension mismatches inside the assembly (a bug, not an
/// input condition).
#[allow(clippy::missing_panics_doc)]
pub fn solve<S: LinearSolver>(
    solver: &mut S,
    grid: &Grid2d,
    state: &mut SolveState,
    params: &ProblemParams,
    config: &NewtonConfig,
) -> Result<(NewtonStatus, Vec<NewtonLog>), rbs_linalg::SolveError> {
    let dim = grid.dim();
    let n = GNUM * dim + 1;
    let mut ws = Workspace::new(grid);
    let pattern = JacobianPattern::build(grid, params);
    let mut vals = vec![0.0; pattern.col_inds.len()];
    let mut f = assemble_residual(grid, &state.fields, params, &mut ws);
    let mut log = Vec::new();

    for k in 0..config.max_iterations {
        // 1. J(u^k) du^k = f(u^k): the C's `LINEAR_SOLVE_1(du[k], J, f[k])`.
        // The assembled Jacobian is the derivative of the RESCALE-FREE
        // residual while `f` carries `rescale = -1`, so the Newton
        // direction comes out of the already-negated right-hand side —
        // negating again would flip the step.
        assemble_jacobian_values(&pattern, grid, &state.fields, params, &mut ws, &mut vals);
        let csr = rbs_linalg::CsrMatrix {
            nrows: pattern.nrows,
            ncols: pattern.ncols,
            row_ptrs: pattern.row_ptrs.clone(),
            col_inds: pattern.col_inds.clone(),
            vals: vals.clone(),
        };
        solver.factor(&csr)?;
        let mut du = vec![0.0; n];
        solver.solve_into(&f, &mut du)?;

        // Convergence test (the C's NORMALIZED branch): stop once
        // ||du|| < epsilon * ||u||, applying the full step.
        let u_flat = state.to_flat();
        let norm_du = norm2_all_variables(grid, &du);
        let norm_u = norm2_all_variables(grid, &u_flat);
        if norm_du < config.epsilon * norm_u {
            state.axpy(1.0, &du);
            return Ok((
                NewtonStatus::Converged {
                    iterations: k + 1,
                    residual_norm: norm_du / norm_u,
                },
                log,
            ));
        }

        // 2. Damped trial iterate with backtracking on the full
        //    residual norm.  The acceptance rule is the C's
        //    sufficient-decrease test (`nleq_err.c`:
        //    `Theta <= 1 - lambda / 4`, with THETA_MAX = 1) applied to
        //    the residual ratio instead of the du_bar ratio (SAN-52
        //    ports the full machinery).
        let f_norm = residual_rms_all(grid, &f);
        let mut lambda = 1.0;
        let mut accepted = false;
        let mut f_trial = Vec::new();
        let mut trial = state.clone();
        while lambda >= config.lambda_min {
            trial.clone_from(state);
            trial.axpy(lambda, &du);
            f_trial = assemble_residual(grid, &trial.fields, params, &mut ws);
            let theta = residual_rms_all(grid, &f_trial) / f_norm;
            if theta <= 1.0 - 0.25 * lambda {
                accepted = true;
                break;
            }
            lambda *= 0.5;
        }
        if !accepted {
            return Ok((NewtonStatus::Diverged, log));
        }
        let theta = residual_rms_all(grid, &f_trial) / f_norm;
        log.push(NewtonLog {
            iter: k,
            norm_du: norm_du / norm_u,
            lambda,
            theta,
        });

        // Accept the trial iterate (the C keeps `u[k+1]` and `f[k+1]`).
        state.clone_from(&trial);
        f.clone_from(&f_trial);
    }

    Ok((NewtonStatus::MaxIterations, log))
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
        assert_ne!(NewtonStatus::MaxIterations, NewtonStatus::Diverged);
    }

    #[test]
    fn config_defaults_match_the_smoke_spec() {
        let c = NewtonConfig::default();
        assert_eq!(c.epsilon, 1e-8);
        assert_eq!(c.max_iterations, 20);
        assert_eq!(c.lambda_min, 1e-5);
    }

    /// The norm excludes log alpha and the omega slot, like the C's
    /// `norm2_all_variables`.
    #[test]
    fn norm_excludes_log_alpha_and_omega() {
        let grid = Grid2d::new(6, 6, 2, 0.25, 0.25);
        let dim = grid.dim();
        let mut flat = vec![0.0; GNUM * dim + 1];
        // Field 0 (log alpha) would contribute 1 per point if included.
        for v in &mut flat[..dim] {
            *v = 1.0;
        }
        // Field 4 (psi): 2 per point.
        for v in &mut flat[4 * dim..5 * dim] {
            *v = 2.0;
        }
        // Expected: sqrt(mean over fields 1..=5 of RMS^2) with only
        // psi nonzero: sqrt((1/5) * 4) .
        let n = norm2_all_variables(&grid, &flat);
        assert!((n - (4.0_f64 / 5.0).sqrt()).abs() < 1e-14);
    }
}
