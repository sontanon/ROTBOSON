//! SAN-52 fault-injection gates: the damped machinery's behavior on
//! a synthetic badly-scaled Jacobian and near-singular perturbations
//! of the real seed system — where it RECOVERS (Deuflhard's affine
//! invariance) and where it fails GRACEFULLY with the C's typed
//! failure statuses, never a silent wrong answer.
//!
//! The fault is injected at the linear-algebra layer (a
//! [`LinearSolver`] decorator scaling Jacobian rows — with or
//! without the matching right-hand-side scaling), so the real
//! assembly, the real drivers, and the real backend all participate.
//!
//! Ignored by default (sparse LU sizes); run with:
//!
//! ```text
//! CARGO_HOME=<cache> cargo test --release -p rbs-elliptic -- --ignored
//! ```

use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::ProblemParams;
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonStatus};
use rbs_linalg::faer_lu::FaerSparseLu;
use rbs_linalg::{CsrMatrix, LinearSolver, SolveError};

/// Number of grid functions (mirrors the C `GNUM`).
const GNUM: usize = 6;

/// The fault decorator: scales a row block of the Jacobian by
/// `jac_scale` in [`LinearSolver::factor`] and the same rows of every
/// right-hand side by `rhs_scale` in [`LinearSolver::solve_into`].
///
/// * `jac_scale == rhs_scale`: an EQUATION row rescaling — the Newton
///   step is exactly invariant, so the whole NLEQ-ERR path must not
///   notice (the badly-scaled-Jacobian recovery case);
/// * `jac_scale != rhs_scale`: an INCONSISTENT scaling — the solve
///   produces a step amplified/attenuated in the faulted block, the
///   near-singular perturbation case.
struct RowScaledSolver<S: LinearSolver> {
    inner: S,
    row_lo: usize,
    row_hi: usize,
    jac_scale: f64,
    rhs_scale: f64,
}

impl<S: LinearSolver> RowScaledSolver<S> {
    /// Fault the Jacobian rows `lo..hi` consistently (equation
    /// rescaling).
    fn consistent(inner: S, row_lo: usize, row_hi: usize, scale: f64) -> Self {
        Self {
            inner,
            row_lo,
            row_hi,
            jac_scale: scale,
            rhs_scale: scale,
        }
    }

    /// Fault the Jacobian rows `lo..hi` WITHOUT the matching RHS
    /// scaling (a near-singular perturbation of the system).
    fn inconsistent(inner: S, row_lo: usize, row_hi: usize, scale: f64) -> Self {
        Self {
            inner,
            row_lo,
            row_hi,
            jac_scale: scale,
            rhs_scale: 1.0,
        }
    }

    fn scale_rows(a: &mut CsrMatrix, row_lo: usize, row_hi: usize, scale: f64) {
        for r in row_lo..row_hi {
            for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
                a.vals[k] *= scale;
            }
        }
    }
}

impl<S: LinearSolver> LinearSolver for RowScaledSolver<S> {
    fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError> {
        let mut faulted = a.clone();
        Self::scale_rows(&mut faulted, self.row_lo, self.row_hi, self.jac_scale);
        self.inner.factor(&faulted)
    }

    // rhs_scale is either exactly 1.0 (no fault) or a set fault factor.
    #[allow(clippy::float_cmp)]
    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        if self.rhs_scale == 1.0 {
            return self.inner.solve_into(b, x);
        }
        let mut faulted = b.to_vec();
        for v in &mut faulted[self.row_lo..self.row_hi] {
            *v *= self.rhs_scale;
        }
        self.inner.solve_into(&faulted, x)
    }

    fn name(&self) -> &'static str {
        "row-scaled-fault"
    }
}

/// The N=16 seed spec (dr = 1.0, domain ~16 — the smoke family on a
/// fast grid; the fault gates need the machinery exercised, not
/// oracle parity).
fn setup() -> (Grid2d, SolveState, ProblemParams) {
    let grid = Grid2d::new(20, 20, 2, 1.0, 1.0);
    let state = initial_guess(&grid, &InitialGuess::l1_smoke());
    let params = ProblemParams {
        l: 1,
        m: 1.0,
        xi: rbs_elliptic::omega::inverse_omega_calc(0.95, 1.0),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: None,
    };
    (grid, state, params)
}

/// The badly-scaled Jacobian, injected consistently: the whole psi
/// equation block rescaled by 1e-6. NLEQ-ERR's damping is
/// affine-invariant (Deuflhard), so the solve must RECOVER with the
/// same iteration path and the same solution.
#[test]
#[ignore = "N=16 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn badly_scaled_jacobian_recovers_affine_invariant() {
    let (grid, _, params) = setup();
    let dim = grid.dim();

    let mut clean_state = initial_guess(&grid, &InitialGuess::l1_smoke());
    let (clean_status, clean_log) = solve(
        &mut FaerSparseLu::new(),
        &grid,
        &mut clean_state,
        &params,
        &NewtonConfig::default(),
    )
    .unwrap();
    assert!(
        matches!(clean_status, NewtonStatus::Converged { .. }),
        "clean baseline did not converge: {clean_status:?}"
    );
    let mut faulted_state = initial_guess(&grid, &InitialGuess::l1_smoke());
    let mut solver = RowScaledSolver::consistent(FaerSparseLu::new(), 4 * dim, 5 * dim, 1e-6);
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut faulted_state,
        &params,
        &NewtonConfig::default(),
    )
    .unwrap();
    assert!(
        matches!(status, NewtonStatus::Converged { .. }),
        "faulted solve failed: {status:?}"
    );

    // The path is invariant: same row count, same events, the same
    // damping schedule — the numeric columns to the faer-vs-PARDISO
    // rounding the 1e-6 row scaling injects into the LU (observed
    // ~2e-5 relative drift on lambda').
    assert_eq!(log.len(), clean_log.len(), "row count changed under scaling");
    for (a, b) in log.iter().zip(&clean_log) {
        assert_eq!(a.event, b.event, "event mismatch");
        assert!(
            (a.lambda - b.lambda).abs() <= 1e-3 * b.lambda.abs().max(1e-300),
            "lambda {} vs {}",
            a.lambda,
            b.lambda
        );
        assert!(
            (a.theta - b.theta).abs() <= 1e-3 * b.theta.abs().max(1e-300),
            "theta {} vs {}",
            a.theta,
            b.theta
        );
    }
    // And the same solution (the faulted block's LU rounding shows at
    // the ~1e-10 absolute level — observed 5.8e-10 on psi itself).
    for k in 0..GNUM {
        let a = faulted_state.fields.field(k);
        let b = clean_state.fields.field(k);
        let max_diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "field {k} differs by {max_diff:.3e} (clean scale {})",
            b.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        );
    }
}

/// The near-singular perturbation: psi's Jacobian rows shrunk by
/// 1e-9 with the right-hand sides left alone. The linear solve
/// produces steps amplified ~1e9 in the psi direction, the trial
/// monitors go non-finite, and the port's non-finite guard (the C
/// would ACCEPT such a trial — its reject tests are NaN-blind) walks
/// the damping down through seven REJECT A halvings to the
/// regularity floor: the C's -1, with the state left at the clean
/// last-accepted iterate (the initial ansatz).
#[test]
#[ignore = "N=16 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn near_singular_perturbation_fails_gracefully() {
    let (grid, mut state, params) = setup();
    let dim = grid.dim();
    let mut solver = RowScaledSolver::inconsistent(FaerSparseLu::new(), 4 * dim, 5 * dim, 1e-9);
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut state,
        &params,
        &NewtonConfig::default(),
    )
    .unwrap();

    assert_eq!(
        status,
        NewtonStatus::Diverged,
        "expected the regularity-floor failure, got {status:?} ({} rows)",
        log.len()
    );
    // The damping collapsed through the REJECT A rule, and the
    // returned iterate is the last accepted one — finite and exactly
    // the initial ansatz (no trial was ever accepted).
    assert!(
        log.iter()
            .any(|r| r.event == rbs_elliptic::newton::NewtonEvent::RejectA),
        "no REJECT A in the path"
    );
    assert_eq!(
        log.last().expect("rows").event,
        rbs_elliptic::newton::NewtonEvent::RegularityFailure
    );
    assert!(state.fields.as_slice().iter().all(|v| v.is_finite()));
    let fresh = initial_guess(&grid, &InitialGuess::l1_smoke());
    for k in 0..GNUM {
        let max_diff = state
            .fields
            .field(k)
            .iter()
            .zip(fresh.fields.field(k))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert_eq!(max_diff, 0.0, "field {k} moved off the ansatz");
    }
}
