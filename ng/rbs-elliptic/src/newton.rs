//! The global Newton machinery — the SAN-52 port of the C solver's
//! full damping and relaxation semantics.
//!
//! Machinery map (C source → item in this module):
//!
//! | C source | Rust |
//! |---|---|
//! | `src/nleq_err.c` | [`nleq_err`] — the error-based adaptive-damping driver (`solverType = 1`) |
//! | `src/qnerr.c` | `nleq_err_qnerr` — the quasi-Newton error-reduction tail (`localSolver = 1`) |
//! | `src/nleq_res.c` | [`nleq_res`] — the residual-based damping driver (`solverType = 2`) |
//! | `src/qnres.c` | `nleq_res_qnres` — the residual-reduction tail |
//! | `src/newton.c` | [`classic_newton`] — the fixed-damping loop (`solverType = 3`) |
//! | `src/vector_algebra.c` | [`norm2_all_variables`], [`dot_all_variables`], [`norm2_interior_all_variables`] |
//! | `src/main.c` `run_newton` | the [`NewtonKind`] dispatch in [`solve`] (the C's `solverType` switch) |
//!
//! # The damping semantics (`nleq_err.c`)
//!
//! NLEQ-ERR is Deuflhard's affine-invariant damped Newton. Each outer
//! step solves `J(u^k) du^k = -f(u^k)`, then tries the damped iterate
//! `u^{k+1} = u^k + lambda_k du^k`. The trial is MONITORED by the
//! second (cheap, stale-factor) solve `J(u^k) du_bar^{k+1} = -f(u^{k+1})`
//! through `Theta_k = ||du_bar^{k+1}|| / ||du^k||`:
//!
//! * `Theta_k > 1 - lambda_k / 4` (REJECT A): the trial is rejected,
//!   `lambda_k <- min(lambda_k / 2, mu'_k)` and retried against the
//!   regularity floor `lambda_min`;
//! * otherwise `lambda'_k = min(1, mu'_k)` with the predicted
//!   affine-invariant step bound `mu'_k = (1/2) ||du^k|| lambda_k^2 /
//!   ||du_bar^{k+1} - (1 - lambda_k) du^k||`; if `lambda'_k > 4
//!   lambda_k` (REJECT B) the damping is GROWN geometrically and the
//!   trial is retried; if `lambda'_k = lambda_k = 1` the iterate is in
//!   the local (undamped) region and, once `Theta_k < 1/2`, QNERR
//!   takes over with full steps against the frozen factorization.
//!
//! For `k > prediction_start` the damping is predicted from the
//! previous step's monitors (`nleq_err.c` step 1: `mu_k = ||du^{k-1}||
//! ||du_bar^k|| lambda_{k-1} / (||du_bar^k - du^k|| ||du^k||)`,
//! `lambda_k = min(1, mu_k)`; the C's differentiated 2x/1.25x growth
//! by `Theta_{k-1}` is commented out in the source — both branches
//! reduce to the same `min`).
//!
//! # The QNERR tail (qnerr.c)
//!
//! Once NLEQ-ERR reaches the local region with `Theta < 1/2`, QNERR
//! runs full Newton-like steps against the single factorization of
//! `J(u^{k+1})` (the stale factor — `LINEAR_SOLVE_2`), correcting the
//! frozen-Jacobian update `du_bar^{l+1}` by its projections onto the
//! previous updates (`alpha_bar` over `du^{i-1}`, `i = 1..=l`) and
//! accelerating through `du^{l+1} = du_bar^{l+1} / (1 - alpha_{l+1})`.
//! If the contraction monitor `Theta_l = ||du_bar^{l+1}||/||du^l||`
//! exceeds `1/2`, QNERR exits and NLEQ-ERR RESTARTS from the last
//! QNERR update with `lambda = 1` and the prediction disabled for one
//! round — the restart path the C oracle exercises on the seed spec at
//! `epsilon = 1e-12`.
//!
//! # The residual pair (`nleq_res.c` / qnres.c)
//!
//! NLEQ-RES monitors the trial by the residual ratio `Theta_k =
//! ||f(u^{k+1})|| / ||f(u^k)||` instead of a second solve, rejects on
//! `Theta_k >= 1` with a quarter shrink (`lambda' = min(lambda/4,
//! mu')`), and enters QNRES in the local region (`lambda = 1` and
//! `Theta < 1/2`). QNRES is a secant-updated residual-reduction loop
//! (the `gamma_l = <dF_{l+1}, dF_{l+1}>` bookkeeping and the `v`
//! recursion of `qnres.c`) with the preconditioner monitor
//! `kappa /= 1 - 2 Theta_l` (exit at `KAPPA_MAX = 1e6`), restarting
//! NLEQ-RES with `lambda = 1/4` when either sanity test trips.
//!
//! # Statuses and the C's exit codes
//!
//! Every driver exit maps onto [`NewtonStatus`] with the C's
//! `err_code` available through [`NewtonStatus::error_code`]: the
//! regularity floor (`-1`), the trial budgets (`-11`/`-13`), the
//! iteration budget (`-3`). Backend failures (singular Jacobian,
//! factorization failure) propagate through the [`SolveError`]
//! channel — never a silent wrong answer.
//!
//! # Documented C quirks kept faithfully
//!
//! * the monitoring norm [`norm2_all_variables`] averages fields
//!   1..=5 (beta, log h, log a, psi, lambda) — it is LOG-ALPHA-BLIND
//!   and omega-blind (`GRID_VARIABLE_START..GRID_VARIABLE_END` in
//!   `src/vector_algebra.c`). The convergence tests inherit this,
//!   exactly like the C;
//! * `nleq_res` receives its initial residual norm from the caller
//!   (`norm_f[0]` in the C signature). The C DRIVER computes it from
//!   the initial GUESS (`main.c:184`,
//!   `norm2_interior_all_variables(ctx, u[0])` — a different vector
//!   AND a different norm than the loop's own monitor). [`solve`]
//!   reproduces that value for iteration-path parity, and documents
//!   the hazard: on a near-flat initial guess the k=0 convergence test
//!   of the residual path sees `||u^0||`, not `||f(u^0)||` — inherited
//!   from the oracle, degenerate-input-only;
//! * NLEQ-ERR's local-region branch accepts implicitly when
//!   `lambda = lambda' = 1` but `Theta_k >= 1/2` (the `if/else-if`
//!   chain of `nleq_err.c` skips both reject and accept, so the loop
//!   continues without resetting the B counter);
//! * on failure the drivers return the last ACCEPTED iterate (the C
//!   returns an index into its arrays — `u[k+1]` on trial-budget
//!   exhaustion holds the REJECTED trial; a retry built on a rejected
//!   trial would be wrong, so this port returns `u[k]`, documented for
//!   the campaign retry semantics).

use crate::assembly::{
    assemble_jacobian_values, assemble_residual, JacobianPattern, ProblemParams, Workspace,
};
use crate::initial::SolveState;
use rbs_core::fields::FieldSet;
use rbs_core::grid::Grid2d;
use rbs_linalg::{CsrMatrix, LinearSolver, SolveError};

/// Number of grid functions (`GNUM`).
const GNUM: usize = crate::generated::GNUM;

// ===========================================================================
// The nonlinear-system contract (the C's function-pointer bundle).
// ===========================================================================

/// The nonlinear problem the Newton drivers code against — the C's
/// `rb_context` plus its function-pointer bundle (`RHS_CALC`,
/// `JACOBIAN_CALC`, `NORM`, `DOT`, `LINEAR_SOLVE_1`, `LINEAR_SOLVE_2`).
///
/// The two linear entry points mirror the C backend split: the
/// factorization half of `LINEAR_SOLVE_1` lives in
/// [`NewtonSystem::factor_jacobian`] (assemble `J(u)` and factor it),
/// and the solve half of `LINEAR_SOLVE_1` plus all of `LINEAR_SOLVE_2`
/// (the stale-factor reuse the campaign leans on near folds) lives in
/// [`NewtonSystem::solve_into`].
pub trait NewtonSystem {
    /// The system dimension (`ctx->dim`, the length of every iterate).
    fn dim(&self) -> usize;

    /// `RHS_CALC(ctx, out, u)`: evaluate the residual at `u` into `out`.
    ///
    /// Infallible like the C's: the elliptic residual evaluation has no
    /// failure mode of its own.
    fn rhs_into(&mut self, u: &[f64], f: &mut [f64]);

    /// `JACOBIAN_CALC` + the factorization half of `LINEAR_SOLVE_1`:
    /// assemble `J(u)` and factor it. Subsequent
    /// [`NewtonSystem::solve_into`] calls reuse this factorization.
    ///
    /// # Errors
    ///
    /// Whatever the backend reports (singular Jacobian, factorization
    /// failure) — propagated unchanged.
    fn factor_jacobian(&mut self, u: &[f64]) -> Result<(), SolveError>;

    /// The solve half of `LINEAR_SOLVE_1` and all of `LINEAR_SOLVE_2`:
    /// solve `J x = b` against the current factorization.
    ///
    /// # Errors
    ///
    /// Backend-specific failures (dimension mismatch, lost
    /// factorization) — propagated unchanged.
    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError>;

    /// `NORM`: the monitoring norm (the C's `norm2_all_variables`).
    fn norm(&self, x: &[f64]) -> f64;

    /// `DOT`: the monitoring inner product (the C's
    /// `dot_all_variables`).
    fn dot(&self, x: &[f64], y: &[f64]) -> f64;
}

// ===========================================================================
// vector_algebra.c
// ===========================================================================

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

/// The C `dot_all_variables` (`src/vector_algebra.c`): the average
/// over fields 1..=5 of the plain dot product scaled by `1/dim`.
///
/// With [`norm2_all_variables`] the `1/(5 dim)` scaling cancels in
/// every ratio the drivers build on it (`alpha_bar`, `alpha`, the
/// QNRES `beta` recursion) — the machinery is scale-free; the C's
/// scaled quantities are reproduced for monitoring parity.
#[must_use]
pub fn dot_all_variables(grid: &Grid2d, x: &[f64], y: &[f64]) -> f64 {
    let dim = grid.dim();
    let mut sum = 0.0;
    for k in 1..GNUM {
        let a = &x[k * dim..(k + 1) * dim];
        let b = &y[k * dim..(k + 1) * dim];
        sum += a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>() / dim as f64;
    }
    sum / 5.0
}

/// The C `norm2_interior_all_variables` (`src/vector_algebra.c`): the
/// per-field interior-rectangle RMS (ghost rows and ghost columns
/// excluded), averaged over fields 1..=5.
///
/// The C driver feeds this — computed from the initial GUESS — to
/// NLEQ-RES as `norm_f[0]` (`main.c:184`); see the module docs.
#[must_use]
pub fn norm2_interior_all_variables(grid: &Grid2d, flat: &[f64]) -> f64 {
    let dim = grid.dim();
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    let (nr_int, nz_int) = (nr - 2 * g, nz - 2 * g);
    let mut sum = 0.0;
    for k in 1..GNUM {
        let field = &flat[k * dim..(k + 1) * dim];
        let mut s = 0.0;
        for i in g..g + nr_int {
            for j in g..g + nz_int {
                let v = field[grid.idx(i, j)];
                s += v * v;
            }
        }
        sum += s / (nr_int * nz_int) as f64;
    }
    (sum / 5.0).sqrt()
}

/// The C `ARRAY_SUM(out, alpha, x, beta, y)`: `out = alpha * x + beta * y`
/// across distinct arrays.
fn array_sum(out: &mut [f64], alpha: f64, x: &[f64], beta: f64, y: &[f64]) {
    for (o, (xi, yi)) in out.iter_mut().zip(x.iter().zip(y)) {
        *o = alpha * xi + beta * yi;
    }
}

/// The C `ARRAY_SUM(arr[dst], alpha, arr[src], beta, y)` with
/// `src < dst`: the same-history form (`u[k + 1] = u[k] + lambda du`).
fn array_sum_from(
    arr: &mut [Vec<f64>],
    dst: usize,
    src: usize,
    alpha: f64,
    beta: f64,
    y: &[f64],
) {
    let (head, tail) = arr.split_at_mut(dst);
    let out = &mut tail[0];
    let x = &head[src];
    for (o, (xi, yi)) in out.iter_mut().zip(x.iter().zip(y)) {
        *o = alpha * xi + beta * yi;
    }
}

/// Non-finite monitors refuse the trial they describe.
///
/// IEEE comparisons with NaN are false, so the C's reject tests
/// (`Theta > 1 - lambda/4`, `Theta > 1/2`, `kappa > KAPPA_MAX`) all
/// PASS a NaN monitor — the C would accept the trial and then feed
/// PARDISO a poisoned matrix, a genuine oracle robustness hole. This
/// port keeps every damping rule but treats a non-finite monitor as
/// the worst contraction, engaging the reject/exit machinery instead:
/// the failure then surfaces as a typed status with the last accepted
/// (finite) iterate, never a poisoned state.
fn monitors_finite(vals: &[f64]) -> bool {
    vals.iter().all(|v| v.is_finite())
}

/// The C `ARRAY_SUM(arr[i], 1.0, arr[i], beta, y)` — the in-place
/// accumulate (`u[k + 1] += du_bar[k + 1]`).
fn array_sum_accum(arr: &mut [Vec<f64>], i: usize, beta: f64, y: &[f64]) {
    for (o, yi) in arr[i].iter_mut().zip(y) {
        *o += beta * yi;
    }
}

// ===========================================================================
// Statuses, events, logs, configuration.
// ===========================================================================

/// Outcome of one Newton solve. The variants carry the C solver's
/// failure conditions; [`NewtonStatus::error_code`] exposes the C's
/// `err_code` for campaign classification (`failed:newton`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NewtonStatus {
    /// Converged: the driver's normalized exit test passed
    /// (`CONVERGED A`/`B` of `nleq_err.c`, `A` of `nleq_res.c`,
    /// `CONVERGED C` of the QNERR/QNRES tails, or the classic loop's
    /// test).
    Converged {
        /// Total iterations spent (the C's returned index / printed
        /// count, QNERR/QNRES tails included).
        iterations: usize,
        /// Final normalized update norm (`||du|| / ||u||`; for the
        /// residual path, the final residual monitoring norm).
        residual_norm: f64,
    },
    /// The damping factor hit the regularity floor
    /// (`ERROR_CODE_FAILED_REGULARITY_TEST`, `-1`): no acceptable
    /// step above `lambda_min` exists from the returned iterate.
    Diverged,
    /// The iteration budget ran out
    /// (`ERROR_CODE_EXCEEDED_MAX_ITERATIONS`, `-3`).
    MaxIterations,
    /// The REJECT A budget ran out
    /// (`ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_A`, `-11`): the
    /// trial kept failing `Theta > 1 - lambda/4` while halving the
    /// damping — the C's stuck-in-a-loop guard.
    ExceededTrialA,
    /// The REJECT B budget ran out
    /// (`ERROR_CODE_EXCEEDED_MAX_TRIAL_ITERATIONS_B`, `-13`): the
    /// damping kept quadrupling without an accepted trial.
    ExceededTrialB,
}

impl NewtonStatus {
    /// The C's `err_code` for this outcome (the value the C driver
    /// writes to the `error_code` dataset).
    #[must_use]
    pub fn error_code(self) -> i32 {
        match self {
            Self::Converged { .. } => 0,
            Self::Diverged => -1,
            Self::MaxIterations => -3,
            Self::ExceededTrialA => -11,
            Self::ExceededTrialB => -13,
        }
    }
}

/// One monitoring row (the C's table rows; the `STATUS` column is
/// [`NewtonEvent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NewtonEvent {
    /// `ACCEPT` — a trial iterate was accepted.
    Accept,
    /// `REJECT A` — `Theta` violated the contraction rule; the damping
    /// halves.
    RejectA,
    /// `REJECT B` — the predicted damping grew by more than 4x; the
    /// damping is replaced by the prediction and the trial retried.
    RejectB,
    /// `ENTER QNERR` — the local region reached, the QNERR tail takes
    /// over.
    EnterQnerr,
    /// A QNERR tail row (the C's QNERR table `ACCEPT`).
    QnerrAccept,
    /// `EXIT QNERR` — the tail's contraction monitor exceeded 1/2;
    /// NLEQ-ERR restarts.
    ExitQnerr,
    /// `CONVERGED A` — exit on `||du|| < epsilon ||u||`.
    ConvergedA,
    /// `CONVERGED B` — exit on `||du_bar|| < epsilon ||u||` in the
    /// local region (error path only).
    ConvergedB,
    /// `CONVERGED C` — exit inside a QNERR/QNRES tail.
    ConvergedC,
    /// `ENTER QNRES` — the residual path's local region.
    EnterQnres,
    /// A QNRES tail row.
    QnresAccept,
    /// `EXIT QNRES` — the tail's sanity test tripped; NLEQ-RES
    /// restarts.
    ExitQnres,
    /// `ERROR -1` — the regularity floor.
    RegularityFailure,
}

impl NewtonEvent {
    /// The C's `STATUS` string.
    #[must_use]
    pub fn status_str(self) -> &'static str {
        match self {
            Self::Accept | Self::QnerrAccept | Self::QnresAccept => "ACCEPT",
            Self::RejectA => "REJECT A",
            Self::RejectB => "REJECT B",
            Self::EnterQnerr => "ENTER QNERR",
            Self::ExitQnerr => "EXIT QNERR",
            Self::ConvergedA => "CONVERGED A",
            Self::ConvergedB => "CONVERGED B",
            Self::ConvergedC => "CONVERGED C",
            Self::EnterQnres => "ENTER QNRES",
            Self::ExitQnres => "EXIT QNRES",
            Self::RegularityFailure => "ERROR -1",
        }
    }
}

/// One accepted iteration's monitoring record (the C's table row).
///
/// The columns mirror the C's per-driver tables: `norm_du` is the
/// first numeric column (`||du[k]||/||u||` on error-path rows,
/// `||f[k]||` on residual rows, `||du[l+1]||/||u||` on QNERR rows),
/// and on QNERR rows `lambda`/`lambda_prime` carry `alpha_bar` /
/// `alpha[l+1]` — exactly the C's two-table layout folded into one
/// stream, distinguished by [`NewtonEvent`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonLog {
    /// Iteration index (`k` on NLEQ rows; the QNERR/QNRES tails log
    /// their global index `k + 1 + l`).
    pub iter: usize,
    /// The row's norm column (see the type docs).
    pub norm_du: f64,
    /// The damping factor (`lambda[k]`), or `alpha_bar` on QNERR rows.
    pub lambda: f64,
    /// The contraction monitor (`Theta`), or `alpha[l+1]` on QNERR
    /// rows.
    pub theta: f64,
    /// The predicted damping (`lambda'[k]`), unused on some rows.
    pub lambda_prime: f64,
    /// The row's `STATUS`.
    pub event: NewtonEvent,
}

/// The solver flavor — the C config's `solverType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonKind {
    /// `solverType = 1`: the error-based NLEQ-ERR driver (with the
    /// QNERR tail when [`NewtonConfig::local_solver`] is set).
    Error,
    /// `solverType = 2`: the residual-based NLEQ-RES driver (with the
    /// QNRES tail when [`NewtonConfig::local_solver`] is set).
    Residual,
    /// `solverType = 3`: the classic fixed-damping loop of
    /// `src/newton.c`.
    Classic,
}

/// Solver settings — the C config's solver block (`epsilon`,
/// `maxNewtonIter`, `lambda0`, `lambdaMin`, `localSolver`) plus the
/// trial budgets `main.c` hardcodes (`MAX_TRIAL_A/B_ITERATIONS = 8`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonConfig {
    /// Exit tolerance on the normalized update norm (`epsilon`).
    pub epsilon: f64,
    /// Maximum Newton iterations (`maxNewtonIter`), QNERR/QNRES tails
    /// included.
    pub max_iterations: usize,
    /// Regularity floor for the damping factor (`lambdaMin`).
    pub lambda_min: f64,
    /// Initial damping factor (`lambda0`).
    pub lambda0: f64,
    /// The driver flavor (`solverType`).
    pub kind: NewtonKind,
    /// Enable the QNERR/QNRES local tail (`localSolver`).
    pub local_solver: bool,
    /// The REJECT A budget (`MAX_TRIAL_A_ITERATIONS`).
    pub max_trial_a: usize,
    /// The REJECT B budget (`MAX_TRIAL_B_ITERATIONS`).
    pub max_trial_b: usize,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        // The smoke specs' solver block (`configs/l1_smoke_ci.toml`):
        // solverType = 1, localSolver = 1, epsilon = 1e-8,
        // maxNewtonIter = 20, lambda0 = 1e-3, lambdaMin = 1e-5, and
        // main.c's hardcoded trial budgets.
        Self {
            epsilon: 1e-8,
            max_iterations: 20,
            lambda_min: 1e-5,
            lambda0: 1e-3,
            kind: NewtonKind::Error,
            local_solver: true,
            max_trial_a: 8,
            max_trial_b: 8,
        }
    }
}

// nleq_err.c constants.
/// `LOCAL_LAMBDA`: the full-step damping factor.
const LOCAL_LAMBDA: f64 = 1.0;
/// The two stages of a damped Newton step: the C's `REGULARITY_TEST` /
/// `TRIAL_ITERATE` goto pair, written as a loop over this state.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    RegularityTest,
    TrialIterate,
}

/// How a trial iterate ended: accepted, or restart at the given iteration.
#[derive(Clone, Copy)]
enum TrialOutcome {
    Accept,
    Restart(usize),
}

/// `THETA_MAX` of `nleq_err.c`: the contraction ceiling of the reject
/// rule.
const THETA_MAX: f64 = 1.0;
/// `RESTRICTED` of `nleq_err.c`: the reject rule is
/// `Theta > THETA_MAX - RESTRICTED * lambda / 4`.
const RESTRICTED: f64 = 1.0;
/// `THETA_MAX` of `qnres.c`/`nleq_res.c`: the local-region contraction
/// ceiling.
const RESIDUAL_THETA_MAX: f64 = 0.5;
/// `KAPPA_MAX` of `qnres.c`: the preconditioner monitor ceiling.
const KAPPA_MAX: f64 = 1.0e6;
/// `MIN_ITERATIONS` of `qnerr.c` (0 — the tail's Theta exit is always
/// armed).
const QNERR_MIN_ITERATIONS: usize = 0;

// ===========================================================================
// nleq_err.c — the error-based adaptive-damping driver.
// ===========================================================================

/// The per-solve iteration histories (the `u`/`f`/`du`/`du_bar` vector
/// arrays and the `lambda`/`Theta`/`mu_prime` scalar arrays of
/// `main.c`'s allocation, index 0..=`max_iterations`).
struct History {
    u: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    du: Vec<Vec<f64>>,
    du_bar: Vec<Vec<f64>>,
    norm_du: Vec<f64>,
    norm_du_bar: Vec<f64>,
    norm_f: Vec<f64>,
    lambda: Vec<f64>,
    theta: Vec<f64>,
    mu_prime: Vec<f64>,
}

impl History {
    /// Allocate `len = max_iterations + 1` entries of `dim`-vectors.
    fn new(dim: usize, max_iterations: usize) -> Self {
        let len = max_iterations + 1;
        Self {
            u: vec![vec![0.0; dim]; len],
            f: vec![vec![0.0; dim]; len],
            du: vec![vec![0.0; dim]; len],
            du_bar: vec![vec![0.0; dim]; len],
            norm_du: vec![0.0; len],
            norm_du_bar: vec![0.0; len],
            norm_f: vec![0.0; len],
            lambda: vec![0.0; len],
            theta: vec![0.0; len],
            mu_prime: vec![0.0; len],
        }
    }
}

/// The QNERR tail's exit (qnerr.c's return value + `err_code` pair).
struct QnerrExit {
    /// Iterations consumed (the magnitude of the C's return).
    consumed: usize,
    /// Why the tail exited.
    reason: QnerrReason,
}

enum QnerrReason {
    /// `CONVERGED C`, carrying the final normalized update norm.
    Converged(f64),
    /// `ERROR_CODE_QNERR_THETA_INCREASE_EXIT` (`-2`): the caller
    /// restarts NLEQ-ERR from the last tail update.
    ThetaIncrease,
    /// `ERROR_CODE_EXCEEDED_MAX_ITERATIONS` (`-3`).
    MaxIterations,
}

/// The QNRES tail's exit (qnres.c's return value + `err_code` pair).
struct QnresExit {
    /// Iterations consumed.
    consumed: usize,
    /// Why the tail exited.
    reason: QnresReason,
}

enum QnresReason {
    /// `CONVERGED C`, carrying the final residual monitoring norm.
    Converged(f64),
    /// `ERROR_CODE_QNRES_THETA_INCREASE_EXIT` (`-2`): restart.
    ThetaIncrease,
    /// `ERROR_CODE_QNRES_ILL_CONDITIONED` (`-4`): the kappa monitor
    /// tripped; restart.
    IllConditioned,
    /// `ERROR_CODE_EXCEEDED_MAX_ITERATIONS` (`-3`).
    MaxIterations,
}

/// The error-based global Newton driver — the port of `nleq_err`
/// (`src/nleq_err.c`). See the module docs for the damping semantics.
///
/// `u` carries the initial guess in and the final iterate out (the
/// C's `u[0]` in, `u[|k|]` out): on failure the last ACCEPTED iterate
/// (see the module docs for the one deliberate divergence from the
/// C's array indexing).
///
/// The driver is one function on purpose: it maps the C's loop
/// statement-for-statement (the goto pair included), and splitting it
/// would blur the line-by-line correspondence the port is gated on.
///
/// # Errors
///
/// Propagates the backend's [`SolveError`] (singular Jacobian,
/// factorization failure) — never a silent wrong answer.
///
/// # Panics
///
/// Panics if `u` does not have [`NewtonSystem::dim`] entries (a bug in
/// the calling code, not an input condition).
#[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
pub fn nleq_err<S: NewtonSystem>(
    sys: &mut S,
    u: &mut [f64],
    config: &NewtonConfig,
) -> Result<(NewtonStatus, Vec<NewtonLog>), SolveError> {
    let dim = sys.dim();
    debug_assert_eq!(u.len(), dim, "initial guess length");
    let max = config.max_iterations;
    let mut h = History::new(dim, max);
    let mut aux = vec![0.0; dim];
    let mut log = Vec::new();

    h.u[0].copy_from_slice(u);
    sys.rhs_into(&h.u[0], &mut h.f[0]);
    h.lambda[0] = config.lambda0;
    // `prediction_start` (nleq_err.c): the prediction is disabled for
    // the restart round after a QNERR exit.
    let mut prediction_start = 0usize;
    // Trial budgets (main.c's counters).
    let (mut l_a, mut l_b) = (0usize, 0usize);

    let mut k = 0usize;
    while k < max {
        // 1. Step k: J(u^k) du^k = -f(u^k) — `JACOBIAN_CALC` +
        //    `LINEAR_SOLVE_1`.
        sys.factor_jacobian(&h.u[k])?;
        sys.solve_into(&h.f[k], &mut h.du[k])?;
        h.norm_du[k] = sys.norm(&h.du[k]);
        // NORMALIZED: the solution norm for the relative exit test.
        let mut norm_u = sys.norm(&h.u[k]);

        // Convergence test A: ||du^k|| < epsilon ||u^k||.
        if h.norm_du[k] < config.epsilon * norm_u {
            array_sum_from(&mut h.u, k + 1, k, 1.0, 1.0, &h.du[k]);
            let norm = h.norm_du[k] / norm_u;
            log.push(NewtonLog {
                iter: k,
                norm_du: norm,
                lambda: h.lambda[k],
                theta: h.theta[k],
                lambda_prime: h.lambda[k],
                event: NewtonEvent::ConvergedA,
            });
            u.copy_from_slice(&h.u[k + 1]);
            return Ok((
                NewtonStatus::Converged {
                    iterations: k + 1,
                    residual_norm: norm,
                },
                log,
            ));
        }

        // For k > prediction_start: predict the damping from the
        // previous step's monitors. Both C branches (Theta > 1/2 and
        // the rest) reduce to the same min — the differentiated
        // growth factors are commented out in nleq_err.c.
        if k > prediction_start {
            array_sum(&mut aux, 1.0, &h.du_bar[k], -1.0, &h.du[k]);
            let norm_du_bar_minus_du = sys.norm(&aux);
            let mu = (h.norm_du[k - 1] * h.norm_du_bar[k] * h.lambda[k - 1])
                / (norm_du_bar_minus_du * h.norm_du[k]);
            h.lambda[k] = LOCAL_LAMBDA.min(mu);
        }

        // The trial-iterate machinery: the C's REGULARITY_TEST /
        // TRIAL_ITERATE goto pair as a stage loop.
        let mut stage = Stage::RegularityTest;
        let outcome = loop {
            match stage {
                Stage::RegularityTest => {
                    // Regularity test: lambda_k < lambda_min stops with
                    // the -1 failure code.
                    if h.lambda[k] < config.lambda_min {
                        log.push(NewtonLog {
                            iter: k,
                            norm_du: h.norm_du[k] / norm_u,
                            lambda: h.lambda[k],
                            theta: h.theta[k],
                            lambda_prime: 0.0,
                            event: NewtonEvent::RegularityFailure,
                        });
                        u.copy_from_slice(&h.u[k]);
                        return Ok((NewtonStatus::Diverged, log));
                    }
                    stage = Stage::TrialIterate;
                }
                Stage::TrialIterate => {
                    // 2. Trial iterate u^{k+1} = u^k + lambda_k du^k,
                    //    then the stale-factor second solve
                    //    J(u^k) du_bar^{k+1} = -f(u^{k+1})
                    //    (`LINEAR_SOLVE_2`).
                    array_sum_from(&mut h.u, k + 1, k, 1.0, h.lambda[k], &h.du[k]);
                    sys.rhs_into(&h.u[k + 1], &mut h.f[k + 1]);
                    sys.solve_into(&h.f[k + 1], &mut h.du_bar[k + 1])?;
                    h.norm_du_bar[k + 1] = sys.norm(&h.du_bar[k + 1]);
                    norm_u = sys.norm(&h.u[k + 1]);

                    // 3. The monitoring quantities.
                    array_sum(&mut aux, 1.0, &h.du_bar[k + 1], h.lambda[k] - 1.0, &h.du[k]);
                    let norm_du_bar_minus_one_minus_lambda_du = sys.norm(&aux);
                    h.theta[k] = h.norm_du_bar[k + 1] / h.norm_du[k];
                    h.mu_prime[k] = 0.5 * h.norm_du[k] * h.lambda[k] * h.lambda[k]
                        / norm_du_bar_minus_one_minus_lambda_du;

                    // REJECT A: Theta_k > 1 - lambda_k / 4, or a
                    // non-finite monitor (the worst contraction; see
                    // `monitors_finite`).
                    if h.theta[k] > THETA_MAX - RESTRICTED * 0.25 * h.lambda[k]
                        || !monitors_finite(&[h.theta[k], h.mu_prime[k], h.norm_du_bar[k + 1]])
                    {
                        let lambda_prime = (0.5 * h.lambda[k]).min(h.mu_prime[k]);
                        log.push(NewtonLog {
                            iter: k,
                            norm_du: h.norm_du[k] / norm_u,
                            lambda: h.lambda[k],
                            theta: h.theta[k],
                            lambda_prime,
                            event: NewtonEvent::RejectA,
                        });
                        h.lambda[k] = lambda_prime;
                        l_a += 1;
                        if l_a >= config.max_trial_a {
                            // -11: exceeded the A-type trial budget.
                            u.copy_from_slice(&h.u[k]);
                            return Ok((NewtonStatus::ExceededTrialA, log));
                        }
                        stage = Stage::RegularityTest;
                    } else {
                        let lambda_prime = LOCAL_LAMBDA.min(h.mu_prime[k]);
                        // A proper lambda was found: reset the A budget.
                        l_a = 0;

                        // Local region: lambda' = lambda = 1. Exact float
                        // comparison on purpose: in the local region both
                        // are set to exactly 1.0, as in the C.
                        #[allow(clippy::float_cmp)]
                        let in_local_region =
                            lambda_prime == h.lambda[k] && h.lambda[k] == LOCAL_LAMBDA;
                        if in_local_region {
                            // Convergence test B on the trial's second
                            // solve.
                            if h.norm_du_bar[k + 1] < config.epsilon * norm_u {
                                array_sum_accum(&mut h.u, k + 1, 1.0, &h.du_bar[k + 1]);
                                let norm = h.norm_du_bar[k + 1] / norm_u;
                                log.push(NewtonLog {
                                    iter: k,
                                    norm_du: norm,
                                    lambda: h.lambda[k],
                                    theta: h.theta[k],
                                    lambda_prime,
                                    event: NewtonEvent::ConvergedB,
                                });
                                u.copy_from_slice(&h.u[k + 1]);
                                return Ok((
                                    NewtonStatus::Converged {
                                        iterations: k + 1,
                                        residual_norm: norm,
                                    },
                                    log,
                                ));
                            }
                            if h.theta[k] < 0.5 {
                                if config.local_solver {
                                    log.push(NewtonLog {
                                        iter: k,
                                        norm_du: h.norm_du_bar[k + 1] / norm_u,
                                        lambda: h.lambda[k],
                                        theta: h.theta[k],
                                        lambda_prime,
                                        event: NewtonEvent::EnterQnerr,
                                    });
                                    // Call QNERR with the iterate
                                    // u[k+1] / residual f[k+1]; the
                                    // histories are shared so the
                                    // restart state lands in place.
                                    let exit = nleq_err_qnerr(
                                        sys,
                                        &mut h,
                                        &mut log,
                                        k + 1,
                                        config.epsilon,
                                        max - (k + 1),
                                    )?;
                                    match exit.reason {
                                        QnerrReason::Converged(res) => {
                                            u.copy_from_slice(&h.u[k + 1 + exit.consumed]);
                                            return Ok((
                                                NewtonStatus::Converged {
                                                    iterations: k + 1 + exit.consumed,
                                                    residual_norm: res,
                                                },
                                                log,
                                            ));
                                        }
                                        QnerrReason::ThetaIncrease => {
                                            // Restart NLEQ-ERR from the
                                            // last tail update: full
                                            // damping, prediction off
                                            // for one round.
                                            h.lambda[k + exit.consumed + 1] = LOCAL_LAMBDA;
                                            break TrialOutcome::Restart(exit.consumed);
                                        }
                                        QnerrReason::MaxIterations => {
                                            u.copy_from_slice(&h.u[k + 1 + exit.consumed]);
                                            return Ok((NewtonStatus::MaxIterations, log));
                                        }
                                    }
                                }
                                // The no-QNERR global algorithm.
                                log.push(NewtonLog {
                                    iter: k,
                                    norm_du: h.norm_du[k] / norm_u,
                                    lambda: h.lambda[k],
                                    theta: h.theta[k],
                                    lambda_prime,
                                    event: NewtonEvent::Accept,
                                });
                                break TrialOutcome::Accept;
                            }
                            // Theta >= 1/2 in the local region: the C's
                            // if/else chain skips every branch — the
                            // trial is accepted implicitly without
                            // resetting the B budget (a faithful quirk;
                            // see the module docs).
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_du[k] / norm_u,
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::Accept,
                            });
                            break TrialOutcome::Accept;
                        }
                        if lambda_prime > 4.0 * h.lambda[k] {
                            // REJECT B: the predicted damping is more
                            // than 4x the current — replace and retry
                            // the trial (no regularity test: the
                            // damping only grows).
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_du[k] / norm_u,
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::RejectB,
                            });
                            h.lambda[k] = lambda_prime;
                            l_b += 1;
                            if l_b >= config.max_trial_b {
                                // -13: exceeded the B-type budget.
                                u.copy_from_slice(&h.u[k]);
                                return Ok((NewtonStatus::ExceededTrialB, log));
                            }
                            stage = Stage::TrialIterate;
                        } else {
                            // Accept u^{k+1}.
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_du[k] / norm_u,
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::Accept,
                            });
                            l_b = 0;
                            break TrialOutcome::Accept;
                        }
                    }
                }
            }
        };
        match outcome {
            TrialOutcome::Accept => {}
            TrialOutcome::Restart(consumed) => {
                // The C's `k += (-qnerr_stop); prediction_start = k + 1;
                // continue;` with the loop's `++k` folded in: the next
                // iteration index is the restart index.
                k += consumed + 1;
                prediction_start = k;
                continue;
            }
        }
        k += 1;
    }

    // -3: the iteration budget is spent.
    u.copy_from_slice(&h.u[max]);
    Ok((NewtonStatus::MaxIterations, log))
}

// ===========================================================================
// qnerr.c — the error-reduction tail with the frozen factorization.
// ===========================================================================

/// The QNERR tail (`nleq_err_qnerr`, `src/qnerr.c`) — called by
/// [`nleq_err`] once the local region is reached with `Theta < 1/2`.
///
/// Operates on the caller's shared histories from index `base = k + 1`
/// (the C's `u + k + 1` pointer offsets): full steps against the ONE
/// factorization of `J(u[base])`, the `alpha_bar` correction over the
/// update history, the acceleration `du = du_bar / (1 - alpha)`, exit
/// on `Theta > 1/2` (restart) or the normalized convergence test.
#[allow(clippy::too_many_lines)]
fn nleq_err_qnerr<S: NewtonSystem>(
    sys: &mut S,
    h: &mut History,
    log: &mut Vec<NewtonLog>,
    base: usize,
    epsilon: f64,
    max_iterations: usize,
) -> Result<QnerrExit, SolveError> {
    let dim = sys.dim();
    // The tail-local arrays: `alpha` history, `du_bar` updates and
    // their norms (qnerr.c writes these into the caller's slots; only
    // u/f/du/norm_du flow back).
    let mut alpha = vec![0.0; max_iterations + 1];
    let mut du_bar = vec![vec![0.0; dim]; max_iterations + 1];
    let mut norm_du_bar = vec![0.0; max_iterations + 1];

    // Calculate J(u^0) and solve J du^0 = -f(u^0) (the only
    // factorization inside the tail).
    sys.factor_jacobian(&h.u[base])?;
    sys.solve_into(&h.f[base], &mut h.du[base])?;
    h.norm_du[base] = sys.norm(&h.du[base]);

    let mut alpha_bar = 0.0;
    // The solution norm for the normalized exit test, refreshed after
    // each full step (qnerr.c line 124).
    let mut norm_u: f64;

    for l in 0..max_iterations {
        // New iterate u^{l+1} = u^l + du^l (full step).
        array_sum_from(&mut h.u, base + l + 1, base + l, 1.0, 1.0, &h.du[base + l]);
        // f(u^{l+1}) and the stale-factor solve
        // J(u^0) du_bar^{l+1} = -f(u^{l+1}).
        sys.rhs_into(&h.u[base + l + 1], &mut h.f[base + l + 1]);
        sys.solve_into(&h.f[base + l + 1], &mut du_bar[l + 1])?;

        // The correction over the update history (i = 1..=l).
        if l > 0 {
            for i in 1..=l {
                alpha_bar = sys.dot(&du_bar[l + 1], &h.du[base + i - 1])
                    / (h.norm_du[base + i - 1] * h.norm_du[base + i - 1]);
                array_sum_accum(&mut du_bar, l + 1, alpha_bar, &h.du[base + i]);
            }
        }

        // alpha_{l+1} = (du_bar^{l+1} . du^l) / ||du^l||^2.
        alpha[l + 1] = sys.dot(&du_bar[l + 1], &h.du[base + l])
            / (h.norm_du[base + l] * h.norm_du[base + l]);
        norm_du_bar[l + 1] = sys.norm(&du_bar[l + 1]);

        // Theta_l = ||du_bar^{l+1}|| / ||du^l||; exit if > 1/2 (or on a
        // non-finite monitor — the tail is broken, restart; see
        // `monitors_finite`).
        let theta = norm_du_bar[l + 1] / h.norm_du[base + l];
        if (theta > 0.5 && (l + 1) > QNERR_MIN_ITERATIONS)
            || !monitors_finite(&[theta, alpha[l + 1], norm_du_bar[l + 1]])
        {
            log.push(NewtonLog {
                iter: base + l,
                norm_du: 0.0,
                lambda: alpha_bar,
                theta,
                lambda_prime: alpha[l + 1],
                event: NewtonEvent::ExitQnerr,
            });
            return Ok(QnerrExit {
                consumed: l + 1,
                reason: QnerrReason::ThetaIncrease,
            });
        }

        // du^{l+1} = du_bar^{l+1} / (1 - alpha_{l+1}).
        let scale = 1.0 / (1.0 - alpha[l + 1]);
        for (out, bar) in h.du[base + l + 1]
            .iter_mut()
            .zip(du_bar[l + 1].iter())
        {
            *out = scale * bar;
        }
        h.norm_du[base + l + 1] = sys.norm(&h.du[base + l + 1]);
        norm_u = sys.norm(&h.u[base + l + 1]);

        // Convergence test: ||du^{l+1}|| < epsilon ||u^{l+1}||.
        if h.norm_du[base + l + 1] < epsilon * norm_u {
            array_sum_accum(&mut h.u, base + l + 1, 1.0, &h.du[base + l + 1]);
            let norm = h.norm_du[base + l + 1] / norm_u;
            log.push(NewtonLog {
                iter: base + l,
                norm_du: norm,
                lambda: alpha_bar,
                theta,
                lambda_prime: alpha[l + 1],
                event: NewtonEvent::ConvergedC,
            });
            return Ok(QnerrExit {
                consumed: l + 1,
                reason: QnerrReason::Converged(norm),
            });
        }
        log.push(NewtonLog {
            iter: base + l,
            norm_du: h.norm_du[base + l + 1] / norm_u,
            lambda: alpha_bar,
            theta,
            lambda_prime: alpha[l + 1],
            event: NewtonEvent::QnerrAccept,
        });
    }

    // The tail spent all remaining iterations.
    Ok(QnerrExit {
        consumed: max_iterations,
        reason: QnerrReason::MaxIterations,
    })
}

// ===========================================================================
// nleq_res.c — the residual-based adaptive-damping driver.
// ===========================================================================

/// The residual-based global Newton driver — the port of `nleq_res`
/// (`src/nleq_res.c`). See the module docs for the semantics; the
/// differences from [`nleq_err`] are the residual-ratio monitor
/// (no second solve), the `Theta >= 1` reject with a QUARTER shrink,
/// the `Theta < 1/2` gate inside the local test, and the QNRES tail.
///
/// `norm_f0` is the initial residual monitoring norm — the C's
/// `norm_f[0]`, which the C driver computes from the initial GUESS
/// with the interior norm (`main.c:184`, a C quirk this port
/// reproduces through [`solve`]; see the module docs).
///
/// Like [`nleq_err`], the driver is one function on purpose — it maps
/// `nleq_res.c` statement-for-statement.
///
/// # Errors
///
/// Propagates the backend's [`SolveError`].
///
/// # Panics
///
/// Panics if `u` does not have [`NewtonSystem::dim`] entries.
#[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
pub fn nleq_res<S: NewtonSystem>(
    sys: &mut S,
    u: &mut [f64],
    norm_f0: f64,
    config: &NewtonConfig,
) -> Result<(NewtonStatus, Vec<NewtonLog>), SolveError> {
    let dim = sys.dim();
    debug_assert_eq!(u.len(), dim, "initial guess length");
    let max = config.max_iterations;
    let mut h = History::new(dim, max);
    let mut aux = vec![0.0; dim];
    let mut log = Vec::new();

    h.u[0].copy_from_slice(u);
    sys.rhs_into(&h.u[0], &mut h.f[0]);
    h.norm_f[0] = norm_f0;
    h.lambda[0] = config.lambda0;
    let mut prediction_start = 0usize;
    let (mut l_a, mut l_b) = (0usize, 0usize);

    let mut k = 0usize;
    while k < max {
        // 1. Convergence test on the residual: ||f(u^k)|| < epsilon.
        if h.norm_f[k] < config.epsilon {
            log.push(NewtonLog {
                iter: k,
                norm_du: h.norm_f[k],
                lambda: 0.0,
                theta: 0.0,
                lambda_prime: 0.0,
                event: NewtonEvent::ConvergedA,
            });
            u.copy_from_slice(&h.u[k]);
            return Ok((
                NewtonStatus::Converged {
                    iterations: k + 1,
                    residual_norm: h.norm_f[k],
                },
                log,
            ));
        }

        // Jacobian + solve (the residual driver assembles only after
        // the convergence test).
        sys.factor_jacobian(&h.u[k])?;
        sys.solve_into(&h.f[k], &mut h.du[k])?;

        // For k > prediction_start: predict the damping.
        if k > prediction_start {
            let mu = h.norm_f[k - 1] * h.mu_prime[k - 1] / h.norm_f[k];
            h.lambda[k] = 1.0_f64.min(mu);
        }

        let mut stage = Stage::RegularityTest;
        let outcome = loop {
            match stage {
                Stage::RegularityTest => {
                    if h.lambda[k] < config.lambda_min {
                        log.push(NewtonLog {
                            iter: k,
                            norm_du: h.norm_f[k],
                            lambda: h.lambda[k],
                            theta: h.theta[k],
                            lambda_prime: 0.0,
                            event: NewtonEvent::RegularityFailure,
                        });
                        u.copy_from_slice(&h.u[k]);
                        return Ok((NewtonStatus::Diverged, log));
                    }
                    stage = Stage::TrialIterate;
                }
                Stage::TrialIterate => {
                    // Trial iterate + residual norm (no second solve).
                    array_sum_from(&mut h.u, k + 1, k, 1.0, h.lambda[k], &h.du[k]);
                    sys.rhs_into(&h.u[k + 1], &mut h.f[k + 1]);
                    h.norm_f[k + 1] = sys.norm(&h.f[k + 1]);

                    // Monitoring: Theta_k = ||f^{k+1}|| / ||f^k||.
                    array_sum(&mut aux, 1.0, &h.f[k + 1], h.lambda[k] - 1.0, &h.f[k]);
                    let norm_f_minus_one_minus_lambda_f = sys.norm(&aux);
                    h.theta[k] = h.norm_f[k + 1] / h.norm_f[k];
                    h.mu_prime[k] = 0.5 * h.norm_f[k] * h.lambda[k] * h.lambda[k]
                        / norm_f_minus_one_minus_lambda_f;

                    // REJECT A: Theta_k >= 1 (a stricter test than the
                    // error driver's, with a quarter shrink), or a
                    // non-finite monitor (the worst contraction; see
                    // `monitors_finite`).
                    if h.theta[k] >= 1.0
                        || !monitors_finite(&[h.theta[k], h.mu_prime[k], h.norm_f[k + 1]])
                    {
                        let lambda_prime = (0.25 * h.lambda[k]).min(h.mu_prime[k]);
                        log.push(NewtonLog {
                            iter: k,
                            norm_du: h.norm_f[k],
                            lambda: h.lambda[k],
                            theta: h.theta[k],
                            lambda_prime,
                            event: NewtonEvent::RejectA,
                        });
                        h.lambda[k] = lambda_prime;
                        l_a += 1;
                        if l_a >= config.max_trial_a {
                            u.copy_from_slice(&h.u[k]);
                            return Ok((NewtonStatus::ExceededTrialA, log));
                        }
                        stage = Stage::RegularityTest;
                    } else {
                        let lambda_prime = 1.0_f64.min(h.mu_prime[k]);
                        l_a = 0;

                        // Local region: lambda' = lambda = 1 and
                        // Theta < 1/2 (the gate is part of the test
                        // here, unlike the error driver).
                        // Exact float comparison on purpose, as in the C.
                        #[allow(clippy::float_cmp)]
                        let in_local_region = lambda_prime == h.lambda[k]
                            && h.lambda[k] == 1.0
                            && h.theta[k] < RESIDUAL_THETA_MAX;
                        if in_local_region {
                            if config.local_solver {
                                log.push(NewtonLog {
                                    iter: k,
                                    norm_du: h.norm_f[k],
                                    lambda: h.lambda[k],
                                    theta: h.theta[k],
                                    lambda_prime,
                                    event: NewtonEvent::EnterQnres,
                                });
                                let exit = nleq_res_qnres(
                                    sys,
                                    &mut h,
                                    &mut log,
                                    k + 1,
                                    config.epsilon,
                                    max - (k + 1),
                                )?;
                                match exit.reason {
                                    QnresReason::Converged(res) => {
                                        u.copy_from_slice(&h.u[k + 1 + exit.consumed]);
                                        return Ok((
                                            NewtonStatus::Converged {
                                                iterations: k + 1 + exit.consumed,
                                                residual_norm: res,
                                            },
                                            log,
                                        ));
                                    }
                                    QnresReason::ThetaIncrease
                                    | QnresReason::IllConditioned => {
                                        // Restart NLEQ-RES with the
                                        // quarter damping (nleq_res.c
                                        // uses 0.25, not 1.0).
                                        h.lambda[k + exit.consumed + 1] = 0.25;
                                        break TrialOutcome::Restart(exit.consumed);
                                    }
                                    QnresReason::MaxIterations => {
                                        u.copy_from_slice(&h.u[k + 1 + exit.consumed]);
                                        return Ok((NewtonStatus::MaxIterations, log));
                                    }
                                }
                            }
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_f[k],
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::Accept,
                            });
                            break TrialOutcome::Accept;
                        }
                        if lambda_prime > 4.0 * h.lambda[k] {
                            // REJECT B.
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_f[k],
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::RejectB,
                            });
                            h.lambda[k] = lambda_prime;
                            l_b += 1;
                            if l_b >= config.max_trial_b {
                                u.copy_from_slice(&h.u[k]);
                                return Ok((NewtonStatus::ExceededTrialB, log));
                            }
                            stage = Stage::TrialIterate;
                        } else {
                            log.push(NewtonLog {
                                iter: k,
                                norm_du: h.norm_f[k],
                                lambda: h.lambda[k],
                                theta: h.theta[k],
                                lambda_prime,
                                event: NewtonEvent::Accept,
                            });
                            l_b = 0;
                            break TrialOutcome::Accept;
                        }
                    }
                }
            }
        };
        match outcome {
            TrialOutcome::Accept => {}
            TrialOutcome::Restart(consumed) => {
                k += consumed + 1;
                prediction_start = k;
                continue;
            }
        }
        k += 1;
    }

    u.copy_from_slice(&h.u[max]);
    Ok((NewtonStatus::MaxIterations, log))
}

// ===========================================================================
// qnres.c — the residual-reduction tail.
// ===========================================================================

/// The QNRES tail (`nleq_res_qnres`, `src/qnres.c`) — called by
/// [`nleq_res`] in the local region. Full steps against the one
/// factorization of `J(u[base])`, with the secant-style `v` recursion
/// over the residual history (`gamma_l = <dF_{l+1}, dF_{l+1}>`), the
/// contraction gate `Theta_l > 1/2`, and the preconditioner monitor
/// `kappa /= 1 - 2 Theta_l` (exit at `KAPPA_MAX`).
#[allow(clippy::too_many_lines)]
fn nleq_res_qnres<S: NewtonSystem>(
    sys: &mut S,
    h: &mut History,
    log: &mut Vec<NewtonLog>,
    base: usize,
    epsilon: f64,
    max_iterations: usize,
) -> Result<QnresExit, SolveError> {
    let dim = sys.dim();
    let mut v = vec![0.0; dim];
    let mut w = vec![0.0; dim];
    let mut gamma = vec![0.0; max_iterations + 1];
    let mut du = vec![vec![0.0; dim]; max_iterations + 1];
    let mut kappa = 1.0;

    // J(u^0) + the only factorization of the tail.
    sys.factor_jacobian(&h.u[base])?;
    sys.solve_into(&h.f[base], &mut du[0])?;

    for l in 0..max_iterations {
        // Full step + residual.
        array_sum_from(&mut h.u, base + l + 1, base + l, 1.0, 1.0, &du[l]);
        sys.rhs_into(&h.u[base + l + 1], &mut h.f[base + l + 1]);
        h.norm_f[base + l + 1] = sys.norm(&h.f[base + l + 1]);

        // Theta and the preconditioner monitor.
        let theta = h.norm_f[base + l + 1] / h.norm_f[base + l];
        kappa /= 1.0 - 2.0 * theta;

        // w = dF[l+1] = F[l+1] - F[l]; gamma[l] = <w, w>.
        array_sum(&mut w, 1.0, &h.f[base + l + 1], -1.0, &h.f[base + l]);
        gamma[l] = sys.dot(&w, &w);

        // Convergence test on the residual norm.
        if h.norm_f[base + l + 1] < epsilon {
            log.push(NewtonLog {
                iter: base + l,
                norm_du: h.norm_f[base + l + 1],
                lambda: gamma[l],
                theta,
                lambda_prime: kappa,
                event: NewtonEvent::ConvergedC,
            });
            return Ok(QnresExit {
                consumed: l + 1,
                reason: QnresReason::Converged(h.norm_f[base + l + 1]),
            });
        }

        // Sanity tests: the contraction ceiling, then the
        // preconditioner monitor (either non-finite trips it; see
        // `monitors_finite`).
        if theta > RESIDUAL_THETA_MAX || !theta.is_finite() {
            log.push(NewtonLog {
                iter: base + l,
                norm_du: h.norm_f[base + l + 1],
                lambda: gamma[l],
                theta,
                lambda_prime: kappa,
                event: NewtonEvent::ExitQnres,
            });
            return Ok(QnresExit {
                consumed: l + 1,
                reason: QnresReason::ThetaIncrease,
            });
        }
        if kappa > KAPPA_MAX || !kappa.is_finite() {
            log.push(NewtonLog {
                iter: base + l,
                norm_du: h.norm_f[base + l + 1],
                lambda: gamma[l],
                theta,
                lambda_prime: kappa,
                event: NewtonEvent::ExitQnres,
            });
            return Ok(QnresExit {
                consumed: l + 1,
                reason: QnresReason::IllConditioned,
            });
        }

        // v = (1 - <w, F[l+1]>/gamma[l]) * F[l+1].
        let z = sys.dot(&w, &h.f[base + l + 1]);
        let c = 1.0 - z / gamma[l];
        for (out, fv) in v.iter_mut().zip(h.f[base + l + 1].iter()) {
            *out = c * fv;
        }

        // The recursion over the residual history (i = l-1 .. 0):
        // v = v - beta * F[i+1].
        for i in (0..l).rev() {
            let beta =
                (sys.dot(&h.f[base + i + 1], &v) - sys.dot(&h.f[base + i], &v)) / gamma[i];
            for (o, fv) in v.iter_mut().zip(h.f[base + i + 1].iter()) {
                *o -= beta * fv;
            }
        }

        // J(u^0) du^{l+1} = -v (the signs work out; qnres.c comment).
        sys.solve_into(&v, &mut du[l + 1])?;

        log.push(NewtonLog {
            iter: base + l,
            norm_du: h.norm_f[base + l + 1],
            lambda: gamma[l],
            theta,
            lambda_prime: kappa,
            event: NewtonEvent::QnresAccept,
        });
    }

    Ok(QnresExit {
        consumed: max_iterations,
        reason: QnresReason::MaxIterations,
    })
}

// ===========================================================================
// newton.c — the classic fixed-damping loop.
// ===========================================================================

/// The classic error-based Newton loop — the port of `newton`
/// (`src/newton.c`, `solverType = 3`): every step applies the SAME
/// damping `lambda0` (the C never updates `lambda[k+1]` beyond copying
/// it forward), exits on the normalized update test, and logs the
/// residual ratio `Theta_k = ||f^{k+1}|| / ||f^k||` without using it
/// (the C's `THETA_MAX` branch is empty — a faithful dead letter).
///
/// # Errors
///
/// Propagates the backend's [`SolveError`].
///
/// # Panics
///
/// Panics if `u` does not have [`NewtonSystem::dim`] entries.
#[allow(clippy::missing_panics_doc)]
pub fn classic_newton<S: NewtonSystem>(
    sys: &mut S,
    u: &mut [f64],
    config: &NewtonConfig,
) -> Result<(NewtonStatus, Vec<NewtonLog>), SolveError> {
    let dim = sys.dim();
    debug_assert_eq!(u.len(), dim, "initial guess length");
    let max = config.max_iterations;
    let mut h = History::new(dim, max);
    let mut log = Vec::new();

    h.u[0].copy_from_slice(u);
    sys.rhs_into(&h.u[0], &mut h.f[0]);

    for k in 0..max {
        sys.factor_jacobian(&h.u[k])?;
        sys.solve_into(&h.f[k], &mut h.du[k])?;
        // NORMALIZED: the classic driver normalizes inside the norm
        // slot itself.
        let norm_u = sys.norm(&h.u[k]);
        h.norm_du[k] = sys.norm(&h.du[k]) / norm_u;

        if h.norm_du[k] < config.epsilon {
            array_sum_from(&mut h.u, k + 1, k, 1.0, 1.0, &h.du[k]);
            log.push(NewtonLog {
                iter: k,
                norm_du: h.norm_du[k],
                lambda: config.lambda0,
                theta: h.theta[k],
                lambda_prime: config.lambda0,
                event: NewtonEvent::ConvergedA,
            });
            u.copy_from_slice(&h.u[k + 1]);
            return Ok((
                NewtonStatus::Converged {
                    iterations: k + 1,
                    residual_norm: h.norm_du[k],
                },
                log,
            ));
        }

        // Fixed-damping trial iterate.
        array_sum_from(&mut h.u, k + 1, k, 1.0, config.lambda0, &h.du[k]);
        sys.rhs_into(&h.u[k + 1], &mut h.f[k + 1]);
        h.theta[k] = sys.norm(&h.f[k + 1]) / sys.norm(&h.f[k]);
        log.push(NewtonLog {
            iter: k,
            norm_du: h.norm_du[k],
            lambda: config.lambda0,
            theta: h.theta[k],
            lambda_prime: config.lambda0,
            event: NewtonEvent::Accept,
        });
    }

    u.copy_from_slice(&h.u[max]);
    Ok((NewtonStatus::MaxIterations, log))
}

// ===========================================================================
// The elliptic adapter + the public solve entry point.
// ===========================================================================

/// The elliptic problem as a [`NewtonSystem`]: the assembly (owned by
/// the grid/fixedPhi lane) over a [`LinearSolver`] backend, bridging
/// the C's single `u` buffer to the split field/xi state.
///
/// The Jacobian and residual follow the C's frequency convention:
/// `w = omega_calc(u[w_idx])` evaluated at the CURRENT iterate
/// (`csr_grid_fill.c:108`, `rhs.c:27`) — the adapter refreshes
/// `params.xi` from the omega slot on every call, so free-omega
/// solves (SAN-65) follow the iterate.
///
/// The linear solve keeps the slice's sign convention: the assembled
/// Jacobian is the derivative of the RESCALE-FREE residual while `f`
/// carries `rescale = -1`, so the Newton direction comes out of the
/// already-negated right-hand side — mirroring
/// `LINEAR_SOLVE_1(du[k], J, f[k])`.
struct EllipticSystem<'a, S: LinearSolver> {
    grid: &'a Grid2d,
    params: ProblemParams,
    pattern: JacobianPattern,
    csr: CsrMatrix,
    fields: FieldSet,
    ws: Workspace,
    solver: &'a mut S,
}

impl<'a, S: LinearSolver> EllipticSystem<'a, S> {
    /// Build the adapter: the CSR pattern is fixed for the whole solve
    /// (the C's `csr_allocate` once), only the values refresh per
    /// iterate.
    fn new(grid: &'a Grid2d, params: &ProblemParams, solver: &'a mut S) -> Self {
        let pattern = JacobianPattern::build(grid, params);
        let csr = CsrMatrix {
            nrows: pattern.nrows,
            ncols: pattern.ncols,
            row_ptrs: pattern.row_ptrs.clone(),
            col_inds: pattern.col_inds.clone(),
            vals: vec![0.0; pattern.col_inds.len()],
        };
        Self {
            grid,
            params: params.clone(),
            pattern,
            csr,
            fields: FieldSet::zeros(grid),
            ws: Workspace::new(grid),
            solver,
        }
    }

    /// Bridge the C's single `u` vector into the assembly view: copy
    /// the field values and refresh the frequency parameter from the
    /// omega slot.
    fn sync_state(&mut self, u: &[f64]) {
        let dim = self.grid.dim();
        self.fields.as_mut_slice().copy_from_slice(&u[..GNUM * dim]);
        self.params.xi = u[GNUM * dim];
    }
}

impl<S: LinearSolver> NewtonSystem for EllipticSystem<'_, S> {
    fn dim(&self) -> usize {
        GNUM * self.grid.dim() + 1
    }

    fn rhs_into(&mut self, u: &[f64], f: &mut [f64]) {
        self.sync_state(u);
        let grid = self.grid;
        let out = assemble_residual(grid, &self.fields, &self.params, &mut self.ws);
        f.copy_from_slice(&out);
    }

    fn factor_jacobian(&mut self, u: &[f64]) -> Result<(), SolveError> {
        self.sync_state(u);
        let grid = self.grid;
        assemble_jacobian_values(
            &self.pattern,
            grid,
            &self.fields,
            &self.params,
            &mut self.ws,
            &mut self.csr.vals,
        );
        self.solver.factor(&self.csr)
    }

    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        self.solver.solve_into(b, x)
    }

    fn norm(&self, x: &[f64]) -> f64 {
        norm2_all_variables(self.grid, x)
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        dot_all_variables(self.grid, x, y)
    }
}

/// Solve the elliptic system with `solver` as the linear backend,
/// mutating `state` in place and returning the status plus the
/// iteration log.
///
/// The driver flavor comes from [`NewtonConfig::kind`] (the C's
/// `solverType` switch in `main.c`'s `run_newton`); the default
/// [`NewtonConfig`] is the seed specs' NLEQ-ERR + QNERR. On failure
/// statuses the state is the last accepted iterate (the module docs'
/// retry note); on a [`SolveError`] the state is untouched.
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
    let mut u = state.to_flat();
    let mut sys = EllipticSystem::new(grid, params, solver);
    let (status, log) = match config.kind {
        NewtonKind::Error => nleq_err(&mut sys, &mut u, config)?,
        NewtonKind::Residual => {
            // The C driver's initial residual norm (main.c:184):
            // computed from the initial GUESS with the interior norm —
            // a C quirk reproduced for iteration-path parity; see the
            // module docs for the degenerate-input hazard.
            let norm_f0 = norm2_interior_all_variables(grid, &u);
            nleq_res(&mut sys, &mut u, norm_f0, config)?
        }
        NewtonKind::Classic => classic_newton(&mut sys, &mut u, config)?,
    };
    // Write the final iterate back into the split state.
    let dim = grid.dim();
    state.fields.as_mut_slice().copy_from_slice(&u[..GNUM * dim]);
    state.xi = u[GNUM * dim];
    Ok((status, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // A tiny dense Newton system: residual/Jacobian function pointers
    // plus a hand-rolled LU with partial pivoting. Only an EXACT zero
    // pivot is singular (near-singular passes through as a huge step,
    // the case the damping machinery must survive).
    // -------------------------------------------------------------------

    struct DenseSystem {
        dim: usize,
        rhs: fn(&[f64], &mut [f64]),
        jac: fn(&[f64], &mut [f64]),
        a: Vec<f64>,
        piv: Vec<usize>,
    }

    impl DenseSystem {
        fn new(dim: usize, rhs: fn(&[f64], &mut [f64]), jac: fn(&[f64], &mut [f64])) -> Self {
            Self {
                dim,
                rhs,
                jac,
                a: vec![0.0; dim * dim],
                piv: Vec::new(),
            }
        }
    }

    impl NewtonSystem for DenseSystem {
        fn dim(&self) -> usize {
            self.dim
        }

        fn rhs_into(&mut self, u: &[f64], f: &mut [f64]) {
            // The C driver's sign convention (`rescale = -1` from
            // `src/main.c`): the solve receives -F(u), so
            // `solve_into(f)` produces the Newton direction.
            (self.rhs)(u, f);
            for v in f.iter_mut() {
                *v *= -1.0;
            }
        }

        fn factor_jacobian(&mut self, u: &[f64]) -> Result<(), SolveError> {
            self.a = vec![0.0; self.dim * self.dim];
            (self.jac)(u, &mut self.a);
            lu_factor(&mut self.a, self.dim, &mut self.piv)
        }

        fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            lu_solve(&self.a, &self.piv, b, x)
        }

        fn norm(&self, x: &[f64]) -> f64 {
            (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
        }

        fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
            x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / x.len() as f64
        }
    }

    /// In-place LU with partial pivoting (row-major). An exact zero
    /// pivot is `SolveError::Singular`.
    fn lu_factor(a: &mut [f64], n: usize, piv: &mut Vec<usize>) -> Result<(), SolveError> {
        piv.clear();
        for k in 0..n {
            let (mut p, mut best) = (k, a[k * n + k].abs());
            for i in k + 1..n {
                if a[i * n + k].abs() > best {
                    best = a[i * n + k].abs();
                    p = i;
                }
            }
            if best == 0.0 {
                return Err(SolveError::Singular);
            }
            piv.push(p);
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            for i in k + 1..n {
                let m = a[i * n + k] / a[k * n + k];
                a[i * n + k] = m;
                for j in k + 1..n {
                    a[i * n + j] -= m * a[k * n + j];
                }
            }
        }
        Ok(())
    }

    /// Forward/back substitution replaying the factor's row swaps.
    fn lu_solve(a: &[f64], piv: &[usize], b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        let n = piv.len();
        if b.len() != n || x.len() != n {
            return Err(SolveError::DimensionMismatch {
                expected: n,
                got: b.len().min(x.len()),
            });
        }
        x.copy_from_slice(b);
        for (k, &p) in piv.iter().enumerate() {
            if p != k {
                x.swap(k, p);
            }
        }
        // Forward (unit lower).
        for i in 1..n {
            let mut s = x[i];
            for j in 0..i {
                s -= a[i * n + j] * x[j];
            }
            x[i] = s;
        }
        // Back (upper).
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in i + 1..n {
                s -= a[i * n + j] * x[j];
            }
            x[i] = s / a[i * n + i];
        }
        Ok(())
    }

    // A smooth separable cubic with a far start: enough nonlinearity to
    // trigger the REJECT B growth at k = 0 and the QNERR entry later.
    fn smooth_rhs(u: &[f64], f: &mut [f64]) {
        f[0] = u[0] + u[0] * u[0] * u[0] / 3.0 - 1.0;
        f[1] = u[1] + u[1] * u[1] * u[1] / 3.0 + 2.0;
    }
    fn smooth_jac(u: &[f64], a: &mut [f64]) {
        a.fill(0.0);
        a[0] = 1.0 + u[0] * u[0];
        a[3] = 1.0 + u[1] * u[1];
    }

    // The same system with every equation row-rescaled by a wild
    // factor: NLEQ-ERR's affine invariance must make the path
    // identical.
    fn scaled_rhs(u: &[f64], f: &mut [f64]) {
        smooth_rhs(u, f);
        f[0] *= 1.0e-8;
        f[1] *= 1.0e8;
    }
    fn scaled_jac(u: &[f64], a: &mut [f64]) {
        smooth_jac(u, a);
        for j in 0..2 {
            a[j] *= 1.0e-8;
            a[2 + j] *= 1.0e8;
        }
    }

    // A fold: the second equation has no real root (u1^2 = -0.1); the
    // Jacobian degenerates as the iterate slides toward the fold.
    fn fold_rhs(u: &[f64], f: &mut [f64]) {
        f[0] = u[0];
        f[1] = u[1] * u[1] + 0.1;
    }
    fn fold_jac(u: &[f64], a: &mut [f64]) {
        a.fill(0.0);
        a[0] = 1.0;
        a[3] = 2.0 * u[1];
    }

    // Rank-deficient: both equations reach only u0.
    fn singular_rhs(u: &[f64], f: &mut [f64]) {
        f[0] = u[0];
        f[1] = u[0];
    }
    fn singular_jac(_u: &[f64], a: &mut [f64]) {
        a.fill(0.0);
        a[0] = 1.0;
        a[2] = 1.0;
    }

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
        assert_ne!(NewtonStatus::ExceededTrialA, NewtonStatus::ExceededTrialB);
        assert_ne!(
            NewtonStatus::ExceededTrialA,
            NewtonStatus::Converged {
                iterations: 0,
                residual_norm: 0.0
            }
        );
    }

    #[test]
    fn status_error_codes_are_the_c_codes() {
        assert_eq!(
            NewtonStatus::Converged {
                iterations: 1,
                residual_norm: 0.0
            }
            .error_code(),
            0
        );
        assert_eq!(NewtonStatus::Diverged.error_code(), -1);
        assert_eq!(NewtonStatus::MaxIterations.error_code(), -3);
        assert_eq!(NewtonStatus::ExceededTrialA.error_code(), -11);
        assert_eq!(NewtonStatus::ExceededTrialB.error_code(), -13);
    }

    #[test]
    fn config_defaults_match_the_smoke_spec() {
        let c = NewtonConfig::default();
        assert_eq!(c.epsilon, 1e-8);
        assert_eq!(c.max_iterations, 20);
        assert_eq!(c.lambda_min, 1e-5);
        // The smoke spec's lambda0 and main.c's trial budgets.
        assert_eq!(c.lambda0, 1e-3);
        assert!(c.local_solver);
        assert_eq!(c.kind, NewtonKind::Error);
        assert_eq!(c.max_trial_a, 8);
        assert_eq!(c.max_trial_b, 8);
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

    /// `dot_all_variables` is the C's scaled average over fields
    /// 1..=5, and `dot(x, x) == norm2(x)^2` on the same slice.
    #[test]
    fn dot_matches_the_c_scaling_and_the_norm() {
        let grid = Grid2d::new(6, 6, 2, 0.25, 0.25);
        let dim = grid.dim();
        let mut x = vec![0.0; GNUM * dim + 1];
        let mut y = vec![0.0; GNUM * dim + 1];
        for k in 1..GNUM {
            for (i, v) in x[k * dim..(k + 1) * dim].iter_mut().enumerate() {
                *v = f64::from(u32::try_from((k + i) % 7).expect("fits u32")) - 3.0;
            }
            for (i, v) in y[k * dim..(k + 1) * dim].iter_mut().enumerate() {
                *v = f64::from(u32::try_from((k * i) % 5).expect("fits u32")) - 1.0;
            }
        }
        let mut expect = 0.0;
        for k in 1..GNUM {
            let a = &x[k * dim..(k + 1) * dim];
            let b = &y[k * dim..(k + 1) * dim];
            expect += a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>() / dim as f64;
        }
        expect /= 5.0;
        assert!((dot_all_variables(&grid, &x, &y) - expect).abs() < 1e-12);
        let nn = norm2_all_variables(&grid, &x);
        assert!((dot_all_variables(&grid, &x, &x) - nn * nn).abs() < 1e-12);
    }

    /// The interior norm excludes the ghost frame and averages over
    /// the five monitored fields.
    #[test]
    fn interior_norm_excludes_ghosts_and_averages_fields() {
        let grid = Grid2d::new(8, 8, 2, 0.25, 0.25);
        let dim = grid.dim();
        let mut flat = vec![0.0; GNUM * dim + 1];
        // psi = 1 everywhere (ghosts included): the interior norm only
        // sees the interior rectangle, all ones → per-field RMS 1.
        for v in &mut flat[4 * dim..5 * dim] {
            *v = 1.0;
        }
        let n = norm2_interior_all_variables(&grid, &flat);
        assert!((n - (1.0_f64 / 5.0).sqrt()).abs() < 1e-14);
        // A ghost-only bump is invisible.
        let (nr, g) = (grid.nr_total(), grid.ghost());
        flat[4 * dim + grid.idx(g - 1, g - 1)] = 1.0e3;
        let n2 = norm2_interior_all_variables(&grid, &flat);
        assert!((n2 - (1.0_f64 / 5.0).sqrt()).abs() < 1e-14);
        let _ = nr;
    }

    /// The seed-spec damping pattern on a synthetic: from lambda0 =
    /// 1e-3 the REJECT B rule grows the damping, full steps follow,
    /// and the QNERR tail finishes with CONVERGED C.
    #[test]
    fn nleq_err_grows_lambda_then_enters_qnerr() {
        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let (status, log) = nleq_err(&mut sys, &mut u, &NewtonConfig::default())
            .expect("dense backend ok");
        if std::env::var("RBS_NEWTON_DEBUG").is_ok() {
            for r in &log {
                eprintln!(
                    "iter {:2}: norm_du {:.6e} lambda {:.6e} theta {:.6e} lambda' {:.6e} {}",
                    r.iter,
                    r.norm_du,
                    r.lambda,
                    r.theta,
                    r.lambda_prime,
                    r.event.status_str()
                );
            }
            eprintln!("status: {status:?}");
        }
        assert!(matches!(status, NewtonStatus::Converged { .. }), "{status:?}");
        let events: Vec<_> = log.iter().map(|r| r.event).collect();
        assert!(events.contains(&NewtonEvent::RejectB), "no REJECT B in {events:?}");
        assert!(events.contains(&NewtonEvent::EnterQnerr), "no QNERR entry");
        assert_eq!(events.last(), Some(&NewtonEvent::ConvergedC));
        // The solution: u + u^3/3 = c has the roots near +-1.1..; the
        // residual must be at machine zero.
        let mut f = vec![0.0; 2];
        smooth_rhs(&u, &mut f);
        assert!(f[0].abs() < 1e-10 && f[1].abs() < 1e-10, "residual {f:?}");
    }

    /// The badly-scaled fault: row-scaling the equations by 1e-8/1e8
    /// leaves the NLEQ-ERR path affine-invariant (Deuflhard's
    /// invariance — the C machinery's designed robustness).
    #[test]
    fn row_scaling_is_affine_invariant() {
        let mut plain = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut scaled = DenseSystem::new(2, scaled_rhs, scaled_jac);
        let mut u1 = vec![-5.0, 5.0];
        let mut u2 = vec![-5.0, 5.0];
        let (s1, log1) = nleq_err(&mut plain, &mut u1, &NewtonConfig::default()).unwrap();
        let (s2, log2) = nleq_err(&mut scaled, &mut u2, &NewtonConfig::default()).unwrap();
        assert_eq!(
            std::mem::discriminant(&s1),
            std::mem::discriminant(&s2),
            "status diverged under row scaling: {s1:?} vs {s2:?}"
        );
        assert_eq!(log1.len(), log2.len(), "row count changed under scaling");
        for (a, b) in log1.iter().zip(&log2) {
            assert_eq!(a.event, b.event, "event mismatch");
            assert!((a.lambda - b.lambda).abs() < 1e-6, "lambda {} vs {}", a.lambda, b.lambda);
            assert!((a.theta - b.theta).abs() < 1e-6, "theta {} vs {}", a.theta, b.theta);
        }
        assert!((u1[0] - u2[0]).abs() < 1e-8 && (u1[1] - u2[1]).abs() < 1e-8);
    }

    /// The fold fault: no real root, the Jacobian degenerates along
    /// the path — observed deterministically as the typed backend
    /// error (an iterate lands on the fold vertex, J[u1] = 0), never
    /// a silent wrong answer.
    #[test]
    fn fold_surfaces_the_degenerate_jacobian() {
        let mut sys = DenseSystem::new(2, fold_rhs, fold_jac);
        let mut u = vec![1.0, 1.0];
        let err = nleq_err(&mut sys, &mut u, &NewtonConfig::default()).unwrap_err();
        assert_eq!(err, SolveError::Singular, "err {err:?}");
    }

    /// The fold near its vertex (u1 = 0.05): the accepted damping
    /// walks the iterate INTO the degenerate zone, the next trial
    /// overshoots it by orders of magnitude, REJECT A fires, and the
    /// shrunk damping falls through the regularity floor — the C's
    /// -1, with the iterate stopped at the last accepted point.
    #[test]
    fn near_vertex_fold_hits_the_regularity_floor() {
        let mut sys = DenseSystem::new(2, fold_rhs, fold_jac);
        let mut u = vec![1.0, 0.05];
        let (status, log) = nleq_err(&mut sys, &mut u, &NewtonConfig::default()).unwrap();
        assert_eq!(status, NewtonStatus::Diverged, "status {status:?}");
        assert!(log.iter().any(|r| r.event == NewtonEvent::RejectA));
        assert_eq!(log.last().expect("rows").event, NewtonEvent::RegularityFailure);
        // The returned iterate is the last accepted one — finite, and
        // not a fake solution (the fold has no root).
        assert!(u.iter().all(|v| v.is_finite()));
        let mut f = vec![0.0; 2];
        fold_rhs(&u, &mut f);
        assert!(f[1].abs() > 1e-2, "spurious convergence at u = {u:?}, f = {f:?}");
    }

    /// The fold with a high regularity floor: the damping hits
    /// `lambda_min` first and the status is the C's -1.
    #[test]
    fn fold_hits_the_regularity_floor_when_lambda_min_is_high() {
        let mut sys = DenseSystem::new(2, fold_rhs, fold_jac);
        let mut u = vec![1.0, 1.0];
        let config = NewtonConfig {
            lambda_min: 1e-2,
            ..NewtonConfig::default()
        };
        let (status, log) = nleq_err(&mut sys, &mut u, &config).unwrap();
        assert_eq!(status, NewtonStatus::Diverged, "status {status:?}");
        assert_eq!(log.last().expect("rows").event, NewtonEvent::RegularityFailure);
    }

    /// An exactly singular Jacobian surfaces as the typed backend
    /// error — propagated, never swallowed.
    #[test]
    fn singular_jacobian_is_a_typed_error() {
        let mut sys = DenseSystem::new(2, singular_rhs, singular_jac);
        let mut u = vec![1.0, 1.0];
        let err = nleq_err(&mut sys, &mut u, &NewtonConfig::default()).unwrap_err();
        assert_eq!(err, SolveError::Singular);
    }

    /// The REJECT B budget is a typed status (the C's -13), reachable
    /// with the config knob (main.c hardcodes 8; here 1).
    #[test]
    fn trial_b_budget_exhaustion_is_typed() {
        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let config = NewtonConfig {
            max_trial_b: 1,
            ..NewtonConfig::default()
        };
        let (status, log) = nleq_err(&mut sys, &mut u, &config).unwrap();
        assert_eq!(status, NewtonStatus::ExceededTrialB, "status {status:?}");
        assert_eq!(log.last().expect("rows").event, NewtonEvent::RejectB);
    }

    /// The REJECT A budget likewise (the C's -11): the near-vertex
    /// fold's first REJECT A trips the budget-1 check before the
    /// shrunk damping is retested against the floor.
    #[test]
    fn trial_a_budget_exhaustion_is_typed() {
        let mut sys = DenseSystem::new(2, fold_rhs, fold_jac);
        let mut u = vec![1.0, 0.05];
        let config = NewtonConfig {
            max_trial_a: 1,
            ..NewtonConfig::default()
        };
        let (status, log) = nleq_err(&mut sys, &mut u, &config).unwrap();
        assert_eq!(status, NewtonStatus::ExceededTrialA, "status {status:?}");
        assert_eq!(log.last().expect("rows").event, NewtonEvent::RejectA);
    }

    /// The residual driver converges with the QNRES tail on a smooth
    /// synthetic, and its k = 0 monitoring uses the caller-supplied
    /// `norm_f0` (the C's driver quirk is the CALLER's choice).
    #[test]
    fn nleq_res_enters_qnres_and_converges() {
        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let mut f0 = vec![0.0; 2];
        smooth_rhs(&u, &mut f0);
        let norm_f0 = sys.norm(&f0);
        let (status, log) = nleq_res(&mut sys, &mut u, norm_f0, &NewtonConfig::default()).unwrap();
        assert!(matches!(status, NewtonStatus::Converged { .. }), "{status:?}");
        let events: Vec<_> = log.iter().map(|r| r.event).collect();
        eprintln!("{events:?}");
        assert!(events.contains(&NewtonEvent::EnterQnres), "no QNRES entry");
        assert_eq!(events.last(), Some(&NewtonEvent::ConvergedC));
        // The residual driver exits at ||f|| < epsilon = 1e-8 (not at
        // machine zero — that is the C's exit semantics).
        let mut f = vec![0.0; 2];
        smooth_rhs(&u, &mut f);
        assert!(f[0].abs() < 1e-7 && f[1].abs() < 1e-7, "residual {f:?}");
    }

    /// The residual path on the fold fails gracefully too.
    #[test]
    fn nleq_res_fails_gracefully_on_the_fold() {
        let mut sys = DenseSystem::new(2, fold_rhs, fold_jac);
        let mut u = vec![1.0, 1.0];
        let (status, _log) = nleq_res(&mut sys, &mut u, 1.0, &NewtonConfig::default()).unwrap();
        assert!(
            status == NewtonStatus::ExceededTrialA
                || status == NewtonStatus::Diverged
                || status == NewtonStatus::MaxIterations,
            "status {status:?}"
        );
        assert!(u.iter().all(|v| v.is_finite()));
        let mut f = vec![0.0; 2];
        fold_rhs(&u, &mut f);
        assert!(f[1].abs() > 1e-2, "spurious convergence at {u:?}");
    }

    /// The classic driver: fixed lambda0 = 1e-3 never converges on the
    /// far start within the budget (the C oracle's own classic path on
    /// the seed); with lambda0 = 1 it does.
    #[test]
    #[allow(clippy::float_cmp)] // lambda is copied, never computed
    fn classic_newton_fixed_damping_semantics() {
        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let (status, log) = classic_newton(&mut sys, &mut u, &NewtonConfig::default()).unwrap();
        assert_eq!(status, NewtonStatus::MaxIterations);
        // Every row is an ACCEPT at the same fixed damping.
        assert!(log.iter().all(|r| r.event == NewtonEvent::Accept));
        assert!(log.iter().all(|r| r.lambda == 1e-3));

        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let config = NewtonConfig {
            lambda0: 1.0,
            ..NewtonConfig::default()
        };
        let (status, _log) = classic_newton(&mut sys, &mut u, &config).unwrap();
        assert!(matches!(status, NewtonStatus::Converged { .. }), "{status:?}");
    }

    /// No-QNERR mode: the local region falls back to plain accepted
    /// full steps (the C's `localSolver = 0` global algorithm).
    #[test]
    fn nleq_err_without_the_local_solver_still_converges() {
        let mut sys = DenseSystem::new(2, smooth_rhs, smooth_jac);
        let mut u = vec![-5.0, 5.0];
        let config = NewtonConfig {
            local_solver: false,
            ..NewtonConfig::default()
        };
        let (status, log) = nleq_err(&mut sys, &mut u, &config).unwrap();
        assert!(matches!(status, NewtonStatus::Converged { .. }), "{status:?}");
        assert!(!log.iter().any(|r| r.event == NewtonEvent::EnterQnerr));
        let mut f = vec![0.0; 2];
        smooth_rhs(&u, &mut f);
        assert!(f[0].abs() < 1e-9 && f[1].abs() < 1e-9, "residual {f:?}");
    }
}
