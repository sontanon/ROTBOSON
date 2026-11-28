//! The four experiments of `docs/specs/A0.5-near-fold-spike.md`.
//!
//! Each returns CSV rows (header first). Distances are `psi0* - psi0`
//! against the campaign's fold estimate.

use std::time::Instant;

use rbs_core::fields::NUM_FIELDS;
use rbs_elliptic::assembly::{assemble_jacobian, ProblemParams, Workspace};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonStatus};
use rbs_elliptic::omega::omega_calc;
use rbs_linalg::faer_lu::FaerSparseLu;
use rbs_linalg::{
    BlockFieldKrylov, BlockKrylovConfig, CsrMatrix, LinearSolver, Preconditioner, SolveError,
};

use crate::state::NearFoldState;

/// Relative tolerance of the linear experiments.
const LINEAR_TOLERANCE: f64 = 1e-10;
/// Inverse iteration: at most this many steps ...
const EIG_MAX_ITERATIONS: usize = 30;
/// ... or until the eigenvalue estimate changes by less than this, relatively.
const EIG_TOLERANCE: f64 = 1e-6;

/// The short name used on the command line and in the CSVs.
pub fn short_name(p: Preconditioner) -> &'static str {
    match p {
        Preconditioner::BlockJacobi => "jacobi",
        Preconditioner::BorderedBlockJacobi => "bordered",
        Preconditioner::BlockGaussSeidel => "gs",
    }
}

fn krylov(preconditioner: Preconditioner, tolerance: f64) -> BlockFieldKrylov {
    BlockFieldKrylov::new(BlockKrylovConfig {
        tolerance,
        preconditioner,
        // Record a miss in the CSV instead of failing the run.
        strict: false,
        ..BlockKrylovConfig::default()
    })
}

fn jacobian(s: &NearFoldState, params: &ProblemParams) -> CsrMatrix {
    let mut ws = Workspace::new(&s.grid);
    assemble_jacobian(&s.grid, &s.state.fields, params, &mut ws)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let ax = a.matvec(x);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
    norm(&r) / norm(b)
}

fn rel_error(x: &[f64], x_true: &[f64]) -> f64 {
    let e: Vec<f64> = x.iter().zip(x_true).map(|(p, q)| p - q).collect();
    norm(&e) / norm(x_true)
}

/// For each state, the index of its neighbor along the branch: the next
/// state with the same N, or the previous one for the last.
fn neighbors(states: &[NearFoldState]) -> Vec<usize> {
    (0..states.len())
        .map(|i| {
            let same_n = |j: usize| states[j].row.n == states[i].row.n;
            if i + 1 < states.len() && same_n(i + 1) {
                i + 1
            } else {
                i - 1
            }
        })
        .collect()
}

/// A direction along the branch: `x_true = neighbor - this`, flattened
/// with the xi slot, and the right-hand side `b = J x_true`.
fn branch_problem(
    states: &[NearFoldState],
    i: usize,
    j: usize,
    a: &CsrMatrix,
) -> (Vec<f64>, Vec<f64>) {
    let x_true: Vec<f64> = states[j]
        .state
        .to_flat()
        .iter()
        .zip(states[i].state.to_flat())
        .map(|(p, q)| p - q)
        .collect();
    let b = a.matvec(&x_true);
    (x_true, b)
}

fn common_columns(s: &NearFoldState) -> String {
    format!(
        "{},{},{:?},{:?},{:.3e}",
        s.row.n, s.row.step, s.row.psi0, s.row.dist, s.residual
    )
}

const COMMON_HEADER: &str = "N,step,psi0,dist,self_check_residual";

/// `linear`: one fresh factorization and one solve per state and preconditioner.
pub fn linear(
    states: &[NearFoldState],
    preconditioners: &[Preconditioner],
    threads: usize,
) -> Vec<String> {
    let mut rows = vec![format!(
        "{COMMON_HEADER},solver,threads,iterations,converged,residual,rel_error,factor_s,solve_s"
    )];
    let neighbor = neighbors(states);
    for (i, s) in states.iter().enumerate() {
        let a = jacobian(s, &s.params);
        let (x_true, b) = branch_problem(states, i, neighbor[i], &a);
        for &p in preconditioners {
            let mut solver = krylov(p, LINEAR_TOLERANCE);
            let t = Instant::now();
            let factored = solver.factor(&a);
            let factor_s = t.elapsed().as_secs_f64();
            let solved = factored.and_then(|()| {
                let t = Instant::now();
                let x = solver.solve(&b)?;
                Ok((x, t.elapsed().as_secs_f64()))
            });
            let stats = solver.last_stats();
            let row = match solved {
                Ok((x, solve_s)) => format!(
                    "{},{},{threads},{},{},{:.3e},{:.3e},{factor_s:.3},{solve_s:.3}",
                    common_columns(s),
                    short_name(p),
                    stats.iterations,
                    stats.converged,
                    rel_residual(&a, &b, &x),
                    rel_error(&x, &x_true)
                ),
                Err(e) => format!(
                    "{},{},{threads},,failed: {e},,,{factor_s:.3},",
                    common_columns(s),
                    short_name(p)
                ),
            };
            eprintln!("linear {row}");
            rows.push(row);
        }
        if s.row.n <= 128 {
            let row = direct_reference(s, &a, &b, &x_true, threads);
            eprintln!("linear {row}");
            rows.push(row);
        }
    }
    rows
}

/// The faer full-LU reference row of `linear` (N=128 only).
fn direct_reference(
    s: &NearFoldState,
    a: &CsrMatrix,
    b: &[f64],
    x_true: &[f64],
    threads: usize,
) -> String {
    let mut lu = FaerSparseLu::new();
    let t = Instant::now();
    let factored = lu.factor(a);
    let factor_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let x = factored.and_then(|()| lu.solve(b));
    let solve_s = t.elapsed().as_secs_f64();
    match x {
        Ok(x) => format!(
            "{},lu,{threads},0,true,{:.3e},{:.3e},{factor_s:.3},{solve_s:.3}",
            common_columns(s),
            rel_residual(a, b, &x),
            rel_error(&x, x_true)
        ),
        Err(e) => format!(
            "{},lu,{threads},,LU failed: {e},,,{factor_s:.3},",
            common_columns(s)
        ),
    }
}

/// `stale`: per N, factor at the state farthest from the fold, then solve
/// every state's system against those blocks; also solve fresh, for the ratio.
pub fn stale(
    states: &[NearFoldState],
    preconditioners: &[Preconditioner],
    threads: usize,
) -> Vec<String> {
    let mut rows = vec![format!(
        "{COMMON_HEADER},solver,threads,factored_at_step,stale_iterations,stale_converged,stale_residual,fresh_iterations,fresh_converged"
    )];
    let neighbor = neighbors(states);
    let mut ns: Vec<usize> = states.iter().map(|s| s.row.n).collect();
    ns.dedup();
    for n in ns {
        let group: Vec<usize> = (0..states.len())
            .filter(|&i| states[i].row.n == n)
            .collect();
        let Some(&farthest) = group.iter().max_by(|&&a, &&b| {
            states[a]
                .row
                .dist
                .abs()
                .total_cmp(&states[b].row.dist.abs())
        }) else {
            continue;
        };
        let a_far = jacobian(&states[farthest], &states[farthest].params);
        for &p in preconditioners {
            let mut stale_solver = krylov(p, LINEAR_TOLERANCE);
            if let Err(e) = stale_solver.factor(&a_far) {
                eprintln!(
                    "stale: factor at step {} failed: {e}",
                    states[farthest].row.step
                );
                continue;
            }
            for &i in &group {
                let s = &states[i];
                let a = jacobian(s, &s.params);
                let (_, b) = branch_problem(states, i, neighbor[i], &a);
                let mut x = vec![0.0; b.len()];
                let stale_result = stale_solver.solve_stale(&a, &b, &mut x);
                let st = stale_solver.last_stats();
                let mut fresh_solver = krylov(p, LINEAR_TOLERANCE);
                let fresh = fresh_solver
                    .factor(&a)
                    .and_then(|()| fresh_solver.solve(&b).map(|_| fresh_solver.last_stats()));
                let (fresh_iterations, fresh_converged) = match fresh {
                    Ok(f) => (f.iterations.to_string(), f.converged.to_string()),
                    Err(e) => (String::new(), format!("failed: {e}")),
                };
                let row = match stale_result {
                    Ok(()) => format!(
                        "{},{},{threads},{},{},{},{:.3e},{fresh_iterations},{fresh_converged}",
                        common_columns(s),
                        short_name(p),
                        states[farthest].row.step,
                        st.iterations,
                        st.converged,
                        st.residual
                    ),
                    Err(e) => format!(
                        "{},{},{threads},{},,failed: {e},,{fresh_iterations},{fresh_converged}",
                        common_columns(s),
                        short_name(p),
                        states[farthest].row.step
                    ),
                };
                eprintln!("stale {row}");
                rows.push(row);
            }
        }
    }
    rows
}

/// Counts the Krylov iterations of every linear solve inside a Newton run.
struct Counting {
    inner: BlockFieldKrylov,
    krylov_iterations: usize,
    linear_solves: usize,
}

impl LinearSolver for Counting {
    fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError> {
        self.inner.factor(a)
    }

    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        let result = self.inner.solve_into(b, x);
        self.krylov_iterations += self.inner.last_stats().iterations;
        self.linear_solves += 1;
        result
    }

    fn name(&self) -> &'static str {
        "counting block-field-krylov"
    }
}

/// `newton`: at the state closest to the fold for each N, run the full
/// Newton solve from the neighbor's fields with this state's pin value.
pub fn newton(
    states: &[NearFoldState],
    preconditioners: &[Preconditioner],
    threads: usize,
) -> Vec<String> {
    let mut rows = vec![format!(
        "{COMMON_HEADER},solver,threads,start_step,status,newton_iterations,linear_solves,krylov_iterations,omega,omega_c,max_rel_field_diff,worst_field,max_rel_field_diff_interior,worst_field_interior,seconds"
    )];
    let neighbor = neighbors(states);
    let mut ns: Vec<usize> = states.iter().map(|s| s.row.n).collect();
    ns.dedup();
    let config = NewtonConfig {
        epsilon: 1e-10,
        max_iterations: 60,
        ..NewtonConfig::default()
    };
    for n in ns {
        let Some(i) = (0..states.len())
            .filter(|&i| states[i].row.n == n)
            .min_by(|&a, &b| {
                states[a]
                    .row
                    .dist
                    .abs()
                    .total_cmp(&states[b].row.dist.abs())
            })
        else {
            continue;
        };
        let target = &states[i];
        let start = &states[neighbor[i]];
        let (pin_r, pin_z) = target.params.fixed_phi.unwrap_or((2, 2));
        let pin = target.grid.idx(pin_r, pin_z);
        for &p in preconditioners {
            let mut state = start.state.clone();
            state.fields.set(4, pin, target.state.fields.at(4, pin));
            let mut solver = Counting {
                inner: krylov(p, 1e-8),
                krylov_iterations: 0,
                linear_solves: 0,
            };
            let t = Instant::now();
            let result = solve(
                &mut solver,
                &target.grid,
                &mut state,
                &target.params,
                &config,
            );
            let seconds = t.elapsed().as_secs_f64();
            let (status, newton_iterations) = match &result {
                Ok((NewtonStatus::Converged { iterations, .. }, _)) => {
                    ("converged".to_string(), iterations.to_string())
                }
                Ok((other, log)) => (format!("{other:?}"), log.len().to_string()),
                Err(e) => (format!("linear solver failed: {e}"), String::new()),
            };
            let omega = omega_calc(state.xi, target.params.m);
            let (max_rel, worst) = field_difference(target, &state.fields, true);
            let (max_rel_interior, worst_interior) = field_difference(target, &state.fields, false);
            let row = format!(
                "{},{},{threads},{},{},{newton_iterations},{},{},{omega:?},{:?},{max_rel:.3e},{worst},{max_rel_interior:.3e},{worst_interior},{seconds:.1}",
                common_columns(target),
                short_name(p),
                start.row.step,
                status.replace(',', ";"),
                solver.linear_solves,
                solver.krylov_iterations,
                target.omega
            );
            eprintln!("newton {row}");
            rows.push(row);
        }
    }
    rows
}

/// The largest field-relative difference `max|u - u_C| / max|u_C|` over
/// the six fields, and the field where it occurs. With `ghosts = false`,
/// only interior points count.
fn field_difference(
    target: &NearFoldState,
    fields: &rbs_core::fields::FieldSet,
    ghosts: bool,
) -> (f64, &'static str) {
    const NAMES: [&str; NUM_FIELDS] = ["log_alpha", "beta", "log_h", "log_a", "psi", "lambda"];
    let grid = &target.grid;
    let g = grid.ghost();
    let mut worst = (0.0_f64, NAMES[0]);
    for (k, name) in NAMES.iter().enumerate() {
        let c = target.state.fields.field(k);
        let u = fields.field(k);
        let scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut diff = 0.0_f64;
        for i in 0..grid.nr_total() {
            for j in 0..grid.nz_total() {
                let interior =
                    i >= g && j >= g && i < grid.nr_total() - g && j < grid.nz_total() - g;
                if ghosts || interior {
                    let idx = grid.idx(i, j);
                    diff = diff.max((u[idx] - c[idx]).abs());
                }
            }
        }
        if diff / scale > worst.0 {
            worst = (diff / scale, name);
        }
    }
    worst
}

/// Inverse iteration for the smallest-magnitude eigenvalue of `a`, with
/// faer's full sparse LU. Returns (estimate, iterations, converged).
fn smallest_eigenvalue(a: &CsrMatrix) -> Result<(f64, usize, bool), SolveError> {
    let mut lu = FaerSparseLu::new();
    lu.factor(a)?;
    let n = a.nrows;
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut estimate = f64::NAN;
    for k in 1..=EIG_MAX_ITERATIONS {
        let y = lu.solve(&x)?;
        // x has unit norm, so lambda ~ 1 / (x . y), with its sign.
        let xy: f64 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
        let next = 1.0 / xy;
        let ny = norm(&y);
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi = yi / ny;
        }
        if (next - estimate).abs() <= EIG_TOLERANCE * next.abs() {
            return Ok((next, k, true));
        }
        estimate = next;
    }
    Ok((estimate, EIG_MAX_ITERATIONS, false))
}

/// `eig`: smallest-magnitude eigenvalue of the fixed-omega and fixed-phi
/// Jacobians at every N=128 state.
pub fn eig(states: &[NearFoldState], threads: usize) -> Vec<String> {
    let mut rows = vec![format!(
        "{COMMON_HEADER},threads,lambda_fixed_omega,iterations_fixed_omega,converged_fixed_omega,lambda_fixed_phi,iterations_fixed_phi,converged_fixed_phi,seconds"
    )];
    for s in states.iter().filter(|s| s.row.n == 128) {
        let t = Instant::now();
        let mut cells = Vec::new();
        for params in [s.fixed_omega_params(), s.params.clone()] {
            match smallest_eigenvalue(&jacobian(s, &params)) {
                Ok((lambda, k, ok)) => cells.push(format!("{lambda:.6e},{k},{ok}")),
                Err(e) => cells.push(format!("failed: {e},,false")),
            }
        }
        let row = format!(
            "{},{threads},{},{},{:.1}",
            common_columns(s),
            cells[0],
            cells[1],
            t.elapsed().as_secs_f64()
        );
        eprintln!("eig {row}");
        rows.push(row);
    }
    rows
}
