//! SAN-67 measurement modes on the REAL first-Newton system
//! (rbs-elliptic assembly at the l=1 smoke ansatz — the same system the
//! N=64 parity gate builds):
//!
//! * `equil` — the faer-LU row-equilibration experiment: factor the
//!   full 6-field Jacobian raw vs row-equilibrated vs two-sided
//!   equilibrated, and measure the solve residual trend with N.
//! * `blocknk` — the field-blocked Newton–Krylov backend
//!   (`rbs_linalg::BlockFieldKrylov`): fresh-block factor+solve plus
//!   stale-block campaign steps against a drifted Jacobian, with
//!   per-solve iteration counts, wall, and RSS, and a fill-law memory
//!   guard before large numeric factorizations.
//!
//! Grid semantics: the C ladder keeps the DOMAIN fixed (dr * N = 16:
//! 0.25 at N=64, 0.125 at N=128, 0.0625 at N=256 — `configs/l1_seed.toml`,
//! `probe_l1_umfp_n256.toml`), which is exactly the refinement path that
//! shrinks the Robin/decay boundary rows like 1/N². `--domain 16` is the
//! default so the measured systems are the production ones.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams, NumericLu, LuRef};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};
use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::{
    assemble_jacobian_values, assemble_residual, JacobianPattern, ProblemParams, Workspace,
};
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::omega::inverse_omega_calc;
use rbs_linalg::block_field::{BlockFieldKrylov, BlockKrylovConfig};
use rbs_linalg::{CsrMatrix, LinearSolver};
use std::io::Write;
use std::time::Instant;

/// faer version measured (kept in sync with Cargo.toml).
const FAER_VERSION: &str = "0.24.4";

fn emit(kv: &str, v: impl std::fmt::Display) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{}={}", kv, v).ok();
    out.flush().ok();
}

fn now() -> Instant {
    Instant::now()
}

fn elapsed_s(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Current and peak RSS of this process (KiB, `/proc/self/status`).
fn rss_kib(peak: bool) -> u64 {
    let want = if peak { "VmHWM:" } else { "VmRSS:" };
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(want) {
            let num: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            return num.parse().unwrap_or(0);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// The real first-Newton system
// ---------------------------------------------------------------------------

/// The real system under measurement: grid, state, parameters, the
/// fixed Jacobian pattern, the current Jacobian values, and the
/// current residual.
struct RealSystem {
    grid: Grid2d,
    params: ProblemParams,
    state: SolveState,
    pattern: JacobianPattern,
    jac: CsrMatrix,
    vals: Vec<f64>,
    b: Vec<f64>,
    ws: Workspace,
}

/// Build the smoke-spec system at grid `n` (interior points per axis)
/// on the fixed `domain`.
fn build_real_system(n: usize, domain: f64) -> RealSystem {
    let dr = domain / n as f64;
    let grid = Grid2d::new(n + 4, n + 4, 2, dr, dr);
    let params = ProblemParams {
        l: 1,
        m: 1.0,
        xi: inverse_omega_calc(0.95, 1.0),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: None,
    };
    let state = initial_guess(&grid, &InitialGuess::l1_smoke());
    let pattern = JacobianPattern::build(&grid, &params);
    let mut ws = Workspace::new(&grid);
    let b = assemble_residual(&grid, &state.fields, &params, &mut ws);
    let mut vals = vec![0.0; pattern.col_inds.len()];
    assemble_jacobian_values(&pattern, &grid, &state.fields, &params, &mut ws, &mut vals);
    let jac = CsrMatrix {
        nrows: pattern.nrows,
        ncols: pattern.ncols,
        row_ptrs: pattern.row_ptrs.clone(),
        col_inds: pattern.col_inds.clone(),
        vals: vals.clone(),
    };
    RealSystem {
        grid,
        params,
        state,
        pattern,
        jac,
        vals,
        b,
        ws,
    }
}

impl RealSystem {
    /// Refresh the Jacobian values and the residual at the current
    /// state (the `csr_gen_jacobian` + `rhs` per-iterate flow).
    fn refresh(&mut self) {
        let RealSystem {
            grid,
            params,
            state,
            pattern,
            jac,
            vals,
            b,
            ws,
        } = self;
        assemble_jacobian_values(pattern, grid, &state.fields, params, ws, vals);
        jac.vals.copy_from_slice(vals);
        *b = assemble_residual(grid, &state.fields, params, ws);
    }

    /// One damped Newton acceptance (the `nleq_err.c` rule the slice's
    /// Newton loop uses: sufficient decrease `||f_trial|| <= (1 -
    /// lambda/4) ||f||`, backtracking by halves to `lambda_min`).
    /// Applies the accepted step and returns the damping factor, or
    /// `None` at the regularity floor.
    #[allow(clippy::range_plus_one)]
    fn accept_step(&mut self, du: &[f64]) -> Option<f64> {
        let f_norm = rms_residual(&self.b);
        let mut lambda = 1.0;
        while lambda >= 1e-5 {
            let mut trial = self.state.clone();
            trial.axpy(lambda, du);
            let f_trial = assemble_residual(&self.grid, &trial.fields, &self.params, &mut self.ws);
            if rms_residual(&f_trial) <= (1.0 - 0.25 * lambda) * f_norm {
                self.state = trial;
                self.refresh();
                return Some(lambda);
            }
            lambda *= 0.5;
        }
        None
    }
}

/// Full six-field residual RMS (the slice's line-search merit norm).
fn rms_residual(f: &[f64]) -> f64 {
    let n = f.len() - 1;
    let sum: f64 = f[..n].iter().map(|v| v * v).sum();
    (sum / n as f64).sqrt()
}

/// ‖Ax − b‖ / ‖b‖ on a CSR matrix.
fn residual_rel(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let ax = a.matvec(x);
    let mut num = 0.0;
    let mut den = 0.0;
    for (r, bi) in ax.iter().zip(b) {
        num += (r - bi) * (r - bi);
        den += bi * bi;
    }
    (num / den).sqrt()
}

/// Convert the real CSR matrix to faer's CSC arrays.
fn csr_to_csc(a: &CsrMatrix) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
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
    let mut row_idx = vec![0usize; a.nnz()];
    let mut vals = vec![0.0; a.nnz()];
    for r in 0..a.nrows {
        for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
            let c = a.col_inds[k];
            let dest = next[c];
            row_idx[dest] = r;
            vals[dest] = a.vals[k];
            next[c] += 1;
        }
    }
    (col_ptrs, row_idx, vals)
}

// ---------------------------------------------------------------------------
// Mode `equil`
// ---------------------------------------------------------------------------

/// The equilibration variants.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EquilVariant {
    /// The raw matrix.
    None,
    /// One-sided row-max scaling: factor `D·A`, solve `D·A x = D·b`
    /// (row scaling does not move the solution).
    Row,
    /// Row-max then column-max scaling: factor `D·A·C`, solve for `y`
    /// and unscale `x = C·y`.
    TwoSided,
}

/// `equil` mode entry point (args come from the process argv).
pub fn run_equil_from_env() {
    let mut n = 64usize;
    let mut domain = 16.0;
    let mut variant = EquilVariant::None;
    let mut threads = 4usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut take = || args.next().unwrap_or_default();
        match a.as_str() {
            "--n" => n = take().parse().unwrap(),
            "--domain" => domain = take().parse().unwrap(),
            "--variant" => {
                variant = match take().as_str() {
                    "row" => EquilVariant::Row,
                    "2sided" => EquilVariant::TwoSided,
                    _ => EquilVariant::None,
                }
            }
            "--threads" => threads = take().parse().unwrap(),
            _ => {}
        }
    }
    run_equil(n, domain, variant, threads);
}

fn run_equil(n: usize, domain: f64, variant: EquilVariant, threads: usize) {
    emit("faer_version", FAER_VERSION);
    emit("mode", "equil");
    emit("n", n);
    emit("domain", domain);
    emit("variant", format!("{variant:?}").to_lowercase());
    emit("threads", threads);

    let sys = build_real_system(n, domain);
    let jac = &sys.jac;
    let b = &sys.b;
    let rows = jac.nrows;
    emit("rows", rows);
    emit("nnz", jac.nnz());
    emit("dr", domain / n as f64);

    // Row-max diagnostics: the mechanism under test is the 1/N^2
    // boundary-row shrinkage relative to the O(1) interior rows.
    let mut row_max = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut m = 0.0_f64;
        for k in jac.row_ptrs[r]..jac.row_ptrs[r + 1] {
            m = m.max(jac.vals[k].abs());
        }
        row_max[r] = m;
    }
    let mut sorted = row_max.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    emit("rowmax_min", format!("{:.3e}", sorted[0]));
    emit("rowmax_median", format!("{:.3e}", sorted[rows / 2]));
    emit("rowmax_max", format!("{:.3e}", sorted[rows - 1]));

    // Assemble the scaled matrix.
    let (col_ptrs, row_idx, mut vals) = csr_to_csc(jac);
    let mut b_scaled = b.clone();
    let mut col_scale: Option<Vec<f64>> = None;
    if variant != EquilVariant::None {
        // D: row-max over the FULL row (all fields + omega column),
        // exactly the scaling a direct solver would apply.
        for p in 0..vals.len() {
            let r = row_idx[p];
            vals[p] /= row_max[r];
        }
        for (bi, rm) in b_scaled.iter_mut().zip(&row_max) {
            *bi /= rm;
        }
    }
    if variant == EquilVariant::TwoSided {
        // C: column-max of D·A. A single (row, col) sweep is the
        // standard lazy equilibration; no iteration to convergence.
        let mut col_max = vec![0.0_f64; rows];
        for c in 0..rows {
            for p in col_ptrs[c]..col_ptrs[c + 1] {
                col_max[c] = col_max[c].max(vals[p].abs());
            }
        }
        let c_scale: Vec<f64> = col_max
            .iter()
            .map(|m| if *m > 0.0 { 1.0 / m } else { 1.0 })
            .collect();
        for c in 0..rows {
            for p in col_ptrs[c]..col_ptrs[c + 1] {
                vals[p] *= c_scale[c];
            }
        }
        col_scale = Some(c_scale);
    }

    // Factor and solve with faer's sparse LU (AUTO variant).
    let par = Par::rayon(threads);
    let sym_ref = SymbolicSparseColMatRef::<usize>::new_checked(rows, rows, &col_ptrs, None, &row_idx);
    let t = now();
    let symbolic = factorize_symbolic_lu(
        sym_ref,
        LuSymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            ..Default::default()
        },
    )
    .expect("symbolic factorization failed");
    emit("symbolic_s", elapsed_s(t));

    let mat = SparseColMatRef::<usize, f64>::new(sym_ref, &vals);
    let t = now();
    let req = symbolic.factorize_numeric_lu_scratch::<f64>(par, Default::default());
    let mut mem = MemBuffer::try_new(req).expect("numeric scratch alloc");
    let mut stack = MemStack::new(&mut mem);
    let mut numeric = NumericLu::<usize, f64>::new();
    symbolic
        .factorize_numeric_lu::<f64>(&mut numeric, mat, par, &mut stack, Default::default())
        .expect("numeric factorization failed");
    emit("numeric_s", elapsed_s(t));

    let t = now();
    let lu = LuRef::new_unchecked(&symbolic, &numeric);
    let req = symbolic.solve_in_place_scratch::<f64>(1, par);
    let mut mem = MemBuffer::try_new(req).expect("solve scratch alloc");
    let mut stack = MemStack::new(&mut mem);
    let mut x = b_scaled.clone();
    let rhs = MatMut::<f64>::from_column_major_slice_mut(&mut x, rows, 1);
    lu.solve_in_place_with_conj(Conj::No, rhs, par, &mut stack);
    emit("solve_s", elapsed_s(t));

    // Unscale the two-sided unknowns: x = C y.
    if let Some(c) = &col_scale {
        for (xi, ci) in x.iter_mut().zip(c) {
            *xi *= ci;
        }
    }

    // The verdict quantity: the residual of the ORIGINAL system.
    emit("residual_rel", format!("{:.6e}", residual_rel(jac, b, &x)));
    emit("vmhwm_peak_kib", rss_kib(true));
}

// ---------------------------------------------------------------------------
// Mode `blocknk`
// ---------------------------------------------------------------------------

/// `blocknk` mode entry point (args from the process argv).
pub fn run_blocknk_from_env() {
    let mut n = 64usize;
    let mut domain = 16.0;
    let mut threads = 4usize;
    let mut tol = 1e-8;
    let mut restart = 30usize;
    let mut max_cycles = 40usize;
    let mut equilibrate = true;
    let mut stale_steps = 3usize;
    let mut budget_gb = 8.0;
    // Run the stale loop to the C's Newton exit (`||du|| < eps ||u||`)
    // instead of a fixed step count; `stale-steps` becomes the cap.
    let mut newton = false;
    let mut epsilon = 1e-8;
    // Compare the converged state against the pinned N=64 oracle
    // fixture (the parity gate's own reference).
    let mut fixture = false;
    // Memory-projection inputs (measured at smaller N, passed in by the
    // orchestration; 0 = no guard).
    let mut delta64_mib = 0.0_f64;
    let mut delta128_mib = 0.0_f64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut take = || args.next().unwrap_or_default();
        match a.as_str() {
            "--n" => n = take().parse().unwrap(),
            "--domain" => domain = take().parse().unwrap(),
            "--threads" => threads = take().parse().unwrap(),
            "--tol" => tol = take().parse().unwrap(),
            "--restart" => restart = take().parse().unwrap(),
            "--max-cycles" => max_cycles = take().parse().unwrap(),
            "--no-equil" => equilibrate = false,
            "--stale-steps" => stale_steps = take().parse().unwrap(),
            "--budget-gb" => budget_gb = take().parse().unwrap(),
            "--newton" => newton = true,
            "--epsilon" => epsilon = take().parse().unwrap(),
            "--fixture" => fixture = true,
            "--delta64-mib" => delta64_mib = take().parse().unwrap(),
            "--delta128-mib" => delta128_mib = take().parse().unwrap(),
            _ => {}
        }
    }
    run_blocknk(
        n,
        domain,
        threads,
        tol,
        restart,
        max_cycles,
        equilibrate,
        stale_steps,
        budget_gb,
        delta64_mib,
        delta128_mib,
        newton,
        epsilon,
        fixture,
    );
}

/// Set RLIMIT_AS to `bytes` (linux; the hard net of the compute guards).
fn set_rlimit_as(bytes: u64) -> bool {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Rlimit {
        cur: u64,
        max: u64,
    }
    extern "C" {
        fn getrlimit(resource: i32, rlim: *mut Rlimit) -> i32;
        fn setrlimit(resource: i32, rlim: *const Rlimit) -> i32;
    }
    const RLIMIT_AS: i32 = 9;
    unsafe {
        let mut r = Rlimit::default();
        if getrlimit(RLIMIT_AS, &mut r) != 0 {
            return false;
        }
        r.cur = bytes.min(r.max);
        setrlimit(RLIMIT_AS, &r) == 0
    }
}

#[allow(clippy::too_many_arguments)]
fn run_blocknk(
    n: usize,
    domain: f64,
    threads: usize,
    tol: f64,
    restart: usize,
    max_cycles: usize,
    equilibrate: bool,
    stale_steps: usize,
    budget_gb: f64,
    delta64_mib: f64,
    delta128_mib: f64,
    newton: bool,
    epsilon: f64,
    fixture: bool,
) {
    emit("faer_version", FAER_VERSION);
    emit("mode", "blocknk");
    emit("n", n);
    emit("domain", domain);
    emit("threads", threads);
    emit("tol", tol);
    emit("restart", restart);
    emit("max_cycles", max_cycles);
    emit("block_equil", equilibrate);
    emit("dr", domain / n as f64);

    // Pin faer's process-global parallelism (the block factorizations
    // and block solves honor it): the 8-hyperthread default is the
    // 2.5x oversubscription trap of docs/profiling.md SS4.
    faer::set_global_parallelism(Par::rayon(threads));

    // ---- memory guard (compute guards: never factor large blocks
    // over budget; abort-and-record instead) ----
    if delta64_mib > 0.0 && delta128_mib > 0.0 && n > 128 {
        // Numeric-factor RSS delta grows ~ fill ~ rows^exp; exp from
        // the 64->128 measurements, projected to this n's block size.
        let dim = |k: usize| (k + 4) * (k + 4);
        let exp = (delta128_mib / delta64_mib).ln() / (dim(128) as f64 / dim(64) as f64).ln();
        let proj_delta = delta128_mib * (dim(n) as f64 / dim(128) as f64).powf(exp);
        // The 128 run measured all 6 blocks; add the operator + arenas.
        let base_mib = rss_kib(false) as f64 / 1024.0;
        let projected_peak = base_mib + proj_delta;
        emit("fill_law_exp", exp);
        emit("projected_delta_numeric_mib", proj_delta);
        emit("projected_peak_mib", projected_peak);
        emit("budget_gb", budget_gb);
        if projected_peak >= budget_gb * 1024.0 {
            emit("guarded_decision", "stop_over_budget");
            emit("vmhwm_peak_kib", rss_kib(true));
            return;
        }
        emit("guarded_decision", "attempt_numeric");
        // Hard net so a wrong projection cannot take the box down.
        let net = ((budget_gb + 4.0) * 1024.0 * 1024.0 * 1024.0) as u64;
        emit("rlimit_as_set", set_rlimit_as(net));
    }

    let mut sys = build_real_system(n, domain);
    emit("rows", sys.jac.nrows);
    emit("nnz", sys.jac.nnz());
    emit("dim_per_block", sys.grid.dim());
    // The pre-factor RSS baseline (assembly + pattern + residual); the
    // N=256 guard projects the factor-phase delta from smaller sizes.
    emit("vmrss_after_assembly_kib", rss_kib(false));

    let mut backend = BlockFieldKrylov::new(BlockKrylovConfig {
        nfields: 6,
        restart_length: restart,
        max_cycles,
        tolerance: tol,
        equilibrate_blocks: equilibrate,
        strict: true,
        ..BlockKrylovConfig::default()
    });

    // ---- fresh-block cold solve: factor(J0) + Krylov ----
    let t = now();
    backend.factor(&sys.jac).expect("block factorization");
    emit("block_factor_s", elapsed_s(t));
    emit("vmrss_after_factor_kib", rss_kib(false));
    emit("vmhwm_after_factor_kib", rss_kib(true));

    let mut x = vec![0.0; sys.jac.nrows];
    let t = now();
    backend.solve_into(&sys.b, &mut x).expect("fresh solve");
    let solve_s = elapsed_s(t);
    let stats = backend.last_stats();
    emit("fresh_solve_s", solve_s);
    emit("fresh_iters", stats.iterations);
    emit("fresh_cycles", stats.cycles);
    emit("fresh_residual_rel", format!("{:.6e}", stats.residual));
    // Independent check against the backend's own accounting.
    emit(
        "fresh_residual_rel_independent",
        format!("{:.6e}", residual_rel(&sys.jac, &sys.b, &x)),
    );

    // ---- stale-block campaign steps: reuse the blocks, drift the
    // operator by real (damped) Newton steps — the stale-LU CGS
    // analogue of docs/profiling.md SS3. In `--newton` mode the loop
    // runs to the C's normalized exit (`||du|| < eps ||u||`, the
    // `norm2_all_variables` norm over fields 1..=5), with the blocks
    // factored ONCE at the ansatz. ----
    let dim = sys.grid.dim();
    let norm2_all_vars = |flat: &[f64]| -> f64 {
        // The C `norm2_all_variables`: RMS over fields 1..=5,
        // averaged over the five fields (log alpha and the omega slot
        // excluded).
        let mut sum = 0.0;
        for k in 1..6 {
            let field = &flat[k * dim..(k + 1) * dim];
            sum += field.iter().map(|v| v * v).sum::<f64>() / dim as f64;
        }
        (sum / 5.0).sqrt()
    };
    let mut newton_converged_step: Option<usize> = None;
    if newton {
        let norm_du = norm2_all_vars(&x);
        let norm_u = norm2_all_vars(&sys.state.to_flat());
        emit("newton_check", format!("0 {:.3e}", norm_du / norm_u));
        if norm_du < epsilon * norm_u {
            sys.state.axpy(1.0, &x);
            sys.refresh();
            newton_converged_step = Some(0);
        }
    }

    for k in 1..=stale_steps {
        if newton_converged_step.is_some() {
            break;
        }
        let Some(lambda) = sys.accept_step(&x) else {
            emit("stale_regularity_floor", k);
            break;
        };
        emit("stale_lambda", format!("{} {}", k, lambda));
        let t = now();
        let ok = backend.solve_stale(&sys.jac, &sys.b, &mut x);
        let solve_s = elapsed_s(t);
        match ok {
            Ok(()) => {
                let stats = backend.last_stats();
                emit(
                    "stale_solve",
                    format!(
                        "step {} iters {} cycles {} residual {:.3e} wall {:.4} s",
                        k, stats.iterations, stats.cycles, stats.residual, solve_s
                    ),
                );
                emit(
                    "stale_residual_rel_independent",
                    format!("{} {:.6e}", k, residual_rel(&sys.jac, &sys.b, &x)),
                );
            }
            Err(e) => {
                emit("stale_solve_error", format!("{} {:?}", k, e));
                break;
            }
        }
        if newton {
            let norm_du = norm2_all_vars(&x);
            let norm_u = norm2_all_vars(&sys.state.to_flat());
            emit("newton_check", format!("{k} {:.3e}", norm_du / norm_u));
            if norm_du < epsilon * norm_u {
                sys.state.axpy(1.0, &x);
                sys.refresh();
                newton_converged_step = Some(k);
            }
        }
    }

    if newton {
        match newton_converged_step {
            Some(k) => emit("newton_converged", k + 1),
            None => emit("newton_not_converged", stale_steps),
        }
    }

    // Optional end-to-end check against the pinned N=64 oracle fixture
    // (the parity gate's own reference; the gate itself stays with the
    // faer backend in rbs-elliptic — this records the block backend's
    // campaign-shape agreement).
    if fixture {
        if n != 64 {
            emit("fixture_skipped", "n != 64");
        } else {
            match load_oracle_fixture() {
                Ok(fields) => {
                    let names = [
                        "log_alpha_f",
                        "beta_f",
                        "log_h_f",
                        "log_a_f",
                        "psi_f",
                        "lambda_f",
                    ];
                    let mut worst_rel = 0.0_f64;
                    let mut worst_abs = 0.0_f64;
                    for (kk, name) in names.iter().enumerate() {
                        let oracle = &fields[kk];
                        let mine = sys.state.fields.field(kk);
                        let scale = oracle.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                        let max_abs = oracle
                            .iter()
                            .zip(mine)
                            .map(|(o, m)| (o - m).abs())
                            .fold(0.0_f64, f64::max);
                        worst_rel = worst_rel.max(max_abs / scale);
                        worst_abs = worst_abs.max(max_abs);
                        emit(
                            "fixture_field",
                            format!(
                                "{name} rel {:.3e} abs {:.3e}",
                                max_abs / scale,
                                max_abs
                            ),
                        );
                    }
                    emit(
                        "fixture_worst",
                        format!("rel {:.3e} abs {:.3e}", worst_rel, worst_abs)
                    );
                }
                Err(why) => emit("fixture_error", why),
            }
        }
    }

    emit("vmhwm_peak_kib", rss_kib(true));
}

/// The pinned C-oracle N=64 solution fixture (the parity gate's
/// `tests/fixtures/l1_smoke_n64_solution.txt`, loaded read-only).
fn load_oracle_fixture() -> Result<Vec<Vec<f64>>, String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../rbs-elliptic/tests/fixtures/l1_smoke_n64_solution.txt"
    );
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("fixture unreadable: {e}"))?;
    let mut fields: Vec<Vec<f64>> = Vec::new();
    let mut current: Option<Vec<f64>> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("field") => {
                if let Some(done) = current.take() {
                    fields.push(done);
                }
                current = Some(Vec::new());
            }
            // Header lines of the fixture (provenance block).
            Some("nr_total") | Some("nz_total") | Some("omega") => {}
            Some(value) => {
                current
                    .as_mut()
                    .ok_or("values before a field header")?
                    .push(value.parse().map_err(|e| format!("fixture float: {e}"))?);
            }
            None => {}
        }
    }
    if let Some(done) = current.take() {
        fields.push(done);
    }
    if fields.len() != 6 {
        return Err(format!("expected 6 fields, got {}", fields.len()));
    }
    Ok(fields)
}
