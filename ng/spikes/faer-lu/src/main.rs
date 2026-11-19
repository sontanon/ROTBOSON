//! SAN-54 spike: measure faer's sparse-LU envelope on the ROTBOSON 6-field
//! Jacobian pattern.
//!
//! Modes
//!   full      pattern -> symbolic -> numeric -> one solve (repeatable)
//!   symbolic  pattern -> symbolic only
//!   guarded   pattern -> symbolic -> memory projection -> numeric only if the
//!             projected peak stays under the budget (with an RLIMIT_AS net)
//!
//! Output: `key=value` lines (KiB/s units noted per key), greppable.

mod pattern;

use dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{
    factorize_symbolic_lu, simplicial, LuSymbolicParams, NumericLu,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};
use std::io::Write;
use std::time::Instant;

/// faer version measured by this spike (kept in sync with Cargo.toml).
const FAER_VERSION: &str = "0.24.4";

type I = u32;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Force {
    Auto,
    Simplicial,
    Supernodal,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Full,
    Symbolic,
    Guarded,
}

fn vmhwm_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            return num.parse().unwrap_or(0);
        }
    }
    0
}

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

// RLIMIT_AS safety net (linux, no libc dependency: link the raw symbols).
#[cfg(target_os = "linux")]
mod rlimit {
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

    /// Set RLIMIT_AS to `bytes` (keeping the existing hard limit).
    pub fn set_as(bytes: u64) -> bool {
        unsafe {
            let mut r = Rlimit::default();
            if getrlimit(RLIMIT_AS, &mut r) != 0 {
                return false;
            }
            r.cur = bytes.min(r.max);
            setrlimit(RLIMIT_AS, &r) == 0
        }
    }
}

struct RunConfig {
    n: usize,
    domain: f64,
    mode: Mode,
    force: Force,
    runs: usize,
    threads: usize,
    // guarded-mode inputs
    budget_gib: f64,
    fill64: f64,
    fill128: f64,
    delta_num_mib_128: f64,
}

#[derive(Default, Clone)]
struct RunStats {
    assembly_s: Vec<f64>,
    symbolic_s: Vec<f64>,
    numeric_s: Vec<f64>,
    solve_s: Vec<f64>,
    residual: Vec<f64>,
    vmhwm_after_symbolic_kib: Vec<u64>,
    vmhwm_after_numeric_kib: Vec<u64>,
    vmhwm_peak_kib: Vec<u64>,
    fill_lu: Vec<u64>,
    fill_l: Vec<u64>,
    fill_u: Vec<u64>,
    variant: Option<String>,
}

fn symbolic_params(force: Force) -> LuSymbolicParams<'static> {
    let threshold = match force {
        Force::Auto => SupernodalThreshold::AUTO,
        Force::Simplicial => SupernodalThreshold::FORCE_SIMPLICIAL,
        Force::Supernodal => SupernodalThreshold::FORCE_SUPERNODAL,
    };
    LuSymbolicParams {
        supernodal_flop_ratio_threshold: threshold,
        ..Default::default()
    }
}

fn variant_of(symbolic: &faer::sparse::linalg::lu::SymbolicLu<I>) -> String {
    // The variant enum is private; Debug renders it. Encapsulated here on
    // purpose (spike-only affordance).
    let dbg = format!("{:?}", symbolic);
    if dbg.contains("Supernodal") {
        "supernodal".to_string()
    } else {
        "simplicial".to_string()
    }
}

fn main() {
    let cfg = parse_args();
    emit("faer_version", FAER_VERSION);
    emit("n", cfg.n);
    emit("rows", pattern::expected_rows(cfg.n));
    emit("nnz", pattern::expected_total_nnz(cfg.n));
    emit("mode", format!("{:?}", cfg.mode).to_lowercase());
    emit("force", format!("{:?}", cfg.force).to_lowercase());
    emit("threads", cfg.threads);

    let mut stats = RunStats::default();

    match cfg.mode {
        Mode::Symbolic => {
            let jac = build_pattern(&cfg);
            let t = now();
            let (col_ptr, row_idx, vals) = pattern::csr_to_csc(&jac);
            emit("csc_convert_s", elapsed_s(t));
            let symbolic = run_symbolic(&cfg, &col_ptr, &row_idx);
            let variant = variant_of(&symbolic);
            emit("variant", &variant);
            emit("vmhwm_peak_kib", vmhwm_kib());
        }
        Mode::Full => {
            for _ in 0..cfg.runs {
                let one = run_full(&cfg);
                stats.merge(one);
            }
            report_best(&cfg, &stats);
        }
        Mode::Guarded => run_guarded(&cfg),
    }
}

fn build_pattern(cfg: &RunConfig) -> pattern::Jacobian {
    let t = now();
    let jac = pattern::build_jacobian(cfg.n, cfg.domain);
    emit("pattern_build_s", elapsed_s(t));
    jac
}

fn run_symbolic(
    cfg: &RunConfig,
    col_ptr: &[I],
    row_idx: &[I],
) -> faer::sparse::linalg::lu::SymbolicLu<I> {
    let n = pattern::expected_rows(cfg.n);
    let sym_mat =
        SymbolicSparseColMatRef::<I>::new_checked(n, n, col_ptr, None, row_idx);
    let t = now();
    let symbolic = factorize_symbolic_lu(sym_mat, symbolic_params(cfg.force))
        .expect("symbolic factorization failed");
    emit("symbolic_s", elapsed_s(t));
    symbolic
}

fn solve_into(
    symbolic: &faer::sparse::linalg::lu::SymbolicLu<I>,
    numeric: &NumericLu<I, f64>,
    b: &mut [f64],
    threads: usize,
) {
    let n = b.len();
    let par = Par::rayon(threads);
    let lu_ref = faer::sparse::linalg::lu::LuRef::new_unchecked(symbolic, numeric);
    let req = symbolic.solve_in_place_scratch::<f64>(1, par);
    let mut mem = MemBuffer::try_new(req).expect("solve scratch alloc");
    let mut stack = MemStack::new(&mut mem);
    let rhs = MatMut::<f64>::from_column_major_slice_mut(b, n, 1);
    lu_ref.solve_in_place_with_conj(Conj::No, rhs, par, &mut stack);
}

fn run_full(cfg: &RunConfig) -> RunStats {
    let mut st = RunStats::default();

    let t = now();
    let jac = pattern::build_jacobian(cfg.n, cfg.domain);
    let (col_ptr, row_idx, vals) = pattern::csr_to_csc(&jac);
    st.assembly_s.push(elapsed_s(t));

    let symbolic = run_symbolic(cfg, &col_ptr, &row_idx);
    st.variant = Some(variant_of(&symbolic));
    st.vmhwm_after_symbolic_kib.push(vmhwm_kib());

    let n = pattern::expected_rows(cfg.n);
    let sym_mat =
        SymbolicSparseColMatRef::<I>::new_checked(n, n, &col_ptr, None, &row_idx);
    let mat = SparseColMatRef::<I, f64>::new(sym_mat, &vals);
    let par = Par::rayon(cfg.threads);

    let t = now();
    match cfg.force {
        Force::Simplicial => {
            // Own the SimplicialLu so we can read the exact L/U fill and
            // solve with the same factors (single numeric factorization).
            let mut lu = simplicial::SimplicialLu::<I, f64>::new();
            let mut row_perm = vec![0 as I; n];
            let mut row_perm_inv = vec![0 as I; n];
            let req = simplicial::factorize_simplicial_numeric_lu_scratch::<I, f64>(
                n, n,
            );
            let mut mem = MemBuffer::try_new(req).expect("numeric scratch alloc");
            let mut stack = MemStack::new(&mut mem);
            simplicial::factorize_simplicial_numeric_lu(
                &mut row_perm,
                &mut row_perm_inv,
                &mut lu,
                mat,
                symbolic.col_perm(),
                &mut stack,
            )
            .expect("simplicial numeric factorization failed");
            st.fill_l.push(lu.l_factor_unsorted().compute_nnz() as u64);
            st.fill_u.push(lu.u_factor_unsorted().compute_nnz() as u64);
            st.fill_lu
                .push(st.fill_l.last().unwrap() + st.fill_u.last().unwrap());
            st.numeric_s.push(elapsed_s(t));

            let mut b = pattern::make_rhs(&jac);
            let t = now();
            let row_perm_ref =
                faer::perm::PermRef::new_checked(&row_perm, &row_perm_inv, n);
            let req = simplicial::solve_in_place_scratch::<I, f64>(n, 1, par);
            let mut mem = MemBuffer::try_new(req).expect("solve scratch alloc");
            let mut stack = MemStack::new(&mut mem);
            let mut rhs = MatMut::<f64>::from_column_major_slice_mut(&mut b, n, 1);
            lu.solve_in_place_with_conj(
                row_perm_ref,
                symbolic.col_perm(),
                Conj::No,
                rhs,
                par,
                &mut stack,
            );
            st.solve_s.push(elapsed_s(t));
            st.residual.push(residual_rel(&jac, &b));
        }
        _ => {
            let mut numeric = NumericLu::<I, f64>::new();
            let req = symbolic.factorize_numeric_lu_scratch::<f64>(
                par,
                Default::default(),
            );
            let mut mem = MemBuffer::try_new(req).expect("numeric scratch alloc");
            let mut stack = MemStack::new(&mut mem);
            symbolic
                .factorize_numeric_lu::<f64>(
                    &mut numeric,
                    mat,
                    par,
                    &mut stack,
                    Default::default(),
                )
                .expect("numeric factorization failed");
            st.numeric_s.push(elapsed_s(t));

            let mut b = pattern::make_rhs(&jac);
            let t = now();
            solve_into(&symbolic, &numeric, &mut b, cfg.threads);
            st.solve_s.push(elapsed_s(t));
            st.residual.push(residual_rel(&jac, &b));
        }
    }
    st.vmhwm_after_numeric_kib.push(vmhwm_kib());
    st.vmhwm_peak_kib.push(vmhwm_kib());
    st
}

fn residual_rel(jac: &pattern::Jacobian, x: &[f64]) -> f64 {
    let b = pattern::make_rhs(jac);
    let ax = pattern::csr_matvec(jac, x);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..ax.len() {
        let d = ax[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    (num / den).sqrt()
}

fn run_guarded(cfg: &RunConfig) {
    let jac = build_pattern(cfg);
    let t = now();
    let (col_ptr, row_idx, vals) = pattern::csr_to_csc(&jac);
    emit("csc_convert_s", elapsed_s(t));

    let symbolic = run_symbolic(cfg, &col_ptr, &row_idx);
    let variant = variant_of(&symbolic);
    emit("variant", &variant);
    let vmhwm_sym = vmhwm_kib();
    emit("vmhwm_after_symbolic_kib", vmhwm_sym);

    // ---- fill-law projection from the N=64/128 measurements ----
    let rows64 = pattern::expected_rows(64) as f64;
    let rows128 = pattern::expected_rows(128) as f64;
    let rows256 = pattern::expected_rows(256) as f64;
    let b_exp = (cfg.fill128 / cfg.fill64).ln() / (rows128 / rows64).ln();
    let fill256 = cfg.fill128 * (rows256 / rows128).powf(b_exp);
    let fill_ratio = fill256 / cfg.fill128;

    emit("fill_law_exp_rows", b_exp);
    emit("projected_fill_lu_256", fill256.round() as u64);
    emit("fill_ratio_256_over_128", fill_ratio);

    // numeric-phase memory scales with the fill; 15% safety margin
    let safety = 1.15;
    let proj_delta_num_mib = cfg.delta_num_mib_128 * fill_ratio * safety;
    let proj_peak_mib = vmhwm_sym as f64 / 1024.0 + proj_delta_num_mib;
    emit("projected_delta_numeric_mib_256", proj_delta_num_mib);
    emit("projected_peak_mib_256", proj_peak_mib);
    emit("budget_gib", cfg.budget_gib);
    let budget_mib = cfg.budget_gib * 1024.0;

    if proj_peak_mib >= budget_mib {
        emit("guarded_decision", "stop_over_budget");
        emit("vmhwm_peak_kib", vmhwm_kib());
        return;
    }

    emit("guarded_decision", "attempt_numeric");
    // hard net so a wrong projection cannot take the box down
    let rlimit_ok = rlimit::set_as((budget_mib as u64 + 1024) * 1024 * 1024);
    emit("rlimit_as_set", rlimit_ok);

    let n = pattern::expected_rows(cfg.n);
    let sym_mat =
        SymbolicSparseColMatRef::<I>::new_checked(n, n, &col_ptr, None, &row_idx);
    let mat = SparseColMatRef::<I, f64>::new(sym_mat, &vals);
    let par = Par::rayon(cfg.threads);
    let mut numeric = NumericLu::<I, f64>::new();
    let t = now();
    let req = symbolic.factorize_numeric_lu_scratch::<f64>(par, Default::default());
    match MemBuffer::try_new(req) {
        Ok(mut mem) => {
            let mut stack = MemStack::new(&mut mem);
            match symbolic.factorize_numeric_lu::<f64>(
                &mut numeric,
                mat,
                par,
                &mut stack,
                Default::default(),
            ) {
                Ok(_) => {
                    emit("numeric_s", elapsed_s(t));
                    emit("vmhwm_after_numeric_kib", vmhwm_kib());
                    let mut b = pattern::make_rhs(&jac);
                    let t = now();
                    solve_into(&symbolic, &numeric, &mut b, cfg.threads);
                    emit("solve_s", elapsed_s(t));
                    emit("residual_rel", residual_rel(&jac, &b));
                }
                Err(e) => {
                    emit("numeric_error", format!("{:?}", e));
                }
            }
        }
        Err(_) => emit("numeric_error", "scratch_alloc_failed"),
    }
    emit("vmhwm_peak_kib", vmhwm_kib());
}

fn report_best(cfg: &RunConfig, st: &RunStats) {
    if let Some(v) = &st.variant {
        emit("variant", v);
    }
    let best = |xs: &[f64]| xs.iter().cloned().fold(f64::INFINITY, f64::min);
    emit("runs", cfg.runs);
    emit("assembly_s_best", best(&st.assembly_s));
    emit("symbolic_s_best", best(&st.symbolic_s));
    if !st.numeric_s.is_empty() {
        emit("numeric_s_best", best(&st.numeric_s));
        emit("solve_s_best", best(&st.solve_s));
        emit(
            "residual_rel_best",
            st.residual.iter().cloned().fold(f64::INFINITY, f64::min),
        );
    }
    if !st.fill_lu.is_empty() {
        emit("fill_l", st.fill_l.iter().max().copied().unwrap_or(0));
        emit("fill_u", st.fill_u.iter().max().copied().unwrap_or(0));
        emit("fill_lu", st.fill_lu.iter().max().copied().unwrap_or(0));
    }
    let vmax = |xs: &[u64]| xs.iter().copied().max().unwrap_or(0);
    emit("vmhwm_after_symbolic_kib", vmax(&st.vmhwm_after_symbolic_kib));
    emit("vmhwm_after_numeric_kib", vmax(&st.vmhwm_after_numeric_kib));
    emit("vmhwm_peak_kib", vmax(&st.vmhwm_peak_kib));
}

impl RunStats {
    fn merge(&mut self, other: RunStats) {
        let Self {
            assembly_s,
            symbolic_s,
            numeric_s,
            solve_s,
            residual,
            vmhwm_after_symbolic_kib,
            vmhwm_after_numeric_kib,
            vmhwm_peak_kib,
            fill_lu,
            fill_l,
            fill_u,
            variant,
        } = other;
        self.assembly_s.extend(assembly_s);
        self.symbolic_s.extend(symbolic_s);
        self.numeric_s.extend(numeric_s);
        self.solve_s.extend(solve_s);
        self.residual.extend(residual);
        self.vmhwm_after_symbolic_kib.extend(vmhwm_after_symbolic_kib);
        self.vmhwm_after_numeric_kib.extend(vmhwm_after_numeric_kib);
        self.vmhwm_peak_kib.extend(vmhwm_peak_kib);
        self.fill_lu.extend(fill_lu);
        self.fill_l.extend(fill_l);
        self.fill_u.extend(fill_u);
        if self.variant.is_none() {
            self.variant = variant;
        }
    }
}

fn parse_args() -> RunConfig {
    let mut cfg = RunConfig {
        n: 64,
        domain: 8.0,
        mode: Mode::Full,
        force: Force::Auto,
        runs: 1,
        threads: 4,
        budget_gib: 6.0,
        fill64: 0.0,
        fill128: 0.0,
        delta_num_mib_128: 0.0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut take = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--n" => cfg.n = take().parse().unwrap(),
            "--domain" => cfg.domain = take().parse().unwrap(),
            "--mode" => {
                cfg.mode = match take().as_str() {
                    "symbolic" => Mode::Symbolic,
                    "guarded" => Mode::Guarded,
                    _ => Mode::Full,
                }
            }
            "--force" => {
                cfg.force = match take().as_str() {
                    "simplicial" => Force::Simplicial,
                    "supernodal" => Force::Supernodal,
                    _ => Force::Auto,
                }
            }
            "--runs" => cfg.runs = take().parse().unwrap(),
            "--threads" => cfg.threads = take().parse().unwrap(),
            "--budget-gb" => cfg.budget_gib = take().parse().unwrap(),
            "--fill64" => cfg.fill64 = take().parse().unwrap(),
            "--fill128" => cfg.fill128 = take().parse().unwrap(),
            "--delta-num-mib-128" => cfg.delta_num_mib_128 = take().parse().unwrap(),
            _ => {}
        }
        i += 1;
    }
    cfg
}
