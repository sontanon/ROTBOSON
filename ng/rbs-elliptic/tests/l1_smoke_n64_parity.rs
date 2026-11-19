//! SAN-62 parity gate: the N=64 l=1 fixed-omega smoke solve in pure Rust
//! vs the pinned C-oracle solution.
//!
//! The oracle fixture (`tests/fixtures/l1_smoke_n64_solution.txt`) is the
//! committed output of the C binary on the exact smoke spec (see the
//! fixture header for provenance), so this test is fully self-contained:
//! no C binary, no Python, no HDF5.
//!
//! Ignored by default because the N=64 sparse LU is slow in debug builds;
//! run the gate with:
//!
//! ```text
//! CARGO_HOME=<cache> cargo test --release -p rbs-elliptic -- --ignored
//! ```
//!
//! Gates asserted:
//!
//! * convergence of the NLEQ-ERR + QNERR machinery (the C's normalized
//!   `||du|| < eps * ||u||` family of exits) on the smoke spec's
//!   `epsilon = 1e-8`, with the C oracle's ITERATION PATH: the k = 0
//!   REJECT B damping growth (lambda0 = 1e-3 -> 0.128601), the k = 1
//!   prediction recovery (lambda = 0.662003), the k = 2 QNERR entry,
//!   and the CONVERGED C exit — 12 iterations, exactly the oracle's
//!   count (SAN-52);
//! * the initial residual at the ansatz matches the oracle's logged
//!   `||f0||..||f5||` to 10 digits (the strongest single check that the
//!   whole residual chain — derivatives, regularization auxiliaries,
//!   kernels, boundary rows — is the C's);
//! * the SOLUTION fields match the oracle at the exit points:
//!   - at `epsilon = 1e-12` (the deeper exit: the QNERR tail's
//!     Theta-increase restart, then the CONVERGED B on a fresh-J
//!     second solve at `||du_bar||/||u|| ~ 2e-14`) to the original
//!     tight gates — field-relative below 1e-10, absolute below
//!     5e-14 (measured: 2.9e-14 / 1.5e-15);
//!   - at the smoke spec's `epsilon = 1e-8` (the C's own 12-iteration
//!     CONVERGED C exit) to 1e-8 / 1e-9 gates: that exit applies a
//!     stale-J *corrected* step at `||du|| ~ 1e-9`, and the alpha
//!     correction chain amplifies faer-vs-PARDISO solve rounding to
//!     the ~1e-10 level (measured: 7.0e-10) — the C's own exit
//!     state carries the same-order error in ITS backend's rounding,
//!     which is why the tight pin lives at the deeper exit;
//! * per-point relative deviation is printed for the record; its worst
//!   value sits on near-zero axis points of `lambda` where an
//!   absolute ~1.5e-15 difference is a large relative number and means
//!   nothing;
//! * the faer linear backend's solve residual `||Ax - b|| / ||b||` on
//!   the real first-Newton-step system.  faer's sparse LU has no
//!   matrix scaling, and the C's Robin/decay rows lose magnitude as
//!   the grid refines — at THIS spec's fixed `dr = dz = 0.25` the
//!   pattern is benign (the solve residual check passes at ~1e-13),
//!   but any coarser grid or backend change must re-run this gate
//!   (faer 0.24.4; see the SAN-62 report and the ng-spike verdict).

use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::{
    assemble_jacobian, assemble_residual, JacobianPattern, ProblemParams, Workspace,
};
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonEvent, NewtonStatus};
use rbs_linalg::faer_lu::FaerSparseLu;
use rbs_linalg::LinearSolver;

/// The oracle's logged per-field initial-residual RMS (run.log of the
/// smoke spec; epsilon-independent — it is the ansatz state).
const ORACLE_INITIAL_NORMS: [f64; 6] = [
    6.458_734_832_1e-5,
    4.400_626_312_2e-5,
    9.549_604_719_8e-5,
    7.003_752_271_9e-5,
    4.601_532_077_1e-5,
    5.168_841_808_5e-6,
];

struct Fixture {
    nr_total: usize,
    nz_total: usize,
    omega: f64,
    fields: Vec<Vec<f64>>,
}

fn load_fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_smoke_n64_solution.txt"
    );
    let text = std::fs::read_to_string(path).expect("fixture present");
    let mut fix = Fixture {
        nr_total: 0,
        nz_total: 0,
        omega: 0.0,
        fields: Vec::new(),
    };
    let mut current: Option<Vec<f64>> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("nr_total") => fix.nr_total = parts.next().unwrap().parse().unwrap(),
            Some("nz_total") => fix.nz_total = parts.next().unwrap().parse().unwrap(),
            Some("omega") => fix.omega = parts.next().unwrap().parse().unwrap(),
            Some("field") => {
                if let Some(done) = current.take() {
                    fix.fields.push(done);
                }
                current = Some(Vec::new());
            }
            Some(value) => {
                current
                    .as_mut()
                    .expect("values follow a field header")
                    .push(value.parse().expect("fixture float"));
            }
            None => {}
        }
    }
    if let Some(done) = current.take() {
        fix.fields.push(done);
    }
    fix
}

#[test]
#[ignore = "N=64 sparse solve: ~15 s in --release, minutes in debug. Run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n64_l1_smoke_matches_the_c_oracle() {
    let fix = load_fixture();
    assert_eq!(fix.nr_total, 68);
    assert_eq!(fix.nz_total, 68);
    assert_eq!(fix.fields.len(), 6);

    let grid = Grid2d::new(fix.nr_total, fix.nz_total, 2, 0.25, 0.25);
    let params = ProblemParams {
        l: 1,
        m: 1.0,
        xi: rbs_elliptic::omega::inverse_omega_calc(0.95, 1.0),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: None,
    };
    let mut state: SolveState = initial_guess(&grid, &InitialGuess::l1_smoke());

    // Initial-residual parity: the ansatz residual RMS per field must
    // reproduce the oracle's logged values to the printed precision.
    let mut ws = Workspace::new(&grid);
    let f0 = assemble_residual(&grid, &state.fields, &params, &mut ws);
    let dim = grid.dim();
    for (k, expect) in ORACLE_INITIAL_NORMS.iter().enumerate() {
        let rms = (f0[k * dim..(k + 1) * dim])
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            / (dim as f64).sqrt();
        assert!(
            (rms - expect).abs() / expect < 1e-9,
            "initial ||f{k}||: {rms:.10e} vs oracle {expect:.10e}"
        );
    }

    // Linear-backend sanity on the real first Newton system (the
    // ng-spike faer verdict makes this check mandatory).
    {
        let jac = assemble_jacobian(&grid, &state.fields, &params, &mut ws);
        let mut backend = FaerSparseLu::new();
        backend.factor(&jac).expect("faer factorization");
        let x = backend.solve(&f0).expect("faer solve");
        let ax = jac.matvec(&x);
        let mut b_norm = 0.0;
        let mut diff_norm = 0.0;
        for (r, b) in ax.iter().zip(&f0) {
            diff_norm += (r - b) * (r - b);
            b_norm += b * b;
        }
        let rel = (diff_norm / b_norm).sqrt();
        eprintln!("faer solve residual ||Ax-b||/||b|| = {rel:.3e}");
        assert!(rel < 1e-10, "faer solve residual too large: {rel:.3e}");
    }

    // The solve: the smoke spec's epsilon = 1e-8 first (the C's own
    // 12-iteration exit path).
    let mut solver = FaerSparseLu::new();
    let t0 = std::time::Instant::now();
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut state,
        &params,
        &NewtonConfig::default(),
    )
    .expect("linear backend ok");
    let elapsed = t0.elapsed();
    for entry in &log {
        eprintln!(
            "iter {:2}: norm_du {:.3e} lambda {:.3e} theta {:.3e} lambda' {:.3e} {}",
            entry.iter,
            entry.norm_du,
            entry.lambda,
            entry.theta,
            entry.lambda_prime,
            entry.event.status_str()
        );
    }
    let iterations = match status {
        NewtonStatus::Converged { iterations, .. } => iterations,
        other => panic!("solve did not converge: {other:?}"),
    };
    eprintln!(
        "converged in {iterations} iterations, {elapsed:?} wall (C oracle: 12 at epsilon=1e-8)"
    );

    // The oracle's iteration path (SAN-52): the REJECT B growth out of
    // lambda0 = 1e-3, the damped first step, the prediction recovery,
    // the QNERR entry, and the CONVERGED C exit after 12 iterations.
    assert_eq!(iterations, 12, "the C oracle exits after 12 iterations");
    let events: Vec<_> = log.iter().map(|r| r.event).collect();
    assert_eq!(
        events[..4],
        [
            NewtonEvent::RejectB,
            NewtonEvent::Accept,
            NewtonEvent::Accept,
            NewtonEvent::EnterQnerr,
        ],
        "events: {events:?}"
    );
    assert_eq!(events.last(), Some(&NewtonEvent::ConvergedC));
    // The oracle's k = 0 row: lambda' = 0.128601 (the mu-prime
    // regularization capping the REJECT B growth).
    assert!(
        (log[0].lambda_prime - 0.128601).abs() < 1e-3 * 0.128601,
        "k=0 lambda' {} vs oracle 0.128601",
        log[0].lambda_prime
    );

    // Field-by-field parity vs the pinned oracle solution.
    let names = [
        "log_alpha_f",
        "beta_f",
        "log_h_f",
        "log_a_f",
        "psi_f",
        "lambda_f",
    ];
    let mut worst_field_rel = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for (k, name) in names.iter().enumerate() {
        let oracle = &fix.fields[k];
        assert_eq!(oracle.len(), dim, "{name} length");
        let mine = state.fields.field(k);
        let scale = oracle.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut max_abs = 0.0_f64;
        let mut worst_point_rel = 0.0_f64;
        for idx in 0..dim {
            let diff = (mine[idx] - oracle[idx]).abs();
            max_abs = max_abs.max(diff);
            worst_point_rel = worst_point_rel.max(diff / oracle[idx].abs().max(1e-12));
        }
        eprintln!(
            "{name:12}: field-rel {:.3e}, max|diff| {:.3e}, per-point rel {:.3e}",
            max_abs / scale,
            max_abs,
            worst_point_rel
        );
        // Relaxed gates for the eps = 1e-8 exit: that exit applies a
        // stale-J CORRECTED step at ||du|| ~ 1e-9 and the alpha
        // correction chain amplifies faer-vs-PARDISO solve rounding
        // to the ~1e-10 level (measured 7.0e-10); the tight pin lives
        // at the eps = 1e-12 exit below.
        assert!(
            max_abs / scale < 1e-8,
            "{name}: field-relative deviation {} exceeds the eps=1e-8 exit gate",
            max_abs / scale
        );
        assert!(
            max_abs < 1e-9,
            "{name}: absolute deviation {max_abs:.3e} exceeds the eps=1e-8 exit gate"
        );
        worst_field_rel = worst_field_rel.max(max_abs / scale);
        worst_abs = worst_abs.max(max_abs);
    }
    eprintln!(
        "eps=1e-8 exit: worst field-rel {worst_field_rel:.3e}, worst abs {worst_abs:.3e} (gate 1e-8 / 1e-9)"
    );

    // Omega is pinned by the constraint row; both solvers carry it
    // exactly.
    let w = rbs_elliptic::omega::omega_calc(state.xi, 1.0);
    assert!((w - fix.omega).abs() < 1e-15, "omega drifted: {w} vs {}", fix.omega);
    let _ = JacobianPattern::build(&grid, &params);

    // The tight solution pin at the deeper exit: epsilon = 1e-12 (the
    // same spec, the QNERR Theta-increase restart the C oracle walks
    // to a CONVERGED B after 15 iterations). The final state there is
    // a fresh-J du_bar update at ||du_bar||/||u|| ~ 2e-14 —
    // machine-level — so the ORIGINAL SAN-62 gates hold with full
    // margin (measured: worst field-rel 2.9e-14, worst abs 1.5e-15).
    let mut state12: SolveState = initial_guess(&grid, &InitialGuess::l1_smoke());
    let t12 = std::time::Instant::now();
    let (status12, log12) = solve(
        &mut FaerSparseLu::new(),
        &grid,
        &mut state12,
        &params,
        &NewtonConfig {
            epsilon: 1e-12,
            ..NewtonConfig::default()
        },
    )
    .expect("linear backend ok");
    let iterations12 = match status12 {
        NewtonStatus::Converged { iterations, .. } => iterations,
        other => panic!("eps=1e-12 solve did not converge: {other:?}"),
    };
    eprintln!(
        "eps=1e-12: converged in {iterations12} iterations, {:?} wall (C oracle: 15, CONVERGED B)",
        t12.elapsed()
    );
    assert_eq!(iterations12, 15, "the C oracle exits after 15 iterations");
    let events12: Vec<_> = log12.iter().map(|r| r.event).collect();
    // 4 NLEQ rows, 11 QNERR rows (the last the Theta-increase EXIT),
    // then the restarted iteration's CONVERGED B.
    assert_eq!(events12.len(), 16, "events: {events12:?}");
    assert_eq!(events12[3], NewtonEvent::EnterQnerr);
    assert_eq!(events12[14], NewtonEvent::ExitQnerr, "the l = 10 restart");
    assert_eq!(events12.last(), Some(&NewtonEvent::ConvergedB));

    let mut worst_field_rel12 = 0.0_f64;
    let mut worst_abs12 = 0.0_f64;
    for (k, name) in names.iter().enumerate() {
        let oracle = &fix.fields[k];
        let mine = state12.fields.field(k);
        let scale = oracle.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut max_abs = 0.0_f64;
        for idx in 0..dim {
            max_abs = max_abs.max((mine[idx] - oracle[idx]).abs());
        }
        eprintln!(
            "{name:12}: eps=1e-12 field-rel {:.3e}, max|diff| {:.3e}",
            max_abs / scale,
            max_abs
        );
        assert!(
            max_abs / scale < 1e-10,
            "{name}: field-relative deviation {} exceeds the tight gate",
            max_abs / scale
        );
        assert!(max_abs < 5e-14, "{name}: absolute deviation {max_abs:.3e}");
        worst_field_rel12 = worst_field_rel12.max(max_abs / scale);
        worst_abs12 = worst_abs12.max(max_abs);
    }
    eprintln!(
        "eps=1e-12 gate: worst field-rel {worst_field_rel12:.3e}, worst abs {worst_abs12:.3e} (gate 1e-10 / 5e-14)"
    );
    let w12 = rbs_elliptic::omega::omega_calc(state12.xi, 1.0);
    assert!(
        (w12 - fix.omega).abs() < 1e-15,
        "omega drifted: {w12} vs {}",
        fix.omega
    );
}
