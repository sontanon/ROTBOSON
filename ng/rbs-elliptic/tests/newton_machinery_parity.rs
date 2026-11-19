//! SAN-52 machinery-parity gates: the Rust Newton drivers' iteration
//! paths vs the C oracle's, pinned from captured oracle logs.
//!
//! Every constant below is a row of the C binary's own monitoring
//! table (build/release/ROTBOSON, the configs listed per test; run
//! `../build/release/ROTBOSON <config>` and read the `***** | ...`
//! rows to regenerate).  The gates assert the qualitative path —
//! WHERE the C damps, at which λ, and where it takes full steps —
//! with iteration counts exact and the numeric columns loose enough
//! for faer-vs-PARDISO rounding (the assembly itself is bit-parity
//! per the SAN-62 gate, so the margins are all backend noise).
//!
//! Ignored by default (sparse LU sizes); run with:
//!
//! ```text
//! CARGO_HOME=<cache> cargo test --release -p rbs-elliptic -- --ignored
//! ```

use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::ProblemParams;
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonEvent, NewtonKind, NewtonStatus};
use rbs_linalg::faer_lu::FaerSparseLu;

/// The l=1 smoke family's physical + solver block (w = 0.95, fixed
/// omega; solverType/localSolver/lambda0/lambdaMin per the configs).
fn smoke_params() -> ProblemParams {
    ProblemParams {
        l: 1,
        m: 1.0,
        xi: rbs_elliptic::omega::inverse_omega_calc(0.95, 1.0),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: None,
    }
}

/// The event sequence of a solve's log.
fn events(log: &[rbs_elliptic::newton::NewtonLog]) -> Vec<NewtonEvent> {
    log.iter().map(|r| r.event).collect()
}

/// Relative agreement of two nonzero table values.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-300)
}

/// N = 32, dr = dz = 0.5 (domain ~16 like the smoke family): the
/// fast grid for the three-driver path comparison.
fn n32_setup() -> (Grid2d, SolveState) {
    let grid = Grid2d::new(36, 36, 2, 0.5, 0.5);
    let state = initial_guess(&grid, &InitialGuess::l1_smoke());
    (grid, state)
}

/// NLEQ-ERR + QNERR on the N=32 seed spec vs the C oracle
/// (`l1_smoke_ci.toml` at NrInterior = 32, dr = 0.5):
///
/// ```text
/// | 0 | 6.31924E+01 | 1.00000E-03 | 9.98996E-01 | 1.28233E-01 | REJECT B    |
/// | 0 | 7.73848E+00 | 1.28233E-01 | 8.04358E-01 | 1.21907E-01 | ACCEPT      |
/// | 1 | 1.07677E+00 | 6.59216E-01 | 2.24841E-01 | 1.00000E+00 | ACCEPT      |
/// | 2 | 5.77280E-02 | 1.00000E+00 | 2.43639E-01 | 1.00000E+00 | ENTER QNERR|
/// (9 QNERR rows) ... CONVERGED C — converged after 12 iterations
/// ```
#[test]
#[ignore = "N=32 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n32_nleq_err_matches_the_oracle_path() {
    let (grid, mut state) = n32_setup();
    let params = smoke_params();
    let mut solver = FaerSparseLu::new();
    let (status, log) =
        solve(&mut solver, &grid, &mut state, &params, &NewtonConfig::default()).unwrap();

    let NewtonStatus::Converged { iterations, .. } = status else {
        panic!("did not converge: {status:?}");
    };
    assert_eq!(iterations, 12, "oracle: 12 iterations");

    let events = events(&log);
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
    assert_eq!(events.len(), 4 + 9, "4 NLEQ rows + 9 QNERR rows");

    // The k = 0 REJECT B: lambda0 = 1e-3, the mu-prime regularization
    // caps the growth at lambda' = 0.128233.
    assert!(close(log[0].lambda, 1e-3, 1e-9), "lambda {}", log[0].lambda);
    assert!(close(log[0].theta, 9.98996e-1, 1e-3), "theta {}", log[0].theta);
    assert!(
        close(log[0].lambda_prime, 1.28233e-1, 1e-3),
        "lambda' {} vs oracle 0.128233",
        log[0].lambda_prime
    );
    // The accepted damping of iteration 0.
    assert!(close(log[1].lambda, 1.28233e-1, 1e-3), "{}", log[1].lambda);
    // The k = 1 prediction recovers geometrically (oracle 0.659216).
    assert!(close(log[2].lambda, 6.59216e-1, 1e-3), "{}", log[2].lambda);
    // Full steps from k = 2 on, and the QNERR tail's contraction
    // monitor below 1/2.
    assert!(close(log[3].lambda, 1.0, 1e-12), "{}", log[3].lambda);
    assert!(
        log[4..].iter().all(|r| r.theta < 0.5),
        "a QNERR row exceeded Theta = 1/2"
    );
}

/// NLEQ-RES + QNRES on the same spec vs the C oracle
/// (solverType = 2):
///
/// ```text
/// | 0 | 9.90834E-04 | 1.00000E-03 | 2.29202E-01 | 1.00000E+00 | REJECT B    |
/// | 0 | 9.90834E-04 | 1.00000E+00 | 1.48584E-01 | 1.00000E+00 | ENTER QNRES |
/// | 0 | 1.13952E-04 | 1.45513E-08 | 7.74015E-01 | -1.82471E+00 | EXIT QNRES  |
/// | 2 | 1.13952E-04 | 2.50000E-01 | 8.45747E-01 | 1.71980E-01 | ACCEPT      |
/// | 3..5 ACCEPT | | 6 ENTER QNRES | (3 rows) CONVERGED C — after 10
/// ```
///
/// Note the k = 0 `||f[k]||` column: 9.90834E-04 is the initial
/// GUESS's interior norm — the C driver's `main.c:184` quirk this
/// port reproduces (the loop's own monitor would be ~5e-5).
#[test]
#[ignore = "N=32 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n32_nleq_res_matches_the_oracle_path() {
    let (grid, mut state) = n32_setup();
    let params = smoke_params();
    let config = NewtonConfig {
        kind: NewtonKind::Residual,
        ..NewtonConfig::default()
    };
    let mut solver = FaerSparseLu::new();
    let (status, log) = solve(&mut solver, &grid, &mut state, &params, &config).unwrap();

    let NewtonStatus::Converged { iterations, .. } = status else {
        panic!("did not converge: {status:?}");
    };
    assert_eq!(iterations, 10, "oracle: 10 iterations");

    let events = events(&log);
    assert_eq!(
        events[..3],
        [
            NewtonEvent::RejectB,
            NewtonEvent::EnterQnres,
            NewtonEvent::ExitQnres,
        ],
        "events: {events:?}"
    );
    assert_eq!(events.last(), Some(&NewtonEvent::ConvergedC));

    // The main.c:184 quirk, reproduced: the k = 0 row's norm column
    // is the interior norm of the initial GUESS.
    assert!(
        close(log[0].norm_du, 9.90834e-4, 1e-4),
        "norm_f0 {} vs oracle 9.90834e-4",
        log[0].norm_du
    );
    // REJECT B straight to lambda = 1, then the QNRES exit restarts
    // at the quarter damping.
    assert!(close(log[1].lambda, 1.0, 1e-12));
    assert!(close(log[3].lambda, 0.25, 1e-9), "{}", log[3].lambda);
    // The QNRES rows' kappa monitor (the C's -1.82471 at the exit).
    assert!(
        close(log[2].lambda_prime, -1.82471, 1e-3),
        "kappa {} vs oracle -1.82471",
        log[2].lambda_prime
    );
}

/// The classic fixed-damping loop (solverType = 3) on the same spec:
/// the oracle prints 20 ACCEPT rows at lambda = 1e-3 with
/// Theta ~= 0.999 and fails on the iteration budget.
#[test]
#[ignore = "N=32 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n32_classic_newton_matches_the_oracle_path() {
    let (grid, mut state) = n32_setup();
    let params = smoke_params();
    let config = NewtonConfig {
        kind: NewtonKind::Classic,
        ..NewtonConfig::default()
    };
    let mut solver = FaerSparseLu::new();
    let (status, log) = solve(&mut solver, &grid, &mut state, &params, &config).unwrap();

    assert_eq!(status, NewtonStatus::MaxIterations, "{status:?}");
    assert_eq!(log.len(), 20, "the oracle prints 20 rows");
    assert!(log.iter().all(|r| r.event == NewtonEvent::Accept));
    assert!(log.iter().all(|r| close(r.lambda, 1e-3, 1e-9)));
    assert!(
        log.iter().all(|r| close(r.theta, 0.999, 5e-3)),
        "theta drifted from the oracle's ~0.999"
    );
}

/// All three drivers reach the same fixed point (the residual and
/// error exits both sit within the quadratic tail of the solution),
/// and the pinned omega is carried exactly.
#[test]
#[ignore = "N=32 sparse solves; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n32_all_drivers_agree_on_the_solution() {
    let (grid, _) = n32_setup();
    let params = smoke_params();
    let mut reference = None;
    for (name, kind) in [
        ("error", NewtonKind::Error),
        ("residual", NewtonKind::Residual),
    ] {
        let mut state = initial_guess(&grid, &InitialGuess::l1_smoke());
        let config = NewtonConfig {
            kind,
            ..NewtonConfig::default()
        };
        let mut solver = FaerSparseLu::new();
        let (status, _log) = solve(&mut solver, &grid, &mut state, &params, &config).unwrap();
        assert!(
            matches!(status, NewtonStatus::Converged { .. }),
            "{name}: {status:?}"
        );
        let w = rbs_elliptic::omega::omega_calc(state.xi, params.m);
        assert!(
            (w - 0.95).abs() < 1e-12,
            "{name}: omega drifted: {w:.15} vs 0.95"
        );
        match reference {
            None => reference = Some(state),
            Some(ref r) => {
                for k in 0..6 {
                    let a = state.fields.field(k);
                    let b = r.fields.field(k);
                    let max_diff = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0_f64, f64::max);
                    if k == 0 {
                        // Log alpha is the monitoring norm's BLIND SPOT
                        // (the C's `norm2_all_variables` covers fields
                        // 1..=5 only): the residual driver exits ~1e-8
                        // from the fixed point in this direction, so the
                        // two drivers' log-alpha fields legitimately
                        // differ at that scale — the C's own
                        // solverType=1/2 outputs show the same gap.
                        assert!(
                            max_diff < 1e-5,
                            "{name}: field 0 (log alpha, norm-blind) differs by {max_diff:.3e}"
                        );
                    } else {
                        // The residual driver exits within ~epsilon of
                        // the fixed point (its test is ||f|| < 1e-8);
                        // the error driver rides the quadratic tail
                        // to ~1e-14 — the two solutions agree to the
                        // residual exit distance in ABSOLUTE terms
                        // (the small-magnitude fields would inflate a
                        // relative measure).
                        assert!(
                            max_diff < 1e-6,
                            "{name}: field {k} differs by {max_diff:.3e}"
                        );
                    }
                }
            }
        }
    }
}

/// The epsilon = 1e-12 restart path on the N=64 smoke spec vs the
/// C oracle (`l1_smoke_ci.toml` with epsilon = 1e-12): the QNERR
/// tail's contraction monitor exceeds 1/2 at l = 10 (Θ = 0.532525),
/// the tail EXITS, NLEQ-ERR RESTARTS from its last update at k = 14
/// with lambda = 1, and the restarted iteration converges on the
/// du_bar test (CONVERGED B, 15 iterations total):
///
/// ```text
/// | 0 REJECT B | 0 ACCEPT | 1 ACCEPT | 2 ENTER QNERR |
/// (QNERR l = 0..9 ACCEPT, | 10 | ... | 5.32525E-01 | EXIT QNERR  |)
/// | 14 | 2.09587E-14 | 1.00000E+00 | 9.24710E-05 | 1.00000E+00 | CONVERGED B |
/// converged after 15 iterations
/// ```
#[test]
#[ignore = "N=64 sparse solve with the restart tail; run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n64_eps12_restart_path_matches_the_oracle() {
    let grid = Grid2d::new(68, 68, 2, 0.25, 0.25);
    let params = smoke_params();
    let mut state = initial_guess(&grid, &InitialGuess::l1_smoke());
    let config = NewtonConfig {
        epsilon: 1e-12,
        ..NewtonConfig::default()
    };
    let mut solver = FaerSparseLu::new();
    let (status, log) = solve(&mut solver, &grid, &mut state, &params, &config).unwrap();

    let NewtonStatus::Converged { iterations, .. } = status else {
        panic!("did not converge: {status:?}");
    };
    assert_eq!(iterations, 15, "oracle: 15 iterations");

    let events = events(&log);
    // 4 NLEQ rows + 11 QNERR rows (l = 0..10, the last the exit) +
    // the restarted k = 14 row.
    assert_eq!(events.len(), 4 + 11 + 1, "events: {events:?}");
    assert_eq!(events[3], NewtonEvent::EnterQnerr);
    assert_eq!(events[14], NewtonEvent::ExitQnerr, "the l = 10 exit");
    assert_eq!(events.last(), Some(&NewtonEvent::ConvergedB));
    // The exit's Theta: 0.532525 > 1/2 (the restart trigger).
    assert!(
        close(log[14].theta, 0.532525, 1e-3),
        "exit theta {} vs oracle 0.532525",
        log[14].theta
    );
    // The restarted iteration: full damping, CONVERGED B on du_bar.
    // The deep-tail norm is backend-noise dominated at the 1e-14
    // level (observed 1.5e-14 vs the oracle's 2.1e-14 — both exits
    // are O(||du_bar||^2) from the fixed point), so gate the band,
    // not the digit.
    let last = log.last().expect("rows");
    assert!(close(last.lambda, 1.0, 1e-12));
    assert!(
        (1e-16..1e-12).contains(&last.norm_du),
        "converged-B norm {} outside the oracle band (2.09587e-14)",
        last.norm_du
    );
}
