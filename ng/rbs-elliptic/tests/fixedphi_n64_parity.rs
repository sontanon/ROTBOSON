//! SAN-58 fixedPhi parity: the N=64 l=1 mode vs the C oracle.
//!
//! The fixedPhi spec flips the constraint row of the smoke solve:
//! `fixedPhi = 1, fixedPhiR = 2, fixedPhiZ = 2, fixedOmega = 0` pins
//! psi at raw grid node (2, 2) (r = z = 0.5*dr) at its seed value and
//! makes the frequency the free eigenvalue (`src/csr_omega_constraint.c`:
//! the omega row's single entry is 1.0 at column
//! `4*dim + IDX(fixedPhiR, fixedPhiZ)`; the row's residual is zero, so
//! Newton updates never move the pinned value).
//!
//! The oracle fixture (`tests/fixtures/fixedphi_n64_solution.txt`) is
//! the committed output of the C binary on the canonical from-scratch
//! spec (the smoke grid/ansatz with the fixedPhi constraint: the C
//! walks omega 0.95 -> 0.88325... in ~9 NLEQ-ERR steps; see the fixture
//! header). Two gates live here:
//!
//! * **Plumbing parity** (`..._fixed_point_seeded_...`): seed both
//!   solvers AT the oracle's converged star (the C reads its own
//!   output via `readInitialData = 1`; the Rust loads the same
//!   fixture) and run the fixedPhi solve. The Rust must converge,
//!   carry the frequency at machine precision, preserve the pinned
//!   node exactly, and reproduce every field — proving the whole
//!   fixedPhi pipeline (pattern with the pin column, Jacobian values
//!   with the live omega column, constraint row, Newton exit).
//!
//! * **From-scratch parity** (`..._from_scratch_matches_the_c_oracle`):
//!   start from the smoke ansatz with the frequency guess w0 = 0.95 and
//!   let Newton find the eigenvalue, as the C oracle does (it walks
//!   omega 0.95 -> 0.88325...). The Rust solve must converge to the
//!   oracle's omega at 1e-10 relative and pass the same field gates.
//!   This needs the NLEQ-ERR/QNERR Newton machinery: the simpler
//!   residual-RMS line search cannot accept the eigenvalue-dominated
//!   steps (the damped direction is an ascent direction for that merit
//!   function), so it stalls.
//!
//! Ignored by default because the N=64 sparse LU is slow in debug builds;
//! run the gates with:
//!
//! ```text
//! CARGO_HOME=<cache> cargo test --release -p rbs-elliptic -- --ignored
//! ```

use rbs_core::fields::FieldSet;
use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::{
    assemble_residual, nnz_jacobian_4th, JacobianPattern, ProblemParams, Workspace,
};
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonStatus};
use rbs_linalg::faer_lu::FaerSparseLu;

/// The pinned node, raw grid coordinates (`fixedPhiR`, `fixedPhiZ`):
/// grid node (2, 2) sits at r = z = 0.5*dr.
const PIN: (usize, usize) = (2, 2);

/// The oracle's solver block: `epsilon = 1.0E-12`, `maxNewtonIter = 40`
/// (`lambdaMin = 1.0E-05`).
fn oracle_solver_config() -> NewtonConfig {
    NewtonConfig {
        epsilon: 1e-12,
        max_iterations: 40,
        lambda_min: 1e-5,
        ..NewtonConfig::default()
    }
}

/// One parsed solution fixture: the six field-major grids plus the
/// frequency.
struct Fixture {
    omega: f64,
    fields: Vec<Vec<f64>>,
}

fn load_fixture(rel: &str) -> Fixture {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/");
    let text = std::fs::read_to_string(format!("{path}{rel}")).expect("fixture present");
    let mut fix = Fixture {
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
            Some("nr_total" | "nz_total") | None => {}
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
        }
    }
    if let Some(done) = current.take() {
        fix.fields.push(done);
    }
    fix
}

/// Asserts that every one of the six solved fields matches the oracle
/// fixture: field-relative deviation below 1e-10 and absolute deviation
/// below 5e-14 (the seed-solve gates).
fn assert_fields_match_oracle(fix: &Fixture, fields: &FieldSet, dim: usize) {
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
        let mine = fields.field(k);
        let scale = oracle.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let max_abs = mine
            .iter()
            .zip(oracle)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{name:12}: field-rel {:.3e}, max|diff| {:.3e}", max_abs / scale, max_abs);
        assert!(
            max_abs / scale < 1e-10,
            "{name}: field-relative deviation {} exceeds the gate",
            max_abs / scale
        );
        assert!(max_abs < 5e-14, "{name}: absolute deviation {max_abs:.3e}");
        worst_field_rel = worst_field_rel.max(max_abs / scale);
        worst_abs = worst_abs.max(max_abs);
    }
    eprintln!(
        "gate: worst field-rel {worst_field_rel:.3e}, worst abs {worst_abs:.3e} (gate 1e-10 / 5e-14)"
    );
}

/// The grid and params of the canonical fixedPhi spec (the smoke
/// grid/ansatz with the constraint row flipped).
fn fixedphi_grid() -> Grid2d {
    Grid2d::new(68, 68, 2, 0.25, 0.25)
}

fn fixedphi_params(xi: f64) -> ProblemParams {
    ProblemParams {
        l: 1,
        m: 1.0,
        xi,
        q1: 1.0,
        q2: 1.0,
        fixed_phi: Some(PIN),
    }
}

/// The state whose fields are the oracle fixture's solution — the
/// in-memory image of the C's `readInitialData = 1` seed.
fn oracle_seeded_state(grid: &Grid2d) -> SolveState {
    let seed = load_fixture("tests/fixtures/fixedphi_n64_solution.txt");
    assert_eq!(seed.fields.len(), 6);
    let dim = grid.dim();
    let mut fields = FieldSet::zeros(grid);
    for k in 0..6 {
        assert_eq!(seed.fields[k].len(), dim, "seed field {k} length");
        fields.field_mut(k).copy_from_slice(&seed.fields[k]);
    }
    SolveState {
        fields,
        xi: rbs_elliptic::omega::inverse_omega_calc(seed.omega, 1.0),
    }
}

/// SAN-58 plumbing gate: seeded at the oracle's own converged star,
/// the fixedPhi solve must converge at machine parity — the frequency
/// carried exactly, the pinned node preserved, every field a match.
#[test]
#[ignore = "N=64 sparse solve: ~30 s in --release, minutes in debug. Run with `cargo test --release -p rbs-elliptic -- --ignored`"]
fn n64_l1_fixedphi_fixed_point_seeded_matches_the_c_oracle() {
    let fix = load_fixture("tests/fixtures/fixedphi_n64_solution.txt");
    assert_eq!(fix.fields.len(), 6);

    let grid = fixedphi_grid();
    let dim = grid.dim();
    // The frequency slot starts at the oracle's eigenvalue (w_i =
    // w_f.asc of the seed); with fixed_phi set it is a free unknown.
    let params = fixedphi_params(rbs_elliptic::omega::inverse_omega_calc(fix.omega, 1.0));
    let mut state = oracle_seeded_state(&grid);

    // The pattern with the pin column: total nnz must stay the C's
    // `nnz_jacobian` (the omega row has one entry either way).
    let pattern = JacobianPattern::build(&grid, &params);
    assert_eq!(pattern.nnz(), nnz_jacobian_4th(64, 64));

    // The pin: the seed's psi at (2,2) is the constraint value.
    let pin_seed = state.fields.at(4, grid.idx(PIN.0, PIN.1));
    eprintln!("pin value psi(2,2) = {pin_seed:.17e}");

    // Seed parity: the oracle's converged star is a fixed point, so
    // the residual must sit at the oracle's final level (~1e-16
    // per-field RMS).
    let mut ws = Workspace::new(&grid);
    let f0 = assemble_residual(&grid, &state.fields, &params, &mut ws);
    for k in 0..6 {
        let rms = (f0[k * dim..(k + 1) * dim])
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            / (dim as f64).sqrt();
        eprintln!("initial ||f{k}||: {rms:.10e}");
        assert!(rms < 1e-14, "seed state is not a fixed point: ||f{k}|| = {rms:.3e}");
    }

    // The solve, at the oracle's solver settings.
    let mut solver = FaerSparseLu::new();
    let t0 = std::time::Instant::now();
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut state,
        &params,
        &oracle_solver_config(),
    )
    .expect("linear backend ok");
    let elapsed = t0.elapsed();
    for entry in &log {
        eprintln!(
            "iter {:2}: norm_du {:.3e} lambda {:.3e} theta {:.3e}",
            entry.iter, entry.norm_du, entry.lambda, entry.theta
        );
    }
    let iterations = match status {
        NewtonStatus::Converged { iterations, .. } => iterations,
        other => panic!("fixed-point-seeded solve did not converge: {other:?}"),
    };
    eprintln!("converged in {iterations} iterations, {elapsed:?} wall");

    // The frequency: carried through the free-eigenvalue pipeline at
    // machine precision.
    let w = rbs_elliptic::omega::omega_calc(state.xi, params.m);
    eprintln!(
        "omega: rust {w:.16e} vs oracle {:.16e}, diff {:.3e}",
        fix.omega,
        (w - fix.omega).abs()
    );
    assert!(
        (w - fix.omega).abs() < 1e-12,
        "omega parity failed: rust {w:.16e} vs oracle {:.16e}",
        fix.omega
    );

    // The pin held through the solve (the constraint row forces
    // du_psi(pin) = 0 exactly).
    let pin_solved = state.fields.at(4, grid.idx(PIN.0, PIN.1));
    eprintln!(
        "pin psi(2,2): solved {pin_solved:.17e} vs seed {pin_seed:.17e}, drift {:.3e}",
        (pin_solved - pin_seed).abs()
    );
    assert!(
        (pin_solved - pin_seed).abs() < 1e-15,
        "the pinned node drifted: {pin_solved:.17e} vs {pin_seed:.17e}"
    );

    assert_fields_match_oracle(&fix, &state.fields, dim);
}

/// The from-scratch fixedPhi gate: start from the smoke ansatz with the
/// frequency guess w0 = 0.95, pin psi at (2,2), and let Newton find the
/// eigenvalue. The C oracle walks this spec to a clearly different
/// omega; the Rust solve must land on the same omega and fields.
#[test]
#[ignore = "slow: about 10 s in --release"]
fn n64_l1_fixedphi_from_scratch_matches_the_c_oracle() {
    let fix = load_fixture("tests/fixtures/fixedphi_n64_solution.txt");

    let grid = fixedphi_grid();
    let params = fixedphi_params(rbs_elliptic::omega::inverse_omega_calc(0.95, 1.0));
    let mut state: SolveState = initial_guess(&grid, &InitialGuess::l1_smoke());

    // The oracle's eigenvalue really moved away from the guess, so this
    // test exercises the omega walk and not a fixed point.
    assert!(
        (fix.omega - 0.95).abs() > 1e-2,
        "the oracle fixture must hold the moved eigenvalue, got {}",
        fix.omega
    );

    let mut solver = FaerSparseLu::new();
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut state,
        &params,
        &oracle_solver_config(),
    )
    .expect("linear backend ok");
    for entry in &log {
        eprintln!(
            "iter {:2}: norm_du {:.3e} lambda {:.3e} theta {:.3e}",
            entry.iter, entry.norm_du, entry.lambda, entry.theta
        );
    }
    assert!(
        matches!(status, NewtonStatus::Converged { .. }),
        "Newton did not converge: {status:?}"
    );

    let w = rbs_elliptic::omega::omega_calc(state.xi, params.m);
    let rel = ((w - fix.omega) / fix.omega).abs();
    eprintln!("omega: rust {w:.16e} vs oracle {:.16e}, rel {rel:.3e}", fix.omega);
    assert!(rel < 1e-10, "omega parity failed: rel {rel:.3e}");

    assert_fields_match_oracle(&fix, &state.fields, grid.dim());
}
