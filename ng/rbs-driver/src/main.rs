//! Campaign entry point.
//!
//! SAN-62 status: the l=1 fixed-omega vertical slice (the smoke spec of
//! `configs/l1_smoke_ci.toml`) solves end-to-end in pure Rust and is
//! parity-pinned against the C oracle by
//! `rbs-elliptic/tests/l1_smoke_n64_parity.rs`.  This binary runs that
//! slice and prints the Newton monitoring table; TOML campaign parsing,
//! continuation and output staging are SAN-65.

use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::ProblemParams;
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonStatus};
use rbs_linalg::faer_lu::FaerSparseLu;

fn main() {
    // The vertical-slice spec: l=1, w=0.95, N=64, dr=dz=0.25, order 4
    // (configs/l1_smoke_ci.toml).
    let n = 64;
    let grid = Grid2d::new(n + 4, n + 4, 2, 0.25, 0.25);
    let params = ProblemParams {
        l: 1,
        m: 1.0,
        xi: rbs_elliptic::omega::inverse_omega_calc(0.95, 1.0),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: None,
    };
    let mut state: SolveState = initial_guess(&grid, &InitialGuess::l1_smoke());

    println!("rbs-driver: SAN-62 vertical slice (l=1, w=0.95, N={n})");
    let mut solver = FaerSparseLu::new();
    let t0 = std::time::Instant::now();
    let (status, log) = solve(
        &mut solver,
        &grid,
        &mut state,
        &params,
        &NewtonConfig::default(),
    )
    .unwrap_or_else(|err| {
        eprintln!("rbs-driver: linear backend failure: {err}");
        std::process::exit(70);
    });
    for entry in &log {
        println!(
            "iter {:2}: norm_du {:.3e} lambda {:.3e} theta {:.3e}",
            entry.iter, entry.norm_du, entry.lambda, entry.theta
        );
    }
    match status {
        NewtonStatus::Converged { iterations, .. } => {
            let w = rbs_elliptic::omega::omega_calc(state.xi, params.m);
            println!(
                "rbs-driver: converged in {iterations} iterations, {:.2?} wall, omega = {w:.12}",
                t0.elapsed()
            );
        }
        other => {
            eprintln!("rbs-driver: solve failed: {other:?}");
            std::process::exit(1);
        }
    }
}
