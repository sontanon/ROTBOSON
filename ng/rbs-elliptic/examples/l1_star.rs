//! Solve an l=1 rotating boson star from scratch, entirely in Rust, and
//! write it out for inspection. This is the milestone A0 demo.
//!
//! The spec is the canonical fixedPhi smoke problem: l = 1, m = 1, a
//! Gaussian ansatz with psi0 = 0.01 and w0 = 0.95 as the frequency guess,
//! and the scalar field pinned at grid node (2, 2). The frequency is the
//! free eigenvalue, so the solve has to walk omega away from the guess.
//!
//! ```sh
//! cd ng && cargo run --release -p rbs-elliptic --example l1_star -- ../out/demo/a0/rust
//! cd ng && cargo run --release -p rbs-elliptic --example l1_star -- ../out/demo/a0/rust --backend lu
//! ```
//!
//! It writes into the output directory:
//!
//! - `solution.vtr`: the six fields on the (r, z) grid; open it in `ParaView`.
//! - `solution.txt`: the same fields as plain text, in the format of
//!   `tools/export_nearfold_states.py`, for Python plotting and comparison.
//!
//! Backends: `krylov` (default) is the block-field Krylov solver the plan
//! builds on; `lu` is faer's sparse LU, the direct reference.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rbs_core::fields::NUM_FIELDS;
use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::ProblemParams;
use rbs_elliptic::initial::{initial_guess, InitialGuess, SolveState};
use rbs_elliptic::newton::{solve, NewtonConfig, NewtonLog, NewtonStatus};
use rbs_elliptic::omega::{inverse_omega_calc, omega_calc};
use rbs_io::vtr::VtrWriter;
use rbs_io::{FieldBlock, FieldWriter, GridTopology};
use rbs_linalg::faer_lu::FaerSparseLu;
use rbs_linalg::{BlockFieldKrylov, BlockKrylovConfig, SolveError};

/// Field names, in the solver's field order (`rbs_core::fields`).
const FIELD_NAMES: [&str; NUM_FIELDS] = ["log_alpha", "beta", "log_h", "log_a", "psi", "lambda"];

/// Interior points per direction and the grid step (domain edge at 16).
const N_INTERIOR: usize = 64;
const GHOST: usize = 2;
const STEP: f64 = 0.25;

/// The node where psi is pinned (`fixedPhiR`, `fixedPhiZ`).
const PIN: (usize, usize) = (2, 2);

/// Initial frequency guess and scalar-field mass.
const OMEGA_GUESS: f64 = 0.95;
const MASS: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Krylov,
    Lu,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let (out_dir, backend) = parse_args()?;
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let n_total = N_INTERIOR + 2 * GHOST;
    let grid = Grid2d::new(n_total, n_total, GHOST, STEP, STEP);
    let params = ProblemParams {
        l: 1,
        m: MASS,
        xi: inverse_omega_calc(OMEGA_GUESS, MASS),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: Some(PIN),
    };
    let config = NewtonConfig {
        epsilon: 1e-12,
        max_iterations: 40,
        lambda_min: 1e-5,
        ..NewtonConfig::default()
    };
    let mut state = initial_guess(&grid, &InitialGuess::l1_smoke());

    println!("l=1 star, N={N_INTERIOR}, dr=dz={STEP}, backend {backend:?}");
    let started = Instant::now();
    let (status, log) = match backend {
        Backend::Krylov => {
            let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
                // Relative to the Newton residual, so the error in each
                // step shrinks with the step; 1e-13 stagnates near 7e-12.
                tolerance: 1e-10,
                ..BlockKrylovConfig::default()
            });
            solve(&mut solver, &grid, &mut state, &params, &config)
        }
        Backend::Lu => {
            let mut solver = FaerSparseLu::new();
            solve(&mut solver, &grid, &mut state, &params, &config)
        }
    }
    .map_err(|e: SolveError| format!("linear solver failed: {e}"))?;
    let seconds = started.elapsed().as_secs_f64();

    print_log(&log);
    if !matches!(status, NewtonStatus::Converged { .. }) {
        return Err(format!("Newton did not converge: {status:?}"));
    }
    let omega = omega_calc(state.xi, MASS);
    let iterations = log.last().map_or(0, |entry| entry.iter + 1);
    println!("converged in {iterations} Newton iterations, {seconds:.1} s");
    println!("omega = {omega:.15}");

    write_vtr(&out_dir.join("solution.vtr"), &grid, &state)?;
    write_text(&out_dir.join("solution.txt"), &grid, &state, omega)?;
    Ok(())
}

/// Read `<out-dir> [--backend krylov|lu]` from the command line.
fn parse_args() -> Result<(PathBuf, Backend), String> {
    let usage = "usage: l1_star <out-dir> [--backend krylov|lu]";
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().ok_or(usage)?;
    let backend = match (args.next().as_deref(), args.next().as_deref()) {
        (None, None) | (Some("--backend"), Some("krylov")) => Backend::Krylov,
        (Some("--backend"), Some("lu")) => Backend::Lu,
        _ => return Err(usage.to_string()),
    };
    Ok((PathBuf::from(out_dir), backend))
}

fn print_log(log: &[NewtonLog]) {
    for entry in log {
        println!(
            "  iter {:2}: |du| {:.3e}  {:?}",
            entry.iter, entry.norm_du, entry.event
        );
    }
}

fn field_blocks(state: &SolveState) -> Vec<FieldBlock> {
    FIELD_NAMES
        .iter()
        .enumerate()
        .map(|(k, name)| FieldBlock {
            name: (*name).to_string(),
            values: state.fields.field(k).to_vec(),
        })
        .collect()
}

fn write_vtr(path: &Path, grid: &Grid2d, state: &SolveState) -> Result<(), String> {
    let topology = GridTopology::centered(
        grid.nr_total(),
        grid.nz_total(),
        grid.dr(),
        grid.dz(),
        grid.ghost(),
    );
    VtrWriter::new(path)
        .write_snapshot(&topology, &field_blocks(state))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Write the text format of `tools/export_nearfold_states.py`: a header
/// of `key value` lines, then `field <name>_f` and one value per line,
/// row-major over the full grid including ghosts. `{:?}` prints the
/// shortest string that reads back to the same f64.
fn write_text(path: &Path, grid: &Grid2d, state: &SolveState, omega: f64) -> Result<(), String> {
    let mut text = String::new();
    let header = [
        "# written by ng/rbs-elliptic/examples/l1_star.rs".to_string(),
        format!("nr_total {}", grid.nr_total()),
        format!("nz_total {}", grid.nz_total()),
        format!("ghost {}", grid.ghost()),
        format!("dr {:?}", grid.dr()),
        format!("dz {:?}", grid.dz()),
        "l 1".to_string(),
        format!("m {MASS:?}"),
        format!("pin_r {}", PIN.0),
        format!("pin_z {}", PIN.1),
        format!("psi0 {:?}", state.fields.at(4, grid.idx(PIN.0, PIN.1))),
        format!("omega {omega:?}"),
    ];
    for line in header {
        text.push_str(&line);
        text.push('\n');
    }
    for (k, name) in FIELD_NAMES.iter().enumerate() {
        let _ = writeln!(text, "field {name}_f");
        for value in state.fields.field(k) {
            let _ = writeln!(text, "{value:?}");
        }
    }
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}
