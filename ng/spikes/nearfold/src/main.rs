//! Near-fold solver spike (NR-11): does the block-field Krylov solver
//! still work near the l=1 `omega_min` fold? See `README.md` and
//! `docs/specs/A0.5-near-fold-spike.md`.

mod experiments;
mod state;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use faer::Par;
use rbs_linalg::Preconditioner;

/// Threads for faer's block factorizations and solves.
const THREADS: usize = 8;

const USAGE: &str = "usage: nearfold-spike <linear|stale|newton|eig> --manifest <manifest.csv> --out <results.csv> [--preconditioner jacobi|bordered|gs|all] [--n <N>]";

struct Args {
    command: String,
    manifest: PathBuf,
    out: PathBuf,
    preconditioners: Vec<Preconditioner>,
    only_n: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or(USAGE)?;
    let (mut manifest, mut out, mut preconditioner, mut only_n) =
        (None, None, "all".to_string(), None);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--preconditioner" => preconditioner = value,
            "--n" => only_n = Some(value.parse().map_err(|_| format!("bad --n {value}"))?),
            _ => return Err(format!("unknown option {flag}\n{USAGE}")),
        }
    }
    let preconditioners = match preconditioner.as_str() {
        "jacobi" => vec![Preconditioner::BlockJacobi],
        "bordered" => vec![Preconditioner::BorderedBlockJacobi],
        "gs" => vec![Preconditioner::BlockGaussSeidel],
        "all" => vec![
            Preconditioner::BlockJacobi,
            Preconditioner::BorderedBlockJacobi,
            Preconditioner::BlockGaussSeidel,
        ],
        other => return Err(format!("unknown preconditioner {other}\n{USAGE}")),
    };
    Ok(Args {
        command,
        manifest: manifest.ok_or(USAGE)?,
        out: out.ok_or(USAGE)?,
        preconditioners,
        only_n,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    faer::set_global_parallelism(Par::rayon(THREADS));

    let dir = args.manifest.parent().unwrap_or(Path::new("."));
    let rows = state::read_manifest(&args.manifest).map_err(|e| e.to_string())?;
    let mut states = Vec::new();
    for row in rows.iter().filter(|r| args.only_n.is_none_or(|n| r.n == n)) {
        let s = state::load(dir, row).map_err(|e| e.to_string())?;
        eprintln!(
            "loaded {} (N={}, psi0={:.5}, dist={:+.5}): self-check residual {:.2e}",
            row.file, row.n, row.psi0, row.dist, s.residual
        );
        states.push(s);
    }
    if states.len() < 2 {
        return Err("need at least two states".to_string());
    }

    let lines = match args.command.as_str() {
        "linear" => experiments::linear(&states, &args.preconditioners, THREADS),
        "stale" => experiments::stale(&states, &args.preconditioners, THREADS),
        "newton" => experiments::newton(&states, &args.preconditioners, THREADS),
        "eig" => experiments::eig(&states, THREADS),
        other => return Err(format!("unknown experiment {other}\n{USAGE}")),
    };
    let mut text = lines.join("\n");
    text.push('\n');
    std::fs::write(&args.out, text)
        .map_err(|e| format!("cannot write {}: {e}", args.out.display()))?;
    println!("wrote {} ({} rows)", args.out.display(), lines.len() - 1);
    Ok(())
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
