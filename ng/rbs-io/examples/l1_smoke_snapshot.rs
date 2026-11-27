//! Write the l=1 N=64 smoke snapshot (solution.h5 + solution.vtr) from
//! the vertical slice's parity fixture — the six-field solve state
//! already proven at C-oracle parity (SAN-62).
//!
//! This is the NG output layer's reference invocation (the shape the
//! SAN-65 campaign loop writes per step) and the generator of
//! `tests/fixtures/l1_smoke_n64_rust.h5`:
//!
//! ```sh
//! cargo run -p rbs-io --features hdf5 --example l1_smoke_snapshot -- <out-dir>
//! ```
//!
//! Provenance: `git_hash` comes from `$GIT_HASH` (fallback "unknown",
//! like the C binary's build default); set it when regenerating the
//! committed fixture:
//!
//! ```sh
//! GIT_HASH=$(git rev-parse --short HEAD) \
//!   cargo run -p rbs-io --features hdf5 --example l1_smoke_snapshot -- tests/fixtures
//! ```
//!
//! The HDF5 path needs the `hdf5` feature (native libhdf5); without it
//! only the VTR half runs.

// Shared with the integration tests so the fixture parsing has exactly
// one implementation.
#[path = "../tests/common/mod.rs"]
mod common;

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "hdf5")]
use rbs_io::hdf5::Hdf5Writer;
use rbs_io::vtr::VtrWriter;
use rbs_io::FieldWriter;

fn main() -> ExitCode {
    let out_dir = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {e}", out_dir.display());
        return ExitCode::FAILURE;
    }

    let snap = common::load_smoke();

    // solution.vtr — ParaView rectilinear grid, always available.
    let vtr_path = out_dir.join("solution.vtr");
    let mut vtr = VtrWriter::new(&vtr_path);
    match vtr.write_snapshot(&snap.grid, &snap.fields) {
        Ok(()) => println!("wrote {}", vtr_path.display()),
        Err(e) => {
            eprintln!("VTR write failed: {e}");
            return ExitCode::FAILURE;
        }
    }

    // solution.h5 — the C-oracle HDF5 contract, behind the `hdf5` feature.
    #[cfg(feature = "hdf5")]
    {
        let git_hash =
            std::env::var("GIT_HASH").unwrap_or_else(|_| "unknown".to_string());
        let parfile = "configs/l1_smoke_ci.toml (epsilon=1.0E-12, maxNewtonIter=40 \
                       scratch protocol; fields: ng/rbs-elliptic parity fixture)"
            .to_string();
        let attrs = common::smoke_attrs(&git_hash, &parfile);
        let h5_path = out_dir.join("solution.h5");
        let mut h5 = Hdf5Writer::new(&h5_path);
        let w_f = snap.w_f();
        match h5.write_snapshot_with(&snap.grid, &snap.fields, &[w_f], &[], &attrs) {
            Ok(()) => println!("wrote {}", h5_path.display()),
            Err(e) => {
                eprintln!("HDF5 write failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    #[cfg(not(feature = "hdf5"))]
    eprintln!("note: built without the `hdf5` feature; skipped solution.h5");

    ExitCode::SUCCESS
}
