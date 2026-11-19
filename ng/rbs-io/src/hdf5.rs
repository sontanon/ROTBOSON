//! HDF5 output — the oracle's format contract, feature `hdf5`.
//!
//! The C writer (`src/hdf5_writer.c`) produces one self-describing
//! `solution.h5` per solve: field datasets named `<field>.asc`
//! (2D dims `(nr, nz)` in C order — row-major over `IDX(i, j)`), the
//! coordinate arrays as 1D `.asc` datasets, and scalar parameters as root
//! attributes (`format_version = 1`, GRID block: `dr`, `dz`, `NrTotal`,
//! `NzTotal`, `ghost`, `dim`, SCALAR FIELD block: `l`, `m`, ...). The Rust
//! writer mirrors that layout so `tools/rotboson_io.py` reads Rust output
//! unchanged; the full C attribute set lands with the driver port
//! (`rbs-driver`), which owns the parameter source of truth.
//!
//! The feature needs a native libhdf5 (via hdf5-sys). It is OFF in CI by
//! design (ADR 0001 keeps runners dependency-light; the C CI already pins
//! the format contract). The dependency is the maintained
//! `hdf5-metno` fork (imported as `hdf5`): the original `hdf5` 0.8.1
//! crate rejects modern system `HDF5` releases (its build script fails on
//! 1.14.5 with "Invalid `H5_VERSION`").

use std::io;
use std::path::{Path, PathBuf};

use crate::{FieldBlock, FieldWriter, GridTopology};

/// Dataset name of a field, mirroring the C convention (`"<name>.asc"`).
#[must_use]
pub fn dataset_name(field: &str) -> String {
    format!("{field}.asc")
}

/// Root-group attributes written by every snapshot (the stable subset of
/// the C contract; see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotAttrs {
    /// `format_version` (the C writer pins 1).
    pub format_version: i64,
    /// `output_backend`.
    pub backend: String,
    /// `l` (azimuthal number).
    pub l: i64,
    /// `m` (scalar-field mass).
    pub m: f64,
}

impl Default for SnapshotAttrs {
    fn default() -> Self {
        Self {
            format_version: 1,
            backend: "hdf5-rust".to_string(),
            l: 0,
            m: 0.0,
        }
    }
}

/// HDF5 file writer: one `.h5` file per snapshot.
#[derive(Debug)]
pub struct Hdf5Writer {
    path: PathBuf,
}

impl Hdf5Writer {
    /// Write snapshots to `path` (replacing any existing file, like the C
    /// backend's idempotent re-run semantics).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The target file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl FieldWriter for Hdf5Writer {
    fn write_snapshot(
        &mut self,
        grid: &GridTopology,
        fields: &[FieldBlock],
    ) -> io::Result<()> {
        self.write_snapshot_with(grid, fields, &SnapshotAttrs::default())
    }
}

impl Hdf5Writer {
    /// Write a snapshot with explicit root attributes.
    ///
    /// # Errors
    ///
    /// IO/HDF5 failures, or — when built without the `hdf5` feature — the
    /// disabled-backend error.
    pub fn write_snapshot_with(
        &mut self,
        grid: &GridTopology,
        fields: &[FieldBlock],
        attrs: &SnapshotAttrs,
    ) -> io::Result<()> {
        #[cfg(feature = "hdf5")]
        return write_hdf5(&self.path, grid, fields, attrs)
            .map_err(hdf5_to_io);
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (grid, fields, attrs);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "rbs-io was built without the `hdf5` feature; \
                 rebuild with: cargo build -p rbs-io --features hdf5",
            ))
        }
    }
}

#[cfg(feature = "hdf5")]
fn hdf5_to_io(e: hdf5::Error) -> io::Error {
    io::Error::other(e)
}

#[cfg(feature = "hdf5")]
fn write_hdf5(
    path: &Path,
    grid: &GridTopology,
    fields: &[FieldBlock],
    attrs: &SnapshotAttrs,
) -> hdf5::Result<()> {
    let file = hdf5::File::create(path)?;

    // Coordinates as 1D `.asc` datasets, like the C writer's grid arrays.
    file.new_dataset::<f64>()
        .shape([grid.nr])
        .create("r.asc")?
        .write_raw(&grid.r)?;
    file.new_dataset::<f64>()
        .shape([grid.nz])
        .create("z.asc")?
        .write_raw(&grid.z)?;

    // Fields as 2D `(nr, nz)` C-order datasets — row-major over
    // IDX(i, j), the exact layout tools/rotboson_io.py expects.
    for f in fields {
        debug_assert_eq!(f.values.len(), grid.nr * grid.nz);
        let name = dataset_name(&f.name);
        file.new_dataset::<f64>()
            .shape([grid.nr, grid.nz])
            .create(name.as_str())?
            .write_raw(&f.values)?;
    }

    // Root attributes: the stable subset of the C contract.
    file.new_attr::<i64>()
        .create("format_version")?
        .write_scalar(&attrs.format_version)?;
    file.new_attr::<i64>().create("l")?.write_scalar(&attrs.l)?;
    file.new_attr::<f64>().create("m")?.write_scalar(&attrs.m)?;
    let backend = attrs
        .backend
        .parse::<hdf5::types::VarLenUnicode>()
        .map_err(|e| hdf5::Error::Internal(e.to_string()))?;
    file.new_attr::<hdf5::types::VarLenUnicode>()
        .create("output_backend")?
        .write_scalar(&backend)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "hdf5")]
    fn hdf5_snapshot_roundtrips() {
        let dir = std::env::temp_dir().join(format!("rbs-io-h5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let grid = GridTopology::uniform(4, 3, 2.0, 1.0);
        let fields = vec![FieldBlock {
            name: "psi".to_string(),
            values: (0..12).map(f64::from).collect(),
        }];
        let path = dir.join("solution.h5");
        let mut w = Hdf5Writer::new(&path);
        let attrs = SnapshotAttrs {
            l: 1,
            m: 1.0,
            ..SnapshotAttrs::default()
        };
        w.write_snapshot_with(&grid, &fields, &attrs).expect("write");

        let file = hdf5::File::open(&path).expect("reopen");
        let psi = file.dataset("psi.asc").expect("dataset");
        assert_eq!(psi.shape(), [4, 3]);
        assert_eq!(
            psi.read_raw::<f64>().expect("values"),
            (0..12_i32).map(f64::from).collect::<Vec<f64>>()
        );
        let r = file.dataset("r.asc").expect("r").read_1d::<f64>().expect("r vals");
        assert_eq!(r.len(), 4);
        assert!(
            (file.attr("m")
                .expect("attr")
                .read_scalar::<f64>()
                .expect("m")
                - 1.0)
                .abs()
                < 1e-15
        );
        assert_eq!(
            file.attr("format_version")
                .expect("fmt")
                .read_scalar::<i64>()
                .expect("fmt val"),
            1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(not(feature = "hdf5"))]
    fn disabled_backend_reports_the_feature_flag() {
        let mut w = Hdf5Writer::new("/nonexistent/solution.h5");
        let err = w
            .write_snapshot(&GridTopology::uniform(2, 2, 1.0, 1.0), &[])
            .expect_err("must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("--features hdf5"));
    }

    #[test]
    fn dataset_names_match_the_c_convention() {
        assert_eq!(dataset_name("psi"), "psi.asc");
    }
}
