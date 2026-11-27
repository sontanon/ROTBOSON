//! HDF5 output — the oracle's format contract, feature `hdf5`.
//!
//! The C writer (`src/hdf5_writer.c`) produces one self-describing
//! `solution.h5` per solve: datasets named `<name>.asc` mirroring the ASCII
//! backend 1:1 — 2D fields shaped `(NrTotal, NzTotal)` in C order
//! (row-major over `IDX(i, j)`), the coordinate grids as 2D `r.asc`/`z.asc`
//! meshgrids (`r[i, j] = r_i`, `z[i, j] = z_j`), scalars/histories as
//! 1D arrays — plus every parameter, solver setting and provenance item as
//! a root attribute. This module mirrors that layout so
//! `tools/rotboson_io.py` reads Rust output unchanged (ADR 0001 §6).
//!
//! Deliberate deviations, Python-reader-invisible (both load as `str`):
//! string attributes are written as HDF5 vlen UTF-8 (the `hdf5` crate's
//! `VarLenUnicode`) instead of the C writer's fixed-size null-terminated
//! strings, and `output_backend` is `"hdf5-rust"` to mark NG provenance.
//! Attribute names, types (f64/i64) and semantics are exact.
//!
//! The feature needs a native libhdf5 (via hdf5-sys). It is OFF in CI by
//! design (ADR 0001 keeps runners dependency-light; the C CI already pins
//! the format contract). The dependency is the maintained `hdf5-metno`
//! fork (imported as `hdf5`): the original `hdf5` 0.8.1 crate rejects
//! modern system `HDF5` releases (its build script fails on 1.14.5 with
//! "Invalid `H5_VERSION`").

use std::io;
use std::path::{Path, PathBuf};

use crate::{FieldBlock, FieldWriter, GridTopology, IntSeriesBlock, SeriesBlock};

/// Dataset name of a field, mirroring the C convention (`"<name>.asc"`).
#[must_use]
pub fn dataset_name(field: &str) -> String {
    format!("{field}.asc")
}

/// Root-group attributes of one snapshot: the C writer's self-describing
/// contract, one struct field per HDF5 attribute.
///
/// `Default` mirrors the pre-parse defaults of `src/context.c` (the state
/// the C oracle writes for a parameter file that omits a key); the driver
/// fills the fields it parsed. [`Hdf5Writer::write_snapshot_with`]
/// auto-syncs `NrTotal`, `NzTotal` and `dim` to the grid actually written,
/// so the file can never misdescribe its own datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotAttrs {
    /// Build git hash (`git_hash`; the C binary's `-DGIT_HASH`).
    pub git_hash: String,
    /// Parameter-file provenance (`parfile`).
    pub parfile: String,
    /// Output backend tag (`output_backend`; `"hdf5-rust"` marks NG).
    pub backend: String,
    /// Creation timestamp, ISO 8601 UTC (`created`). Empty is stamped at
    /// write time (like the C writer's `write_attributes`).
    pub created: String,
    /// Format version (`format_version`; the C writer pins 1).
    pub format_version: i64,
    /// GRID block.
    pub grid: GridAttrs,
    /// SCALAR FIELD + FIXED VARIABLE blocks.
    pub field: ScalarFieldAttrs,
    /// INITIAL DATA block (+ optional input paths).
    pub initial_data: InitialDataAttrs,
    /// SCALE INITIAL DATA block (`scale_u0`..=`scale_u6`).
    pub scale_u: [f64; 7],
    /// SOLVER + INITIAL GUESS CHECK blocks.
    pub solver: SolverAttrs,
    /// Spherical analysis grid (`NrrTotal`, `NthTotal`, `p_dim`, `drr`,
    /// `dth`, `rr_inf`) — written only when the snapshot carries the
    /// corresponding `sph_*` datasets.
    pub analysis_grid: Option<SphericalGridAttrs>,
    /// Analysis results (`M_KOMAR`, `J_KOMAR`, `GRV2`, `GRV3`, `phi_max`,
    /// `rr_phi_max`, `hwl_res`) — written only when computed.
    pub analysis: Option<AnalysisAttrs>,
}

/// GRID root attributes (exact C names in doc comments).
#[derive(Debug, Clone, PartialEq)]
pub struct GridAttrs {
    /// `dr` — radial step.
    pub dr: f64,
    /// `dz` — axial step.
    pub dz: f64,
    /// `NrInterior` — interior points along r.
    pub nr_interior: i64,
    /// `NzInterior` — interior points along z.
    pub nz_interior: i64,
    /// `NrTotal` — total (ghost-inclusive) points along r.
    pub nr_total: i64,
    /// `NzTotal` — total (ghost-inclusive) points along z.
    pub nz_total: i64,
    /// `dim` — interior points per field (`NrInterior * NzInterior`).
    pub dim: i64,
    /// `ghost` — ghost layers per side (order/2).
    pub ghost: i64,
    /// `order` — finite-difference order.
    pub order: i64,
}

impl Default for GridAttrs {
    fn default() -> Self {
        // src/context.c GRID defaults.
        Self {
            dr: 0.0625,
            dz: 0.0625,
            nr_interior: 128,
            nz_interior: 128,
            nr_total: 130,
            nz_total: 130,
            dim: 16_900,
            ghost: 1,
            order: 2,
        }
    }
}

/// SCALAR FIELD + FIXED VARIABLE root attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarFieldAttrs {    /// `l` — azimuthal number.
    pub l: i64,
    /// `m` — scalar-field mass.
    pub m: f64,
    /// `psi0` — initial-guess amplitude.
    pub psi0: f64,
    /// `sigmaR` — Gaussian width along r.
    pub sigma_r: f64,
    /// `sigmaZ` — Gaussian width along z.
    pub sigma_z: f64,
    /// `rExt` — asymptotic-cutoff radius.
    pub r_ext: f64,
    /// `w0` — initial frequency.
    pub w0: f64,
    /// `w_idx` — omega slot in the unknown vector (`GNUM * dim`).
    pub w_idx: i64,
    /// `fixedPhi` — 1 when phi is pinned.
    pub fixed_phi: i64,
    /// `fixedPhiR` — phi pin cell along r (context default 4).
    pub fixed_phi_r: i64,
    /// `fixedPhiZ` — phi pin cell along z (context default 4).
    pub fixed_phi_z: i64,
    /// `fixedOmega` — 1 when omega is pinned.
    pub fixed_omega: i64,
}

impl Default for ScalarFieldAttrs {
    fn default() -> Self {
        // src/context.c SCALAR FIELD + FIXED VARIABLE defaults.
        Self {
            l: 1,
            m: 1.0,
            psi0: 0.2,
            sigma_r: 4.0,
            sigma_z: 4.0,
            r_ext: 16.0,
            w0: 0.7,
            w_idx: 135_200,
            fixed_phi: 1,
            fixed_phi_r: 4,
            fixed_phi_z: 4,
            fixed_omega: 0,
        }
    }
}

/// INITIAL DATA root attributes; the input-path strings exist only when
/// the solve read initial data (the C writer emits them conditionally).
#[derive(Debug, Clone, PartialEq)]
pub struct InitialDataAttrs {
    /// `readInitialData` — 0 (from-scratch ansatz), 1/2/3 (from files).
    pub read_initial_data: i64,
    /// `NrTotalInitial` — input grid total along r.
    pub nr_total_initial: i64,
    /// `NzTotalInitial` — input grid total along z.
    pub nz_total_initial: i64,
    /// `ghost_i` — input-grid ghost layers.
    pub ghost_i: i64,
    /// `order_i` — input-grid finite-difference order.
    pub order_i: i64,
    /// `dr_i` — input-grid radial step.
    pub dr_i: f64,
    /// `dz_i` — input-grid axial step.
    pub dz_i: f64,
    /// Optional input paths (`log_alpha_i` .. `w_i`).
    pub paths: Option<InitialDataPaths>,
}

impl Default for InitialDataAttrs {
    fn default() -> Self {
        // src/context.c INITIAL DATA defaults (unset input-grid values).
        Self {
            read_initial_data: 0,
            nr_total_initial: 0,
            nz_total_initial: 0,
            ghost_i: 1,
            order_i: 2,
            dr_i: 1.0,
            dz_i: 1.0,
            paths: None,
        }
    }
}

/// The seven optional `*_i` input-path attributes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InitialDataPaths {
    /// Path to the initial `log_alpha` field file.
    pub log_alpha_i: String,
    /// Path to the initial `beta` field file.
    pub beta_i: String,
    /// Path to the initial `log_h` field file.
    pub log_h_i: String,
    /// Path to the initial `log_a` field file.
    pub log_a_i: String,
    /// Path to the initial `psi` field file.
    pub psi_i: String,
    /// Path to the initial `lambda` field file.
    pub lambda_i: String,
    /// Path to the initial omega value file.
    pub w_i: String,
}

/// SOLVER + INITIAL GUESS CHECK root attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverAttrs {
    /// `solverType` — 1: error-based Newton (NLEQ-ERR).
    pub solver_type: i64,
    /// `localSolver` — sparse-direct backend selector.
    pub local_solver: i64,
    /// `epsilon` — Newton exit tolerance.
    pub epsilon: f64,
    /// `maxNewtonIter` — Newton iteration cap.
    pub max_newton_iter: i64,
    /// `lambda0` — initial damping.
    pub lambda0: f64,
    /// `lambdaMin` — damping floor.
    pub lambda_min: f64,
    /// `useLowRank` — low-rank Jacobian update flag.
    pub use_low_rank: i64,
    /// `max_initial_guess_checks` — guess-repair retry cap.
    pub max_initial_guess_checks: i64,
    /// `norm_f0_target` — guess-repair residual target.
    pub norm_f0_target: f64,
}

impl Default for SolverAttrs {
    fn default() -> Self {
        // src/context.c SOLVER + INITIAL GUESS CHECK defaults.
        Self {
            solver_type: 1,
            local_solver: 1,
            epsilon: 1e-5,
            max_newton_iter: 10,
            lambda0: 1e-3,
            lambda_min: 1e-8,
            use_low_rank: 0,
            max_initial_guess_checks: 8,
            norm_f0_target: 1e-5,
        }
    }
}

/// SPHERICAL ANALYSIS GRID root attributes (written only when the
/// snapshot carries `sph_*` datasets).
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalGridAttrs {
    /// `NrrTotal` — spherical analysis grid total along rr.
    pub nrr_total: i64,
    /// `NthTotal` — spherical analysis grid total along theta.
    pub nth_total: i64,
    /// `p_dim` — interior points (`NrrTotal * NthTotal`).
    pub p_dim: i64,
    /// `drr` — spherical grid radial step.
    pub drr: f64,
    /// `dth` — spherical grid angular step.
    pub dth: f64,
    /// `rr_inf` — analysis outer-boundary radius.
    pub rr_inf: f64,
}

/// ANALYSIS RESULTS root attributes (written only when computed; the C
/// writer fills them at close time after the analysis pass).
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisAttrs {
    /// `M_KOMAR` — Komar mass.
    pub m_komar: f64,
    /// `J_KOMAR` — Komar angular momentum.
    pub j_komar: f64,
    /// `GRV2` — accuracy identifier (2nd order).
    pub grv2: f64,
    /// `GRV3` — accuracy identifier (3rd order).
    pub grv3: f64,
    /// `phi_max` — scalar-field maximum.
    pub phi_max: f64,
    /// `rr_phi_max` — radius of the scalar-field maximum.
    pub rr_phi_max: f64,
    /// `hwl_res` — half-wave-length resolution at the maximum.
    pub hwl_res: i64,
}

impl Default for SnapshotAttrs {
    fn default() -> Self {
        // Composed from the src/context.c defaults of each block.
        Self {
            git_hash: "unknown".to_string(),
            parfile: String::new(),
            backend: "hdf5-rust".to_string(),
            created: String::new(),
            format_version: 1,
            grid: GridAttrs::default(),
            field: ScalarFieldAttrs::default(),
            initial_data: InitialDataAttrs::default(),
            scale_u: [1.0; 7],
            solver: SolverAttrs::default(),
            analysis_grid: None,
            analysis: None,
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
        self.write_snapshot_with(grid, fields, &[], &[], &SnapshotAttrs::default())
    }
}

impl Hdf5Writer {
    /// Write a full self-describing snapshot: 2D `fields`, 1D `series`
    /// (scalars, histories, profiles) and `int_series` (error codes,
    /// flags), all under their `<name>.asc` dataset names.
    ///
    /// `NrTotal`, `NzTotal` and `dim` are synced from `grid` before
    /// writing, and an empty `attrs.created` is stamped with the current
    /// UTC time — the file always describes itself truthfully.
    ///
    /// # Errors
    ///
    /// IO/HDF5 failures, malformed blocks (wrong value counts, empty
    /// series), or — when built without the `hdf5` feature — the
    /// disabled-backend error.
    pub fn write_snapshot_with(
        &mut self,
        grid: &GridTopology,
        fields: &[FieldBlock],
        series: &[SeriesBlock],
        int_series: &[IntSeriesBlock],
        attrs: &SnapshotAttrs,
    ) -> io::Result<()> {
        #[cfg(feature = "hdf5")]
        {
            let mut attrs = attrs.clone();
            sync_grid_attrs(&mut attrs, grid);
            validate_blocks(grid, fields, series, int_series)?;
            write_hdf5(&self.path, grid, fields, series, int_series, &attrs)
                .map_err(hdf5_to_io)
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (grid, fields, series, int_series, attrs);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "rbs-io was built without the `hdf5` feature; \
                 rebuild with: cargo build -p rbs-io --features hdf5",
            ))
        }
    }
}

#[cfg(feature = "hdf5")]
/// Fill the topology-derived GRID attributes from the grid being written
/// (the C writer emits `ctx->NrTotal` for exactly these datasets).
fn sync_grid_attrs(attrs: &mut SnapshotAttrs, grid: &GridTopology) {
    let nr = grid.nr as i64;
    let nz = grid.nz as i64;
    attrs.grid.nr_total = nr;
    attrs.grid.nz_total = nz;
    attrs.grid.dim = nr * nz;
}

/// Reject blocks that cannot be written as described.
#[cfg(feature = "hdf5")]
fn validate_blocks(
    grid: &GridTopology,
    fields: &[FieldBlock],
    series: &[SeriesBlock],
    int_series: &[IntSeriesBlock],
) -> io::Result<()> {
    let n = grid.nr * grid.nz;
    for f in fields {
        if f.values.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "field \"{}\" has {} values for a {}x{} grid",
                    f.name,
                    f.values.len(),
                    grid.nr,
                    grid.nz
                ),
            ));
        }
    }
    for s in series {
        if s.values.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("series \"{}\" is empty", s.name),
            ));
        }
    }
    for s in int_series {
        if s.values.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("int series \"{}\" is empty", s.name),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "hdf5")]
fn hdf5_to_io(e: hdf5::Error) -> io::Error {
    io::Error::other(e)
}

/// One root scalar attribute of f64s (mirrors `put_double_attr`).
#[cfg(feature = "hdf5")]
fn put_f64(file: &hdf5::File, name: &str, v: f64) -> hdf5::Result<()> {
    file.new_attr::<f64>()
        .create(name)?
        .write_scalar(&v)?;
    Ok(())
}

/// One root scalar attribute of i64s (mirrors `put_llong_attr`).
#[cfg(feature = "hdf5")]
fn put_i64(file: &hdf5::File, name: &str, v: i64) -> hdf5::Result<()> {
    file.new_attr::<i64>()
        .create(name)?
        .write_scalar(&v)?;
    Ok(())
}

/// One root string attribute (the C writer uses fixed-size
/// null-terminated; vlen UTF-8 loads identically through h5py).
#[cfg(feature = "hdf5")]
fn put_str(file: &hdf5::File, name: &str, v: &str) -> hdf5::Result<()> {
    let s = v
        .parse::<hdf5::types::VarLenUnicode>()
        .map_err(|e| hdf5::Error::Internal(e.to_string()))?;
    file.new_attr::<hdf5::types::VarLenUnicode>()
        .create(name)?
        .write_scalar(&s)?;
    Ok(())
}

/// Creation timestamp, ISO 8601 UTC (the C writer's `created` format).
#[must_use]
pub fn iso8601_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z",
        hh = sod / 3600,
        mm = (sod % 3600) / 60,
        ss = sod % 60
    )
}

/// Gregorian (year, month, day) from days since 1970-01-01 — Howard
/// Hinnant's `civil_from_days` (public-domain algorithm reference).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(feature = "hdf5")]
fn write_hdf5(
    path: &Path,
    grid: &GridTopology,
    fields: &[FieldBlock],
    series: &[SeriesBlock],
    int_series: &[IntSeriesBlock],
    attrs: &SnapshotAttrs,
) -> hdf5::Result<()> {
    let file = hdf5::File::create(path)?;

    // Coordinate meshgrids, 2D (nr, nz) like the C writer's r/z grids:
    // r[i, j] = r_i, z[i, j] = z_j (main.c fills them this way).
    let nz = grid.nz;
    let mut r2 = Vec::with_capacity(grid.nr * nz);
    for &ri in &grid.r {
        r2.extend(std::iter::repeat(ri).take(nz));
    }
    let z2 = grid.z.repeat(grid.nr);
    file.new_dataset::<f64>()
        .shape([grid.nr, nz])
        .create("r.asc")?
        .write_raw(&r2)?;
    file.new_dataset::<f64>()
        .shape([grid.nr, nz])
        .create("z.asc")?
        .write_raw(&z2)?;

    // Fields as 2D (nr, nz) C-order datasets — row-major over
    // IDX(i, j), the exact layout tools/rotboson_io.py expects.
    for f in fields {
        let name = dataset_name(&f.name);
        file.new_dataset::<f64>()
            .shape([grid.nr, grid.nz])
            .create(name.as_str())?
            .write_raw(&f.values)?;
    }

    // 1D double series (scalars, histories, profiles) and int series
    // (error codes, flags) as shape-(n,) datasets, like the C 1D writers.
    for s in series {
        let name = dataset_name(&s.name);
        file.new_dataset::<f64>()
            .shape([s.values.len()])
            .create(name.as_str())?
            .write_raw(&s.values)?;
    }
    for s in int_series {
        let name = dataset_name(&s.name);
        file.new_dataset::<i64>()
            .shape([s.values.len()])
            .create(name.as_str())?
            .write_raw(&s.values)?;
    }

    // Root attributes: provenance first, then the C writer's blocks in
    // write_attributes order (src/hdf5_writer.c).
    put_str(&file, "git_hash", &attrs.git_hash)?;
    put_str(&file, "parfile", &attrs.parfile)?;
    put_str(&file, "output_backend", &attrs.backend)?;
    let created = if attrs.created.is_empty() {
        iso8601_utc_now()
    } else {
        attrs.created.clone()
    };
    put_str(&file, "created", &created)?;
    put_i64(&file, "format_version", attrs.format_version)?;

    // GRID.
    let g = &attrs.grid;
    put_f64(&file, "dr", g.dr)?;
    put_f64(&file, "dz", g.dz)?;
    put_i64(&file, "NrInterior", g.nr_interior)?;
    put_i64(&file, "NzInterior", g.nz_interior)?;
    put_i64(&file, "NrTotal", g.nr_total)?;
    put_i64(&file, "NzTotal", g.nz_total)?;
    put_i64(&file, "dim", g.dim)?;
    put_i64(&file, "ghost", g.ghost)?;
    put_i64(&file, "order", g.order)?;

    // SCALAR FIELD + FIXED VARIABLE.
    let f = &attrs.field;
    put_i64(&file, "l", f.l)?;
    put_f64(&file, "m", f.m)?;
    put_f64(&file, "psi0", f.psi0)?;
    put_f64(&file, "sigmaR", f.sigma_r)?;
    put_f64(&file, "sigmaZ", f.sigma_z)?;
    put_f64(&file, "rExt", f.r_ext)?;
    put_f64(&file, "w0", f.w0)?;
    put_i64(&file, "w_idx", f.w_idx)?;
    put_i64(&file, "fixedPhi", f.fixed_phi)?;
    put_i64(&file, "fixedPhiR", f.fixed_phi_r)?;
    put_i64(&file, "fixedPhiZ", f.fixed_phi_z)?;
    put_i64(&file, "fixedOmega", f.fixed_omega)?;

    // INITIAL DATA (+ paths when the solve read them).
    let id = &attrs.initial_data;
    if let Some(p) = &id.paths {
        put_str(&file, "log_alpha_i", &p.log_alpha_i)?;
        put_str(&file, "beta_i", &p.beta_i)?;
        put_str(&file, "log_h_i", &p.log_h_i)?;
        put_str(&file, "log_a_i", &p.log_a_i)?;
        put_str(&file, "psi_i", &p.psi_i)?;
        put_str(&file, "lambda_i", &p.lambda_i)?;
        put_str(&file, "w_i", &p.w_i)?;
    }
    put_i64(&file, "readInitialData", id.read_initial_data)?;
    put_i64(&file, "NrTotalInitial", id.nr_total_initial)?;
    put_i64(&file, "NzTotalInitial", id.nz_total_initial)?;
    put_i64(&file, "ghost_i", id.ghost_i)?;
    put_i64(&file, "order_i", id.order_i)?;
    put_f64(&file, "dr_i", id.dr_i)?;
    put_f64(&file, "dz_i", id.dz_i)?;

    // SCALE INITIAL DATA.
    for (i, s) in attrs.scale_u.iter().enumerate() {
        put_f64(&file, &format!("scale_u{i}"), *s)?;
    }

    // SOLVER + INITIAL GUESS CHECK.
    let s = &attrs.solver;
    put_i64(&file, "solverType", s.solver_type)?;
    put_i64(&file, "localSolver", s.local_solver)?;
    put_f64(&file, "epsilon", s.epsilon)?;
    put_i64(&file, "maxNewtonIter", s.max_newton_iter)?;
    put_f64(&file, "lambda0", s.lambda0)?;
    put_f64(&file, "lambdaMin", s.lambda_min)?;
    put_i64(&file, "useLowRank", s.use_low_rank)?;
    put_i64(&file, "max_initial_guess_checks", s.max_initial_guess_checks)?;
    put_f64(&file, "norm_f0_target", s.norm_f0_target)?;

    // SPHERICAL ANALYSIS GRID (only when the snapshot carries sph_* data).
    if let Some(a) = &attrs.analysis_grid {
        put_i64(&file, "NrrTotal", a.nrr_total)?;
        put_i64(&file, "NthTotal", a.nth_total)?;
        put_i64(&file, "p_dim", a.p_dim)?;
        put_f64(&file, "drr", a.drr)?;
        put_f64(&file, "dth", a.dth)?;
        put_f64(&file, "rr_inf", a.rr_inf)?;
    }

    // ANALYSIS RESULTS (only when computed).
    if let Some(a) = &attrs.analysis {
        put_f64(&file, "M_KOMAR", a.m_komar)?;
        put_f64(&file, "J_KOMAR", a.j_komar)?;
        put_f64(&file, "GRV2", a.grv2)?;
        put_f64(&file, "GRV3", a.grv3)?;
        put_f64(&file, "phi_max", a.phi_max)?;
        put_f64(&file, "rr_phi_max", a.rr_phi_max)?;
        put_i64(&file, "hwl_res", a.hwl_res)?;
    }
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
        let series = vec![
            SeriesBlock {
                name: "w_f".to_string(),
                values: vec![0.95],
            },
            SeriesBlock {
                name: "norm_du".to_string(),
                values: vec![1e-3, 1e-6, 1e-9],
            },
        ];
        let int_series = vec![IntSeriesBlock {
            name: "error_code".to_string(),
            values: vec![0],
        }];
        let path = dir.join("solution.h5");
        let mut w = Hdf5Writer::new(&path);
        let attrs = SnapshotAttrs {
            created: "2026-01-01T00:00:00Z".to_string(),
            field: ScalarFieldAttrs {
                l: 1,
                m: 1.0,
                w0: 0.95,
                fixed_phi: 0,
                fixed_omega: 1,
                ..ScalarFieldAttrs::default()
            },
            solver: SolverAttrs {
                epsilon: 1e-12,
                max_newton_iter: 40,
                ..SolverAttrs::default()
            },
            ..SnapshotAttrs::default()
        };
        w.write_snapshot_with(&grid, &fields, &series, &int_series, &attrs)
            .expect("write");

        let file = hdf5::File::open(&path).expect("reopen");
        let psi = file.dataset("psi.asc").expect("dataset");
        assert_eq!(psi.shape(), [4, 3]);
        assert_eq!(
            psi.read_raw::<f64>().expect("values"),
            (0..12_i32).map(f64::from).collect::<Vec<f64>>()
        );

        // r/z are 2D meshgrids: r constant along rows, z along columns.
        let r = file.dataset("r.asc").unwrap().read_2d::<f64>().unwrap();
        let z = file.dataset("z.asc").unwrap().read_2d::<f64>().unwrap();
        assert_eq!(r.shape(), [4, 3]);
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(r[[i, j]], grid.r[i]);
                assert_eq!(z[[i, j]], grid.z[j]);
            }
        }

        // Series and int series keep their C 1D shapes.
        let w_f = file.dataset("w_f.asc").unwrap();
        assert_eq!(w_f.shape(), [1]);
        assert_eq!(w_f.read_raw::<f64>().unwrap(), vec![0.95]);
        let nd = file.dataset("norm_du.asc").unwrap();
        assert_eq!(nd.shape(), [3]);
        let ec = file.dataset("error_code.asc").unwrap();
        assert_eq!(ec.shape(), [1]);
        assert_eq!(ec.read_raw::<i64>().unwrap(), vec![0]);

        // Attributes: names and values round-trip.
        let f = &attrs.field;
        assert!(
            (file.attr("m").unwrap().read_scalar::<f64>().unwrap() - 1.0).abs() < 1e-15
        );
        assert_eq!(
            file.attr("format_version")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            1
        );
        assert_eq!(file.attr("l").unwrap().read_scalar::<i64>().unwrap(), f.l);
        assert_eq!(
            file.attr("w0").unwrap().read_scalar::<f64>().unwrap(),
            f.w0
        );
        assert_eq!(
            file.attr("fixedOmega")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            1
        );
        assert_eq!(
            file.attr("epsilon")
                .unwrap()
                .read_scalar::<f64>()
                .unwrap(),
            attrs.solver.epsilon
        );
        assert_eq!(
            file.attr("sigmaR").unwrap().read_scalar::<f64>().unwrap(),
            attrs.field.sigma_r
        );
        let created = file
            .attr("created")
            .unwrap()
            .read_scalar::<hdf5::types::VarLenUnicode>()
            .unwrap();
        assert_eq!(created.to_string(), "2026-01-01T00:00:00Z");

        // GRID totals are synced from the grid actually written.
        assert_eq!(
            file.attr("NrTotal").unwrap().read_scalar::<i64>().unwrap(),
            4
        );
        assert_eq!(
            file.attr("NzTotal").unwrap().read_scalar::<i64>().unwrap(),
            3
        );
        assert_eq!(
            file.attr("dim").unwrap().read_scalar::<i64>().unwrap(),
            12
        );

        // Optional analysis blocks are absent by default.
        assert!(file.attr("M_KOMAR").is_err());
        assert!(file.attr("NrrTotal").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "hdf5")]
    fn analysis_blocks_roundtrip_when_present() {
        let dir = std::env::temp_dir().join(format!("rbs-io-h5b-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let grid = GridTopology::uniform(2, 2, 1.0, 1.0);
        let mut w = Hdf5Writer::new(dir.join("solution.h5"));
        let attrs = SnapshotAttrs {
            analysis_grid: Some(SphericalGridAttrs {
                nrr_total: 65,
                nth_total: 63,
                p_dim: 4095,
                drr: 0.25,
                dth: std::f64::consts::PI / 124.0,
                rr_inf: 16.5,
            }),
            analysis: Some(AnalysisAttrs {
                m_komar: 0.788,
                j_komar: 0.79,
                grv2: 1.3e-3,
                grv3: 0.917,
                phi_max: 0.0124,
                rr_phi_max: 6.94,
                hwl_res: 43,
            }),
            ..SnapshotAttrs::default()
        };
        w.write_snapshot_with(&grid, &[], &[], &[], &attrs)
            .expect("write");
        let file = hdf5::File::open(w.path()).expect("reopen");
        assert_eq!(
            file.attr("NrrTotal").unwrap().read_scalar::<i64>().unwrap(),
            65
        );
        assert!(
            (file.attr("rr_inf").unwrap().read_scalar::<f64>().unwrap() - 16.5).abs() < 1e-15
        );
        assert!(
            (file.attr("M_KOMAR").unwrap().read_scalar::<f64>().unwrap() - 0.788).abs() < 1e-15
        );
        assert_eq!(
            file.attr("hwl_res").unwrap().read_scalar::<i64>().unwrap(),
            43
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "hdf5")]
    fn mismatched_field_lengths_are_rejected() {
        let mut w = Hdf5Writer::new("/dev/null");
        let grid = GridTopology::uniform(2, 2, 1.0, 1.0);
        let fields = vec![FieldBlock {
            name: "psi".to_string(),
            values: vec![0.0; 5],
        }];
        let err = w
            .write_snapshot_with(&grid, &fields, &[], &[], &SnapshotAttrs::default())
            .expect_err("must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("psi"));

        let err = w
            .write_snapshot_with(
                &grid,
                &[],
                &[SeriesBlock {
                    name: "w_f".to_string(),
                    values: vec![],
                }],
                &[],
                &SnapshotAttrs::default(),
            )
            .expect_err("must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("w_f"));
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
        assert_eq!(dataset_name("w_f"), "w_f.asc");
    }

    #[test]
    fn timestamps_are_iso8601_utc() {
        // Pinned epoch dates: civil_from_days round-trips known values.
        let cases = [
            (0i64, (1970, 1, 1)),
            (86_400, (1970, 1, 2)),
            (1_740_470_400, (2025, 2, 25)),
            (1_794_508_800, (2026, 11, 12)),
        ];
        for (secs, (y, m, d)) in cases {
            let days = secs.div_euclid(86_400);
            assert_eq!(civil_from_days(days), (y, m, d));
        }
        // The live stamp is well-formed: 2026-11-19T..:..:..Z.
        let now = iso8601_utc_now();
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.starts_with("20") && now.ends_with('Z'), "{now}");
        assert_eq!(now.as_bytes().get(10), Some(&b'T'));
        assert_eq!(now.as_bytes().get(4), Some(&b'-'));
        assert_eq!(now.as_bytes().get(13), Some(&b':'));
    }
}
