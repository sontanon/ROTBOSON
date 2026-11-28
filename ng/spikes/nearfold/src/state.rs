//! Loading the near-fold states exported by `tools/export_nearfold_states.py`.

use std::fmt;
use std::path::{Path, PathBuf};

use rbs_core::fields::{FieldSet, NUM_FIELDS};
use rbs_core::grid::Grid2d;
use rbs_elliptic::assembly::{assemble_residual, ProblemParams, Workspace};
use rbs_elliptic::initial::SolveState;
use rbs_elliptic::omega::inverse_omega_calc;

/// Field names in the solver's field order, as written by the exporter.
const FIELD_NAMES: [&str; NUM_FIELDS] = [
    "log_alpha_f",
    "beta_f",
    "log_h_f",
    "log_a_f",
    "psi_f",
    "lambda_f",
];

/// The self-check bound: a converged C state must have a tiny Rust residual.
const SELF_CHECK_MAX_RESIDUAL: f64 = 1e-6;

/// An error while loading a state, with the file it came from.
#[derive(Debug)]
pub struct LoadError(pub String);

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One row of `manifest.csv`.
#[derive(Debug, Clone)]
pub struct ManifestRow {
    /// File name of the state, relative to the manifest's directory.
    pub file: String,
    /// Campaign step number.
    pub step: usize,
    /// Interior points per direction.
    pub n: usize,
    /// Central scalar-field amplitude of the state.
    pub psi0: f64,
    /// Distance to the fold estimate, `psi0* - psi0` (negative past it).
    pub dist: f64,
}

/// A loaded, self-checked state: grid, fixed-phi parameters, and solution.
pub struct NearFoldState {
    /// Where it came from.
    pub row: ManifestRow,
    /// The grid.
    pub grid: Grid2d,
    /// Parameters with the psi pin (fixed-phi mode) and this state's xi.
    pub params: ProblemParams,
    /// The converged C solution.
    pub state: SolveState,
    /// The solved frequency.
    pub omega: f64,
    /// Max-norm of the Rust residual at this state (the self-check value).
    pub residual: f64,
}

impl NearFoldState {
    /// The same parameters in fixed-omega mode (no psi pin).
    pub fn fixed_omega_params(&self) -> ProblemParams {
        ProblemParams {
            fixed_phi: None,
            ..self.params
        }
    }
}

/// Read `manifest.csv`: header `file,step,N,dr,psi0,omega,dist`.
///
/// # Errors
///
/// A missing file or a malformed row.
pub fn read_manifest(path: &Path) -> Result<Vec<ManifestRow>, LoadError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| LoadError(format!("cannot read {}: {e}", path.display())))?;
    let mut rows = Vec::new();
    for (k, line) in text.lines().enumerate().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        let bad = || {
            LoadError(format!(
                "{}:{}: malformed row {line:?}",
                path.display(),
                k + 1
            ))
        };
        if cols.len() != 7 {
            return Err(bad());
        }
        rows.push(ManifestRow {
            file: cols[0].to_string(),
            step: cols[1].parse().map_err(|_| bad())?,
            n: cols[2].parse().map_err(|_| bad())?,
            psi0: cols[4].parse().map_err(|_| bad())?,
            dist: cols[6].parse().map_err(|_| bad())?,
        });
    }
    Ok(rows)
}

/// Load one state and run the self-check.
///
/// # Errors
///
/// A missing or malformed file, or a residual above the self-check bound
/// (which means the layout or the parameters are wrong).
pub fn load(dir: &Path, row: &ManifestRow) -> Result<NearFoldState, LoadError> {
    let path: PathBuf = dir.join(&row.file);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| LoadError(format!("cannot read {}: {e}", path.display())))?;
    let bad = |what: &str| LoadError(format!("{}: {what}", path.display()));

    let mut header = std::collections::HashMap::new();
    let mut values: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(first) = parts.next() else { continue };
        if first.starts_with('#') {
            continue;
        }
        if first == "field" {
            let name = parts.next().ok_or_else(|| bad("field without a name"))?;
            if FIELD_NAMES.get(values.len()) != Some(&name) {
                return Err(bad(&format!("unexpected field {name}")));
            }
            values.push(Vec::new());
        } else if let Some(field) = values.last_mut() {
            field.push(first.parse().map_err(|_| bad("bad value"))?);
        } else {
            let value = parts
                .next()
                .ok_or_else(|| bad("header key without value"))?;
            header.insert(first.to_string(), value.to_string());
        }
    }
    let get = |key: &str| -> Result<&String, LoadError> {
        header
            .get(key)
            .ok_or_else(|| bad(&format!("missing header {key}")))
    };
    let int = |key: &str| -> Result<usize, LoadError> {
        get(key)?.parse().map_err(|_| bad(&format!("bad {key}")))
    };
    let float = |key: &str| -> Result<f64, LoadError> {
        get(key)?.parse().map_err(|_| bad(&format!("bad {key}")))
    };

    let (nr_total, nz_total, ghost) = (int("nr_total")?, int("nz_total")?, int("ghost")?);
    let grid = Grid2d::new(nr_total, nz_total, ghost, float("dr")?, float("dz")?);
    let (m, omega) = (float("m")?, float("omega")?);
    let params = ProblemParams {
        l: i64::try_from(int("l")?).map_err(|_| bad("bad l"))?,
        m,
        xi: inverse_omega_calc(omega, m),
        q1: 1.0,
        q2: 1.0,
        fixed_phi: Some((int("pin_r")?, int("pin_z")?)),
    };

    if values.len() != NUM_FIELDS {
        return Err(bad(&format!(
            "{} fields, expected {NUM_FIELDS}",
            values.len()
        )));
    }
    let mut fields = FieldSet::zeros(&grid);
    for (k, field) in values.iter().enumerate() {
        if field.len() != grid.dim() {
            return Err(bad(&format!("field {k} has {} values", field.len())));
        }
        fields.field_mut(k).copy_from_slice(field);
    }

    let state = SolveState {
        fields,
        xi: params.xi,
    };
    let mut ws = Workspace::new(&grid);
    let f = assemble_residual(&grid, &state.fields, &params, &mut ws);
    // f64::max skips NaN, so a non-finite residual is mapped to NaN first.
    let residual = if f.iter().all(|v| v.is_finite()) {
        f.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
    } else {
        f64::NAN
    };
    // NaN fails the check too.
    if residual.is_nan() || residual >= SELF_CHECK_MAX_RESIDUAL {
        return Err(bad(&format!(
            "self-check failed: residual max-norm {residual:.3e} >= {SELF_CHECK_MAX_RESIDUAL:e}; \
             the export layout or the parameters do not match the solver"
        )));
    }
    Ok(NearFoldState {
        row: row.clone(),
        grid,
        params,
        state,
        omega,
        residual,
    })
}
