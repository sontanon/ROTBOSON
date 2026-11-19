//! state.json model (design §3.2): atomic writes, spec-hash-guarded
//! resume — port of the frozen-dataclass layer in tools/sweep_driver.py.
//!
//! Two contracts live here:
//!
//! * **Byte compatibility.** `to_jv()` emits keys in exactly the order
//!   the Python `to_dict` methods do, and the writer
//!   ([`Jv::write_json`]) reproduces `json.dumps(..., indent=2)`, so a
//!   state.json written by the Python driver re-serializes byte-identically
//!   from Rust (pinned by the round-trip test against the archived
//!   `state_v2_current.json` fixture).
//! * **Lenient resume.** `from_jv` mirrors the Python `from_dict`
//!   deliberately: legacy state.json files (early format, missing keys,
//!   extra keys like `step_factor`) load, unknown keys are dropped, and a
//!   re-save upgrades the file. A spec-hash change aborts resume
//!   ([`StateError::SpecHashMismatch`]).
//!
//! The Python's uncaught-crash paths (corrupt scalars blocks, a
//! non-dict element) become [`crate::error::StateError::Corrupt`].

use crate::error::StateError;
use crate::pyjson::Jv;
use crate::vocab::{Status, StepMode};
use std::path::PathBuf;

/// Truncation-proxy rel-diff map of an accepted regrid
/// (`omega`/`M_Komar`/`J_Komar` → relative difference). Order-preserving
/// so the emitted `rel_diff` object stays byte-identical.
pub type RelDiff = Vec<(String, f64)>;

/// Scalar observables of a solution (`float | int` — the int/float
/// identity survives a round-trip, e.g. `error_code.asc`).
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
}

impl Scalar {
    fn to_jv(&self) -> Jv {
        match self {
            Scalar::Int(i) => Jv::Int(*i),
            Scalar::Float(f) => Jv::Float(*f),
        }
    }
}

// ---------------------------------------------------------------------------
// Coercions — the Python `_st_*` helpers (state.json is machine-written,
// but a corrupted/truncated file must fail with a clear message).
// ---------------------------------------------------------------------------

fn corrupt(expected: &str, got: &Jv) -> StateError {
    StateError::Corrupt(format!(
        "state.json: expected {expected}, got {}",
        got.py_type_name()
    ))
}

fn st_float(v: &Jv) -> Result<f64, StateError> {
    match v {
        Jv::Int(i) => Ok(*i as f64),
        Jv::Float(f) => Ok(*f),
        other => Err(corrupt("a number", other)),
    }
}

fn st_int(v: &Jv) -> Result<i64, StateError> {
    match v {
        Jv::Int(i) => Ok(*i),
        other => Err(corrupt("an integer", other)),
    }
}

fn st_str(v: &Jv) -> Result<String, StateError> {
    match v {
        Jv::Str(s) => Ok(s.clone()),
        other => Err(corrupt("a string", other)),
    }
}

/// Python truthiness `bool(v)` — the `accepted` flag is coerced, not
/// type-checked, in the Python `from_dict`.
fn truthy(v: &Jv) -> bool {
    match v {
        Jv::Null => false,
        Jv::Bool(b) => *b,
        Jv::Int(i) => *i != 0,
        Jv::Float(f) => *f != 0.0,
        Jv::Str(s) => !s.is_empty(),
        Jv::Arr(a) => !a.is_empty(),
        Jv::Obj(o) => !o.is_empty(),
    }
}

fn st_scalar(v: &Jv) -> Result<Scalar, StateError> {
    match v {
        Jv::Int(i) => Ok(Scalar::Int(*i)),
        Jv::Float(f) => Ok(Scalar::Float(*f)),
        other => Err(corrupt("a number", other)),
    }
}

/// A key's value, or the Python-side default when the key is missing
/// (a *present* null is an error, exactly like `raw.get(key, default)`
/// feeding `_st_*`).
fn get_or<'a>(raw: &'a Jv, key: &str, default: &'a Jv) -> Result<&'a Jv, StateError> {
    let _ = raw.as_obj().ok_or_else(|| {
        StateError::Corrupt(format!(
            "state.json: expected an object, got {}",
            raw.py_type_name()
        ))
    })?;
    Ok(raw.get(key).unwrap_or(default))
}

fn st_num_key(raw: &Jv, key: &str, default: f64) -> Result<f64, StateError> {
    st_float(get_or(raw, key, &Jv::Float(default))?)
}

fn st_int_key(raw: &Jv, key: &str, default: i64) -> Result<i64, StateError> {
    st_int(get_or(raw, key, &Jv::Int(default))?)
}

fn st_str_key(raw: &Jv, key: &str, default: &str) -> Result<String, StateError> {
    st_str(get_or(raw, key, &Jv::Str(default.to_string()))?)
}

fn st_opt_float(raw: &Jv, key: &str) -> Result<Option<f64>, StateError> {
    // Python's raw.get(key) yields None for a MISSING key *and* for a
    // present null — both load as None.
    match raw.get(key) {
        None | Some(Jv::Null) => Ok(None),
        Some(v) => st_float(v).map(Some),
    }
}

fn st_opt_int(raw: &Jv, key: &str) -> Result<Option<i64>, StateError> {
    match raw.get(key) {
        None | Some(Jv::Null) => Ok(None),
        Some(v) => st_int(v).map(Some),
    }
}

fn st_opt_str(raw: &Jv, key: &str) -> Result<Option<String>, StateError> {
    match raw.get(key) {
        None | Some(Jv::Null) => Ok(None),
        Some(v) => st_str(v).map(Some),
    }
}

fn st_opt_i32(raw: &Jv, key: &str) -> Result<Option<i32>, StateError> {
    match st_opt_int(raw, key)? {
        None => Ok(None),
        Some(i) => i32::try_from(i).map(Some).map_err(|_| {
            StateError::Corrupt(format!("state.json: exit code {i} out of range"))
        }),
    }
}

fn obj_pairs(v: &Jv) -> Option<&Vec<(String, Jv)>> {
    match v {
        Jv::Obj(pairs) => Some(pairs),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// State records
// ---------------------------------------------------------------------------

/// The campaign's current grid (dr drifts via refinements).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPosition {
    pub dr: f64,
    pub n: i64,
}

impl GridPosition {
    pub fn to_jv(&self) -> Jv {
        Jv::obj(vec![("dr", Jv::Float(self.dr)), ("N", Jv::Int(self.n))])
    }

    pub fn from_jv(raw: &Jv) -> Result<GridPosition, StateError> {
        Ok(GridPosition {
            dr: st_num_key(raw, "dr", 0.0)?,
            n: st_int_key(raw, "N", 0)?,
        })
    }
}

/// Accepted-regrid provenance attached to a [`StepRecord`].
#[derive(Debug, Clone, PartialEq)]
pub struct RegridInfo {
    pub from_dr: f64,
    pub to_dr: f64,
    pub source: Option<String>,
    pub rel_diff: RelDiff,
    pub accepted: bool,
    /// N of the regrid target when it differs from the source (the
    /// domain-keeping refinement's N×2); `None` = the legacy fixed-N
    /// regrid.
    pub to_n: Option<i64>,
}

impl RegridInfo {
    pub fn to_jv(&self) -> Jv {
        let rel_diff = Jv::Obj(
            self.rel_diff
                .iter()
                .map(|(k, v)| (k.clone(), Jv::Float(*v)))
                .collect(),
        );
        let mut pairs = vec![
            ("from_dr", Jv::Float(self.from_dr)),
            ("to_dr", Jv::Float(self.to_dr)),
            ("source", Jv::opt_str(self.source.as_deref())),
            ("rel_diff", rel_diff),
            ("accepted", Jv::Bool(self.accepted)),
        ];
        if let Some(n) = self.to_n {
            pairs.push(("to_n", Jv::Int(n)));
        }
        Jv::obj(pairs)
    }

    pub fn from_jv(raw: &Jv) -> Result<RegridInfo, StateError> {
        let rel_diff = match raw.get("rel_diff") {
            Some(v) => match obj_pairs(v) {
                Some(pairs) => pairs
                    .iter()
                    .map(|(k, v)| st_float(v).map(|f| (k.clone(), f)))
                    .collect::<Result<RelDiff, _>>()?,
                None => RelDiff::new(),
            },
            None => RelDiff::new(),
        };
        Ok(RegridInfo {
            from_dr: st_num_key(raw, "from_dr", 0.0)?,
            to_dr: st_num_key(raw, "to_dr", 0.0)?,
            source: st_opt_str(raw, "source")?,
            rel_diff,
            accepted: truthy(get_or(raw, "accepted", &Jv::Bool(false))?),
            to_n: st_opt_int(raw, "to_n")?,
        })
    }
}

/// A rejected/skipped regrid attempt (`rejected_regrids`). Two legacy
/// shapes share this model: a solved-but-rejected probe
/// (scale_u4/log/sol_dir/scalars/psi0) and a support-fit skip
/// (skipped/from_dr/to_dr); `to_jv` emits present fields in the legacy
/// insertion order so both round-trip byte-compatibly.
#[derive(Debug, Clone, PartialEq)]
pub struct RegridProbe {
    pub i: i64,
    pub mode: StepMode,
    pub exit_code: Option<i32>,
    pub scale_u4: Option<f64>,
    pub log: Option<String>,
    pub sol_dir: Option<String>,
    pub scalars: Option<Vec<(String, Scalar)>>,
    pub psi0: Option<f64>,
    pub skipped: Option<String>,
    pub from_dr: Option<f64>,
    pub to_dr: Option<f64>,
}

impl RegridProbe {
    pub fn to_jv(&self) -> Jv {
        let mut out = vec![
            ("i", Jv::Int(self.i)),
            ("mode", Jv::Str(self.mode.to_string())),
            ("exit_code", Jv::opt_int(self.exit_code.map(|c| c as i64))),
        ];
        if self.sol_dir.is_some() {
            // psi0 is always written once a solution exists (null on read
            // failure).
            out.extend([
                ("scale_u4", Jv::opt_float(self.scale_u4)),
                ("log", Jv::opt_str(self.log.as_deref())),
                ("sol_dir", Jv::opt_str(self.sol_dir.as_deref())),
                (
                    "scalars",
                    match &self.scalars {
                        None => Jv::Null,
                        Some(s) => Jv::Obj(
                            s.iter().map(|(k, v)| (k.clone(), v.to_jv())).collect(),
                        ),
                    },
                ),
                ("psi0", Jv::opt_float(self.psi0)),
            ]);
        } else {
            let extras: Vec<(&str, Jv)> = [
                ("scale_u4", Jv::opt_float(self.scale_u4)),
                ("log", Jv::opt_str(self.log.as_deref())),
                ("skipped", Jv::opt_str(self.skipped.as_deref())),
                ("from_dr", Jv::opt_float(self.from_dr)),
                ("to_dr", Jv::opt_float(self.to_dr)),
            ]
            .into_iter()
            .filter(|(_, v)| !matches!(v, Jv::Null))
            .collect();
            out.extend(extras);
        }
        Jv::obj(out)
    }

    pub fn from_jv(raw: &Jv) -> Result<RegridProbe, StateError> {
        let mode_str = st_str_key(raw, "mode", "regrid-probe")?;
        let mode = StepMode::from_name(&mode_str)
            .ok_or_else(|| StateError::Corrupt(format!("state.json: invalid step mode '{mode_str}'")))?;
        let scalars = match raw.get("scalars") {
            Some(v) => match obj_pairs(v) {
                Some(pairs) => Some(
                    pairs
                        .iter()
                        .map(|(k, v)| st_scalar(v).map(|s| (k.clone(), s)))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                None => None,
            },
            None => None,
        };
        Ok(RegridProbe {
            i: st_int_key(raw, "i", 0)?,
            mode,
            exit_code: st_opt_i32(raw, "exit_code")?,
            scale_u4: st_opt_float(raw, "scale_u4")?,
            log: st_opt_str(raw, "log")?,
            sol_dir: st_opt_str(raw, "sol_dir")?,
            scalars,
            psi0: st_opt_float(raw, "psi0")?,
            skipped: st_opt_str(raw, "skipped")?,
            from_dr: st_opt_float(raw, "from_dr")?,
            to_dr: st_opt_float(raw, "to_dr")?,
        })
    }
}

/// A kept measurement at a fold (`fold_fine_grid_measurement`).
#[derive(Debug, Clone, PartialEq)]
pub struct FoldMeasurement {
    pub psi0: Option<f64>,
    pub omega: Option<f64>,
    pub dr: f64,
    pub sol_dir: String,
    pub note: Option<String>,
}

impl FoldMeasurement {
    pub fn to_jv(&self) -> Jv {
        let mut pairs = vec![
            ("psi0", Jv::opt_float(self.psi0)),
            ("omega", Jv::opt_float(self.omega)),
            ("dr", Jv::Float(self.dr)),
            ("sol_dir", Jv::Str(self.sol_dir.clone())),
        ];
        if let Some(note) = &self.note {
            pairs.push(("note", Jv::Str(note.clone())));
        }
        Jv::obj(pairs)
    }

    pub fn from_jv(raw: &Jv) -> Result<FoldMeasurement, StateError> {
        Ok(FoldMeasurement {
            psi0: st_opt_float(raw, "psi0")?,
            omega: st_opt_float(raw, "omega")?,
            dr: st_num_key(raw, "dr", 0.0)?,
            sol_dir: st_str_key(raw, "sol_dir", "")?,
            note: st_opt_str(raw, "note")?,
        })
    }
}

/// An adopted-but-unverified refinement (`pending_refinement`).
#[derive(Debug, Clone, PartialEq)]
pub struct PendingRefinement {
    pub from_dr: f64,
    pub regrid_step: i64,
    pub measurement: FoldMeasurement,
    /// N of the pre-regrid grid (the domain-keeping refinement's revert
    /// target); `None` = the legacy fixed-N refinement, where `from_dr`
    /// alone suffices.
    pub from_n: Option<i64>,
}

impl PendingRefinement {
    pub fn to_jv(&self) -> Jv {
        let mut pairs = vec![
            ("from_dr", Jv::Float(self.from_dr)),
            ("regrid_step", Jv::Int(self.regrid_step)),
            ("measurement", self.measurement.to_jv()),
        ];
        if let Some(n) = self.from_n {
            pairs.push(("from_n", Jv::Int(n)));
        }
        Jv::obj(pairs)
    }

    pub fn from_jv(raw: &Jv) -> Result<PendingRefinement, StateError> {
        let measurement = match raw.get("measurement") {
            Some(v @ Jv::Obj(_)) => FoldMeasurement::from_jv(v)?,
            _ => FoldMeasurement {
                psi0: None,
                omega: None,
                dr: 0.0,
                sol_dir: String::new(),
                note: None,
            },
        };
        Ok(PendingRefinement {
            from_dr: st_num_key(raw, "from_dr", 0.0)?,
            regrid_step: st_int_key(raw, "regrid_step", 0)?,
            measurement,
            from_n: st_opt_int(raw, "from_n")?,
        })
    }
}

/// One campaign step (`steps[i]`). Optional scalars are `None` exactly
/// when the legacy writer omitted the key: a failed step records only
/// (i, exit_code, mode [+ psi0_target]); a step with a solution directory
/// always carries the scalar block (values may be null when a file was
/// unreadable).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepRecord {
    pub i: i64,
    pub exit_code: i32,
    pub mode: StepMode,
    pub sol_dir: Option<String>,
    pub dr: Option<f64>,
    pub n: Option<i64>,
    pub omega: Option<f64>,
    pub m_komar: Option<f64>,
    pub j_komar: Option<f64>,
    pub rr_phi_max: Option<f64>,
    pub r99: Option<f64>,
    pub hwl: Option<f64>,
    pub psi0: Option<f64>,
    pub newton_iters: Option<i64>,
    pub norm_f: Option<f64>,
    pub psi0_target: Option<f64>,
    pub regrid: Option<RegridInfo>,
}

impl StepRecord {
    pub fn to_jv(&self) -> Jv {
        let mut out = vec![
            ("i", Jv::Int(self.i)),
            ("exit_code", Jv::Int(self.exit_code as i64)),
            ("mode", Jv::Str(self.mode.to_string())),
        ];
        if self.sol_dir.is_some() {
            out.extend([
                ("sol_dir", Jv::opt_str(self.sol_dir.as_deref())),
                ("dr", Jv::opt_float(self.dr)),
                ("N", Jv::opt_int(self.n)),
                ("omega", Jv::opt_float(self.omega)),
                ("M_Komar", Jv::opt_float(self.m_komar)),
                ("J_Komar", Jv::opt_float(self.j_komar)),
                ("rr_phi_max", Jv::opt_float(self.rr_phi_max)),
                ("r99", Jv::opt_float(self.r99)),
                ("hwl", Jv::opt_float(self.hwl)),
                ("psi0", Jv::opt_float(self.psi0)),
            ]);
        }
        if let Some(iters) = self.newton_iters {
            out.push(("newton_iters", Jv::Int(iters)));
        }
        if let Some(norm) = self.norm_f {
            out.push(("norm_f", Jv::Float(norm)));
        }
        if let Some(target) = self.psi0_target {
            out.push(("psi0_target", Jv::Float(target)));
        }
        if let Some(regrid) = &self.regrid {
            out.push(("regrid", regrid.to_jv()));
        }
        Jv::obj(out)
    }

    pub fn from_jv(raw: &Jv) -> Result<StepRecord, StateError> {
        let mode_str = st_str_key(raw, "mode", "fixedPhi")?;
        let mode = StepMode::from_name(&mode_str).ok_or_else(|| {
            StateError::Corrupt(format!("state.json: invalid step mode '{mode_str}'"))
        })?;
        let regrid = match raw.get("regrid") {
            Some(v @ Jv::Obj(_)) => Some(RegridInfo::from_jv(v)?),
            _ => None,
        };
        Ok(StepRecord {
            i: st_int_key(raw, "i", 0)?,
            exit_code: st_opt_i32(raw, "exit_code")?.unwrap_or(0),
            mode,
            sol_dir: st_opt_str(raw, "sol_dir")?,
            dr: st_opt_float(raw, "dr")?,
            n: st_opt_int(raw, "N")?,
            omega: st_opt_float(raw, "omega")?,
            m_komar: st_opt_float(raw, "M_Komar")?,
            j_komar: st_opt_float(raw, "J_Komar")?,
            rr_phi_max: st_opt_float(raw, "rr_phi_max")?,
            r99: st_opt_float(raw, "r99")?,
            hwl: st_opt_float(raw, "hwl")?,
            psi0: st_opt_float(raw, "psi0")?,
            newton_iters: st_opt_int(raw, "newton_iters")?,
            norm_f: st_opt_float(raw, "norm_f")?,
            psi0_target: st_opt_float(raw, "psi0_target")?,
            regrid,
        })
    }
}

/// Campaign resume state (state.json).
#[derive(Debug, Clone, PartialEq)]
pub struct CampaignState {
    pub spec_hash: String,
    pub spec_file: Option<String>,
    pub steps: Vec<StepRecord>,
    pub status: Status,
    pub stop_reason: Option<String>,
    pub grid: Option<GridPosition>,
    pub refinements_left: Option<i64>,
    pub widenings_left: Option<i64>,
    /// The turning-point report dict stored when a campaign stops at
    /// `stopped:turning_point` (kept as an arbitrary JSON object, as the
    /// Python does).
    pub turning_point: Option<Jv>,
    pub rejected_regrids: Option<Vec<RegridProbe>>,
    pub pending_refinement: Option<PendingRefinement>,
    pub fold_fine_grid_measurement: Option<Vec<FoldMeasurement>>,
}

impl Default for CampaignState {
    fn default() -> Self {
        CampaignState {
            spec_hash: String::new(),
            spec_file: None,
            steps: Vec::new(),
            status: Status::Running,
            stop_reason: None,
            grid: None,
            refinements_left: None,
            widenings_left: None,
            turning_point: None,
            rejected_regrids: None,
            pending_refinement: None,
            fold_fine_grid_measurement: None,
        }
    }
}

impl CampaignState {
    /// Emit in exactly the Python `to_dict` key order (byte identity).
    pub fn to_jv(&self) -> Jv {
        let mut out = vec![
            ("spec_hash", Jv::Str(self.spec_hash.clone())),
            ("spec_file", Jv::opt_str(self.spec_file.as_deref())),
            (
                "steps",
                Jv::Arr(self.steps.iter().map(StepRecord::to_jv).collect()),
            ),
            ("status", Jv::Str(self.status.to_string())),
            ("stop_reason", Jv::opt_str(self.stop_reason.as_deref())),
        ];
        if let Some(grid) = &self.grid {
            out.push(("grid", grid.to_jv()));
        }
        if let Some(r) = self.refinements_left {
            out.push(("refinements_left", Jv::Int(r)));
        }
        if let Some(w) = self.widenings_left {
            out.push(("widenings_left", Jv::Int(w)));
        }
        out.push((
            "turning_point",
            self.turning_point.clone().unwrap_or(Jv::Null),
        ));
        if let Some(probes) = &self.rejected_regrids {
            out.push((
                "rejected_regrids",
                Jv::Arr(probes.iter().map(RegridProbe::to_jv).collect()),
            ));
        }
        out.push((
            "pending_refinement",
            match &self.pending_refinement {
                None => Jv::Null,
                Some(p) => p.to_jv(),
            },
        ));
        if let Some(fold) = &self.fold_fine_grid_measurement {
            out.push((
                "fold_fine_grid_measurement",
                Jv::Arr(fold.iter().map(FoldMeasurement::to_jv).collect()),
            ));
        }
        Jv::obj(out)
    }

    /// `json.dumps(state.to_dict(), indent=2)` (no trailing newline).
    pub fn to_json_pretty(&self) -> String {
        self.to_jv().write_json()
    }

    pub fn from_jv(raw: &Jv) -> Result<CampaignState, StateError> {
        // Legacy files (pre-regrid format) lack grid/refinements_left and
        // the provenance lists: they stay None and the effective values
        // come from the spec via `current_grid`.
        let steps = match raw.get("steps") {
            Some(Jv::Arr(items)) => items
                .iter()
                .map(StepRecord::from_jv)
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        let grid = match raw.get("grid") {
            Some(v @ Jv::Obj(_)) => Some(GridPosition::from_jv(v)?),
            _ => None,
        };
        let rejected_regrids = match raw.get("rejected_regrids") {
            Some(Jv::Arr(items)) => Some(
                items
                    .iter()
                    .map(RegridProbe::from_jv)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            _ => None,
        };
        let fold = match raw.get("fold_fine_grid_measurement") {
            Some(Jv::Arr(items)) => Some(
                items
                    .iter()
                    .map(FoldMeasurement::from_jv)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            _ => None,
        };
        let pending_refinement = match raw.get("pending_refinement") {
            Some(v @ Jv::Obj(_)) => Some(PendingRefinement::from_jv(v)?),
            _ => None,
        };
        let turning_point = match raw.get("turning_point") {
            Some(Jv::Obj(pairs)) => Some(Jv::Obj(pairs.clone())),
            _ => None,
        };
        let status_str = st_str_key(raw, "status", "running")?;
        let status = Status::from_name(&status_str).ok_or_else(|| {
            StateError::Corrupt(format!("state.json: invalid status '{status_str}'"))
        })?;
        Ok(CampaignState {
            spec_hash: st_str_key(raw, "spec_hash", "")?,
            spec_file: st_opt_str(raw, "spec_file")?,
            steps,
            status,
            stop_reason: st_opt_str(raw, "stop_reason")?,
            grid,
            refinements_left: st_opt_int(raw, "refinements_left")?,
            widenings_left: st_opt_int(raw, "widenings_left")?,
            turning_point,
            rejected_regrids,
            pending_refinement,
            fold_fine_grid_measurement: fold,
        })
    }

    /// Append to `rejected_regrids` (creating the list on first use) —
    /// Python `with_rejected_probe`.
    pub fn with_rejected_probe(&mut self, probe: RegridProbe) {
        self.rejected_regrids
            .get_or_insert_with(Vec::new)
            .push(probe);
    }

    /// Append to `fold_fine_grid_measurement` (creating the list on first
    /// use) — Python `with_fold_measurement`.
    pub fn with_fold_measurement(&mut self, measurement: FoldMeasurement) {
        self.fold_fine_grid_measurement
            .get_or_insert_with(Vec::new)
            .push(measurement);
    }
}

// ---------------------------------------------------------------------------
// File plumbing: atomic writes, spec-hash-guarded resume
// ---------------------------------------------------------------------------

/// `<output.root>/state.json` (Python `state_path`).
pub fn state_path(spec: &crate::spec::Spec) -> PathBuf {
    spec.output.root.join("state.json")
}

/// Load state.json if it exists; a spec-hash mismatch aborts resume
/// (Python `load_state`).
pub fn load_state(spec: &crate::spec::Spec) -> Result<Option<CampaignState>, StateError> {
    let p = state_path(spec);
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p)?;
    let raw = Jv::parse(&text)
        .map_err(|e| StateError::Corrupt(format!("state.json at {} is corrupt: {e}", p.display())))?;
    if raw.as_obj().is_none() {
        return Err(StateError::Corrupt(format!(
            "state.json at {} is corrupt: top level is not a JSON object",
            p.display()
        )));
    }
    match raw.get("spec_hash") {
        Some(Jv::Str(h)) if *h == spec.spec_hash => {}
        _ => return Err(StateError::SpecHashMismatch { path: p }),
    }
    Ok(Some(CampaignState::from_jv(&raw)?))
}

/// Atomically write state.json: write `state.json.tmp`, then rename over
/// the target (Python `save_state`).
pub fn save_state(
    spec: &crate::spec::Spec,
    state: &CampaignState,
) -> Result<(), StateError> {
    let p = state_path(spec);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, format!("{}\n", state.to_json_pretty()))?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

/// A fresh campaign's state: the tracked grid starts at the spec's grid;
/// `refinements_left`/`widenings_left` bound the regrid budgets
/// (Python `fresh_state`).
pub fn fresh_state(spec: &crate::spec::Spec) -> CampaignState {
    CampaignState {
        spec_hash: spec.spec_hash.clone(),
        grid: Some(GridPosition {
            dr: spec.grid.dr,
            n: spec.grid.n,
        }),
        refinements_left: Some(spec.adaptivity.max_refinements),
        widenings_left: Some(spec.adaptivity.max_widenings),
        ..CampaignState::default()
    }
}

/// Effective grid: the state's tracked value, else the spec's initial
/// grid — backwards compatible with state.json files that predate
/// regrids (which lack the `grid` key entirely).
pub fn current_grid(state: &CampaignState, spec: &crate::spec::Spec) -> GridPosition {
    state.grid.unwrap_or(GridPosition {
        dr: spec.grid.dr,
        n: spec.grid.n,
    })
}

/// The last completed step's ψ₀ (Python `psi0_of_last`; the SystemExit
/// message becomes the error Display).
pub fn psi0_of_last(state: &CampaignState) -> Result<f64, crate::error::DriverError> {
    for step in state.steps.iter().rev() {
        if let (Some(psi0), 0) = (step.psi0, step.exit_code) {
            return Ok(psi0);
        }
    }
    Err(crate::error::DriverError::NoCompletedPsi0)
}

/// The last completed step's ω (Python `omega_of_last`).
pub fn omega_of_last(state: &CampaignState) -> Result<f64, crate::error::DriverError> {
    for step in state.steps.iter().rev() {
        if let (Some(omega), 0) = (step.omega, step.exit_code) {
            return Ok(omega);
        }
    }
    Err(crate::error::DriverError::NoCompletedOmega)
}
