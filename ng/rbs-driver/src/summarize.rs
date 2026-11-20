//! `--summarize`: localize `ω_min` from a finished campaign (design §6.2)
//! — port of `tools/sweep_driver.py`'s `summarize`.
//!
//! Post-processing only: reads state.json, fits a low-order polynomial
//! to ω(ψ₀) over the samples bracketing the smallest sampled ω (the
//! degree-3/4 fit, up to the 7-sample window around the minimum), and
//! writes `<root>/summary.json` in the Python's exact JSON shape.
//!
//! The Python additionally prints the payload to stdout
//! (`[driver] summary: {...}`); that CLI concern stays with the future
//! binary front-end, not the library.

use crate::decide::{turning_point_estimate, TurningPointReport};
use crate::error::SummarizeError;
use crate::pyjson::Jv;
use crate::spec::Spec;
use crate::state::{load_state, CampaignState};
use crate::vocab::Status;
use std::path::PathBuf;

/// The default fit degree (Python `turning_point_estimate(..., degree=4)`;
/// the window rarely offers more than 4 + 1 support points, and
/// `turning_point_estimate` caps it at `n - 2` anyway).
pub const SUMMARY_FIT_DEGREE: usize = 4;

/// summary.json payload: the turning-point report plus the campaign
/// bookkeeping keys, in the Python's insertion order (`report` keys
/// first, then `campaign`, `status`, `stop_reason`).
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// `None` when no usable (ψ₀, ω) pairs existed — the Python's empty
    /// `{}` report.
    pub report: Option<TurningPointReport>,
    /// The campaign root, as the Python's `str(spec.output.root)`.
    pub campaign: String,
    /// The campaign's final lifecycle status (`state.json` `status`).
    pub status: Status,
    /// The stored stop-reason string, when the campaign recorded one.
    pub stop_reason: Option<String>,
}

impl Summary {
    /// The payload as a JSON tree, in the Python's insertion order
    /// (report keys first, then the bookkeeping keys).
    pub fn to_jv(&self) -> Jv {
        let mut pairs: Vec<(String, Jv)> = Vec::new();
        if let Some(report) = &self.report {
            pairs.extend(report.to_pairs());
        }
        pairs.push(("campaign".to_string(), Jv::Str(self.campaign.clone())));
        pairs.push(("status".to_string(), Jv::Str(self.status.to_string())));
        pairs.push((
            "stop_reason".to_string(),
            Jv::opt_str(self.stop_reason.as_deref()),
        ));
        Jv::Obj(pairs)
    }

    /// `json.dumps(summary, indent=2)` (no trailing newline).
    pub fn to_json_pretty(&self) -> String {
        self.to_jv().write_json()
    }
}

/// The pure core of `summarize` (Python `sweep_driver.py` L2514-2542):
/// build the report from the state's steps. Errors when no step records
/// ω ("no completed steps record ω; nothing to summarize").
///
/// # Errors
///
/// Returns [`SummarizeError::NoOmegas`] when no step carries an ω —
/// the Python's "no completed steps record ω; nothing to summarize"
/// exit-1 path.
pub fn summarize(spec: &Spec, state: &CampaignState) -> Result<Summary, SummarizeError> {
    let omegas: Vec<Option<f64>> = state.steps.iter().map(|s| s.omega).collect();
    if !omegas.iter().any(|w| w.is_some()) {
        return Err(SummarizeError::NoOmegas);
    }
    let psi0s: Vec<Option<f64>> = state.steps.iter().map(|s| s.psi0).collect();
    let report = turning_point_estimate(&psi0s, &omegas, SUMMARY_FIT_DEGREE);
    Ok(Summary {
        report,
        campaign: spec.output.root.to_string_lossy().into_owned(),
        status: state.status,
        stop_reason: state.stop_reason.clone(),
    })
}

/// The file-backed entry point (the Python `summarize` main path): load
/// state.json under the campaign root, summarize, write
/// `<root>/summary.json` (indent 2, trailing newline — the Python
/// writer's byte shape).
///
/// # Errors
///
/// Returns [`SummarizeError::NoState`] when no state.json exists under
/// the campaign root, or propagates the `load_state`/`summarize` errors
/// (I/O, corrupt state.json, spec-hash mismatch, no recorded ω) and any
/// I/O failure while writing summary.json.
pub fn summarize_campaign(spec: &Spec) -> Result<Summary, SummarizeError> {
    let state = load_state(spec)?.ok_or(SummarizeError::NoState)?;
    let summary = summarize(spec, &state)?;
    let out: PathBuf = spec.output.root.join("summary.json");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, format!("{}\n", summary.to_json_pretty()))?;
    Ok(summary)
}
