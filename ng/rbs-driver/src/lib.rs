//! RBS-NG continuation driver — the pure, solver-independent half
//! (Linear SAN-60, part 1).
//!
//! A Rust port of the logic in `tools/sweep_driver.py` (the Python
//! continuation driver; the Python is the specification) covering
//! everything that does not shell out to the C solver:
//! * **Campaign-spec TOML parsing** — closed vocabulary, unknown
//!   tables/keys rejected as hard parse errors, wrong types hard
//!   errors, tuned constants carried over with citations
//!   ([`spec`] module).
//! * **The state.json model** — atomic step records, byte-identical
//!   re-serialization of Python-written state, lenient legacy resume,
//!   and the spec-hash resume guard ([`state`] module).
//! * **The adaptive decision layer** — step-size policy, `decide_action`,
//!   coarsening/boundary classification, turning-point detection and
//!   the stop-reason taxonomy with exact contract strings
//!   ([`decide`] and [`vocab`] modules).
//! * **The fold summarize** — the degree-3/4 polynomial fit over the
//!   branch points bracketing `ω_min`
//!   ([`summarize`](module@crate::summarize) and [`fit`] modules).
//!
//! Deferred to SAN-60's solver-facing half (blocked on the Rust
//! solver): running the binary, seed-field rendering, the regrid
//! ladder's interpolator plumbing, end-to-end campaign execution.

// The driver ports Python's `not x > y` comparisons verbatim: the
// negated form is load-bearing for NaN semantics (NaN must REJECT —
// `x <= y` would silently accept it), and that behavior is pinned by
// the ported test suite (NaN quirks are part of the Python contract).
#![allow(clippy::neg_cmp_op_on_partial_ord)]
//! See `README.md` for the ported/deferred inventory and the
//! workspace-merge notes.
//!
//! Python line references throughout cite `tools/sweep_driver.py` at the
//! SAN-59 head (`06b4ff2`) — the revision this port was specified
//! against; the master worktree base (`1ca850b`) is identical except
//! that it lacks the `[solver]` `numThreads`/`precision`/`direct`/`fp32_ir`
//! knobs, which are included here so the repo's runbook configs parse.

pub mod decide;
pub mod error;
pub mod fit;
pub mod pyjson;
pub mod sha256;
pub mod spec;
pub mod state;
pub mod summarize;
pub mod vocab;

// Ergonomic flat re-exports: the driver's public vocabulary.
pub use decide::{
    boundary_failure_stop, coarsening_acceptable, decide_action, decide_coarsening,
    detect_turning_point, finished, ghost_of, near_newtonian_limit, newtonian_limit_stop,
    next_target, round15, status_for_stop, step_retryable, turning_point_estimate,
    would_amputate, Action, CoarseningDecision, StepDiagnostics, TurningPointReport,
};
pub use error::{DriverError, SpecError, StateError, SummarizeError};
pub use spec::{
    load_spec, spec_hash, AdaptivitySpec, CampaignSpec, GridSpec, OutputSpec, SeedSpec,
    SolverSpec, Spec,
};
pub use state::{
    current_grid, fresh_state, load_state, omega_of_last, psi0_of_last, save_state, state_path,
    CampaignState, FoldMeasurement, GridPosition, PendingRefinement, RegridInfo, RegridProbe,
    RelDiff, Scalar, StepRecord,
};
pub use summarize::{summarize, summarize_campaign, Summary, SUMMARY_FIT_DEGREE};
pub use vocab::{
    signal_name, stop_family, Direction, OutputFormat, Psi0StepMode, SeedPolicy, Status, StepMode,
    StopReason, TIMEOUT_EXIT,
};
