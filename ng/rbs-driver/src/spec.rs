//! Campaign-spec model: frozen structs parsed + validated in one place
//! (port of `tools/sweep_driver.py`'s `Spec` dataclasses, SAN-60).
//!
//! Contract carried over from the Python:
//!
//! * **Unknown keys are hard parse errors** — `KNOWN` lists the closed
//!   table/key vocabulary; anything else aborts with the Python-verbatim
//!   messages (`unknown spec tables: ['x']`, `[campaign] unknown keys: ['x']`).
//! * **Wrong types are hard parse errors** — TOML's type system gives that
//!   for free via serde (`l = "one"` fails to parse); TOML has no bool/int
//!   confusion, and the two fields the Python coerces with `bool(...)`
//!   (`[solver] direct`, `[adaptivity] refine_keeps_domain`) keep that
//!   *truthiness* behavior via [`de_py_bool`].
//! * **The spec hash** (resume guard) is SHA-256 over the spec text with
//!   `max_steps`/`max_retries` lines and `#` comment lines removed —
//!   raising a runtime limit or editing commentary must not invalidate a
//!   running campaign's physics state (`sweep_driver.py` `load_spec`).
//!
//! Line references cite `tools/sweep_driver.py` at the SAN-59 head
//! (`06b4ff2`), the revision this port was specified against; master
//! (`1ca850b`) is identical except that it lacks the `[solver]`
//! `numThreads`/`precision`/`direct`/`fp32_ir` extension (see `SolverSpec`).

use crate::error::SpecError;
use crate::sha256::sha256_hex;
use crate::vocab::{Direction, OutputFormat, Psi0StepMode, SeedPolicy};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Known spec vocabulary (Python `KNOWN`, sweep_driver.py L277-318)
// ---------------------------------------------------------------------------

/// TOML table names and their closed key sets, in the Python dict's
/// order (the unknown-key scan walks tables in this order).
static KNOWN: &[(&str, &[&str])] = &[
    (
        "campaign",
        &[
            "l",
            "m",
            "direction",
            "psi0_target",
            "omega_target",
            "psi0_step",
            "psi0_step_mode",
            "max_retries",
            "max_steps",
            "fixedPhiR",
            "fixedPhiZ",
            "stop_at_turning_point",
        ],
    ),
    (
        "seed",
        &["policy", "source", "w0", "psi0", "sigmaR", "sigmaZ", "rExt"],
    ),
    ("grid", &["dr", "dr_max", "N", "order"]),
    (
        "solver",
        &[
            "solverType",
            "localSolver",
            "epsilon",
            "maxNewtonIter",
            "lambda0",
            "lambdaMin",
            "useLowRank",
            "numThreads",
            "precision",
            "direct",
            "fp32_ir",
        ],
    ),
    ("output", &["root", "format"]),
    (
        "adaptivity",
        &[
            "hwl_min",
            "max_refinements",
            "newtonian_delta",
            "boundary_fraction",
            "refine_keeps_domain",
            "regrid_rtol",
            "support_fraction",
            "max_widenings",
        ],
    ),
];

fn invalid(msg: impl Into<String>) -> SpecError {
    SpecError::Invalid(msg.into())
}

/// Python truthiness for TOML values: `bool(v)`. Used for the two spec
/// fields the Python coerces with `bool(raw.get(...))` rather than the
/// strict `_bool` helper (`SolverSpec.direct`,
/// `AdaptivitySpec.refine_keeps_domain`) — `direct = 1` is `true` there.
///
/// # Errors
///
/// Returns the deserializer's error when the TOML value is not a scalar
/// (the Python `bool(v)` accepts everything; serde requires the case).
pub fn de_py_bool<'de, D>(d: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = bool;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("any TOML value (Python truthiness)")
        }
        fn visit_bool<E>(self, v: bool) -> Result<bool, E> {
            Ok(v)
        }
        fn visit_i64<E>(self, v: i64) -> Result<bool, E> {
            Ok(v != 0)
        }
        fn visit_u64<E>(self, v: u64) -> Result<bool, E> {
            Ok(v != 0)
        }
        fn visit_f64<E>(self, v: f64) -> Result<bool, E> {
            // bool(float("nan")) is True in Python.
            Ok(v != 0.0)
        }
        fn visit_str<E>(self, v: &str) -> Result<bool, E> {
            Ok(!v.is_empty())
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<bool, A::Error> {
            // bool([]) is False, any non-empty list is True.
            let mut any = false;
            while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                any = true;
            }
            Ok(any)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<bool, A::Error> {
            let mut any = false;
            while map.next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?.is_some() {
                any = true;
            }
            Ok(any)
        }
    }
    d.deserialize_any(V)
}

// ---------------------------------------------------------------------------
// Spec tables
// ---------------------------------------------------------------------------

/// `[campaign]` (Python `CampaignSpec`, `sweep_driver.py` L321-371).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSpec {
    /// Azimuthal harmonic order of the scalar field (C param `l`) — the
    /// φ ∝ r^l power law at the axis.
    pub l: i64,
    /// The branch's stepping direction in ψ₀: up from the seed, or down
    /// toward ω = m.
    pub direction: Direction,
    /// Terminal ψ₀ — the campaign's constraint value; targets are
    /// clamped to land on it exactly (see [`crate::decide::next_target`]).
    pub psi0_target: f64,
    /// Step between consecutive ψ₀ targets (relative by default:
    /// `target = ψ₀·(1 ± step)`).
    pub psi0_step: f64,
    /// Theory mass parameter `m` (C param `m`) — the down branch's ω
    /// ceiling: ω → m as ψ₀ → 0.
    #[serde(default = "d_one_f")]
    pub m: f64,
    /// Optional ω terminal value — the campaign also stops once ω
    /// crosses it in the campaign's direction.
    #[serde(default)]
    pub omega_target: Option<f64>,
    /// How `psi0_step` is applied — relative (the default) or absolute.
    #[serde(default = "d_relative")]
    pub psi0_step_mode: Psi0StepMode,
    /// Failed-attempt retries per step: same target, plain rescaled
    /// seed (extrapolation dropped — the rescale converges where the
    /// extrapolation diverges).
    #[serde(default = "d_max_retries")]
    pub max_retries: i64,
    /// Hard cap on recorded steps (`done:max_steps`).
    #[serde(default = "d_max_steps")]
    pub max_steps: i64,
    /// R index of the fixedPhi grid point whose ψ the C constraint pins
    /// to the target (param `fixedPhiR`).
    #[serde(rename = "fixedPhiR", default = "d_two_i")]
    pub fixed_phi_r: i64,
    /// Z index of the same point (param `fixedPhiZ`).
    #[serde(rename = "fixedPhiZ", default = "d_two_i")]
    pub fixed_phi_z: i64,
    /// Stop the up campaign at the detected minimum-ω turning point
    /// (`stopped:turning_point`).
    #[serde(default = "d_true")]
    pub stop_at_turning_point: bool,
}

impl CampaignSpec {
    /// The semantic checks of Python `CampaignSpec.parse`
    /// (`sweep_driver.py` L345-353): `psi0_step > 0` — written in the
    /// negated form so NaN rejects — and `max_retries >= 0`.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) with the
    /// Python-verbatim messages when `psi0_step <= 0` (NaN rejected)
    /// or `max_retries < 0`.
    pub fn validate(&self) -> Result<(), SpecError> {
        // sweep_driver.py L345-353: `if not psi0_step > 0` (NaN rejected)
        // and `if max_retries < 0`.
        if !(self.psi0_step > 0.0) {
            return Err(invalid("[campaign] psi0_step must be > 0"));
        }
        if self.max_retries < 0 {
            return Err(invalid("[campaign] max_retries must be >= 0"));
        }
        Ok(())
    }
}

/// `[seed]` (Python `SeedSpec`, `sweep_driver.py` L374-414). `source` is
/// resolved against the spec's directory and validated to be an existing
/// solution directory by [`SeedSpec::resolve_source`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedSpec {
    /// How the first solve is seeded: analytic `from_scratch` (a fixed-ω
    /// step 0) or an existing solution directory.
    pub policy: SeedPolicy,
    /// Amplitude/ψ₀ of the analytic seed profile (step-0 param `psi0`).
    #[serde(default = "d_seed_psi0")]
    pub psi0: f64,
    /// Gaussian width of the seed in R (param `sigmaR`).
    #[serde(rename = "sigmaR", default = "d_sigma")]
    pub sigma_r: f64,
    /// Gaussian width of the seed in Z (param `sigmaZ`).
    #[serde(rename = "sigmaZ", default = "d_sigma")]
    pub sigma_z: f64,
    /// Radial extent of the seed profile (param `rExt`).
    #[serde(rename = "rExt", default = "d_r_ext")]
    pub r_ext: f64,
    /// The solution directory a `solution` seed starts from —
    /// resolved to an absolute path and validated by
    /// [`SeedSpec::resolve_source`].
    #[serde(default)]
    pub source: Option<String>,
    /// Fixed ω for the `from_scratch` step-0 seed solve (required for
    /// that policy, unused otherwise).
    #[serde(default)]
    pub w0: Option<f64>,
}

impl SeedSpec {
    /// Validate the seed and resolve `source` (Python `SeedSpec.parse`,
    /// `sweep_driver.py` L384-414): `policy = "solution"` requires a source
    /// that is an existing solution directory; `policy = "from_scratch"`
    /// requires `w0` (the fixedOmega seed solve). Relative sources are
    /// resolved against `base_dir` (the spec file's directory).
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) when
    /// `policy = "solution"` has no `source`, the source is not an
    /// existing solution directory, or `policy = "from_scratch"` lacks
    /// `w0` — each with the Python-verbatim message.
    pub fn resolve_source(&mut self, base_dir: &Path) -> Result<(), SpecError> {
        if self.policy == SeedPolicy::Solution {
            let Some(source) = self.source.clone() else {
                return Err(invalid("[seed] policy=solution requires 'source'"));
            };
            let mut resolved = PathBuf::from(&source);
            let mut text = source;
            if !resolved.is_absolute() {
                resolved = absolute_lexical(&base_dir.join(&resolved))?;
                text = resolved.to_string_lossy().into_owned();
            }
            let name_ok = resolved
                .file_name()
                .is_some_and(|n| is_solution_dir(&n.to_string_lossy()));
            if !name_ok || !resolved.is_dir() {
                return Err(invalid(format!(
                    "[seed] source is not a solution directory: {text}"
                )));
            }
            self.source = Some(text);
        }
        if self.policy == SeedPolicy::FromScratch && self.w0.is_none() {
            return Err(invalid(
                "[seed] policy=from_scratch requires 'w0' (fixedOmega seed solve)",
            ));
        }
        Ok(())
    }
}

/// `[grid]` (Python `GridSpec`, `sweep_driver.py` L417-440).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct GridRaw {
    dr: f64,
    #[serde(rename = "N")]
    n: i64,
    #[serde(default)]
    dr_max: Option<f64>,
    #[serde(default = "d_grid_order")]
    order: i64,
}

/// `[grid]`, finalized: `dr_max` defaults to `4.0 * dr`.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    /// Radial spacing (the C side sets dz = dr).
    pub dr: f64,
    /// Interior points per axis (C params `NrInterior = NzInterior = N`).
    pub n: i64,
    /// Coarseness floor — the widest dr a coarsening regrid may reach;
    /// defaults to 4× the seed dr (the §6.1 two-widening budget).
    pub dr_max: f64,
    /// Finite-difference order (2 or 4); sets the ghost-cell count via
    /// [`crate::decide::ghost_of`].
    pub order: i64,
}

impl GridSpec {
    /// Python `GridSpec.parse` (`sweep_driver.py` L425-440): `dr_max`
    /// defaults to 4× the seed dr — the domain-growth budget of design
    /// §6.1 (two coarsening regrids; going beyond requires an explicit
    /// override) — and must be `>= dr` (comparison exactly as the Python
    /// wrote it).
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) when
    /// `dr_max < dr`.
    pub fn new(dr: f64, n: i64, dr_max: Option<f64>, order: i64) -> Result<GridSpec, SpecError> {
        let dr_max = dr_max.unwrap_or(4.0 * dr);
        if dr_max < dr {
            return Err(invalid("[grid] dr_max must be >= dr"));
        }
        Ok(GridSpec {
            dr,
            n,
            dr_max,
            order,
        })
    }
}

/// `[solver]` (Python `SolverSpec`, `sweep_driver.py` L443-485 at the
/// SAN-59 head).
///
/// The `numThreads`/`precision`/`direct`/`fp32_ir` knobs postdate master
/// (SAN-32/M3, "thread count as explicit config" + the fp32+direct+IR
/// production recipe) but the repo's runbook configs already use them, so
/// this port includes them — master's `KNOWN[solver]` would reject its own
/// production configs otherwise.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverSpec {
    /// C solver-type selector (param `solverType`).
    #[serde(rename = "solverType", default = "d_one_i")]
    pub solver_type: i64,
    /// C local (block) solver selector (param `localSolver`).
    #[serde(rename = "localSolver", default = "d_one_i")]
    pub local_solver: i64,
    /// Newton convergence tolerance (param `epsilon`).
    #[serde(default = "d_epsilon")]
    pub epsilon: f64,
    /// Newton iteration cap (param `maxNewtonIter`).
    #[serde(rename = "maxNewtonIter", default = "d_50_i")]
    pub max_newton_iter: i64,
    /// Initial Newton damping λ (param `lambda0`).
    #[serde(default = "d_lambda0")]
    pub lambda0: f64,
    /// Damping floor (param `lambdaMin`).
    #[serde(default = "d_lambda_min")]
    pub lambda_min: f64,
    /// C low-rank switch (param `useLowRank`).
    #[serde(rename = "useLowRank", default = "d_zero_i")]
    pub use_low_rank: i64,
    /// OpenMP thread count for each C invocation. Default 4 = the measured
    /// sweet spot on this box (4 physical cores / 8 hyperthreads, cgroup
    /// CPU quota 4); 8 threads oversubscribes and is ~2.5x slower
    /// (`sweep_driver.py` L453-456).
    #[serde(rename = "numThreads", default = "d_threads")]
    pub num_threads: i64,
    /// Production precision recipe (M3): `precision = 32`, `direct = 1`,
    /// `fp32_ir` — halves factor memory and reaches the paper's N=400
    /// in-core. Maps onto `ROTBOSON_PRECISION`, `ROTBOSON_PARDISO_DIRECT`
    /// and `ROTBOSON_FP32_IR` env vars (`sweep_driver.py` L456-462).
    #[serde(default = "d_precision")]
    pub precision: i64,
    /// Direct+IR PARDISO path (`ROTBOSON_PARDISO_DIRECT`); coerced with
    /// Python truthiness — `direct = 1` is true.
    #[serde(default, deserialize_with = "de_py_bool")]
    pub direct: bool,
    /// Iterative-refinement passes for the fp32+direct recipe (param
    /// `fp32_ir`); only meaningful with `precision = 32` + `direct`.
    #[serde(rename = "fp32_ir", default = "d_fp32_ir")]
    pub fp32_ir: i64,
}

impl Default for SolverSpec {
    fn default() -> Self {
        SolverSpec {
            solver_type: 1,
            local_solver: 1,
            epsilon: 1.0e-8,
            max_newton_iter: 50,
            lambda0: 1.0e-3,
            lambda_min: 1.0e-5,
            use_low_rank: 0,
            num_threads: 4,
            precision: 64,
            direct: false,
            fp32_ir: 2,
        }
    }
}

impl SolverSpec {
    /// The post-parse checks of Python `SolverSpec.parse`
    /// (`sweep_driver.py` L466-472): `precision` ∈ {32, 64} and
    /// `fp32_ir >= 0`.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) with the
    /// Python-verbatim messages when `precision` is not 32 or 64, or
    /// `fp32_ir < 0`.
    pub fn validate(&self) -> Result<(), SpecError> {
        // sweep_driver.py L466-472.
        if self.precision != 32 && self.precision != 64 {
            return Err(invalid("[solver] precision must be 32 or 64"));
        }
        if self.fp32_ir < 0 {
            return Err(invalid("[solver] fp32_ir must be >= 0"));
        }
        Ok(())
    }
}

/// `[adaptivity]` (Python `AdaptivitySpec`, `sweep_driver.py` L488-541).
/// Design §5: defaults start conservative at the historical C sweep
/// thresholds so early behaviour matches what the old in-C sweep
/// tolerated.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptivitySpec {
    /// Under-resolution threshold: refine when the field's half-max width
    /// drops below this many points.
    #[serde(default = "d_hwl_min")]
    pub hwl_min: f64,
    /// The campaign's dr ÷2 refinement budget (irreversible,
    /// verify-then-commit).
    #[serde(rename = "max_refinements", default = "d_two_i")]
    pub max_refinements: i64,
    /// Down-branch tolerance: an exit-2 failure with ω within this of m
    /// is the branch's physical end (`stopped:newtonian_limit`).
    #[serde(rename = "newtonian_delta", default = "d_newtonian_delta")]
    pub newtonian_delta: f64,
    /// Support fraction at which the campaign stops for the boundary
    /// (`r99/r_bdy`; the down-direction boundary stop).
    #[serde(rename = "boundary_fraction", default = "d_boundary_fraction")]
    pub boundary_fraction: f64,
    /// Keep the interior domain when refining (the legacy fixed-N
    /// refinement shrinks it); coerced with Python truthiness.
    #[serde(
        rename = "refine_keeps_domain",
        default,
        deserialize_with = "de_py_bool"
    )]
    pub refine_keeps_domain: bool,
    /// Truncation gate: a widening (dr ×2) is only adopted when the
    /// coarser solve reproduces `ω/M_Komar/J_Komar` within this relative
    /// tolerance.
    #[serde(rename = "regrid_rtol", default = "d_regrid_rtol")]
    pub regrid_rtol: f64,
    /// Support fraction that triggers a widening (`r99/r_bdy`; must stay
    /// below `boundary_fraction` so coarsening fires before the stop).
    #[serde(rename = "support_fraction", default = "d_support_fraction")]
    pub support_fraction: f64,
    /// The campaign's dr ×2 widening budget.
    #[serde(rename = "max_widenings", default = "d_two_i")]
    pub max_widenings: i64,
}

impl Default for AdaptivitySpec {
    fn default() -> Self {
        AdaptivitySpec {
            hwl_min: 8.0,
            max_refinements: 2,
            newtonian_delta: 1.0e-2,
            boundary_fraction: 0.95,
            refine_keeps_domain: false,
            regrid_rtol: 2.0e-2,
            support_fraction: 0.85,
            max_widenings: 2,
        }
    }
}

impl AdaptivitySpec {
    /// Parse a bare `[adaptivity]` table body (TOML keys, no table
    /// header) — the analogue of the Python unit tests calling
    /// `AdaptivitySpec.parse({...})` with a dict.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) when the body is
    /// not valid TOML for the table (type errors, unknown keys) or when
    /// validation fails.
    pub fn parse_str(body: &str) -> Result<AdaptivitySpec, SpecError> {
        let a: AdaptivitySpec = toml::from_str(body)
            .map_err(|e| SpecError::TomlParse(e.to_string()))?;
        a.validate()?;
        Ok(a)
    }

    /// `sweep_driver.py` L503-528. Each comparison is written exactly as
    /// the Python wrote it, which matters for NaN: the `<=`/`<` forms let
    /// NaN through (faithful to the Python), the `not a < b < c` forms
    /// reject it.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`](crate::error::SpecError) with the
    /// Python-verbatim message for each violated range check
    /// (`hwl_min`, `newtonian_delta`, `boundary_fraction`,
    /// `support_fraction`, `regrid_rtol`, and the fraction ordering).
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.hwl_min <= 0.0 {
            return Err(invalid("[adaptivity] hwl_min must be > 0"));
        }
        if self.max_refinements < 0 {
            return Err(invalid("[adaptivity] max_refinements must be >= 0"));
        }
        if self.newtonian_delta < 0.0 {
            return Err(invalid("[adaptivity] newtonian_delta must be >= 0"));
        }
        if !(0.0 < self.boundary_fraction && self.boundary_fraction <= 1.0) {
            return Err(invalid("[adaptivity] boundary_fraction must be in (0, 1]"));
        }
        if self.regrid_rtol <= 0.0 {
            return Err(invalid("[adaptivity] regrid_rtol must be > 0"));
        }
        if !(0.0 < self.support_fraction && self.support_fraction < 1.0) {
            return Err(invalid("[adaptivity] support_fraction must be in (0, 1)"));
        }
        if self.support_fraction >= self.boundary_fraction {
            return Err(invalid(
                "[adaptivity] support_fraction must be < boundary_fraction (the coarsening \
                 trigger must fire before the campaign's boundary stop)",
            ));
        }
        if self.max_widenings < 0 {
            return Err(invalid("[adaptivity] max_widenings must be >= 0"));
        }
        Ok(())
    }
}

/// `[output]` as parsed (root unresolved).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputRaw {
    #[serde(default)]
    root: Option<String>,
    #[serde(default = "d_hdf5")]
    format: OutputFormat,
}

/// `[output]`, finalized: `root` resolved (Python `OutputSpec`,
/// `sweep_driver.py` L544-562).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSpec {
    /// Campaign root: state.json, summary.json, step param files and
    /// solutions live under it. Defaults to
    /// `<repo>/out/campaigns/<spec-stem>`; relatives resolve against the
    /// spec file's directory.
    pub root: PathBuf,
    /// On-disk solution format — `hdf5` (default) or `ascii`.
    pub format: OutputFormat,
}

// serde default functions (the Python dataclass defaults).

fn d_one_f() -> f64 {
    1.0
}
fn d_relative() -> Psi0StepMode {
    Psi0StepMode::Relative
}
fn d_max_retries() -> i64 {
    3
}
fn d_max_steps() -> i64 {
    50
}
fn d_two_i() -> i64 {
    2
}
fn d_true() -> bool {
    true
}
fn d_seed_psi0() -> f64 {
    0.01
}
fn d_sigma() -> f64 {
    4.0
}
fn d_r_ext() -> f64 {
    12.0
}
fn d_grid_order() -> i64 {
    4
}
fn d_one_i() -> i64 {
    1
}
fn d_50_i() -> i64 {
    50
}
fn d_epsilon() -> f64 {
    1.0e-8
}
fn d_lambda0() -> f64 {
    1.0e-3
}
fn d_lambda_min() -> f64 {
    1.0e-5
}
fn d_zero_i() -> i64 {
    0
}
fn d_threads() -> i64 {
    4
}
fn d_precision() -> i64 {
    64
}
fn d_fp32_ir() -> i64 {
    2
}
fn d_hwl_min() -> f64 {
    8.0
}
fn d_newtonian_delta() -> f64 {
    1.0e-2
}
fn d_boundary_fraction() -> f64 {
    0.95
}
fn d_regrid_rtol() -> f64 {
    2.0e-2
}
fn d_support_fraction() -> f64 {
    0.85
}
fn d_hdf5() -> OutputFormat {
    OutputFormat::Hdf5
}

// ---------------------------------------------------------------------------
// The assembled spec
// ---------------------------------------------------------------------------

/// A fully-validated campaign spec (Python `Spec`).
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// `[campaign]` — direction, targets, stepping and step budgets.
    pub campaign: CampaignSpec,
    /// `[seed]` — how the first solve is seeded.
    pub seed: SeedSpec,
    /// `[grid]` — the initial grid (dr drifts via regrids; see
    /// [`crate::state::current_grid`]).
    pub grid: GridSpec,
    /// `[solver]` — C-solver knobs (defaults apply when the table is
    /// absent).
    pub solver: SolverSpec,
    /// `[output]` — campaign root and solution format.
    pub output: OutputSpec,
    /// `[adaptivity]` — regrid gates and budgets.
    pub adaptivity: AdaptivitySpec,
    /// The resume guard: SHA-256 of the canonicalized spec text (see
    /// [`spec_hash`]) — a state.json carrying a different hash refuses to
    /// resume.
    pub spec_hash: String,
}

impl Spec {
    /// Copy with the grid's dr (and optionally N) replaced — regrid
    /// rendering re-solves on a different grid (Python `Spec.with_grid`).
    #[must_use]
    pub fn with_grid(&self, dr: f64, n: Option<i64>) -> Spec {
        let mut out = self.clone();
        out.grid.dr = dr;
        if let Some(n) = n {
            out.grid.n = n;
        }
        out
    }
}

/// The whole TOML document with `deny_unknown_fields` as a second line of
/// defense (the explicit `KNOWN` scan above already rejects unknown
/// tables/keys with the Python-verbatim messages).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSpec {
    #[serde(default)]
    campaign: Option<CampaignSpec>,
    #[serde(default)]
    seed: Option<SeedSpec>,
    #[serde(default)]
    grid: Option<GridRaw>,
    #[serde(default)]
    solver: Option<SolverSpec>,
    #[serde(default)]
    output: Option<OutputRaw>,
    #[serde(default)]
    adaptivity: Option<AdaptivitySpec>,
}

/// Load and validate a campaign spec (Python `load_spec`,
/// `sweep_driver.py` L584-616): unknown tables/keys rejected first, then
/// each table parsed and validated in the Python's order, then the
/// spec hash computed over the canonicalized text.
///
/// `repo_root` is the repository root (the Python derives it from its own
/// file location, `REPO = Path(__file__).resolve().parent.parent`); it
/// only feeds the default output root `<repo>/out/campaigns/<stem>`.
///
/// # Errors
///
/// Returns [`SpecError`](crate::error::SpecError) on I/O failure reading
/// the spec file, invalid TOML, unknown tables/keys, or any table's
/// validation failure — each with the Python-verbatim message.
pub fn load_spec(path: &Path, repo_root: &Path) -> Result<Spec, SpecError> {
    let raw_bytes = std::fs::read(path)?;
    let text = std::str::from_utf8(&raw_bytes).map_err(|_| SpecError::NonUtf8)?;

    let doc: toml::Table = toml::from_str(text).map_err(|e| SpecError::TomlParse(e.to_string()))?;
    check_known(&doc)?;
    let raw: RawSpec = toml::from_str(text).map_err(|e| SpecError::TomlParse(e.to_string()))?;

    // Per-table parse + validation, in the Python's assembly order
    // (sweep_driver.py L598-605). A missing table is the Python's
    // "missing required key" path for tables with required fields;
    // solver/output/adaptivity are fully defaulted.
    let Some(campaign) = raw.campaign else {
        return Err(invalid("[campaign] missing required key 'l'"));
    };
    campaign.validate()?;

    let Some(mut seed) = raw.seed else {
        return Err(invalid("[seed] missing required key 'policy'"));
    };
    let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
    seed.resolve_source(base_dir)?;

    let grid = match raw.grid {
        Some(g) => GridSpec::new(g.dr, g.n, g.dr_max, g.order)?,
        None => return Err(invalid("[grid] missing required keys 'dr' and/or 'N'")),
    };

    let solver = match raw.solver {
        Some(s) => {
            s.validate()?;
            s
        }
        None => SolverSpec::default(),
    };

    let output = match raw.output {
        Some(o) => resolve_output(o.root, o.format, path, repo_root)?,
        None => resolve_output(None, OutputFormat::Hdf5, path, repo_root)?,
    };

    let adaptivity = match raw.adaptivity {
        Some(a) => {
            a.validate()?;
            a
        }
        None => AdaptivitySpec::default(),
    };

    Ok(Spec {
        campaign,
        seed,
        grid,
        solver,
        output,
        adaptivity,
        spec_hash: spec_hash(text),
    })
}

fn check_known(doc: &toml::Table) -> Result<(), SpecError> {
    let mut unknown: Vec<String> = doc
        .keys()
        .filter(|k| !KNOWN.iter().any(|(t, _)| *t == k.as_str()))
        .cloned()
        .collect();
    unknown.sort();
    if !unknown.is_empty() {
        return Err(SpecError::UnknownTables(unknown));
    }
    for (table, allowed) in KNOWN {
        if let Some(v) = doc.get(*table) {
            let keys: Vec<String> = match v {
                toml::Value::Table(t) => t.keys().cloned().collect(),
                // The Python crashes on a non-table value here (TypeError);
                // a clean typed error is the no-panic equivalent.
                _ => return Err(invalid(format!("[{table}] must be a TOML table"))),
            };
            let mut extra: Vec<String> = keys
                .into_iter()
                .filter(|k| !allowed.contains(&k.as_str()))
                .collect();
            extra.sort();
            if !extra.is_empty() {
                return Err(SpecError::UnknownKeys {
                    table,
                    keys: extra,
                });
            }
        }
    }
    Ok(())
}

fn resolve_output(
    root: Option<String>,
    format: OutputFormat,
    spec_path: &Path,
    repo_root: &Path,
) -> Result<OutputSpec, SpecError> {
    let resolved = match root {
        // Python default: REPO / "out" / "campaigns" / spec stem.
        None => repo_root
            .join("out")
            .join("campaigns")
            .join(spec_path.file_stem().unwrap_or_default()),
        Some(r) => {
            let p = PathBuf::from(r);
            if p.is_absolute() {
                p
            } else {
                absolute_lexical(&spec_path.parent().unwrap_or_else(|| Path::new("")).join(p))?
            }
        }
    };
    Ok(OutputSpec {
        root: resolved,
        format,
    })
}

// ---------------------------------------------------------------------------
// Spec hash (the resume guard)
// ---------------------------------------------------------------------------

/// The campaign spec's content hash, matching the Python byte-for-byte
/// (`sweep_driver.py` L608-616): SHA-256 over the spec text with
/// `max_steps`/`max_retries` lines and comment lines removed. Raising a
/// runtime limit must not invalidate the physics state of a running
/// campaign; any physics-affecting change does.
pub fn spec_hash(raw_text: &str) -> String {
    sha256_hex(canonical_spec_text(raw_text).as_bytes())
}

/// Strip the runtime-control lines, then the comment lines — the exact
/// semantics of Python's
/// `comments.sub("", control.sub("", raw_bytes.decode()))` with
/// `re.MULTILINE` patterns `^\s*(max_steps|max_retries)\s*=.*$` and
/// `^\s*#.*$`. Each pass runs over the previous pass's output (as `re.sub`
/// does); the greedy `\s*` may swallow the preceding newline when an
/// empty line precedes a control/comment line, which this scanner
/// reproduces.
pub fn canonical_spec_text(raw: &str) -> String {
    strip_lines(&strip_control_lines(raw), false)
}

fn strip_control_lines(text: &str) -> String {
    strip_lines(text, true)
}

/// One `re.sub(r"^\s*<pattern>$", "", text)` pass over a whole string.
/// `control = true` strips `^\s*(max_steps|max_retries)\s*=.*$` lines;
/// `control = false` strips `^\s*#.*$` comment lines. The scan reproduces
/// `re`'s semantics exactly: matches may only start at position 0 or
/// right after a newline, the greedy `\s*` swallows newlines (an empty
/// line directly above a control/comment line goes with it), and a match
/// ends at the next newline (which stays).
fn strip_lines(text: &str, control: bool) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        // `^` in MULTILINE: position 0 or right after a newline.
        if i == 0 || chars[i - 1] == '\n' {
            if let Some(end) = match_line_start(&chars, i, control) {
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn match_line_start(chars: &[char], start: usize, control: bool) -> Option<usize> {
    let mut q = start;
    while q < chars.len() && is_re_space(chars[q]) {
        q += 1;
    }
    if control {
        for kw in ["max_steps", "max_retries"] {
            if starts_with_at(chars, q, kw) {
                let mut r = q + kw.chars().count();
                while r < chars.len() && is_re_space(chars[r]) {
                    r += 1;
                }
                if r < chars.len() && chars[r] == '=' {
                    return Some(line_end(chars, r + 1));
                }
            }
        }
        None
    } else if q < chars.len() && chars[q] == '#' {
        Some(line_end(chars, q + 1))
    } else {
        None
    }
}

fn starts_with_at(chars: &[char], at: usize, pat: &str) -> bool {
    pat.chars()
        .enumerate()
        .all(|(i, c)| chars.get(at + i) == Some(&c))
}

fn line_end(chars: &[char], from: usize) -> usize {
    let mut e = from;
    while e < chars.len() && chars[e] != '\n' {
        e += 1;
    }
    e
}

fn is_re_space(c: char) -> bool {
    // Python `\s` for str patterns: [ \t\n\r\f\v] (+unicode spaces; TOML
    // specs are ASCII — the ASCII set covers the re behaviour here).
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}')
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// `Path.resolve()` for the paths a campaign spec deals with: absolute
/// (via the process CWD when relative), then lexically normalized.
///
/// Deviation from Python: `Path.resolve()` also resolves symlinks; this
/// does not (state.json stores the resolved string either way, and
/// campaign roots/seeds are not symlinked in practice).
///
/// # Errors
///
/// Never fails; the `Result` shape exists for uniform `?` handling at
/// the call sites.
pub fn absolute_lexical(p: &Path) -> Result<PathBuf, SpecError> {
    use std::path::Component;
    let mut abs = p.to_path_buf();
    if !abs.is_absolute() {
        let cwd = std::env::current_dir()?;
        abs = cwd.join(abs);
    }
    let mut stack: Vec<Component> = Vec::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    let mut out = PathBuf::new();
    for c in stack {
        out.push(c.as_os_str());
    }
    Ok(out)
}

/// Solution-directory name check (`rotboson_io.py`'s `SOLUTION_DIR_RE`:
/// `^l=\d+,w=[\d.Ee+-]+,dr=[\d.Ee+-]+,N=\d+$`), used to validate
/// `[seed] source`.
pub fn is_solution_dir(name: &str) -> bool {
    fn take_while(s: &str, pred: impl Fn(u8) -> bool) -> Option<(&str, &str)> {
        let n = s.bytes().position(|b| !pred(b)).unwrap_or(s.len());
        (n > 0).then(|| s.split_at(n))
    }
    let digits = |b: u8| b.is_ascii_digit();
    let expnum = |b: u8| matches!(b, b'0'..=b'9' | b'.' | b'E' | b'e' | b'+' | b'-');

    let Some((_, rest)) = take_while(name.strip_prefix("l=").unwrap_or(""), digits) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(",w=") else {
        return false;
    };
    let Some((_, rest)) = take_while(rest, expnum) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(",dr=") else {
        return false;
    };
    let Some((_, rest)) = take_while(rest, expnum) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(",N=") else {
        return false;
    };
    match take_while(rest, digits) {
        Some((digits, rest)) => rest.is_empty() && !digits.is_empty(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- canonicalization: pinned against Python re.sub ----------------

    /// Expected outputs computed with the actual Python driver logic
    /// (`re.MULTILINE` subs, greedy `\s*` crossing newlines included).
    #[test]
    fn canonical_text_matches_python_re_sub() {
        let cases: &[(&str, &str)] = &[
            ("a = 1\n\n# comment\n", "a = 1\n\n"),
            ("max_steps\n= 5\n", "\n"),
            ("  max_steps=7\nx=1\n", "\nx=1\n"),
            ("x = 1\nmax_retries = 2\n# c\ny = 2\n", "x = 1\n\ny = 2\n"),
            ("\nmax_steps = 5\n", "\n"),
            ("a=1\n\n\n  # tail comment\nb=2\n", "a=1\n\nb=2\n"),
            ("[campaign]\nmax_steps = 300\nl = 1\n", "[campaign]\n\nl = 1\n"),
            ("max_steps=1\nmax_steps=2\n", "\n\n"),
        ];
        for (input, expected) in cases {
            assert_eq!(&canonical_spec_text(input), expected, "input: {input:?}");
        }
    }

    /// The hash of the repo's real campaign configs, computed with the
    /// Python driver's own code (hashlib + the two re subs). If this test
    /// fails after editing those configs, recompute the expectation with:
    ///   python3 -c "import hashlib,re; ..."
    #[test]
    fn spec_hash_matches_python_on_real_configs() {
        let runbook = include_str!("../../../configs/runbook_l1_down.toml");
        assert_eq!(
            spec_hash(runbook),
            "f4fe0ab13ce3629f58e33695698e063a60835b5d5888e31075b0cf5bf6539898"
        );
        let waypoint = include_str!("../../../configs/l2_waypoint_w09.toml");
        assert_eq!(
            spec_hash(waypoint),
            "67e056c2b7ab7befc84cdce89f12d20c546746a1b68a6c718eec0ffdf8bd0fb1"
        );
    }

    /// The hash ignores runtime controls and comment *content* — raising a
    /// limit or rewording commentary must not invalidate a running
    /// campaign's physics state; a physics-affecting change does.
    /// (Faithful to the Python's `re.sub` semantics: a stripped line still
    /// leaves its newline, so adding/removing lines moves the hash — the
    /// documented contract covers *edits*, not line-count changes.)
    #[test]
    fn hash_sensitivity() {
        let base = concat!(
            "# runbook for l=1\n",
            "[campaign]\n",
            "l = 1\n",
            "direction = \"up\"\n",
            "psi0_target = 0.6\n",
            "psi0_step = 0.03\n",
            "max_steps = 300\n",
            "\n",
            "[grid]\n",
            "dr = 0.125\n",
            "N = 128\n",
        );
        let reworded = base.replace("# runbook for l=1", "# runbook for l=1 (retuned wording)");
        assert_eq!(spec_hash(base), spec_hash(&reworded));
        let raised = base.replace("max_steps = 300", "max_steps = 500");
        assert_eq!(spec_hash(base), spec_hash(&raised));
        let physics = base.replace("psi0_step = 0.03", "psi0_step = 0.04");
        assert_ne!(spec_hash(base), spec_hash(&physics));
    }

    // ---- parsing --------------------------------------------------------

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rbs-driver-spec-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_spec(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    const GOOD: &str = concat!(
        "[campaign]\n",
        "l = 1\n",
        "direction = \"up\"\n",
        "psi0_target = 0.6\n",
        "psi0_step = 0.03\n",
        "\n",
        "[seed]\n",
        "policy = \"from_scratch\"\n",
        "w0 = 0.95\n",
        "\n",
        "[grid]\n",
        "dr = 0.125\n",
        "N = 128\n",
        "\n",
        "[solver]\n",
        "precision = 32\n",
        "direct = true\n",
        "\n",
        "[output]\n",
        "root = \"runs/demo\"\n",
        "format = \"hdf5\"\n",
        "\n",
        "[adaptivity]\n",
        "hwl_min = 14\n",
    );

    #[test]
    fn parses_a_full_spec_with_defaults() {
        let dir = tmpdir("full");
        let path = write_spec(&dir, "demo.toml", GOOD);
        let spec = load_spec(&path, &dir).unwrap();
        assert_eq!(spec.campaign.l, 1);
        assert_eq!(spec.campaign.direction, Direction::Up);
        assert_eq!(spec.campaign.m, 1.0);
        assert_eq!(spec.campaign.max_retries, 3);
        assert_eq!(spec.campaign.max_steps, 50);
        assert_eq!(spec.campaign.psi0_step_mode, Psi0StepMode::Relative);
        assert_eq!(spec.campaign.fixed_phi_r, 2);
        assert!(spec.campaign.stop_at_turning_point);
        assert_eq!(spec.seed.policy, SeedPolicy::FromScratch);
        assert_eq!(spec.seed.w0, Some(0.95));
        assert_eq!(spec.seed.sigma_r, 4.0);
        // dr_max defaults to 4x dr (design §6.1 budget).
        assert_eq!(spec.grid.dr_max, 4.0 * 0.125);
        assert_eq!(spec.grid.order, 4);
        // solver: the SAN-32/M3 knobs parse; direct uses Python
        // truthiness; the rest default.
        assert_eq!(spec.solver.precision, 32);
        assert!(spec.solver.direct);
        assert_eq!(spec.solver.num_threads, 4);
        assert_eq!(spec.solver.fp32_ir, 2);
        // adaptivity: defaults except hwl_min.
        assert_eq!(spec.adaptivity.hwl_min, 14.0);
        assert!(!spec.adaptivity.refine_keeps_domain);
        assert_eq!(spec.adaptivity.support_fraction, 0.85);
        // output root: relative to the spec's directory, lexically clean.
        assert_eq!(spec.output.root, dir.join("runs/demo"));
        assert_eq!(spec.output.format.to_string(), "hdf5");
        assert_eq!(spec.spec_hash.len(), 64);
    }

    #[test]
    fn output_root_defaults_to_repo_campaigns_dir() {
        let dir = tmpdir("outroot");
        let text = GOOD.replace("[output]\nroot = \"runs/demo\"\nformat = \"hdf5\"\n\n", "");
        let path = write_spec(&dir, "camp.toml", &text);
        let spec = load_spec(&path, Path::new("/repo")).unwrap();
        assert_eq!(spec.output.root, Path::new("/repo/out/campaigns/camp"));
    }

    #[test]
    fn unknown_tables_and_keys_are_rejected() {
        let dir = tmpdir("unknown");
        let path = write_spec(&dir, "a.toml", "[bogus]\nx = 1\n");
        let err = load_spec(&path, &dir).unwrap_err();
        assert_eq!(err.to_string(), "unknown spec tables: ['bogus']");

        let path = write_spec(&dir, "b.toml", &format!("{GOOD}\n[campaign2]\nx = 1\n"));
        let err = load_spec(&path, &dir).unwrap_err();
        assert_eq!(err.to_string(), "unknown spec tables: ['campaign2']");

        let bad_key = GOOD.replace("psi0_step = 0.03", "psi0_step = 0.03\nbogus = 1");
        let path = write_spec(&dir, "c.toml", &bad_key);
        let err = load_spec(&path, &dir).unwrap_err();
        assert_eq!(err.to_string(), "[campaign] unknown keys: ['bogus']");

        let bad_key = GOOD.replace("hwl_min = 14", "hwl_min = 14\nnoise = 1");
        let path = write_spec(&dir, "d.toml", &bad_key);
        let err = load_spec(&path, &dir).unwrap_err();
        assert_eq!(err.to_string(), "[adaptivity] unknown keys: ['noise']");
    }

    #[test]
    fn wrong_types_are_hard_errors() {
        let dir = tmpdir("types");
        let bad = GOOD.replace("l = 1", "l = \"one\"");
        let path = write_spec(&dir, "t1.toml", &bad);
        assert!(matches!(
            load_spec(&path, &dir),
            Err(SpecError::TomlParse(_))
        ));
        let bad = GOOD.replace("psi0_step = 0.03", "psi0_step = \"fast\"");
        let path = write_spec(&dir, "t2.toml", &bad);
        assert!(load_spec(&path, &dir).is_err());
        // TOML has no int-as-bool confusion: strict booleans reject ints.
        let bad = GOOD.replace("direct = true", "direct = 1");
        let path = write_spec(&dir, "t3.toml", &bad);
        let spec = load_spec(&path, &dir).unwrap();
        // ... but the Python coerces `direct` with bool(): 1 stays true.
        assert!(spec.solver.direct);
    }

    #[test]
    fn vocabulary_misses_use_the_python_messages() {
        let dir = tmpdir("vocab");
        let bad = GOOD.replace("direction = \"up\"", "direction = \"sideways\"");
        let path = write_spec(&dir, "v1.toml", &bad);
        assert!(load_spec(&path, &dir)
            .unwrap_err()
            .to_string()
            .contains(r#"direction must be "up" or "down""#));
        let bad = GOOD.replace("psi0_step = 0.03", "psi0_step = 0.03\npsi0_step_mode = \"geometric\"");
        let path = write_spec(&dir, "v2.toml", &bad);
        assert!(load_spec(&path, &dir)
            .unwrap_err()
            .to_string()
            .contains(r#"psi0_step_mode must be "absolute" or "relative""#));
        let bad = GOOD.replace("format = \"hdf5\"", "format = \"json\"");
        let path = write_spec(&dir, "v3.toml", &bad);
        assert!(load_spec(&path, &dir)
            .unwrap_err()
            .to_string()
            .contains(r#"format must be "hdf5" or "ascii""#));
        let bad = GOOD.replace("policy = \"from_scratch\"", "policy = \"guess\"");
        let path = write_spec(&dir, "v4.toml", &bad);
        assert!(load_spec(&path, &dir)
            .unwrap_err()
            .to_string()
            .contains(r#"policy must be "from_scratch" or "solution""#));
    }

    #[test]
    fn semantic_validations_match_python() {
        let dir = tmpdir("semantic");
        let bad = GOOD.replace("psi0_step = 0.03", "psi0_step = 0.0");
        let path = write_spec(&dir, "s1.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[campaign] psi0_step must be > 0"
        );
        let bad = GOOD.replace("psi0_step = 0.03", "psi0_step = 0.03\nmax_retries = -1");
        let path = write_spec(&dir, "s2.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[campaign] max_retries must be >= 0"
        );
        let bad = GOOD.replace("dr = 0.125", "dr = 0.125\ndr_max = 0.1");
        let path = write_spec(&dir, "s3.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[grid] dr_max must be >= dr"
        );
        let bad = GOOD.replace("precision = 32", "precision = 16");
        let path = write_spec(&dir, "s4.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[solver] precision must be 32 or 64"
        );
        let bad = GOOD.replace("precision = 32", "precision = 32\nfp32_ir = -1");
        let path = write_spec(&dir, "s5.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[solver] fp32_ir must be >= 0"
        );
    }

    #[test]
    fn nan_quirks_are_faithful() {
        // `hwl_min = nan` passes (Python: `hwl_min <= 0` is False for NaN).
        let a = AdaptivitySpec::parse_str("hwl_min = nan").unwrap();
        assert!(a.hwl_min.is_nan());
        let dir = tmpdir("nan");
        let bad = GOOD.replace("hwl_min = 14", "hwl_min = nan");
        let path = write_spec(&dir, "n1.toml", &bad);
        let spec = load_spec(&path, &dir).unwrap();
        assert!(spec.adaptivity.hwl_min.is_nan());
        // `psi0_step = nan` is rejected (Python: `not nan > 0` is True).
        let bad = GOOD.replace("psi0_step = 0.03", "psi0_step = nan");
        let path = write_spec(&dir, "n2.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[campaign] psi0_step must be > 0"
        );
    }

    #[test]
    fn seed_requirements() {
        let dir = tmpdir("seed");
        // from_scratch without w0.
        let bad = GOOD.replace("w0 = 0.95\n", "");
        let path = write_spec(&dir, "w1.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[seed] policy=from_scratch requires 'w0' (fixedOmega seed solve)"
        );
        // solution without source.
        let sol = GOOD.replace("policy = \"from_scratch\"", "policy = \"solution\"");
        let sol = sol.replace("w0 = 0.95\n", "");
        let path = write_spec(&dir, "w2.toml", &sol);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            "[seed] policy=solution requires 'source'"
        );
        // solution with a real solution directory (relative source).
        let sol_name = "l=1,w=9.50000E-01,dr=1.25000E-01,N=0128";
        std::fs::create_dir_all(dir.join(sol_name)).unwrap();
        let sol = sol.replace(
            "[grid]",
            &format!("source = \"{sol_name}\"\n\n[grid]"),
        );
        let path = write_spec(&dir, "w3.toml", &sol);
        let spec = load_spec(&path, &dir).unwrap();
        // resolved to an absolute, lexically-cleaned path.
        assert_eq!(spec.seed.source.as_deref(), Some(dir.join(sol_name).to_str().unwrap()));
        // a name that is not a solution directory (the relative source is
        // resolved first, so the error carries the resolved path — as the
        // Python does).
        let bad = sol.replace(
            &format!("source = \"{sol_name}\""),
            "source = \"not-a-solution\"",
        );
        let path = write_spec(&dir, "w4.toml", &bad);
        assert_eq!(
            load_spec(&path, &dir).unwrap_err().to_string(),
            format!(
                "[seed] source is not a solution directory: {}",
                dir.join("not-a-solution").to_string_lossy()
            )
        );
    }

    #[test]
    fn solution_dir_names() {
        assert!(is_solution_dir("l=1,w=9.50000E-01,dr=1.25000E-01,N=0128"));
        assert!(is_solution_dir("l=2,w=6.45894E-01,dr=6.25000E-02,N=0128"));
        assert!(is_solution_dir("l=4,w=1.5e+00,dr=1.0E-01,N=0400"));
        assert!(!is_solution_dir("logs"));
        assert!(!is_solution_dir("l=1,w=X.XXXXXE-01,dr=1.25000E-01,N=0128"));
        assert!(!is_solution_dir(""));
        assert!(!is_solution_dir("l=1,w=9.5,dr=0.1,N=128,extra"));
    }

    #[test]
    fn with_grid_replaces_dr_and_n() {
        let dir = tmpdir("withgrid");
        let path = write_spec(&dir, "g.toml", GOOD);
        let spec = load_spec(&path, &dir).unwrap();
        let finer = spec.with_grid(0.0625, Some(256));
        assert_eq!(finer.grid.dr, 0.0625);
        assert_eq!(finer.grid.n, 256);
        // dr-only: N unchanged (the legacy fixed-N regrid path).
        let finer = spec.with_grid(0.0625, None);
        assert_eq!(finer.grid.n, 128);
        // The rest of the spec travels along.
        assert_eq!(finer.campaign.l, 1);
        assert_eq!(finer.spec_hash, spec.spec_hash);
    }
}
