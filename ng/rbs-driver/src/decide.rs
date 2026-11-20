//! Adaptive decision layer (design §4–6) — the pure functions of
//! `tools/sweep_driver.py`, unit-tested there in
//! `tests/test_driver_decisions.py` and ported case-for-case in
//! `tests/driver_decisions.rs`.
//!
//! These functions encode campaign semantics agents tuned across
//! SAN-14→SAN-30 (trend-noise regrid, boundary guards yielding to
//! widening, turning-point detection across the last three branch
//! points). The semantics — including the quirks — are preserved
//! exactly; tuned constants are cited against the Python source
//! (`sweep_driver.py` at the SAN-59 head `06b4ff2`).

use crate::error::DriverError;
use crate::fit::polyfit;
use crate::spec::{CampaignSpec, Spec};
use crate::state::{omega_of_last, CampaignState, RelDiff, StepRecord};
use crate::vocab::{signal_name, Direction, StepMode, StopReason, TIMEOUT_EXIT};

/// Ghost-cell count for a finite-difference order (C: order 2 → 1,
/// 4 → 2). Python `ghost_of` (`sweep_driver.py` L1914-1916).
#[must_use]
pub fn ghost_of(order: i64) -> i64 {
    order / 2
}

// ---------------------------------------------------------------------------
// decide_action — v2: the single refinement rule
// ---------------------------------------------------------------------------

/// Diagnostics feeding `decide_action` (design §5; Python
/// `StepDiagnostics`). `hwl`/`rr_phi_max` are `None` when the step's
/// scalars were unreadable — an under-resolved peak cannot be proven, so
/// `None` means "do not refine".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepDiagnostics {
    /// Half-max width of the field in grid points straddling the peak
    /// (C scalar `hwl_resolution`).
    pub hwl: Option<f64>,
    /// Radius of the φ peak (C scalar `rr_phi_max`) — the axis-hug
    /// input.
    pub rr_phi_max: Option<f64>,
    /// Current grid spacing; the axis-hug rule compares the peak radius
    /// against `(hwl/2)·dr`.
    pub dr: f64,
    /// Refinements left in the campaign's dr ÷2 budget; ≤ 0 disables the
    /// rule.
    pub refinements_left: i64,
    /// `[adaptivity] hwl_min` — the under-resolution threshold in points.
    pub hwl_min: f64,
}

/// Adaptive decision layer output (`decide_action`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Keep stepping on the current grid.
    Ok,
    /// Refine: regrid at dr ÷2 before the next solve.
    RegridFiner,
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Ok => f.write_str("ok"),
            Action::RegridFiner => f.write_str("regrid_finer"),
        }
    }
}

/// Decision policy: exactly one active rule — refine when the field's
/// peak is under-resolved (`sweep_driver.py` `decide_action`, L1944-1976).
///
/// Under-resolution means either the half-max width of the field drops
/// below `hwl_min` points (spikes), or the peak sits closer to the axis
/// than half its own width (`rr_phi_max < (hwl/2)·dr`) — a grid-relative
/// form of the old axis-hug rule: within a half-width of the origin the
/// φ ∝ r^l power law dominates the profile and the peak-location fit
/// becomes axis-biased. The criterion is self-consistent under
/// refinement (hwl grows as dr shrinks, so it converges to a statement
/// about the continuum profile, not the grid).
///
/// Everything else is fixed relative stepping: no growth, no damping
/// statistics, no boundary regrids. `refinements_left` bounds the dr ÷2
/// refinements per campaign (irreversible, verify-then-commit).
#[must_use]
pub fn decide_action(diag: &StepDiagnostics) -> Action {
    if diag.refinements_left <= 0 {
        return Action::Ok;
    }
    let Some(hwl) = diag.hwl else {
        return Action::Ok;
    };
    if hwl < diag.hwl_min {
        return Action::RegridFiner;
    }
    if let Some(rr) = diag.rr_phi_max {
        if rr < (hwl / 2.0) * diag.dr {
            return Action::RegridFiner;
        }
    }
    Action::Ok
}

// ---------------------------------------------------------------------------
// Coarsening (down-campaign widening) and boundary classification
// ---------------------------------------------------------------------------

/// Down-campaign coarsening decision (`sweep_driver.py`
/// `decide_coarsening`, L1423-1443). When the field's support fills the
/// domain (`r99/r_bdy` > `support_fraction`), widen the grid (dr ×2,
/// domain grows) — unless the coarseness floor (`dr_max`) or the regrid
/// budget (`widenings_left`) is exhausted, in which case the campaign
/// stops cleanly: `stopped:domain_budget`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarseningDecision {
    /// Support still fits; keep stepping.
    Ok,
    /// Widen: regrid at dr ×2 (the domain grows).
    RegridCoarser,
    /// Support fills the domain but the coarseness floor or the regrid
    /// budget blocks widening — stop (`stopped:domain_budget`).
    StopBudget,
}

impl std::fmt::Display for CoarseningDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoarseningDecision::Ok => f.write_str("ok"),
            CoarseningDecision::RegridCoarser => f.write_str("regrid_coarser"),
            CoarseningDecision::StopBudget => f.write_str("stop_budget"),
        }
    }
}

/// Classify a down-campaign step against the widening policy (Python
/// `decide_coarsening`, `sweep_driver.py` L1423-1443). `r99_fraction` is the
/// field's support relative to the boundary (`r99/r_bdy`); `None` means the
/// scalars were unreadable — full support cannot be proven, so the
/// answer is [`CoarseningDecision::Ok`]. Support over
/// `support_fraction` widens (dr ×2) unless dr ×2 would exceed the
/// coarseness floor `dr_max` or the `widenings_left` budget is spent,
/// either of which stops the campaign.
#[must_use]
pub fn decide_coarsening(
    r99_fraction: Option<f64>,
    support_fraction: f64,
    dr: f64,
    dr_max: f64,
    widenings_left: i64,
) -> CoarseningDecision {
    match r99_fraction {
        None => CoarseningDecision::Ok,
        Some(f) if f <= support_fraction => CoarseningDecision::Ok,
        Some(_) => {
            if dr * 2.0 > dr_max || widenings_left <= 0 {
                CoarseningDecision::StopBudget
            } else {
                CoarseningDecision::RegridCoarser
            }
        }
    }
}

/// Truncation-proxy gate for a COARSENING regrid (dr ×2): the coarser
/// solve must reproduce `ω/`M_Komar`/`J_Komar`` within `regrid_rtol` — widening
/// is only adopted when it did not corrupt the solution. Refinement has
/// no such gate (its purpose is to reduce the old grid's error, which is
/// recorded). Python `coarsening_acceptable` (`sweep_driver.py`
/// L1413-1420): an empty map rejects (unreadable scalars ⇒ no proof).
#[must_use]
pub fn coarsening_acceptable(rel: &RelDiff, regrid_rtol: f64) -> bool {
    !rel.is_empty() && rel.iter().all(|(_, v)| *v <= regrid_rtol)
}

/// Would a finer regrid amputate the field's support? (Python
/// `would_amputate`, `sweep_driver.py` L1382-1410.) Two cases, deliberately
/// asymmetric — the guard decoupling:
///
/// * **Domain-keeping refinement** (`new_n` > `src_n`: dr ÷2 with N×2): the
///   interior domain is unchanged — the field fits by construction.
///   Only outright amputation blocks it; the `boundary_fraction` margin
///   would be self-defeating here (the ghost-inclusive boundary moves
///   inward ~1.5% while the solve stays valid).
/// * **Legacy fixed-N refinement** (domain shrinks): the conservative
///   margin applies — never re-solve onto a domain the support barely
///   fits.
#[must_use]
pub fn would_amputate(
    src_r99: Option<f64>,
    src_dr: f64,
    src_n: i64,
    new_dr: f64,
    new_n: i64,
    order: i64,
    boundary_fraction: f64,
) -> bool {
    let Some(src_r99) = src_r99 else {
        return false;
    };
    if new_dr >= src_dr {
        return false;
    }
    let new_boundary = (new_n + 2 * ghost_of(order)) as f64 * new_dr;
    if new_n > src_n {
        return src_r99 >= new_boundary;
    }
    src_r99 > boundary_fraction * new_boundary
}

/// Should a failed step be classified as a boundary end? (Python
/// `boundary_failure_stop`, `sweep_driver.py` L1446-1463.) The last GOOD
/// solution's support already fills the domain — the failure is
/// boundary-driven (the field has no room; refinement is spent) and the
/// campaign ends cleanly at the best achievable grid instead of
/// recording a solver failure. The `dr`/`N` truthiness of the Python
/// (`not last_good_dr`) is preserved: a zero dr or N means "unknown".
#[must_use]
pub fn boundary_failure_stop(
    last_good_r99: Option<f64>,
    last_good_dr: Option<f64>,
    last_good_n: Option<i64>,
    order: i64,
    boundary_fraction: f64,
) -> bool {
    let (Some(r99), Some(dr), Some(n)) = (last_good_r99, last_good_dr, last_good_n) else {
        return false;
    };
    if dr == 0.0 || n == 0 {
        return false;
    }
    let r_bdy = (n + 2 * ghost_of(order)) as f64 * dr;
    r99 / r_bdy > boundary_fraction
}

// ---------------------------------------------------------------------------
// Turning point (design §6.2)
// ---------------------------------------------------------------------------

/// Design §6.2: has the branch's minimum-ω turning point been crossed?
/// (Python `detect_turning_point`, `sweep_driver.py` L1979-2001.)
///
/// Looks at the last three fixedPhi/seed steps with distinct increasing
/// ψ₀ and checks that the central slope dω/dψ₀ changed sign (decreasing
/// → strictly increasing). Only meaningful in the up direction; ω rises
/// monotonically toward m on the down side.
///
/// Note (faithful to the Python): steps are filtered by mode and the
/// presence of ψ₀/ω only — *not* by exit code; in practice failed steps
/// carry no ψ₀.
#[must_use]
pub fn detect_turning_point(steps: &[StepRecord]) -> bool {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for s in steps {
        if !matches!(s.mode, StepMode::FixedPhi | StepMode::Seed) {
            continue;
        }
        if let (Some(p), Some(w)) = (s.psi0, s.omega) {
            pts.push((p, w));
        }
    }
    if pts.len() < 3 {
        return false;
    }
    let n = pts.len();
    let (p1, w1) = pts[n - 3];
    let (p2, w2) = pts[n - 2];
    let (p3, w3) = pts[n - 1];
    if !(p2 > p1 && p3 > p2) {
        return false;
    }
    let s1 = (w2 - w1) / (p2 - p1);
    let s2 = (w3 - w2) / (p3 - p2);
    s1 < 0.0 && 0.0 < s2
}

/// Design §6.2 localization report (Python `turning_point_estimate`'s
/// dict; the JSON key order matches the Python insertion order).
#[derive(Debug, Clone, PartialEq)]
pub struct TurningPointReport {
    /// ψ₀ of the window sample with the smallest ω.
    pub psi0_sample_min: f64,
    /// The smallest sampled ω in the window.
    pub omega_sample_min: f64,
    /// Samples in the fit window (up to 7: `[k-3, k+4)` around the
    /// minimum).
    pub n_points: usize,
    /// The polynomial extremum, when a fit localized it
    /// (`method = "polyN"`); `None` means `method = "sample"`.
    pub psi0: Option<f64>,
    /// The polynomial's value at its extremum; `None` alongside a
    /// `None` [`psi0`](Self::psi0).
    pub omega: Option<f64>,
    /// `"sample"` (extremum read off the samples) or `"poly{deg}"`
    /// (localized by the polynomial fit) — the Python's `method` value.
    pub method: String,
}

impl TurningPointReport {
    /// The report dict, in the Python's key order
    /// (`psi0_sample_min`, `omega_sample_min`, `n_points`,
    /// [`psi0`, `omega`,] `method`).
    #[must_use]
    pub fn to_jv(&self) -> crate::pyjson::Jv {
        crate::pyjson::Jv::Obj(self.to_pairs())
    }

    /// The report's key/value pairs in the Python insertion order (used
    /// by `summary.json`, which appends its own keys after them).
    #[must_use]
    pub fn to_pairs(&self) -> Vec<(String, crate::pyjson::Jv)> {
        use crate::pyjson::Jv;
        let mut pairs = vec![
            ("psi0_sample_min".to_string(), Jv::Float(self.psi0_sample_min)),
            ("omega_sample_min".to_string(), Jv::Float(self.omega_sample_min)),
            ("n_points".to_string(), Jv::Int(self.n_points as i64)),
        ];
        if let (Some(p), Some(w)) = (self.psi0, self.omega) {
            pairs.push(("psi0".to_string(), Jv::Float(p)));
            pairs.push(("omega".to_string(), Jv::Float(w)));
        }
        pairs.push(("method".to_string(), Jv::Str(self.method.clone())));
        pairs
    }

    /// Inverse of [`TurningPointReport::to_jv`]: read a report dict the
    /// Python wrote (`summary.json` / state.json `turning_point`). An
    /// absent or null `psi0`/`omega` reads as `None`, a present integer
    /// coerces to float; `None` when a required key is missing or has the
    /// wrong JSON kind.
    #[must_use]
    pub fn from_jv(raw: &crate::pyjson::Jv) -> Option<TurningPointReport> {
        use crate::pyjson::Jv;
        let g = |k: &str| -> Option<Option<f64>> {
            match raw.get(k) {
                None | Some(Jv::Null) => Some(None),
                Some(Jv::Float(f)) => Some(Some(*f)),
                Some(Jv::Int(i)) => Some(Some(*i as f64)),
                _ => None,
            }
        };
        Some(TurningPointReport {
            psi0_sample_min: g("psi0_sample_min")??,
            omega_sample_min: g("omega_sample_min")??,
            n_points: match raw.get("n_points") {
                Some(Jv::Int(n)) => *n as usize,
                _ => return None,
            },
            psi0: g("psi0")?,
            omega: g("omega")?,
            method: match raw.get("method") {
                Some(Jv::Str(s)) => s.clone(),
                _ => return None,
            },
        })
    }
}

/// Design §6.2 localization: fit a low-order polynomial ω(ψ₀) over the
/// samples bracketing the smallest sampled ω and take its extremum (the
/// paper used a 4th-degree spline). Python `turning_point_estimate`
/// (`sweep_driver.py` L2004-2037): pairs are formed index-wise (the
/// shorter input truncates), sorted, the window is the up-to-7 samples
/// `[k-3, k+4)` around the minimum, and the fit is sampled on a 4001-point
/// grid. Returns `None` for no usable pairs (the Python's `{}`).
///
/// # Panics
///
/// Panics if any usable ω sample is NaN — the minimum search unwraps
/// `partial_cmp` (the Python's `min` is equally total on floats).
#[must_use]
pub fn turning_point_estimate(
    psi0s: &[Option<f64>],
    omegas: &[Option<f64>],
    degree: usize,
) -> Option<TurningPointReport> {
    // zip(..., strict=False): pairs end at the shorter input.
    let mut pts: Vec<(f64, f64)> = psi0s
        .iter()
        .zip(omegas)
        .filter_map(|(p, w)| match (p, w) {
            (Some(p), Some(w)) => Some((*p, *w)),
            _ => None,
        })
        .collect();
    if pts.is_empty() {
        return None;
    }
    // sorted(): tuples compare element-wise (ψ₀, then ω on ties).
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // k = index of the smallest ω (first on ties, like min()/argmin).
    let mut k = 0usize;
    for (i, (_, w)) in pts.iter().enumerate() {
        if *w < pts[k].1 {
            k = i;
        }
    }
    let lo = k.saturating_sub(3);
    let hi = (k + 4).min(pts.len());
    let xs: Vec<f64> = pts[lo..hi].iter().map(|(p, _)| *p).collect();
    let ys: Vec<f64> = pts[lo..hi].iter().map(|(_, w)| *w).collect();

    // Sample minima (first argmin, like np.argmin).
    let mut j_min = 0usize;
    for (i, w) in ys.iter().enumerate() {
        if *w < ys[j_min] {
            j_min = i;
        }
    }
    let mut report = TurningPointReport {
        psi0_sample_min: xs[j_min],
        omega_sample_min: *ys.iter().min_by(|a, b| a.partial_cmp(b).unwrap())?,
        n_points: xs.len(),
        psi0: None,
        omega: None,
        method: String::new(),
    };

    let deg = degree.min(xs.len().saturating_sub(2));
    let ptp = xs.last().unwrap() - xs[0];
    if deg < 2 || ptp == 0.0 {
        report.method = "sample".to_string();
        return Some(report);
    }

    // Rank-deficiency guard (see fit.rs): numpy's SVD min-norm fit is
    // arbitrary as a function when the window repeats ψ₀ values; fit the
    // distinct-value degree instead.
    let mut distinct: Vec<f64> = xs.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    distinct.dedup();
    let eff_deg = deg.min(distinct.len().saturating_sub(1));

    if let Some(fit) = polyfit(&xs, &ys, eff_deg) {
        // grid = np.linspace(xs[0], xs[-1], 4001); first argmin of the fit.
        let span = xs.last().unwrap() - xs[0];
        let mut best_j = 0usize;
        let mut best_v = f64::INFINITY;
        for i in 0..4001 {
            let x = if i == 4000 {
                *xs.last().unwrap()
            } else {
                xs[0] + span * (i as f64) / 4000.0
            };
            let v = fit.eval(x);
            if v < best_v {
                best_v = v;
                best_j = i;
            }
        }
        let psi0_min = if best_j == 4000 {
            *xs.last().unwrap()
        } else {
            xs[0] + span * (best_j as f64) / 4000.0
        };
        report.psi0 = Some(psi0_min);
        report.omega = Some(best_v);
        report.method = format!("poly{eff_deg}");
    } else {
        report.method = "sample".to_string();
    }
    Some(report)
}

// ---------------------------------------------------------------------------
// Stop classification
// ---------------------------------------------------------------------------

/// Is a failed step the branch's physical Newtonian end? (Python
/// `newtonian_limit_stop`, `sweep_driver.py` L1807-1832.)
///
/// In the down direction ω → m as ψ₀ → 0; the field extends without
/// bound and the Jacobian becomes singular, so the solver exits 2 near
/// ω = m. That is the branch's physical termination, not a backend
/// failure: `stopped:newtonian_limit` when the last step failed with
/// exit 2 and the last *completed* ω sits within
/// `[adaptivity] newtonian_delta` of m. `None` otherwise (including the
/// up direction, whose `M_max` end has no clean ω-based signature).
#[must_use]
pub fn newtonian_limit_stop(state: &CampaignState, spec: &Spec) -> Option<StopReason> {
    if spec.campaign.direction != Direction::Down {
        return None;
    }
    let last = state.steps.last()?;
    if last.exit_code != 2 {
        return None;
    }
    let omega = omega_of_last(state).ok()?;
    let delta = spec.adaptivity.newtonian_delta;
    if omega >= spec.campaign.m - delta {
        Some(StopReason::StoppedNewtonianLimit)
    } else {
        None
    }
}

/// Fixed-grid subset of the §6.1 exit conditions: the stop-reason
/// classifier for the campaign loop (Python `finished`,
/// `sweep_driver.py` L1835-1906). Returns `Ok(None)` while the campaign
/// should keep stepping.
///
/// Order matters and mirrors the Python exactly:
/// ψ₀ target landing (with the 1e-9 relative tolerance — an exact
/// landing must stop the campaign instead of re-solving the same point
/// forever) → ω target → max_steps → the down-direction boundary guard,
/// which **yields to available coarsening** (a just-started dilute
/// campaign can sit marginally over the threshold while widening is
/// still possible; the guard fires only once widening is exhausted) →
/// turning point (up) → failure classification (exit-2 Newtonian limit
/// first, then the timeout sentinel, then signals, then the 1–4 map).
pub fn finished(state: &CampaignState, spec: &Spec) -> Result<Option<StopReason>, DriverError> {
    let c = &spec.campaign;
    let steps_with_psi0: Vec<&StepRecord> =
        state.steps.iter().filter(|s| s.psi0.is_some()).collect();
    if steps_with_psi0.is_empty() {
        return Ok(None);
    }
    let psi0 = steps_with_psi0.last().unwrap().psi0.unwrap();
    let omega = omega_of_last(state)?;
    let direction = c.direction;

    // ψ₀ hits the constraint value exactly, up to last-ulp rounding of
    // the scale factor (sweep_driver.py L1846-1851).
    let tol = 1e-9 * c.psi0_target.abs().max(1.0);
    if direction == Direction::Up && psi0 >= c.psi0_target - tol {
        return Ok(Some(StopReason::DonePsi0Target));
    }
    if direction == Direction::Down && psi0 <= c.psi0_target + tol {
        return Ok(Some(StopReason::DonePsi0Target));
    }
    if let Some(omega_target) = c.omega_target {
        if direction == Direction::Up && omega <= omega_target {
            return Ok(Some(StopReason::DoneOmegaTarget));
        }
        if direction == Direction::Down && omega >= omega_target {
            return Ok(Some(StopReason::DoneOmegaTarget));
        }
    }
    if state.steps.len() as i64 >= c.max_steps {
        return Ok(Some(StopReason::DoneMaxSteps));
    }
    if direction == Direction::Down {
        // Weak-field boundary stop (v2): near ω → m the tail reaches the
        // outer boundary and boundary error dominates — widening cannot
        // fix it, so stop instead of regridding (sweep_driver.py
        // L1860-1883). The guard YIELDS to available coarsening.
        let last = steps_with_psi0.last().unwrap();
        if let (Some(r99), Some(dr), Some(n)) = (last.r99, last.dr, last.n) {
            if dr != 0.0 && n != 0 {
                let r_bdy = (n + 2 * ghost_of(spec.grid.order)) as f64 * dr;
                if r99 / r_bdy > spec.adaptivity.boundary_fraction {
                    let widenings_left = state
                        .widenings_left
                        .unwrap_or(spec.adaptivity.max_widenings);
                    let dr_cur = crate::state::current_grid(state, spec).dr;
                    if dr_cur * 2.0 > spec.grid.dr_max || widenings_left <= 0 {
                        return Ok(Some(StopReason::StoppedBoundary));
                    }
                }
            }
        }
    }
    if c.stop_at_turning_point
        && direction == Direction::Up
        && detect_turning_point(&state.steps)
    {
        return Ok(Some(StopReason::StoppedTurningPoint));
    }

    let last = state.steps.last().unwrap();
    if last.exit_code != 0 {
        let code = last.exit_code;
        if code == 2 {
            // A solver error near ω → m is the branch's physical end.
            if let Some(nl) = newtonian_limit_stop(state, spec) {
                return Ok(Some(nl));
            }
        }
        if code == TIMEOUT_EXIT {
            return Ok(Some(StopReason::FailedTimeout));
        }
        if code < 0 {
            // Killed by a signal (e.g. -11 = SIGSEGV; rare, pre-existing
            // C backend flakiness) — recorded distinctly from exit codes.
            let sig = code.unsigned_abs();
            return match signal_name(sig as i32) {
                Some(name) => Ok(Some(StopReason::signal(name))),
                None => Err(DriverError::UnknownSignal(code)),
            };
        }
        let reason = match code {
            1 => StopReason::FailedNewton,
            2 => StopReason::FailedSolver,
            3 => StopReason::FailedConfig,
            4 => StopReason::FailedIo,
            other => StopReason::exit_code(other),
        };
        return Ok(Some(reason));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Step-size control policy (v2: fixed relative stepping)
// ---------------------------------------------------------------------------

/// Python `round(x, 15)` — correctly-rounded to 15 decimal places.
/// Rust's `{:.15}` formats the exact binary value with correct rounding
/// (half-even on true decimal ties, which cannot occur for a binary
/// double at the 15th place), matching CPython's decimal-based round.
pub fn round15(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.15}").parse().unwrap_or(x)
}

/// ψ₀ target for the next step (Python `next_target`,
/// `sweep_driver.py` L2152-2163): fixed relative stepping —
/// `target = ψ₀·(1 ± psi0_step)` (or the absolute form), the
/// golden-ladder semantics, rounded to 15 decimals and clamped to land
/// exactly on `psi0_target`.
pub fn next_target(c: &CampaignSpec, base_psi0: f64) -> f64 {
    let sign = if c.direction == Direction::Up { 1.0 } else { -1.0 };
    let target = match c.psi0_step_mode {
        crate::vocab::Psi0StepMode::Relative => base_psi0 * (1.0 + sign * c.psi0_step),
        crate::vocab::Psi0StepMode::Absolute => base_psi0 + sign * c.psi0_step,
    };
    let rounded = round15(target);
    if c.direction == Direction::Up {
        rounded.min(c.psi0_target)
    } else {
        rounded.max(c.psi0_target)
    }
}

/// Retry classification for a failed step (Python `run_campaign`'s
/// retry-loop condition, `sweep_driver.py` L2334-2336): on Newton
/// non-convergence (exit 1), solver error (exit 2) or a signal kill,
/// retry from the last good solution with the plain rescaled seed; a
/// timeout also retries; a step already classified as the branch's
/// Newtonian end does not. Solver/config/IO errors (3, 4) never retry.
pub fn step_retryable(code: i32, near_limit: bool) -> bool {
    (code == 1 || code == 2 || code == TIMEOUT_EXIT || code < 0) && !near_limit
}

/// `near_limit` for the retry decision: exit 2 classified as the
/// branch's Newtonian end (`sweep_driver.py` L2334).
pub fn near_newtonian_limit(state: &CampaignState, spec: &Spec, code: i32) -> bool {
    code == 2 && newtonian_limit_stop(state, spec).is_some()
}

/// Helper mirroring the Python loop's status transition on a stop reason
/// (`run_campaign`, `sweep_driver.py` L2166-2175): `done` → Done,
/// `stopped` → Stopped (clean stop), `failed` → Failed.
pub fn status_for_stop(reason: &StopReason) -> crate::vocab::Status {
    use crate::vocab::Status;
    match reason.family() {
        "done" => Status::Done,
        "stopped" => Status::Stopped,
        _ => Status::Failed,
    }
}
