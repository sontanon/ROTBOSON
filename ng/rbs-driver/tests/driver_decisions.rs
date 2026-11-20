//! Port of `tests/test_driver_decisions.py` — the unit tests for the v2
//! adaptive decision layer (design §4–6 rev 2), case for case, same
//! inputs, same expected outcomes. Rust-only additions (`next_target` /
//! retry policy / status mapping, marked "Rust addition") cover pure
//! helpers the Python exercised only through the campaign loop.

// Python-parity fixtures: the raw long digit strings ARE the test data
// (ported from the Python suite verbatim — psi0/omega values, round15
// cases); digit separators would obscure the exact values under test.
#![allow(clippy::unreadable_literal)]

use rbs_driver::decide::turning_point_estimate;
use rbs_driver::spec::{AdaptivitySpec, CampaignSpec, GridSpec, OutputSpec, SeedSpec, SolverSpec, Spec};
use rbs_driver::state::{CampaignState, RegridInfo, RelDiff, StepRecord};
use rbs_driver::vocab::{Direction, OutputFormat, Psi0StepMode, SeedPolicy, StepMode, StopReason, TIMEOUT_EXIT};
use rbs_driver::{
    boundary_failure_stop, coarsening_acceptable, decide_action, decide_coarsening,
    detect_turning_point, finished, ghost_of, newtonian_limit_stop, next_target, round15,
    status_for_stop, step_retryable, would_amputate, Action, CoarseningDecision, StepDiagnostics,
};
use std::path::PathBuf;

fn approx(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a} vs {b}: |diff| = {} > {tol}",
        (a - b).abs()
    );
}

fn diag(hwl: Option<f64>, rr_phi_max: Option<f64>, dr: f64, refinements_left: i64, hwl_min: f64) -> StepDiagnostics {
    // The Python helper: a well-resolved step's diagnostics (defaults
    // → 'ok' in v2).
    StepDiagnostics { hwl, rr_phi_max, dr, refinements_left, hwl_min }
}

fn well_resolved() -> StepDiagnostics {
    diag(Some(21.0), Some(6.9), 0.125, 2, 8.0)
}

// ---------------------------------------------------------------------------
// decide_action — v2: the single refinement rule
// ---------------------------------------------------------------------------

mod test_decide_action_v2 {
    use super::*;

    #[test]
    fn test_well_resolved_step_is_ok() {
        assert_eq!(decide_action(&well_resolved()), Action::Ok);
    }

    #[test]
    fn test_spikes_trigger_refinement() {
        // hwl below the floor: the field's half-max width is under-sampled
        let d = diag(Some(7.0), Some(6.9), 0.125, 2, 8.0);
        assert_eq!(decide_action(&d), Action::RegridFiner);
    }

    #[test]
    fn test_axis_hug_triggers_refinement() {
        // peak closer to the axis than half its own width:
        // (hwl/2)·dr = 2.5·0.125 = 0.3125 > rr_phi_max
        let d = diag(Some(5.0), Some(0.30), 0.125, 2, 8.0);
        assert_eq!(decide_action(&d), Action::RegridFiner);
    }

    #[test]
    fn test_axis_criterion_is_grid_relative() {
        // the SAME physical configuration passes on a finer grid:
        // (hwl/2)·dr = 5.5·0.0625 = 0.344 <= rr_phi_max
        let d = diag(Some(11.0), Some(0.394), 0.0625, 2, 8.0);
        assert_eq!(decide_action(&d), Action::Ok);
    }

    #[test]
    fn test_axis_clear_and_resolved_is_ok() {
        let d = diag(Some(21.0), Some(6.9), 0.125, 2, 8.0);
        assert_eq!(decide_action(&d), Action::Ok);
    }

    #[test]
    fn test_refinements_exhausted_means_ok() {
        // falls through (matching the archived campaign's hwl ≈ 5 tail)
        let d = diag(Some(5.0), Some(0.30), 0.125, 0, 8.0);
        assert_eq!(decide_action(&d), Action::Ok);
    }

    #[test]
    fn test_missing_diagnostics_are_skipped() {
        let d = diag(None, None, 0.125, 2, 8.0);
        assert_eq!(decide_action(&d), Action::Ok);
    }
}

// ---------------------------------------------------------------------------
// finished() — v2 stops: boundary (down), physics limit, timeout
// ---------------------------------------------------------------------------

/// The Python `spec_down` helper.
fn spec_down_cfg(direction: Direction, newtonian_delta: f64, boundary_fraction: f64) -> Spec {
    Spec {
        campaign: CampaignSpec {
            l: 1,
            direction,
            psi0_target: if direction == Direction::Down { 1.0e-5 } else { 1.0 },
            psi0_step: 0.1,
            m: 1.0,
            omega_target: None,
            psi0_step_mode: Psi0StepMode::Relative,
            max_retries: 3,
            max_steps: 100,
            fixed_phi_r: 2,
            fixed_phi_z: 2,
            stop_at_turning_point: true,
        },
        seed: SeedSpec {
            policy: SeedPolicy::FromScratch,
            psi0: 0.01,
            sigma_r: 4.0,
            sigma_z: 4.0,
            r_ext: 12.0,
            source: None,
            w0: Some(0.9),
        },
        grid: GridSpec {
            dr: 0.125,
            n: 128,
            dr_max: 0.5,
            order: 4,
        },
        solver: SolverSpec::default(),
        output: OutputSpec {
            root: PathBuf::from("/tmp/spec-down-test"),
            format: OutputFormat::Hdf5,
        },
        adaptivity: AdaptivitySpec {
            hwl_min: 8.0,
            max_refinements: 2,
            newtonian_delta,
            boundary_fraction,
            refine_keeps_domain: false,
            regrid_rtol: 2.0e-2,
            support_fraction: 0.85,
            max_widenings: 2,
        },
        spec_hash: "test".to_string(),
    }
}

fn spec_down() -> Spec {
    spec_down_cfg(Direction::Down, 1.0e-2, 0.95)
}

/// The Python `state_failed` helper: failed step LAST (most recent); a
/// prior completed step provides ω.
fn state_failed(code: i32, omega: Option<f64>) -> CampaignState {
    let mut steps = Vec::new();
    if let Some(w) = omega {
        steps.push(StepRecord {
            i: 0,
            exit_code: 0,
            psi0: Some(1e-3),
            omega: Some(w),
            mode: StepMode::FixedPhi,
            ..StepRecord::default()
        });
    }
    steps.push(StepRecord {
        i: 1,
        exit_code: code,
        psi0: Some(1e-4),
        mode: StepMode::FixedPhi,
        ..StepRecord::default()
    });
    CampaignState {
        spec_hash: "test".to_string(),
        steps,
        ..CampaignState::default()
    }
}

mod test_newtonian_limit {
    use super::*;

    #[test]
    fn test_exit2_near_m_is_physical_limit() {
        // down direction, exit 2 with last completed ω within δ of m = 1
        assert_eq!(
            newtonian_limit_stop(&state_failed(2, Some(0.995)), &spec_down()),
            Some(StopReason::StoppedNewtonianLimit)
        );
        assert_eq!(
            finished(&state_failed(2, Some(0.995)), &spec_down()).unwrap(),
            Some(StopReason::StoppedNewtonianLimit)
        );
        assert_eq!(
            finished(&state_failed(2, Some(0.995)), &spec_down())
                .unwrap()
                .unwrap()
                .to_string(),
            "stopped:newtonian_limit"
        );
    }

    #[test]
    fn test_boundary_of_delta_counts() {
        assert_eq!(
            newtonian_limit_stop(&state_failed(2, Some(0.99)), &spec_down()),
            Some(StopReason::StoppedNewtonianLimit)
        );
    }

    #[test]
    fn test_exit2_far_from_m_is_solver_failure() {
        assert_eq!(newtonian_limit_stop(&state_failed(2, Some(0.95)), &spec_down()), None);
        assert_eq!(
            finished(&state_failed(2, Some(0.95)), &spec_down()).unwrap(),
            Some(StopReason::FailedSolver)
        );
    }

    #[test]
    fn test_up_direction_never_classifies() {
        let s = spec_down_cfg(Direction::Up, 1.0e-2, 0.95);
        assert_eq!(newtonian_limit_stop(&state_failed(2, Some(0.995)), &s), None);
        assert_eq!(
            finished(&state_failed(2, Some(0.995)), &s).unwrap(),
            Some(StopReason::FailedSolver)
        );
    }

    #[test]
    fn test_only_exit2_triggers() {
        assert_eq!(newtonian_limit_stop(&state_failed(1, Some(0.995)), &spec_down()), None);
    }

    #[test]
    fn test_no_completed_omega_is_unclassifiable() {
        assert_eq!(
            newtonian_limit_stop(&state_failed(2, None), &spec_down()),
            None
        );
    }

    #[test]
    fn test_delta_knob() {
        assert_eq!(
            newtonian_limit_stop(&state_failed(2, Some(0.95)), &spec_down_cfg(Direction::Down, 0.1, 0.95)),
            Some(StopReason::StoppedNewtonianLimit)
        );
    }
}

mod test_boundary_stop {
    use super::*;

    /// The Python `TestBoundaryStop.state_at`.
    fn state_at(r99: f64, dr: f64, n: i64) -> CampaignState {
        CampaignState {
            spec_hash: "test".to_string(),
            steps: vec![StepRecord {
                i: 0,
                exit_code: 0,
                psi0: Some(1e-4),
                omega: Some(0.995),
                r99: Some(r99),
                dr: Some(dr),
                n: Some(n),
                ..StepRecord::default()
            }],
            ..CampaignState::default()
        }
    }

    #[test]
    fn test_boundary_grazing_tail_stops_down_campaign() {
        // r99/domain = 16/16.5 = 0.97 > 0.95 — but the guard now YIELDS to
        // available coarsening (dr×2=0.25 ≤ dr_max, budget left): widening
        // is tried first. It stops only when widening is spent.
        assert_eq!(finished(&state_at(16.0, 0.125, 128), &spec_down()).unwrap(), None);

        let mut state = state_at(16.0, 0.125, 128);
        state.widenings_left = Some(0);
        assert_eq!(
            finished(&state, &spec_down()).unwrap(),
            Some(StopReason::StoppedBoundary)
        );
        // ... or when the coarseness floor blocks the widening outright
        let mut spec = spec_down();
        spec.grid.dr_max = 0.2;
        assert_eq!(
            finished(&state_at(16.0, 0.125, 128), &spec).unwrap(),
            Some(StopReason::StoppedBoundary)
        );
        assert_eq!(
            finished(&state_at(16.0, 0.125, 128), &spec)
                .unwrap()
                .unwrap()
                .to_string(),
            "stopped:boundary"
        );
    }

    #[test]
    fn test_coarsening_budget_roundtrip() {
        // the budget is honored from state (a resumed campaign's remaining
        // widenings), not re-defaulted
        let mut state = state_at(16.0, 0.125, 128);
        state.widenings_left = Some(1);
        assert_eq!(finished(&state, &spec_down()).unwrap(), None);
    }

    #[test]
    fn test_comfortable_support_does_not_stop() {
        // r99/domain = 10/16.5 = 0.61
        assert_eq!(finished(&state_at(10.0, 0.125, 128), &spec_down()).unwrap(), None);
    }

    #[test]
    fn test_up_direction_ignores_boundary() {
        // the up direction's boundary-grazing weak tail is transient
        let s = spec_down_cfg(Direction::Up, 1.0e-2, 0.95);
        assert_eq!(finished(&state_at(16.0, 0.125, 128), &s).unwrap(), None);
    }

    #[test]
    fn test_boundary_fraction_knob() {
        let s = spec_down_cfg(Direction::Down, 1.0e-2, 0.99);
        assert_eq!(finished(&state_at(16.0, 0.125, 128), &s).unwrap(), None);
    }
}

mod test_timeout {
    use super::*;

    #[test]
    fn test_finished_maps_timeout() {
        assert_eq!(
            finished(&state_failed(TIMEOUT_EXIT, Some(0.95)), &spec_down()).unwrap(),
            Some(StopReason::FailedTimeout)
        );
        assert_eq!(
            finished(&state_failed(TIMEOUT_EXIT, Some(0.95)), &spec_down())
                .unwrap()
                .unwrap()
                .to_string(),
            "failed:timeout"
        );
    }

    #[test]
    fn test_timeout_sentinel_not_treated_as_signal() {
        // The Python asserts signal.Signals(-TIMEOUT_EXIT) raises
        // ValueError (stdlib check). The Rust equivalent: the sentinel
        // does not name a signal, and finished() maps it to
        // failed:timeout — never through the signal-name branch.
        assert_eq!(rbs_driver::signal_name(-TIMEOUT_EXIT), None);
        assert_eq!(
            finished(&state_failed(TIMEOUT_EXIT, Some(0.95)), &spec_down()).unwrap(),
            Some(StopReason::FailedTimeout)
        );
        // while a real signal maps through the name branch:
        assert_eq!(
            finished(&state_failed(-11, Some(0.95)), &spec_down())
                .unwrap()
                .unwrap()
                .to_string(),
            "failed:sigsegv"
        );
    }
}

// ---------------------------------------------------------------------------
// detect_turning_point — design §6.2
// ---------------------------------------------------------------------------

fn step(psi0: f64, omega: f64) -> StepRecord {
    step_mode(psi0, omega, StepMode::FixedPhi)
}

fn step_mode(psi0: f64, omega: f64, mode: StepMode) -> StepRecord {
    StepRecord {
        i: 0,
        exit_code: 0,
        mode,
        psi0: Some(psi0),
        omega: Some(omega),
        ..StepRecord::default()
    }
}

mod test_detect_turning_point {
    use super::*;

    #[test]
    fn test_needs_three_branch_points() {
        assert!(!detect_turning_point(&[]));
        assert!(!detect_turning_point(&[step(0.1, 0.9)]));
        assert!(!detect_turning_point(&[step(0.1, 0.9), step(0.2, 0.8)]));
        // a seed step IS a branch point and counts toward the three
        assert!(detect_turning_point(&[
            step_mode(0.1, 0.9, StepMode::Seed),
            step(0.2, 0.8),
            step(0.3, 0.81),
        ]));
    }

    #[test]
    fn test_decreasing_then_increasing_fires() {
        let steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.66)];
        assert!(detect_turning_point(&steps));
    }

    #[test]
    fn test_monotone_decreasing_does_not_fire() {
        let steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.64)];
        assert!(!detect_turning_point(&steps));
    }

    #[test]
    fn test_flat_slope_does_not_fire() {
        let steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.65)];
        assert!(!detect_turning_point(&steps));
    }

    #[test]
    fn test_repeated_psi0_does_not_fire() {
        let steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.11, 0.66), step(0.13, 0.66)];
        assert!(!detect_turning_point(&steps));
    }

    #[test]
    fn test_design_semantics_last_three_steps_only() {
        let steps = [
            step(0.10, 0.70),
            step(0.11, 0.64),
            step(0.12, 0.65),
            step(0.13, 0.66),
            step(0.14, 0.67),
        ];
        assert!(!detect_turning_point(&steps));
        // in a live campaign detection runs every step, so the crossing is
        // caught at the FIRST post-minimum sample (last three straddle it)
        assert!(detect_turning_point(&steps[..3]));
    }
}

// ---------------------------------------------------------------------------
// turning_point_estimate — §6.2 localization
// ---------------------------------------------------------------------------

mod test_turning_point_estimate {
    use super::*;

    #[test]
    fn test_quadratic_bowl_localization() {
        let psi0s: Vec<Option<f64>> = (0..7).map(|i| Some(0.40 + 0.02 * f64::from(i))).collect();
        let omegas: Vec<Option<f64>> = psi0s
            .iter()
            .map(|p| p.map(|x| 0.65 + 50.0 * (x - 0.46).powi(2)))
            .collect();
        let rep = turning_point_estimate(&psi0s, &omegas, 4).unwrap();
        assert!(rep.method.starts_with("poly"));
        approx(rep.omega.unwrap(), 0.65, 1e-8);
        approx(rep.psi0.unwrap(), 0.46, 1e-6);
    }

    #[test]
    fn test_few_points_falls_back_to_sample() {
        let rep = turning_point_estimate(
            &[Some(0.4), Some(0.5)],
            &[Some(0.7), Some(0.6)],
            4,
        )
        .unwrap();
        assert_eq!(rep.method, "sample");
        assert_eq!(rep.omega_sample_min, 0.6);
        assert!(rep.psi0.is_none());
    }

    #[test]
    fn test_empty_is_empty() {
        // The Python returns {} here; None models the empty report.
        assert!(turning_point_estimate(&[], &[], 4).is_none());
        // pairs form only when BOTH sides are present (zip of Options).
        assert!(turning_point_estimate(&[Some(0.4)], &[None], 4).is_none());
    }

    #[test]
    fn test_catalogue_like_minimum() {
        let psi0s: Vec<Option<f64>> = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
            .iter()
            .map(|v| Some(*v))
            .collect();
        let omegas: Vec<Option<f64>> = [0.6485, 0.6478, 0.64745, 0.64742, 0.64750, 0.6479, 0.6486]
            .iter()
            .map(|v| Some(*v))
            .collect();
        let rep = turning_point_estimate(&psi0s, &omegas, 4).unwrap();
        approx(
            rep.omega.unwrap(),
            omegas.iter().flatten().cloned().fold(f64::INFINITY, f64::min),
            2e-4,
        );
        let psi0 = rep.psi0.unwrap();
        assert!((0.44..=0.46).contains(&psi0), "psi0 {psi0} outside [0.44, 0.46]");
    }
}

// ---------------------------------------------------------------------------
// ghost_of — grid bookkeeping used by the refinement path
// ---------------------------------------------------------------------------

#[test]
fn test_ghost_of() {
    assert_eq!(ghost_of(2), 1);
    assert_eq!(ghost_of(4), 2);
}

// ---------------------------------------------------------------------------
// Driver controls: domain-keeping refinement, coarsening + budgets,
// boundary-class failure classification
// ---------------------------------------------------------------------------

mod test_adaptivity_new_knobs {
    use super::*;

    #[test]
    fn test_defaults() {
        let a = AdaptivitySpec::parse_str("").unwrap();
        assert!(!a.refine_keeps_domain);
        assert_eq!(a.regrid_rtol, 2.0e-2);
        assert_eq!(a.support_fraction, 0.85);
        assert_eq!(a.max_widenings, 2);
    }

    #[test]
    fn test_parse() {
        let a = AdaptivitySpec::parse_str(
            "refine_keeps_domain = true\nregrid_rtol = 1e-3\nsupport_fraction = 0.8\nmax_widenings = 3\n",
        )
        .unwrap();
        assert!(a.refine_keeps_domain);
        assert_eq!(a.regrid_rtol, 1e-3);
        assert_eq!(a.support_fraction, 0.8);
        assert_eq!(a.max_widenings, 3);
    }

    #[test]
    fn test_support_fraction_must_precede_boundary_stop() {
        let err = AdaptivitySpec::parse_str("support_fraction = 0.95\nboundary_fraction = 0.95\n")
            .unwrap_err();
        assert!(err.to_string().contains("support_fraction"), "{err}");
        let err = AdaptivitySpec::parse_str("support_fraction = 0.99\n").unwrap_err();
        assert!(err.to_string().contains("support_fraction"), "{err}");
    }

    #[test]
    fn test_regrid_rtol_validation() {
        let err = AdaptivitySpec::parse_str("regrid_rtol = 0.0\n").unwrap_err();
        assert!(err.to_string().contains("regrid_rtol"), "{err}");
    }
}

mod test_would_amputate {
    use super::*;
    /// The refinement-support guard: only domain-shrinking regrids.

    #[test]
    fn test_legacy_shrink_amputates() {
        // source: dr=0.125, N=128, r99=15.6 (94.5% of the 16.5 boundary,
        // the driver's (N+2·ghost)·dr convention); legacy refinement
        // dr=0.0625 at N=128 → boundary 8.25, far inside the support.
        assert!(would_amputate(Some(15.6), 0.125, 128, 0.0625, 128, 4, 0.95));
    }

    #[test]
    fn test_keep_domain_never_amputates() {
        // N×2 with dr÷2 keeps the interior domain (16 = 16); the ghost-incl.
        // boundary moves 16.5 → 16.25 but the support fits outright.
        assert!(!would_amputate(Some(15.6), 0.125, 128, 0.0625, 256, 4, 0.95));
        // even the seed's 96%-of-boundary support (r99=15.685 of 16.25)
        assert!(!would_amputate(Some(15.685), 0.125, 128, 0.0625, 256, 4, 0.95));
    }

    #[test]
    fn test_fit_shrink_ok() {
        // a compact field fits the smaller domain
        assert!(!would_amputate(Some(4.0), 0.125, 128, 0.0625, 128, 4, 0.95));
    }

    #[test]
    fn test_unknown_support_is_safe() {
        assert!(!would_amputate(None, 0.125, 128, 0.0625, 128, 4, 0.95));
    }
}

mod test_coarsening_acceptable {
    use super::*;

    fn rel(pairs: &[(&str, f64)]) -> RelDiff {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }

    #[test]
    fn test_within_rtol() {
        assert!(coarsening_acceptable(
            &rel(&[("omega", 1e-4), ("M_Komar", 2e-3)]),
            2e-2
        ));
    }

    #[test]
    fn test_beyond_rtol() {
        assert!(!coarsening_acceptable(&rel(&[("omega", 5e-2)]), 2e-2));
    }

    #[test]
    fn test_empty_is_reject() {
        assert!(!coarsening_acceptable(&RelDiff::new(), 2e-2));
    }
}

mod test_decide_coarsening {
    use super::*;

    #[test]
    fn test_support_below_trigger() {
        assert_eq!(
            decide_coarsening(Some(0.5), 0.85, 0.125, 0.5, 2),
            CoarseningDecision::Ok
        );
        assert_eq!(
            decide_coarsening(None, 0.85, 0.125, 0.5, 2),
            CoarseningDecision::Ok
        );
    }

    #[test]
    fn test_support_over_trigger_widens() {
        assert_eq!(
            decide_coarsening(Some(0.9), 0.85, 0.125, 0.5, 2),
            CoarseningDecision::RegridCoarser
        );
    }

    #[test]
    fn test_dr_floor_stops() {
        // dr×2 would exceed the coarseness floor
        assert_eq!(
            decide_coarsening(Some(0.9), 0.85, 0.4, 0.5, 2),
            CoarseningDecision::StopBudget
        );
        assert_eq!(
            decide_coarsening(Some(0.9), 0.85, 0.25, 0.5, 2),
            CoarseningDecision::RegridCoarser
        );
    }

    #[test]
    fn test_budget_exhausted_stops() {
        assert_eq!(
            decide_coarsening(Some(0.9), 0.85, 0.125, 0.5, 0),
            CoarseningDecision::StopBudget
        );
    }
}

mod test_boundary_failure_stop {
    use super::*;

    #[test]
    fn test_support_fills_domain() {
        // r99=15.9 of the 16.5 boundary (N=128, dr=0.125, order 4): 96% > 95%
        assert!(boundary_failure_stop(Some(15.9), Some(0.125), Some(128), 4, 0.95));
    }

    #[test]
    fn test_support_fits() {
        assert!(!boundary_failure_stop(Some(13.1), Some(0.125), Some(128), 4, 0.95));
    }

    #[test]
    fn test_unknown_support() {
        assert!(!boundary_failure_stop(None, Some(0.125), Some(128), 4, 0.95));
        assert!(!boundary_failure_stop(Some(15.6), None, Some(128), 4, 0.95));
        assert!(!boundary_failure_stop(Some(15.6), Some(0.125), None, 4, 0.95));
    }
}

mod test_new_state_fields_round_trip {
    use super::*;
    use rbs_driver::pyjson::Jv;

    #[test]
    fn test_regrid_to_n_roundtrip() {
        let info = RegridInfo {
            from_dr: 0.125,
            to_dr: 0.0625,
            source: Some("src".to_string()),
            rel_diff: vec![("omega".to_string(), 1e-4)],
            accepted: true,
            to_n: Some(256),
        };
        let d = info.to_jv();
        assert!(matches!(d.get("to_n"), Some(Jv::Int(256))));
        assert_eq!(RegridInfo::from_jv(&d).unwrap().to_n, Some(256));
    }

    #[test]
    fn test_regrid_to_n_omitted_for_legacy() {
        let info = RegridInfo {
            from_dr: 0.125,
            to_dr: 0.0625,
            source: None,
            rel_diff: RelDiff::new(),
            accepted: true,
            to_n: None,
        };
        let d = info.to_jv();
        assert!(d.get("to_n").is_none());
        assert_eq!(RegridInfo::from_jv(&d).unwrap().to_n, None);
    }

    #[test]
    fn test_widenings_left_omitted_when_unset() {
        let state = CampaignState {
            spec_hash: "t".to_string(),
            ..CampaignState::default()
        };
        assert!(state.to_jv().get("widenings_left").is_none());
        let mut state2 = state.clone();
        state2.widenings_left = Some(3);
        let d = state2.to_jv();
        assert!(matches!(d.get("widenings_left"), Some(Jv::Int(3))));
        assert_eq!(CampaignState::from_jv(&d).unwrap().widenings_left, Some(3));
    }
}

// ---------------------------------------------------------------------------
// Rust additions: pure helpers the Python exercised only through the
// campaign loop (next_target, retry policy, status mapping).
// ---------------------------------------------------------------------------

mod rust_additions {
    use super::*;

    fn campaign(direction: Direction, step: f64, target: f64, mode: Psi0StepMode) -> CampaignSpec {
        CampaignSpec {
            l: 1,
            direction,
            psi0_target: target,
            psi0_step: step,
            m: 1.0,
            omega_target: None,
            psi0_step_mode: mode,
            max_retries: 3,
            max_steps: 50,
            fixed_phi_r: 2,
            fixed_phi_z: 2,
            stop_at_turning_point: true,
        }
    }

    #[test]
    fn round15_matches_python_round() {
        assert_eq!(round15(0.30000000000000004), 0.3);
        assert_eq!(round15(0.0029237157590377197), 0.002923715759038);
        assert_eq!(round15(1.5), 1.5);
        assert_eq!(round15(0.0), 0.0);
        // round-trip identity for values already at 15 decimals.
        let x = 0.123456789012345;
        assert_eq!(round15(x), x);
    }

    #[test]
    fn next_target_relative_up_grows_and_lands_exactly() {
        let c = campaign(Direction::Up, 0.03, 0.6, Psi0StepMode::Relative);
        // golden-ladder: ψ₀·(1 + step), clamped onto the target.
        approx(next_target(&c, 0.5), 0.515, 1e-12);
        // the clamp lands exactly on psi0_target instead of overshooting.
        assert_eq!(next_target(&c, 0.59), 0.6);
    }

    #[test]
    fn next_target_absolute_mode_and_down_direction() {
        let c = campaign(Direction::Up, 0.000346226907097508, 0.000555733907097508, Psi0StepMode::Absolute);
        approx(next_target(&c, 0.000209507), 0.000555733907097508, 1e-18);
        // down: max() clamps onto the (lower) target.
        let c = campaign(Direction::Down, 0.1, 1e-5, Psi0StepMode::Relative);
        approx(next_target(&c, 1e-4), 9e-5, 1e-12);
        assert_eq!(next_target(&c, 1.05e-5), 1e-5);
    }

    #[test]
    fn retry_policy_matches_the_python_loop() {
        // exit 1/2/timeout/signals retry; config/io errors never do;
        // the branch's Newtonian end is not chased.
        assert!(step_retryable(1, false));
        assert!(step_retryable(2, false));
        assert!(step_retryable(TIMEOUT_EXIT, false));
        assert!(step_retryable(-11, false));
        assert!(!step_retryable(3, false));
        assert!(!step_retryable(4, false));
        assert!(!step_retryable(2, true));
        // near_limit is only ever true for exit 2 (it is defined as
        // `code == 2 and newtonian_limit_stop(...)`), so no other code
        // carries it — the Python condition never sees this shape.
    }

    #[test]
    fn near_newtonian_limit_classifies_exit_2_only_in_down() {
        let spec = spec_down();
        let state = state_failed(2, Some(0.995));
        assert!(rbs_driver::near_newtonian_limit(&state, &spec, 2));
        assert!(!rbs_driver::near_newtonian_limit(&state, &spec, 1));
        let up = spec_down_cfg(Direction::Up, 1.0e-2, 0.95);
        assert!(!rbs_driver::near_newtonian_limit(&state, &up, 2));
    }

    #[test]
    fn status_for_stop_matches_the_loop_transition() {
        use rbs_driver::vocab::Status;
        assert_eq!(status_for_stop(&StopReason::DonePsi0Target), Status::Done);
        assert_eq!(status_for_stop(&StopReason::DoneMaxSteps), Status::Done);
        assert_eq!(status_for_stop(&StopReason::StoppedTurningPoint), Status::Stopped);
        assert_eq!(status_for_stop(&StopReason::StoppedBoundary), Status::Stopped);
        assert_eq!(status_for_stop(&StopReason::FailedSolver), Status::Failed);
        assert_eq!(status_for_stop(&StopReason::ExitCode(7)), Status::Failed);
    }

    #[test]
    fn max_steps_and_omega_target_stops() {
        // done:max_steps: step count reaches the campaign budget.
        let spec = spec_down();
        let mut state = state_failed(0, Some(0.5));
        state.steps = (0..100)
            .map(|i| StepRecord {
                i,
                exit_code: 0,
                psi0: Some(1e-4),
                omega: Some(0.995),
                ..StepRecord::default()
            })
            .collect();
        // psi0 never reaches the down target (1e-5) and ω stays above
        // any omega_target, so the step budget is what fires.
        assert_eq!(
            finished(&state, &spec).unwrap(),
            Some(StopReason::DoneMaxSteps)
        );

        // done:omega_target in the down direction: ω rises to the target.
        let mut spec = spec_down();
        spec.campaign.omega_target = Some(0.99);
        let state = state_failed(0, Some(0.99));
        assert_eq!(
            finished(&state, &spec).unwrap(),
            Some(StopReason::DoneOmegaTarget)
        );

        // up direction: ω falls to the target.
        let mut spec = spec_down_cfg(Direction::Up, 1.0e-2, 0.95);
        spec.campaign.omega_target = Some(0.5);
        let mut state = state_failed(0, Some(0.5));
        state.steps[0].psi0 = Some(1e-3); // far from psi0_target = 1.0
        assert_eq!(
            finished(&state, &spec).unwrap(),
            Some(StopReason::DoneOmegaTarget)
        );
    }

    #[test]
    fn psi0_target_landing_uses_the_1e9_relative_tolerance() {
        // exact landing (within last-ulp rounding) stops the campaign.
        let mut spec = spec_down_cfg(Direction::Up, 1.0e-2, 0.95);
        spec.campaign.psi0_target = 0.6;
        let state = CampaignState {
            spec_hash: "test".into(),
            steps: vec![StepRecord {
                i: 0,
                exit_code: 0,
                psi0: Some(0.6 - 5e-10),
                omega: Some(0.65),
                ..StepRecord::default()
            }],
            ..CampaignState::default()
        };
        assert_eq!(
            finished(&state, &spec).unwrap(),
            Some(StopReason::DonePsi0Target)
        );
        // clearly short of the target: keep going.
        let state = CampaignState {
            spec_hash: "test".into(),
            steps: vec![StepRecord {
                i: 0,
                exit_code: 0,
                psi0: Some(0.5),
                omega: Some(0.65),
                ..StepRecord::default()
            }],
            ..CampaignState::default()
        };
        assert_eq!(finished(&state, &spec).unwrap(), None);
    }
}
