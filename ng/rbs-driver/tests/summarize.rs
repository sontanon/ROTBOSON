//! Round-trip tests for the fold summarize (`--summarize`): synthetic
//! branch points with a known minimum must be localized within the
//! Python test's tolerance (`tests/test_driver_decisions.py::
//! TestTurningPointEstimate`), and the summary.json bytes must match the
//! Python writer's shape.

use rbs_driver::pyjson::Jv;
use rbs_driver::spec::{GridSpec, OutputSpec, Spec};
use rbs_driver::state::{CampaignState, StepRecord};
use rbs_driver::vocab::{Direction, OutputFormat, Psi0StepMode, SeedPolicy, Status, StepMode};
use rbs_driver::{
    summarize, summarize_campaign, AdaptivitySpec, CampaignSpec, SeedSpec, SolverSpec,
    SummarizeError,
};
use std::path::PathBuf;

fn approx(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a} vs {b}: |diff| = {} > {tol}",
        (a - b).abs()
    );
}

fn spec_with_root(root: &std::path::Path) -> Spec {
    Spec {
        campaign: CampaignSpec {
            l: 1,
            direction: Direction::Up,
            psi0_target: 0.6,
            psi0_step: 0.03,
            m: 1.0,
            omega_target: None,
            psi0_step_mode: Psi0StepMode::Relative,
            max_retries: 3,
            max_steps: 50,
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
            root: root.to_path_buf(),
            format: OutputFormat::Hdf5,
        },
        adaptivity: AdaptivitySpec::default(),
        spec_hash: "summarize-test".to_string(),
    }
}

/// A campaign state whose steps walk a quadratic bowl with the minimum
/// at ψ₀ = 0.46, ω = 0.65 (the Python localization test's known
/// minimum), including the noise-free slope flip the detector needs.
fn bowl_state() -> CampaignState {
    let psi0s: Vec<f64> = (0..7).map(|i| 0.40 + 0.02 * f64::from(i)).collect();
    let omegas: Vec<f64> = psi0s
        .iter()
        .map(|&x| 0.65 + 50.0 * (x - 0.46).powi(2))
        .collect();
    CampaignState {
        spec_hash: "summarize-test".to_string(),
        steps: psi0s
            .iter()
            .zip(&omegas)
            .enumerate()
            .map(|(i, (&p, &w))| StepRecord {
                i: i as i64,
                exit_code: 0,
                mode: StepMode::FixedPhi,
                // like every real recorded step (record_step), a step with
                // scalars carries a solution directory — the scalar block
                // is only emitted when sol_dir is set (Python to_dict).
                sol_dir: Some(format!(
                    "/tmp/sols/l=1,w=6.50000E-01,dr=1.25000E-01,N=0128/step{i:04}"
                )),
                psi0: Some(p),
                omega: Some(w),
                ..StepRecord::default()
            })
            .collect(),
        status: Status::Stopped,
        stop_reason: Some("stopped:turning_point".to_string()),
        ..CampaignState::default()
    }
}

#[test]
fn summarize_localizes_the_known_minimum() {
    let dir = tmpdir("bowl");
    let spec = spec_with_root(&dir);
    let state = bowl_state();
    let summary = summarize(&spec, &state).expect("state has omegas");
    let report = summary.report.expect("the bowl localizes");
    // The Python tolerances (test_quadratic_bowl_localization):
    // ω within 1e-8, ψ₀ within 1e-6.
    approx(report.omega.expect("fit extremum"), 0.65, 1e-8);
    approx(report.psi0.expect("fit extremum"), 0.46, 1e-6);
    assert!(report.method.starts_with("poly"));
    assert_eq!(report.n_points, 7);
    // sample minima bookkeeping also matches the Python dict fields.
    approx(report.omega_sample_min, 0.65, 1e-12);
    approx(report.psi0_sample_min, 0.46, 1e-12);
    // the campaign bookkeeping keys ride along.
    assert_eq!(summary.status, Status::Stopped);
    assert_eq!(
        summary.stop_reason.as_deref(),
        Some("stopped:turning_point")
    );
    assert_eq!(summary.campaign, dir.to_string_lossy());
}

#[test]
fn summary_json_shape_matches_python() {
    let dir = tmpdir("shape");
    let spec = spec_with_root(&dir);
    let state = bowl_state();
    let summary = summarize(&spec, &state).unwrap();
    // Key order: the report's fields first, then campaign/status/
    // stop_reason (the Python dict insertion order).
    let Jv::Obj(pairs) = summary.to_jv() else {
        panic!("summary is an object")
    };
    let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "psi0_sample_min",
            "omega_sample_min",
            "n_points",
            "psi0",
            "omega",
            "method",
            "campaign",
            "status",
            "stop_reason",
        ]
    );
    // and the exact Python serialization bytes:
    let method = summary.report.as_ref().unwrap().method.clone();
    let psi0 = summary.report.as_ref().unwrap().psi0.unwrap();
    let omega = summary.report.as_ref().unwrap().omega.unwrap();
    let expected = format!(
        concat!(
            "{{\n",
            "  \"psi0_sample_min\": 0.46,\n",
            "  \"omega_sample_min\": 0.65,\n",
            "  \"n_points\": 7,\n",
            "  \"psi0\": {psi0},\n",
            "  \"omega\": {omega},\n",
            "  \"method\": \"{method}\",\n",
            "  \"campaign\": \"{dir}\",\n",
            "  \"status\": \"stopped\",\n",
            "  \"stop_reason\": \"stopped:turning_point\"\n",
            "}}"
        ),
        psi0 = crate_float(psi0),
        omega = crate_float(omega),
        method = method,
        dir = dir.to_string_lossy(),
    );
    assert_eq!(summary.to_json_pretty(), expected);
}

/// Render like the Python repr for the two fitted values (the exact
/// shortest spelling is asserted rather than assumed).
fn crate_float(f: f64) -> String {
    rbs_driver::pyjson::python_repr_f64(f)
}

#[test]
fn summarize_without_omegas_errors_like_python() {
    let dir = tmpdir("noomega");
    let spec = spec_with_root(&dir);
    let state = CampaignState {
        spec_hash: "summarize-test".into(),
        steps: vec![StepRecord {
            i: 0,
            exit_code: 2,
            psi0: Some(0.4),
            ..StepRecord::default()
        }],
        ..CampaignState::default()
    };
    assert!(matches!(
        summarize(&spec, &state),
        Err(SummarizeError::NoOmegas)
    ));
}

#[test]
fn summarize_campaign_reads_state_json_and_writes_summary_json() {
    let dir = tmpdir("pipeline");
    let spec = spec_with_root(&dir);
    let state = bowl_state();
    rbs_driver::save_state(&spec, &state).unwrap();
    let summary = summarize_campaign(&spec).expect("state exists and has omegas");
    let text = std::fs::read_to_string(dir.join("summary.json")).unwrap();
    assert_eq!(text, format!("{}\n", summary.to_json_pretty()));

    // no state.json → the Python's "no state.json to summarize".
    let empty = tmpdir("nostate");
    let spec = spec_with_root(&empty);
    assert!(matches!(
        summarize_campaign(&spec),
        Err(SummarizeError::NoState)
    ));
}

#[test]
fn turning_point_report_round_trips_through_the_state_model() {
    // The stored `turning_point` block is an arbitrary JSON object
    // (Python keeps a plain dict); the report's to_pairs must re-emit it
    // in the Python insertion order.
    let dir = tmpdir("tp");
    let spec = spec_with_root(&dir);
    let mut state = bowl_state();
    let report = rbs_driver::turning_point_estimate(
        &state.steps.iter().map(|s| s.psi0).collect::<Vec<_>>(),
        &state.steps.iter().map(|s| s.omega).collect::<Vec<_>>(),
        rbs_driver::SUMMARY_FIT_DEGREE,
    )
    .unwrap();
    state.turning_point = Some(report.to_jv());
    rbs_driver::save_state(&spec, &state).unwrap();
    let loaded = rbs_driver::load_state(&spec).unwrap().unwrap();
    assert_eq!(loaded.turning_point, state.turning_point);
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rbs-driver-summarize-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
