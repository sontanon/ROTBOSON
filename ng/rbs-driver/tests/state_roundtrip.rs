//! Port of `tests/test_state_roundtrip.py` — state.json round-trip tests.
//!
//! The frozen state model must be compatible with in-flight campaigns:
//!
//! * the CURRENT writer's shape must round-trip **byte-identically**
//!   — resuming must not churn state.json formats mid-campaign. The
//!   fixture bytes were produced by the Python driver itself, so this
//!   is a cross-language byte-compat check, stronger than the Python's
//!   own same-language round-trip;
//! * legacy v1 state.json files (minimal top-level keys, per-step extras
//!   like `step_factor`) must load and resume; a re-save upgrades the
//!   file, dropping keys the v2 model no longer tracks.
//!
//! Rust-only additions (marked as such): the file plumbing — atomic
//! save/load round-trip and the spec-hash resume guard.

use rbs_driver::pyjson::Jv;
use rbs_driver::state::{CampaignState, StepRecord};
use rbs_driver::vocab::{Status, StepMode};
use rbs_driver::{load_state, save_state, state_path, StateError};

const CURRENT: &str = include_str!("fixtures/state_v2_current.json");
const LEGACY: &str = include_str!("fixtures/state_legacy_v1.json");

fn parse(text: &str) -> Jv {
    Jv::parse(text).expect("fixture is valid JSON")
}

fn obj<'a>(v: &'a Jv) -> &'a Vec<(String, Jv)> {
    v.as_obj().expect("fixture top level is an object")
}

fn get<'a>(v: &'a Jv, key: &str) -> &'a Jv {
    v.get(key).expect("key present in fixture")
}

fn raw_steps(raw: &Jv) -> &Vec<Jv> {
    get(raw, "steps").as_arr().expect("steps is a list")
}

fn raw_probes(raw: &Jv) -> &Vec<Jv> {
    get(raw, "rejected_regrids").as_arr().expect("probes is a list")
}

fn raw_fold(raw: &Jv) -> &Vec<Jv> {
    get(raw, "fold_fine_grid_measurement")
        .as_arr()
        .expect("fold measurements is a list")
}

fn opt_f64(v: Option<&Jv>) -> Option<f64> {
    match v {
        Some(Jv::Float(f)) => Some(*f),
        Some(Jv::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The current writer's shape re-serializes byte-identically.
// ---------------------------------------------------------------------------

mod test_current_writer_byte_round_trip {
    use super::*;

    #[test]
    fn test_byte_identical_round_trip() {
        let text = CURRENT;
        let state = CampaignState::from_jv(&parse(text)).expect("fixture parses");
        assert_eq!(format!("{}\n", state.to_json_pretty()), text);
    }

    #[test]
    fn test_double_round_trip_is_stable() {
        let text = CURRENT;
        let once = CampaignState::from_jv(&parse(text)).unwrap().to_jv();
        let twice = CampaignState::from_jv(&once).unwrap().to_jv();
        assert_eq!(format!("{}\n", twice.write_json()), text);
    }

    #[test]
    fn test_data_preserved() {
        let raw = parse(CURRENT);
        let state = CampaignState::from_jv(&raw).unwrap();
        assert_eq!(
            state.spec_hash,
            get(&raw, "spec_hash").as_str().expect("hash is a string")
        );
        assert_eq!(state.steps.len(), raw_steps(&raw).len());
        assert_eq!(
            state.steps.iter().map(|s| s.i).collect::<Vec<_>>(),
            raw_steps(&raw)
                .iter()
                .map(|r| match get(r, "i") {
                    Jv::Int(i) => *i,
                    _ => panic!("i is an int"),
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            state.steps.iter().map(|s| s.omega).collect::<Vec<_>>(),
            raw_steps(&raw)
                .iter()
                .map(|r| opt_f64(Some(get(r, "omega"))))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            state.steps.iter().map(|s| s.psi0).collect::<Vec<_>>(),
            raw_steps(&raw)
                .iter()
                .map(|r| opt_f64(Some(get(r, "psi0"))))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            state.refinements_left,
            match get(&raw, "refinements_left") {
                Jv::Int(i) => Some(*i),
                _ => None,
            }
        );
        let probes = state.rejected_regrids.as_ref().expect("probes present");
        assert_eq!(probes.len(), raw_probes(&raw).len());
        assert_eq!(
            probes.iter().map(|p| p.psi0).collect::<Vec<_>>(),
            raw_probes(&raw)
                .iter()
                .map(|r| opt_f64(r.get("psi0")))
                .collect::<Vec<_>>()
        );
        let fixture_scalars = |r: &Jv| -> Vec<(String, rbs_driver::state::Scalar)> {
            get(r, "scalars")
                .as_obj()
                .unwrap()
                .iter()
                .map(|(k, v)| match v {
                    Jv::Int(i) => (k.clone(), rbs_driver::state::Scalar::Int(*i)),
                    Jv::Float(f) => (k.clone(), rbs_driver::state::Scalar::Float(*f)),
                    _ => panic!("scalar {k} is a number"),
                })
                .collect()
        };
        assert_eq!(probes[0].scalars.as_ref().unwrap(), &fixture_scalars(&raw_probes(&raw)[0]));
        let fold = state.fold_fine_grid_measurement.as_ref().expect("fold present");
        assert_eq!(
            fold.iter().map(|m| m.sol_dir.as_str()).collect::<Vec<_>>(),
            raw_fold(&raw)
                .iter()
                .map(|m| get(m, "sol_dir").as_str().unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fold[0].note.as_deref(),
            Some(get(&raw_fold(&raw)[0], "note").as_str().unwrap())
        );
    }

    #[test]
    fn test_int_scalars_keep_identity() {
        // int-valued scalars (e.g. error_code) must not become floats
        let raw = parse(CURRENT);
        let state = CampaignState::from_jv(&raw).unwrap();
        let probe = state.rejected_regrids.as_ref().unwrap()[0].clone();
        let scalars = probe.scalars.as_ref().unwrap();
        let orig = get(&raw_probes(&raw)[0], "scalars");
        for (key, value) in scalars {
            let raw_value = orig
                .as_obj()
                .unwrap()
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .expect("scalar key present in fixture");
            let same_kind = matches!(
                (value, raw_value),
                (rbs_driver::state::Scalar::Int(_), Jv::Int(_))
                    | (rbs_driver::state::Scalar::Float(_), Jv::Float(_))
            );
            assert!(same_kind, "scalar {key} changed int/float identity");
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy v1 state.json files load; the model keeps the modeled data.
// ---------------------------------------------------------------------------

mod test_legacy_compat {
    use super::*;

    fn legacy() -> (CampaignState, Jv) {
        let raw = parse(LEGACY);
        (CampaignState::from_jv(&raw).expect("legacy parses"), raw)
    }

    #[test]
    fn test_legacy_loads() {
        let (s, raw) = legacy();
        assert_eq!(s.status.to_string(), get(&raw, "status").as_str().unwrap());
        assert_eq!(s.stop_reason.as_deref(), get(&raw, "stop_reason").as_str());
        assert_eq!(s.steps.len(), raw_steps(&raw).len());
        // legacy files predate the adaptive keys
        assert!(s.grid.is_none());
        assert!(s.refinements_left.is_none());
        assert!(s.widenings_left.is_none());
    }

    #[test]
    fn test_legacy_modeled_data_survives() {
        let (_, raw) = legacy();
        for raw_step in raw_steps(&raw) {
            let modeled = StepRecord::from_jv(raw_step).unwrap();
            // every key the current writer emits matches the original
            // value (unknown legacy keys like `step_factor` are dropped)
            for (key, value) in modeled.to_jv().as_obj().unwrap() {
                if key == "mode" {
                    let orig = get(raw_step, "mode").as_str().unwrap();
                    assert_eq!(value.as_str().unwrap(), orig);
                } else {
                    let orig = raw_step.get(key).expect("modeled key exists in legacy");
                    assert_eq!(value, orig, "step key {key}");
                }
            }
            // the legacy `step_factor` key is dropped by the model (only
            // the later v1 steps carry it).
            if raw_step.get("step_factor").is_some() {
                assert!(modeled.to_jv().get("step_factor").is_none());
            }
        }
    }

    #[test]
    fn test_legacy_semantic_round_trip() {
        // to_jv(from_jv(x)) == from_jv(x) (normalized form is stable)
        let (s, _) = legacy();
        let once = s.to_jv();
        let twice = CampaignState::from_jv(&once).unwrap().to_jv();
        assert_eq!(once, twice);
    }
}

// ---------------------------------------------------------------------------
// Failed records and regrid bookkeeping round-trip.
// ---------------------------------------------------------------------------

mod test_synthetic_records {
    use super::*;
    use rbs_driver::state::{RegridInfo, RelDiff};

    #[test]
    fn test_failed_step_record() {
        let rec = StepRecord {
            i: 3,
            exit_code: 2,
            psi0_target: Some(0.42),
            ..StepRecord::default()
        };
        // a StepRecord dict parsed as a CampaignState has no steps key →
        // the lenient loader yields an empty campaign (Python parity).
        assert_eq!(CampaignState::from_jv(&rec.to_jv()).unwrap().steps, Vec::new());
        let parsed = StepRecord::from_jv(&rec.to_jv()).unwrap();
        assert_eq!(parsed, rec);
    }

    #[test]
    fn test_regrid_record() {
        let rel_diff: RelDiff = vec![("omega".to_string(), 1e-4)];
        let rec = StepRecord {
            i: 7,
            exit_code: 0,
            mode: StepMode::Regrid,
            sol_dir: Some("/tmp/sol".to_string()),
            dr: Some(0.0625),
            n: Some(128),
            omega: Some(0.91),
            psi0: Some(0.31),
            regrid: Some(RegridInfo {
                from_dr: 0.125,
                to_dr: 0.0625,
                source: Some("/tmp/src".to_string()),
                rel_diff,
                accepted: true,
                to_n: None,
            }),
            ..StepRecord::default()
        };
        assert_eq!(StepRecord::from_jv(&rec.to_jv()).unwrap(), rec);
    }

    #[test]
    fn test_minimal_state() {
        let state = CampaignState {
            spec_hash: "abc".to_string(),
            ..CampaignState::default()
        };
        let expected = Jv::obj(vec![
            ("spec_hash", Jv::Str("abc".to_string())),
            ("spec_file", Jv::Null),
            ("steps", Jv::Arr(vec![])),
            ("status", Jv::Str("running".to_string())),
            ("stop_reason", Jv::Null),
            ("turning_point", Jv::Null),
            ("pending_refinement", Jv::Null),
        ]);
        assert_eq!(state.to_jv(), expected);
        assert_eq!(CampaignState::from_jv(&state.to_jv()).unwrap(), state);
        // and the exact Python serialization of that dict:
        assert_eq!(
            state.to_json_pretty(),
            concat!(
                "{\n",
                "  \"spec_hash\": \"abc\",\n",
                "  \"spec_file\": null,\n",
                "  \"steps\": [],\n",
                "  \"status\": \"running\",\n",
                "  \"stop_reason\": null,\n",
                "  \"turning_point\": null,\n",
                "  \"pending_refinement\": null\n",
                "}"
            )
        );
    }

    #[test]
    fn test_corrupt_state_fails_with_clear_messages() {
        // A present-but-null scalar is an error (Python: _st_float(None));
        // a missing key takes the default instead.
        let bad = Jv::obj(vec![("dr", Jv::Null), ("N", Jv::Int(1))]);
        let err = rbs_driver::state::GridPosition::from_jv(&bad).unwrap_err();
        assert_eq!(err.to_string(), "state.json: expected a number, got NoneType");

        let bad = Jv::obj(vec![("N", Jv::Str("128".to_string()))]);
        let err = rbs_driver::state::GridPosition::from_jv(&bad).unwrap_err();
        assert_eq!(err.to_string(), "state.json: expected an integer, got str");

        let ok = Jv::obj(vec![("N", Jv::Int(5))]);
        let g = rbs_driver::state::GridPosition::from_jv(&ok).unwrap();
        assert_eq!((g.dr, g.n), (0.0, 5)); // missing dr → the Python default 0.0

        // unknown status / mode strings are hard errors (Python:
        // ValueError from the StrEnum constructor).
        let bad = Jv::obj(vec![("status", Jv::Str("paused".to_string()))]);
        let err = CampaignState::from_jv(&bad).unwrap_err();
        assert!(err.to_string().contains("invalid status 'paused'"), "{err}");
    }
}

// ---------------------------------------------------------------------------
// Rust additions: the file plumbing (atomic save, hash-guarded resume).
// ---------------------------------------------------------------------------

mod rust_additions {
    use super::*;
    use rbs_driver::spec::{GridSpec, OutputSpec, Spec};
    use std::path::PathBuf;

    fn spec_with_root(root: &std::path::Path, hash: &str) -> Spec {
        Spec {
            campaign: rbs_driver::CampaignSpec {
                l: 1,
                direction: rbs_driver::Direction::Down,
                psi0_target: 1e-5,
                psi0_step: 0.1,
                m: 1.0,
                omega_target: None,
                psi0_step_mode: rbs_driver::Psi0StepMode::Relative,
                max_retries: 3,
                max_steps: 50,
                fixed_phi_r: 2,
                fixed_phi_z: 2,
                stop_at_turning_point: true,
            },
            seed: rbs_driver::SeedSpec {
                policy: rbs_driver::SeedPolicy::FromScratch,
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
            solver: rbs_driver::SolverSpec::default(),
            output: OutputSpec {
                root: root.to_path_buf(),
                format: rbs_driver::OutputFormat::Hdf5,
            },
            adaptivity: rbs_driver::AdaptivitySpec::default(),
            spec_hash: hash.to_string(),
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rbs-driver-state-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn save_then_load_round_trips_the_python_bytes() {
        let dir = tmpdir("roundtrip");
        // the persisted state comes from the archived fixture, so the
        // spec hash must be the fixture's own hash for resume to accept it
        let fixture_hash = get(&parse(CURRENT), "spec_hash")
            .as_str()
            .unwrap()
            .to_string();
        let spec = spec_with_root(&dir, &fixture_hash);
        // parse the archived current fixture as the state to persist
        let state = CampaignState::from_jv(&parse(CURRENT)).unwrap();
        save_state(&spec, &state).unwrap();
        assert!(state_path(&spec).exists());
        // the atomic-write leftover is gone
        assert!(!dir.join("state.json.tmp").exists());
        let loaded = load_state(&spec).unwrap().expect("state exists");
        assert_eq!(loaded, state);
        // byte-stable against the Python fixture text:
        assert_eq!(format!("{}\n", loaded.to_json_pretty()), CURRENT);
    }

    #[test]
    fn missing_state_file_is_none() {
        let dir = tmpdir("missing");
        let spec = spec_with_root(&dir, "hash-1");
        assert!(load_state(&spec).unwrap().is_none());
    }

    #[test]
    fn spec_hash_change_aborts_resume() {
        let dir = tmpdir("guard");
        let spec = spec_with_root(&dir, "hash-1");
        let state = CampaignState {
            spec_hash: "hash-1".to_string(),
            ..CampaignState::default()
        };
        save_state(&spec, &state).unwrap();
        // the same campaign root, a changed spec → SystemExit-equivalent.
        let changed = spec_with_root(&dir, "hash-2");
        let err = load_state(&changed).unwrap_err();
        assert!(matches!(err, StateError::SpecHashMismatch { .. }));
        assert_eq!(
            err.to_string(),
            format!(
                "state.json at {} was written for a different spec version; \
                 pass --fresh to discard it.",
                dir.join("state.json").display()
            )
        );
        // unchanged hash still resumes.
        assert!(load_state(&spec).unwrap().is_some());
    }

    #[test]
    fn corrupt_top_level_is_a_clean_error() {
        let dir = tmpdir("corrupt");
        let spec = spec_with_root(&dir, "hash-1");
        std::fs::write(dir.join("state.json"), "[1, 2, 3]").unwrap();
        let err = load_state(&spec).unwrap_err();
        assert!(err.to_string().contains("top level is not a JSON object"), "{err}");
        std::fs::write(dir.join("state.json"), "{not json").unwrap();
        let err = load_state(&spec).unwrap_err();
        assert!(err.to_string().contains("is corrupt"), "{err}");
    }

    #[test]
    fn fresh_state_and_current_grid_track_the_spec() {
        let dir = tmpdir("fresh");
        let spec = spec_with_root(&dir, "hash-1");
        let state = rbs_driver::fresh_state(&spec);
        assert_eq!(state.spec_hash, "hash-1");
        let grid = state.grid.unwrap();
        assert_eq!((grid.dr, grid.n), (spec.grid.dr, spec.grid.n));
        assert_eq!(state.refinements_left, Some(spec.adaptivity.max_refinements));
        assert_eq!(state.widenings_left, Some(spec.adaptivity.max_widenings));
        assert_eq!(state.status, Status::Running);

        // a legacy state without the grid key falls back to the spec's
        // grid via current_grid.
        let legacy = CampaignState {
            spec_hash: "hash-1".into(),
            ..CampaignState::default()
        };
        let effective = rbs_driver::current_grid(&legacy, &spec);
        assert_eq!((effective.dr, effective.n), (spec.grid.dr, spec.grid.n));
        assert_eq!(rbs_driver::current_grid(&state, &spec), grid);
    }

    #[test]
    fn probe_and_fold_appenders_create_on_first_use() {
        let mut state = CampaignState {
            spec_hash: "t".into(),
            ..CampaignState::default()
        };
        assert!(state.rejected_regrids.is_none());
        assert!(state.fold_fine_grid_measurement.is_none());
        state.with_rejected_probe(rbs_driver::RegridProbe {
            i: 5,
            mode: StepMode::RegridProbe,
            exit_code: Some(0),
            scale_u4: None,
            log: None,
            sol_dir: None,
            scalars: None,
            psi0: None,
            skipped: Some("support would not fit".to_string()),
            from_dr: Some(0.125),
            to_dr: Some(0.0625),
        });
        state.with_fold_measurement(rbs_driver::FoldMeasurement {
            psi0: Some(0.37),
            omega: Some(0.646),
            dr: 0.125,
            sol_dir: "/tmp/sol".to_string(),
            note: None,
        });
        assert_eq!(state.rejected_regrids.as_ref().unwrap().len(), 1);
        assert_eq!(state.fold_fine_grid_measurement.as_ref().unwrap().len(), 1);
        // second append grows, never resets (Python with_* semantics).
        state.with_fold_measurement(rbs_driver::FoldMeasurement {
            psi0: Some(0.38),
            omega: Some(0.647),
            dr: 0.125,
            sol_dir: "/tmp/sol2".to_string(),
            note: None,
        });
        assert_eq!(state.fold_fine_grid_measurement.as_ref().unwrap().len(), 2);
    }
}
