# rbs-driver

**Role.** The solver-independent half of the continuation driver, ported from `tools/sweep_driver.py`:
- parsing the campaign spec;
- the `state.json` model;
- the adaptive step decisions;
- the fold fit and summary.

The other half, which runs real solves, is milestone A2 **(planned)**. That half replaces the Python driver. After A2 the Python byte-compatibility described below may be dropped (ADR 0002).

**Tests** (`cargo test -p rbs-driver`, about 1 s, no solver):
- `tests/driver_decisions.rs`: all 53 cases of `tests/test_driver_decisions.py`;
- `tests/state_roundtrip.rs`: all 10 cases of `tests/test_state_roundtrip.py`, with the archived fixtures in `tests/fixtures/`;
- `tests/summarize.rs`: a synthetic branch walk with a known minimum;
- unit tests in each module.

## Ported

| module | contents | Python source (`sweep_driver.py` at commit 06b4ff2) |
|---|---|---|
| `spec` | `[campaign]`/`[seed]`/`[grid]`/`[solver]`/`[output]`/`[adaptivity]` models, `load_spec`, unknown-key rejection, `spec_hash`, `is_solution_dir` | `KNOWN`, `*_Spec.parse`, `load_spec` (L266–616) |
| `state` | `GridPosition`, `StepRecord`, `RegridInfo`, `RegridProbe`, `FoldMeasurement`, `PendingRefinement`, `CampaignState`, atomic `save_state`, hash-guarded `load_state` | L643–1043 |
| `decide` | `decide_action`, `would_amputate`, `coarsening_acceptable`, `decide_coarsening`, `boundary_failure_stop`, `newtonian_limit_stop`, `finished`, `detect_turning_point`, `turning_point_estimate`, `next_target` (step-size policy), retry classification | L1382–1443, 1807–1906, 1944–2037, 2152–2163, 2334–2336 |
| `fit` | hand-rolled least-squares polynomial fit (stands in for `np.polyfit`/`np.polyval`) | `turning_point_estimate` (L2004–2037) |
| `summarize` | `--summarize` post-processing: report + `<root>/summary.json` | `summarize` (L2514–2542) |
| `vocab` | the closed vocabularies + stop-reason taxonomy (exact strings) | `StrEnum`s (L107–190) |
| `pyjson` | `json.dumps(..., indent=2)`-byte-compatible writer + order/number-identity-preserving reader | the `to_dict`/`from_dict` pair |
| `sha256` | FIPS 180-4 by hand (the spec hash) | `hashlib.sha256` in `load_spec` |
| `error` | `SpecError`, `StateError`, `DriverError`, `SummarizeError` — typed errors, no panics | `SpecError`, `SystemExit`s, uncaught `ValueError`s |

## Not yet ported (A2)

Everything that shells out to or feeds the C/Rust solver: `run_binary`
(env plumbing, wall-clock timeout), seed-field rendering
(`render_seed`, `psi0_origin_estimate`, ASCII field writers), the
per-step parameter-file rendering (`render_params`), solution-dir
discovery/scalar reading (`rotboson_io`), `do_regrid`'s interpolator
calls, `run_campaign`'s loop and its retry/verify-then-commit plumbing.
The pure half here hands over at well-defined seams: `finished()` /
`decide_*` / `turning_point_estimate` take the state + spec and return
typed decisions; `next_target`/`step_retryable` encode the loop's step
policy.

## Contract items (byte/string exact)

**Stop-reason strings** (`vocab::StopReason::Display`, pinned by
`vocab::tests::stop_reason_strings_match_python`):

```
done:psi0_target     done:omega_target    done:max_steps
stopped:newtonian_limit  stopped:boundary  stopped:turning_point
stopped:domain_budget
failed:timeout  failed:newton  failed:solver  failed:config  failed:io
failed:sig{name}       (e.g. failed:sigsegv; lowercase, no SIG prefix)
failed:exit{code}      (exit codes outside 1–4)
failed:seed_exit{code} (failed step-0 seed solve)
```

**Spec parse behavior**: unknown top-level tables and unknown keys in
known tables are hard errors with the Python-verbatim messages
(`unknown spec tables: ['x']`, `[campaign] unknown keys: ['x']`,
sorted, single-quoted); wrong TOML types are hard errors (serde);
vocabulary misses use the Python messages
(`[campaign] direction must be "up" or "down"`, ...); the
Python's `bool(...)`-coercion quirks are preserved (`direct = 1` →
`true`; NaN passes `hwl_min`/`regrid_rtol` validation but fails the
chained comparisons — each check is written exactly as the Python wrote
it). `[solver]` accepts `numThreads`/`precision`/`direct`/`fp32_ir`
(a later revision of the Python spec; the older `KNOWN[solver]` would reject the
repo's own runbook configs).

**state.json**: a Python-written state.json re-serializes
**byte-identically** from this crate (pinned against the archived
fixture; float spellings are Python `repr` — `1e-05`, `15.0` — and
int/float scalar identity survives). Legacy v1 files load leniently and
upgrade on re-save. A `spec_hash` mismatch aborts resume with the
Python's SystemExit text; the hash itself reproduces Python's
canonicalization (control/comment lines stripped with `re`'s greedy
`\s` semantics) — pinned against the Python's own hash of
`configs/runbook_l1_down.toml` and `configs/l2_waypoint_w09.toml`.

## Deliberate deviations (all documented in code)

* Python crash paths become typed errors: corrupt state values
  (`state.json: expected a number, got str`), a non-table spec table
  value, a negative exit code that names no signal
  (`DriverError::UnknownSignal`), `omega_of_last` failures
  (`finished` propagates them; the Python raises SystemExit).
* `turning_point_estimate` returns `Option<TurningPointReport>`; the
  Python's empty `{}` report is the `None` case.
* `np.polyfit` under rank deficiency (repeated ψ₀ in the fit window)
  produces an arbitrary min-norm solution; we lower the degree to the
  distinct-x count, and a still-singular system falls back to the
  `"sample"` method. For the monotone windows summarize actually sees,
  behavior is identical.
* `Path.resolve()` is approximated lexically (no symlink resolution);
  `load_spec` takes the repo root as a parameter (the Python derives it
  from `__file__`).
* A comment/control line still leaves its newline behind (Python
  `re.sub` semantics), so *editing* a comment or a runtime-limit line
  keeps the hash stable but *adding/removing* lines does not — faithful
  to the Python, just worth knowing.

## History

Ported as SAN-60 (part 1) in the old Linear team SAN, as a standalone crate that was later adopted into the workspace. Line references in the table above cite the SAN-59 head `06b4ff2` of `tools/sweep_driver.py`.
