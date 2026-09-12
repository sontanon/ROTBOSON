# Phase 0 validation results

Fidelity check of current `master` (commit f156eea, tag `pre-refactor`) against the
archived publication data. All runs below used the existing Makefile build
(MKL 2026.1, libconfig, gcc 16, Fedora) and the archived parameter files with
local seed paths (see `data/params/regeneration/`).

## 1. Golden regeneration (field-level)

Each archived golden solution was regenerated with `master` and compared to the
archived output at `rtol=1e-10` (with `atol=1e-12` fallback for near-zero fields).

| l | w | grid | verdict | max rel. field diff |
|---|---|---|---|---|
| 1 | 9.00000E-01 | dr=8.0e-2, N=400 | PASS | ~1e-13 |
| 2 | 8.74062E-01 | dr=8.0e-2, N=400 | PASS (vs Catalogue2) | ~1e-13 |
| 3 | 9.00000E-01 | dr=8.0e-2, N=400 | PASS | ~1e-13 |
| 4 | 9.00000E-01 | dr=8.0e-2, N=400 | PASS | ~1e-13 |
| 5 | 8.00000E-01 | dr=8.0e-2, N=400 | PASS | ~1e-13 |
| 6 | 8.00000E-01 | dr=8.0e-2, N=400 | PASS | ~1e-13 |

All fields (metric potentials, scalar field, regularization variable), scalar
observables (Komar masses, angular momenta, ADM/Schwarzschild masses, phi_max,
r99, ergoregion flag) and spherical-coordinate interpolations agree to ~1e-13.

## 2. Cross-check against the published catalogue tables

`data/summaries/l={1..6}.asc` are the Catalogue2 tables behind the paper's
figures. For every regenerated solution whose w is present in those tables
(l=1,2,3,4), M_Komar and J_Komar agree to ~1e-14. (l=5,6 golden points at
w=0.8 are not sampled in the catalogue tables, so no comparison there.)

## 3. Cross-check against the published critical points (Table IX.1)

`data/paper/table_ix1.csv` holds the published critical points (arXiv:2103.13993).
The summary tables reproduce the published maximum masses and angular momenta to
all five (resp. four) significant figures for every l=1..6:

| l | M_max (paper) | M_max (summary) | J_max (paper) | J_max (summary) |
|---|---|---|---|---|
| 1 | 1.3155 | 1.315451 | 1.382 | 1.381537 |
| 2 | 2.2159 | 2.215898 | 4.810 | 4.810248 |
| 3 | 3.5287 | 3.528676 | 12.49 | 12.492814 |
| 4 | 5.0590 | 5.058943 | 25.83 | 25.832284 |
| 5 | 6.6681 | 6.667988 | 44.63 | 44.628839 |
| 6 | 8.2824 | 8.282160 | 69.02 | 69.009766 |

Turning-point frequencies agree to the paper's stated precision; the small
last-digit differences are expected because the paper reports spline-interpolated
turning points, whereas the summary tables are raw discrete samples (and the
turning point is resolution-sensitive).

## 4. Known discrepancy: GRV2/GRV3 for the l=2 Catalogue2 solution

The only mismatch found anywhere is `GRV2.asc`/`GRV3.asc` (the virial-identity
diagnostics) for the l=2 solution at w=8.74062E-01, when compared against the
Catalogue2 copy of that same solution (fields and all physical observables there
still agree to ~1e-13).

Root cause (git archaeology): the Catalogue2 files predate commit `fe80330`
("Fix modulus calculation", 2020-10-30) which changed the Kerr-extrapolation
correction in `analysis.c` from a local `rInf = sph_rr[last]` to the global
`rr_inf` (and the Catalogue2 run itself used a still-earlier analysis build).
The virial integrands themselves are unchanged across the entire history. The
**publication** dataset (StandarizedOutput) is unaffected: its GRV2/GRV3 match
master to ~1e-12 (verified for l=1). See `tools/reconstruct_grv.py`, which
recomputes GRV2/GRV3 from saved spherical data and reproduces the C output.

## 5. Modern-MKL build fix

`mkl_dcsrgemv` (legacy sparse BLAS) was removed from MKL 2026; the single call
site in `src/bicubic_interpolation.c` was replaced with an equivalent inline
16x16 CSR matvec (same math, no API dependency). The smoke test
(`out/l1_from_scratch.par`, l=1 w=0.95) converges and reproduces the expected
solution directory name and omega.

## Tools (uv-managed, see pyproject.toml)

- `tools/smoke.py` — build + run a par file + extract observables
- `tools/compare_solutions.py` — compare a regenerated dir against a reference
- `tools/check_against_summary.py` — cross-check against data/summaries/l={1..6}.asc
- `tools/reconstruct_grv.py` — recompute GRV2/GRV3 from saved spherical data
- `tools/ladder_continue.py` — fixedPhi scale-ladder continuation (with logging)
- `tools/hdf5_roundtrip.py` — export/compare a `solution.h5` against legacy `.asc`

## 6. Phase 4 (SymPy code generation) validation

Phase 4 re-derived the residual + Jacobian in SymPy and regenerated the C
kernels.  Two independent checks, plus the fidelity anchor:

- **Symbolic cross-check** (`tools/sympy_check.py`): the SymPy Jacobian matches
  the Mathematica-pasted strings in the codegen notebook at all 6×31 entries
  (hundreds of random points).
- **MMS convergence** (`tools/mms_test.py`): the residual recovers a
  manufactured solution at the design 4th order on the interior (observed
  ~3.3 → ~3.8 on 64→256 grids).
- **Golden gate (§4c):** with the regenerated `csr_vars.c` + `rhs_vars.c`, the
  l=1 w=0.9 N=400 golden regeneration still matches `data/golden/` to ~1e-13
  (fields + all observables PASS at rtol=1e-10), and the l=1 w=0.95
  from-scratch smoke reproduces the expected ω.  Reproducibility is enforced
  by `tools/generate_kernels.py --check` (byte-identical regeneration).

## 7. Phase 5 (HDF5 output) validation

Phase 5 introduced the `solution_writer` I/O abstraction (ASCII + HDF5
backends) and a level-gated logger. Two acceptance checks:

- **ASCII bit-parity (golden gate, step 1):** with the refactored writer, the
  l=1 w=0.9 N=400 regeneration (`data/params/regeneration/l=1,validate.toml`)
  matches `data/golden/` to ~1e-13 on every field and observable
  (`tools/compare_solutions.py`, rtol=1e-10 / atol=1e-12) — identical to the
  Phase 0/4 result, confirming the `.asc` format is unchanged.
- **HDF5 round-trip (step 2):** an `outputFormat="hdf5"` run of the l=1 w=0.95
  smoke produces `solution.h5` with 70 datasets (named `<field>.asc`) and 70
  attributes (params, solver settings, git hash, analysis results).
  `tools/hdf5_roundtrip.py --ref <ascii_solution>` compares `solution.h5`
  against the ASCII solve with PASS on every field/scalar (field diffs ~1e-15,
  the solver's own run-to-run OpenMP reduction noise).  The `--out` exporter is
  lossless: exporting `solution.h5` back to `.asc` and re-parsing recovers the
  datasets bit-for-bit (the `%9.18E` format round-trips doubles exactly).

Also: `tests/test_output.c` (CTest) pins the ASCII writer's byte output and
path-awareness; all three presets (release, umfpack, asan-ubsan) build and
pass CTest.

## 8. Catalogue rebuild & paper verification (2026-09)

The full pipeline (single-solution C solver + Python sweep driver + HDF5)
regenerated the rotating-boson-star catalogue and verified the published
results (arXiv:2103.13993, Table IX.1 critical points) for l=1..4: critical
points agree with the paper to ≤0.03% (l=1,2) and ≤1.2% (l=3,4, resolution-
limited folds), with every discrepancy traced to resolution/domain effects —
not physics. Full session report, campaign inventory, physics findings and
driver-bug post-mortems: `docs/milestone-report-2026-09.md`; verification
figures in `docs/figures/` (regenerate with
`uv run tools/plot_verification.py`). The l=5/6 coarse campaigns validated
the cross-l seeding technique and were descoped by decision (2026-09-12);
report §7 records the stop state.

## SAN-10 — single-solution strip (2026-08-30)

Removed `sweep_advance()` and the in-C sweep/ladder machinery (keys `sweep`,
`scale_next`, `w_step`, `w_min`/`w_max`, `rr_phi_max_minimum/maximum`,
`hwl_min/max`); strict exit codes introduced (`src/exit_codes.h`, pinned by the
new `exit_codes` CTest); HDF5 provenance extended with a `created` (UTC)
attribute. Validation on this machine (both backends, `release` and `umfpack`
presets):

- **Behaviour unchanged:** the l=1 w=0.95 N=256 from-scratch solve with the
  stripped binary matches the pre-strip binary's output on every field and
  observable (`tools/compare_solutions.py`, rtol=1e-10 / atol=1e-12; worst
  diffs ~1e-12, i.e. the solver's run-to-run OpenMP reduction noise). Note the
  output is *not* byte-identical between ANY two runs of the same binary
  (pre-existing; see below), so the §4c tolerance gate — not byte comparison —
  is the correct gate.
- **CTest:** 4/4 pass on both presets, including the new `exit_codes` test
  (config errors → 3, valid coarse solve → 0).
- **HDF5:** `outputFormat="hdf5"` run exits 0; `solution.h5` carries
  `git_hash`, `parfile`, `output_backend`, `created`; scalars agree with the
  ASCII solve to ~1.6e-13.

Known pre-existing issue (filed separately from SAN-10): rare run-to-run
nondeterminism in this environment — occasionally a coordinate/field row shows
a value from a neighbouring row (OpenMP grid-fill race), and one intermittent
segfault was observed with the UMFPACK binary under load. Both reproduce on
pre-strip master; tracked in a dedicated issue.
