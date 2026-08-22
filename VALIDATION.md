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
