# Python tooling

All tooling is managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13):

```bash
uv sync --dev
uv run <script>.py --help
```

Everything lives in this directory, one script per concern. Shared helpers
(`rotboson_io.py`, `logsetup.py`) are imported, not invoked.

## Sweep driver

| script | purpose |
|--------|---------|
| `sweep_driver.py` | Continuation campaign driver: renders parameter files, invokes the single-solution binary one step at a time, tracks `state.json`/`summary.json`, and owns the adaptive policy (step-size control, regrid ladder, turning-point handling). See the root README and `docs/sweep-driver-design.md` for the spec format and decision table. |

## Validation chain (golden / published data)

These scripts verify the current code against the archived publication
solutions (`data/golden/`, restored from the backup drive) and the published
summary tables:

| script | purpose |
|--------|---------|
| `smoke.py` | End-to-end smoke test: builds the binary, runs a parameter file, extracts key observables. Works on freshly generated solutions alone. |
| `compare_solutions.py` | Field-by-field comparison of a regenerated solution against a golden reference (exit code 0 within tolerance). |
| `check_against_summary.py` | Cross-checks a solution's M_Komar/J_Komar/ω against the Catalogue2 summary tables (`data/summaries/l={1..6}.asc`). |
| `hdf5_roundtrip.py` | Exports an HDF5 `solution.h5` back to the legacy `.asc` layout and/or compares it against a legacy ASCII reference — pins the two output backends to each other. |
| `reconstruct_grv.py` | Python reconstruction of the GRV2/GRV3 virial identities from a solution directory, cross-checking `src/analysis.c`. |
| `ladder_continue.py` | Re-applies the archived l=2 fixedPhi ladder template (scale_u4 = 1.125) step by step from a seed solution — the historical validation workflow, now largely superseded by `sweep_driver.py` in `fixedPhi` mode. |
| `plot_verification.py` | Generates the 2026-09 paper-verification figures (M(ω)/J(ω)/ω-vs-ψ₀ branch curves, Table IX.1 deviation bars, grid study) from campaign `state.json` files into `docs/figures/`. Requires `matplotlib` (dev dependency). |

## Symbolic derivation & code generation

| script | purpose |
|--------|---------|
| `sympy_system.py` | Independent SymPy re-derivation of the six Einstein–Klein–Gordon residuals and the Jacobian. Single source of truth for the generated kernels. |
| `generate_kernels.py` | Emits `src/rhs_vars.c` and `src/csr_vars.c` from the SymPy derivation. Checked in; a test re-runs the generator and asserts byte-identical output, an MMS test verifies the residual at design order. |
| `sympy_check.py` | Cross-checks the SymPy Jacobian against the historical Mathematica notebook strings. |
| `mms_test.py` | Manufactured-solution test of the discretized residual (4th-order Fornberg stencils). |

## Utilities

| script | purpose |
|--------|---------|
| `rotboson_io.py` | Shared readers for solution directories (legacy `.asc` layout and HDF5), filename catalogs, typed state/diagnostic models. |
| `logsetup.py` | Shared logging setup; stdout is a tool's data product, logging goes to stderr. |
| `par_to_toml.py` | Converts legacy libconfig `.par` parameter files to the strict TOML format. |
| `strip_notebooks.py` | Strips notebook outputs in-place for version control (full-output originals live on the backup drive). |

## Conventions

- `print()` is a tool's **stdout product** (reports/verdicts consumed as
  data); `logging` is operational diagnostics on **stderr**
  (`tools/logsetup.py`).
- Unknown TOML keys are rejected everywhere — spec files fail fast.
- Scripts exit non-zero on validation failure so CI can gate on them.
