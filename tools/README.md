# Python tooling

All tooling is managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13):

```bash
uv sync --dev
uv run <script>.py --help
```

Everything lives in this directory, one script per concern. Shared helpers
(`rotboson_io.py`, `logsetup.py`) are imported, not invoked.

## Campaign driver

| script | purpose |
|--------|---------|
| `sweep_driver.py` | Continuation campaign driver: renders parameter files, invokes the single-solution binary one step at a time, tracks `state.json`/`summary.json`, and owns the adaptive policy (step-size control, regrid ladder, turning-point handling). See the root README for the spec format and decision table. |

## Running & checking solutions

| script | purpose |
|--------|---------|
| `smoke.py` | End-to-end smoke test: builds the binary, runs a parameter file, extracts key observables. Works on freshly generated solutions alone. |
| `hdf5_roundtrip.py` | Exports an HDF5 `solution.h5` back to the legacy `.asc` layout and/or compares it against a legacy ASCII reference — pins the two output backends to each other. |
| `plot_verification.py` | Generates the paper-verification figures (M(ω)/J(ω)/ω-vs-ψ₀ branch curves, Table IX.1 deviation bars, grid study) from campaign `state.json` files. Requires `matplotlib` (dev dependency). |

## Symbolic derivation & code generation

| script | purpose |
|--------|---------|
| `sympy_system.py` | Independent SymPy re-derivation of the six Einstein–Klein–Gordon residuals and the Jacobian. Single source of truth for the generated kernels. |
| `generate_kernels.py` | Emits `src/rhs_vars.c` and `src/csr_vars.c` from the SymPy derivation. Checked in; CI re-runs the generator and asserts byte-identical output (`--check`), an MMS test verifies the residual at design order. |
| `sympy_check.py` | Cross-checks the SymPy Jacobian against the codegen notebook (`derivations/Mathematica CSR Code Generation.ipynb`). |
| `mms_test.py` | Manufactured-solution test of the discretized residual (4th-order Fornberg stencils). |

## Shared helpers

| script | purpose |
|--------|---------|
| `rotboson_io.py` | Shared readers for solution directories (legacy `.asc` layout and HDF5), filename catalogs, typed state/diagnostic models. |
| `logsetup.py` | Shared logging setup; stdout is a tool's data product, logging goes to stderr. |

## Conventions

- `print()` is a tool's **stdout product** (reports/verdicts consumed as
  data); `logging` is operational diagnostics on **stderr**
  (`tools/logsetup.py`).
- Unknown TOML keys are rejected everywhere — spec files fail fast.
- Scripts exit non-zero on validation failure so CI can gate on them.
