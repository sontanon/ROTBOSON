# ROTBOSON

Numerical initial-data generation for rotating boson stars in axisymmetry
(3+1 decomposition, quasi-isotropic coordinates, axis regularization). Solves a
system of six coupled nonlinear elliptic PDEs plus the scalar-field frequency ω
via global Newton methods and the PARDISO sparse direct solver.

**Start here: [`RUNBOOK.md`](RUNBOOK.md)** — the complete, top-to-bottom path
from a fresh clone to the four-branch catalogue (l = 1..4) and its
verification against the published data (arXiv:2103.13993, Class. Quantum
Grav. **38** 154003 (2021), Table IX.1). This README covers the build, the
tooling, and the repository layout.

## Prerequisites

Linux with GCC, CMake (>= 3.20), and a C compiler with OpenMP. Two libraries:

### oneMKL (Intel Math Kernel Library)

oneMKL is free (no license/serial) and installable via package managers.

Fedora:
```bash
sudo tee /etc/yum.repos.d/oneAPI.repo > /dev/null << 'EOF'
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
sudo dnf install intel-oneapi-mkl-devel
```

Ubuntu/Debian:
```bash
sudo apt-get install -y wget gpg
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/intel-oneapi.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/intel-oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/intel-oneapi.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel
```

Alternatively install via conda (`conda install -c conda-forge onemkl`),
spack, or the Intel installer. Activate the environment (sets `MKLROOT` and the
runtime library path) before configuring/building:

```bash
source /opt/intel/oneapi/setvars.sh
```

### Configuration (TOML)

Parameter files are TOML, parsed by a vendored `tomlc99` (see
`third_party/tomlc99/`); libconfig is not a dependency. Unknown keys are
rejected and wrong types are hard errors.

Two extra keys control output:
- `outputFormat = "hdf5"` writes a single self-describing `solution.h5` per
  solution (root attributes include the build git hash, parameter file,
  backend and creation timestamp). This is the primary format for new work —
  the Python tooling reads it natively.
- `outputFormat = "ascii"` (default) writes the legacy one-file-per-field
  `.asc` layout, byte-identical to the 2020-era catalogue output. It is a
  compatibility backend, kept so solutions remain interchangeable with the
  historical pipeline.
- `loglevel = "error" | "warn" | "info" (default) | "debug"` sets the
  verbosity of the progress/banner output.

### Single-solution contract & exit codes

The binary does one thing: **one invocation = one Newton solve = one solution
directory.** Sweep/continuation orchestration lives in the Python driver
(see the sweep-driver section below); the old in-C `sweep_advance`/ladder
machinery was removed. The process exit code is part of the public contract
(`src/exit_codes.h`):

| code | meaning                                          |
|------|--------------------------------------------------|
| 0    | converged (`error_code = 0`), solution written    |
| 1    | Newton did not converge within `maxNewtonIter`    |
| 2    | linear-solver error (PARDISO/UMFPACK failure)     |
| 3    | configuration/parse error (bad TOML, bad paths)   |
| 4    | I/O error (output directory/file failure)         |

Pinned by the `exit_codes` CTest.

### HDF5 (optional, for `outputFormat = "hdf5"`)

The primary output backend needs the HDF5 C library:

Fedora: `sudo dnf install hdf5-devel`
Ubuntu/Debian: `sudo apt-get install libhdf5-dev`
macOS: `brew install hdf5`

If HDF5 is not found at configure time the build continues without it and
`outputFormat = "hdf5"` fails at runtime with a clear message.

### MKL-free fallback (SuiteSparse/UMFPACK)

To build without Intel oneMKL, use the UMFPACK backend instead:

Fedora: `sudo dnf install suitesparse-devel`
Ubuntu/Debian: `sudo apt-get install libsuitesparse-dev`

Then build with `cmake --preset umfpack`. The UMFPACK backend produces results
identical to the PARDISO backend (the low-rank factorization update is
PARDISO-only and silently falls back to a full refactorization). UMFPACK is
single-threaded, so it is slower than PARDISO for large grids.

macOS (via Homebrew): `brew install suite-sparse` and
`cmake --preset umfpack` (the OSS backend is the natural default there since
oneMKL's macOS distribution is heavier).

## Compilation (CMake)

```bash
source /opt/intel/oneapi/setvars.sh   # sets MKLROOT (pardiso backend only)
cmake --preset release                # or: dev (asan/ubsan), asan-ubsan, umfpack
cmake --build --preset release -j
```

This produces `build/release/ROTBOSON`.

## Running a solve

The binary writes the solution directory under the process working directory,
so run it from a scratch/output directory (`out/` is the convention; it exists
in the repo as a stub):

```bash
cd out && ../build/release/ROTBOSON ../configs/l1_seed.toml && cd ..
```

`configs/` holds the ready-made specs: the starter seed, the regrid template,
the four runbook down-campaigns, the cross-l seed specs, and demo specs for
the driver's refinement/coarsening machinery. The runbook walks them in
order — that is the intended usage path, not ad-hoc parameter editing.

## Python tooling

Tooling lives under `tools/` (see `tools/README.md` for the per-tool
reference) and is managed with `uv`:

```bash
uv sync --dev
```

The final list: `sweep_driver.py` (continuation campaigns), `smoke.py`
(build + run + extract observables), `plot_verification.py` (the
verification figures from campaign state files), `hdf5_roundtrip.py`
(HDF5 ⇄ ASCII roundtrip), `rotboson_io.py` + `logsetup.py` (shared
helpers), and the symbolic/codegen chain `sympy_system.py`,
`generate_kernels.py`, `sympy_check.py`, `mms_test.py`.

The generated C kernels (`src/rhs_vars.c`, `src/csr_vars.c`) are checked in;
CI regenerates them and asserts byte-identical output — the codegen gate is

```bash
uv run tools/generate_kernels.py --check
```

Run it locally after touching `sympy_system.py` or the notebook.

### Sweep driver

`tools/sweep_driver.py` runs continuation campaigns: the C binary solves one
solution per step, Python orchestrates. The continuation parameter is **ψ₀**
(the field value at the fixedPhi grid point); Newton solves ω as an eigenvalue
each step, so the branch crosses the minimum-ω turning point naturally.

```toml
# campaign spec (TOML; unknown keys rejected)
[campaign]
l = 1
direction = "up"          # amplitude growing (ω → ω_min) or "down" (ψ₀ → 0)
psi0_target = 0.008       # stop when ψ₀ crosses this
omega_target = 0.85       # optional ω stop
psi0_step = 0.03          # per-step ratio, ψ₀ → ψ₀·(1 ± psi0_step) — the
                          # default mode (scale-free); psi0_step_mode =
                          # "absolute" switches to fixed Δψ₀
max_retries = 3           # on Newton failure the step shrinks ×½ and retries
max_steps = 20

[seed]
policy = "from_scratch"   # analytic-guess seed solve at fixed ω...
w0 = 0.95                 # ...or policy = "solution" with source = <solution dir>

[grid]
dr = 0.25
N = 64

[output]
root = "out/campaigns/l1-up"
format = "hdf5"

[adaptivity]              # optional — defaults shown
hwl_min = 8               # regrid dr ÷2 when the field's half-width drops below this
max_refinements = 2       # dr ÷2 refinement budget per campaign
refine_keeps_domain = false  # refinement dr ÷2 with N×2 (domain KEPT) instead
                          # of the legacy fixed-N domain shrink — the runbook
                          # setting (the legacy ladder becomes self-defeating
                          # at high M/R: the domain shrinks below the field)
newtonian_delta = 1.0e-2  # ω → m proximity for the newtonian-limit stop
boundary_fraction = 0.95  # down-campaigns stop (stopped:boundary) when
                          # r99/r_bdy crosses this — unless widening is
                          # still possible (the guard yields to coarsening)
support_fraction = 0.85   # down-campaigns WIDEN (dr ×2, domain grows) when
                          # r99/r_bdy crosses this; must be < boundary_fraction
regrid_rtol = 2.0e-2      # a widening is accepted only if ω/M_Komar/J_Komar
                          # stay within this truncation proxy of the source
max_widenings = 2         # dr ×2 widening budget per campaign; with the
                          # dr_max floor (see [grid]) exhausted →
                          # stopped:domain_budget
```

The `[grid]` table's `dr_max` is the coarseness floor: no widening may push
dr past it (e.g. `dr_max = 0.5` caps the grid coarseness), so a dilute-end
campaign ends cleanly once the field's support cannot be accommodated
within the allowed grid range.

```bash
uv run tools/sweep_driver.py <campaign.toml>            # runs the campaign
uv run tools/sweep_driver.py <campaign.toml> --fresh    # discard state, start over
uv run tools/sweep_driver.py <campaign.toml> --summarize  # localize ω_min (post-processing)
```

Each step writes `stepNNNN.toml` + `logs/stepNNNN.log` under the campaign
root; `state.json` is updated atomically after every step, so an interrupted
campaign resumes from the last completed step on the next invocation (a
changed spec aborts resume). Solutions are read back via `tools/rotboson_io.py`
(HDF5 preferred), and each step records ψ₀, ω, Komar mass/angular momentum,
`rr_phi_max`, `r99`, `hwl_resolution` and the Newton health (iteration count,
tail damping λ, final ‖f‖).

Adaptive behaviour:

- **Step-size control** — a persistent step factor grows (×1.25) after fast,
  healthy convergence and shrinks (×½) after grudging convergence or failure;
  Newton non-convergence shrinks and retries, a solver error retries once
  (rule 4), and a signal-killed step retries at the same size.
- **Regrid ladder** — when the decision table calls for it, the driver
  re-solves the *same* ψ₀ on a neighbouring grid, seeding through the C
  interpolator (`readInitialData = 3`) and correcting the constraint scale
  so ψ₀ lands exactly. Refinement (dr ÷2) doubles N when
  `refine_keeps_domain` is set — the domain is KEPT (the runbook setting;
  the legacy fixed-N variant shrinks the domain and becomes self-defeating
  at high M/R) — and is accepted on convergence + exact ψ₀ landing, merely
  recording the old grid's error. Coarsening (dr ×2, domain grows) fires on
  down-campaigns when the field's support fills the domain
  (`support_fraction`), is bounded by the `dr_max` floor and the
  `max_widenings` budget, and is accepted only when ω/M_Komar/J_Komar stay
  within `regrid_rtol` of the source (truncation-error proxy). Budgets
  exhausted → `stopped:domain_budget`; a boundary-class failure with the
  refinement spent → a clean `stopped:boundary` end.
- **Turning point** — dω/dψ₀ is monitored across the last three branch
  points; with `stop_at_turning_point = true` (default) the campaign stops
  cleanly (`stopped:turning_point`, ω_min estimate in `state.json`), otherwise
  it switches to fine sampling and steps through the fold. `--summarize`
  localizes ω_min from any finished campaign with a degree-4 polynomial fit
  (paper §IX) and writes `summary.json`.

The decision logic lives in pure functions (`decide_action`,
`detect_turning_point`, `turning_point_estimate`), unit-tested in
`tests/test_driver_decisions.py` (`uv run pytest`).

## Repository layout

```
src/            the C solver (elliptic system, Newton, PARDISO/UMFPACK backends,
                HDF5/ASCII output, analysis diagnostics)
tests/          CTest executables + Python tests (incl. pinned HDF5/JSON fixtures)
configs/        the 13 campaign/seed/demo specs (TOML; the runbook walks them)
tools/          the 10 Python tools (see tools/README.md)
derivations/    the Mathematica codegen notebook (source of the checked-in kernels)
data/paper/     the published paper (text + Table IX.1 critical points as CSV)
cmake/          FindMKL
third_party/    vendored tomlc99
.github/        CI (both backends + Python checks, per PR)
out/            run outputs (gitignored except the .gitkeep stub)
RUNBOOK.md      the usage path: fresh clone → catalogue → verification
```

## What a fresh clone gets you

Everything needed to build, run the runbook, and test works out of the box
after installing the packages above — no curated data required:

- the full source tree, CMake presets, vendored `tomlc99`, tests and CI config;
- `configs/` — the runbook specs, including the coarse CI smoke spec
  (`configs/l1_smoke_ci.toml`, used by `.github/workflows/ci.yml` so the
  UMFPACK fallback smoke runs in seconds);
- `data/paper/` (the published paper's text and Table IX.1 critical points);
- the codegen notebook (`derivations/`) and all Python tooling;
- an empty `out/` stub — the solver and the driver write all outputs there,
  nothing under it is tracked.

There is no golden/archived dataset in the repo and no `docs/` directory:
the historical validation narrative lives in git history, and the archived
publication solutions (4.7 GB) live on the backup drive. You can build, run
CTest, solve from scratch, use HDF5 output, and reproduce the full catalogue
via the runbook from a fresh clone alone.

## Known issue: GRV2/GRV3 for the l=2 Catalogue2 copy

The only mismatch ever found against the archived publication data is in the
`GRV2.asc`/`GRV3.asc` virial-identity diagnostics of the l=2 solution at
w=8.74062E-01 **as copied into Catalogue2** — the fields and all physical
observables there still agree to ~1e-13. Root cause (git archaeology): the
Catalogue2 files predate commit `fe80330` ("Fix modulus calculation",
2020-10-30), which changed the Kerr-extrapolation correction in `analysis.c`
from a local `rInf = sph_rr[last]` to the global `rr_inf` (and the Catalogue2
run itself used a still-earlier analysis build); the virial integrands
themselves are unchanged across the entire history. The **publication**
dataset (StandarizedOutput) is unaffected: its GRV2/GRV3 match master to
~1e-12 (verified for l=1). The tool that recomputes GRV2/GRV3 from saved
spherical data was retired with the golden chain and is recoverable from git
history.
