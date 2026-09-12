# ROTBOSON

Numerical initial-data generation for rotating boson stars in axisymmetry
(3+1 decomposition, quasi-isotropic coordinates, axis regularization). Solves a
system of six coupled nonlinear elliptic PDEs plus the scalar-field frequency ω
via global Newton methods and the PARDISO sparse direct solver.

See `PLAN.md` for the modernization roadmap and `VALIDATION.md` for validation
results against the published data (arXiv:2103.13993, Class. Quantum Grav.
**38** 154003 (2021)) — Phase 0 golden fidelity plus the 2026-09 catalogue
rebuild & paper verification (l=1..4; session report
`docs/milestone-report-2026-09.md`, figures `docs/figures/`).

## Prerequisites

Linux with GCC, CMake (>= 3.20), and a C compiler with OpenMP. Two libraries:

### oneMKL (Intel Math Kernel Library)

oneMKL is now free (no license/serial) and installable via package managers.

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
`third_party/tomlc99/`); libconfig is no longer a dependency. Unknown keys are
rejected and wrong types are hard errors. Legacy `.par` files can be converted
with `uv run tools/par_to_toml.py`.

Two extra keys control output:
- `outputFormat = "hdf5"` writes a single self-describing `solution.h5` per
  solution (root attributes include the build git hash, parameter file,
  backend and creation timestamp). This is the primary format for new work —
  the Python tooling reads it natively.
- `outputFormat = "ascii"` (default) writes the legacy one-file-per-field
  `.asc` layout, byte-identical to the 2020-era catalogue output. It is a
  **compatibility backend**, kept so the golden-regression validation chain
  (which compares archived solutions byte-for-byte) runs unchanged. New
  workflows should prefer HDF5; the ASCII default stays until the HDF5
  catalogue rebuild makes the regression chain format-independent.
- `loglevel = "error" | "warn" | "info" (default) | "debug"` sets the
  verbosity of the progress/banner output.

### Single-solution contract & exit codes

The binary does one thing: **one invocation = one Newton solve = one solution
directory.** Sweep/continuation orchestration lives in the Python driver (see
`docs/sweep-driver-design.md`); the old in-C `sweep_advance`/ladder machinery
was removed. The process exit code is part of the public contract
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

This produces `build/release/ROTBOSON`. A legacy GNU Makefile is kept at the
repo root but is deprecated.

## Python tooling

Analysis/validation tools live under `tools/` (see `tools/README.md` for the
full inventory) and are managed with `uv`:

```bash
uv sync --dev
uv run tools/smoke.py out/l1_from_scratch.toml
```

### Sweep driver

`tools/sweep_driver.py` runs continuation campaigns: the C binary solves one
solution per step, Python orchestrates. The continuation parameter is **ψ₀**
(the field value at the fixedPhi grid point); Newton solves ω as an eigenvalue
each step, so the branch crosses the minimum-ω turning point naturally (design:
`docs/sweep-driver-design.md`).

```toml
# campaign spec (TOML; unknown keys rejected)
[campaign]
l = 1
direction = "up"          # amplitude growing (ω → ω_min) or "down" (ψ₀ → 0)
psi0_target = 0.008       # stop when ψ₀ crosses this
omega_target = 0.85       # optional ω stop
psi0_step = 0.03          # per-step ratio, ψ₀ → ψ₀·(1 ± psi0_step) — the
                          # default mode (golden-ladder-like, scale-free);
                          # psi0_step_mode = "absolute" switches to fixed Δψ₀
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

Adaptive behaviour (design §4–6):

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

## What a fresh clone gets you

Everything needed to build, run and test works out of the box after installing
the packages above -- no curated data required:

- the full source tree, CMake presets, vendored `tomlc99`, tests and CI config;
- the from-scratch smoke config (`out/l1_from_scratch.toml`), its coarse-grid
  CI variant (`out/l1_smoke_ci.toml`, used by `.github/workflows/ci.yml` so the
  UMFPACK fallback smoke runs in seconds) and its HDF5
  variant (`out/l1_from_scratch_hdf5.toml`), plus the continuation config
  (`out/l1_from_initial_data.toml`);
- `data/paper/` (the published paper's text and Table IX.1 critical
  points) and the runbook configs (`configs/`);
- the derivation notebooks (`derivations/`) and all Python tooling.

The archived publication dataset is **not** in the repo (gitignored;
4.7 GB, restored from the backup drive -- see `data/golden/README.md` for
provenance and `data/golden/MANIFEST.sha256` for checksums):

- `data/golden/` -- the archived publication solutions (the §4c regression
  gate compares against them);
- `data/seeds/` -- the interpolation seeds referenced by
  `data/params/regeneration/*.toml` (which use paths relative to `out/`);
- `data/summaries/`, `data/params/`, `data/convergence/` -- the Catalogue2
  summary tables and the old-pipeline parameter templates/convergence
  studies (no longer tracked; restored with the archive, or recovered from
  git history).

Without them you can still build, run CTest, solve from scratch, and use HDF5
output; you only cannot re-run the golden-regeneration validation chain.
`tools/smoke.py` and `tools/hdf5_roundtrip.py` work on freshly generated
solutions alone.

## Generating l=1 data

Two parameter files generate $l=1$ data in `out`. Run from `out/` (output
directories are created under the process working directory; ROTBOSON no
longer `chdir`s into them):

```bash
cd out
../build/release/ROTBOSON l1_from_scratch.toml
```

For the single-file HDF5 output, run the `l1_from_scratch_hdf5.toml` variant
from a scratch directory and inspect `solution.h5` with
`uv run tools/hdf5_roundtrip.py` (exports back to `.asc` and/or compares
against a legacy `.asc` reference).

This generates initial data for $l=1$, $m=1$, $\omega=0.95$ in a directory named
`l=1,w=9.50000E-01,dr=6.25000E-02,N=0256` (parameters unchanged).

Then use the other parameter file to generate many more solutions by
continuation from the previous "seed":

```bash
../build/release/ROTBOSON l1_from_initial_data.toml
```

This runs for a while (up to $\omega = 0.675222$, where it stops because the
scalar field is too "spiky" for the grid resolution).