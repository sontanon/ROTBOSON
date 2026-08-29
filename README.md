# ROTBOSON

Numerical initial-data generation for rotating boson stars in axisymmetry
(3+1 decomposition, quasi-isotropic coordinates, axis regularization). Solves a
system of six coupled nonlinear elliptic PDEs plus the scalar-field frequency ω
via global Newton methods and the PARDISO sparse direct solver.

See `PLAN.md` for the modernization roadmap and `VALIDATION.md` for the
Phase 0 fidelity results against the published data
(arXiv:2103.13993, Class. Quantum Grav. **38** 154003 (2021)).

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
- `outputFormat = "ascii"` (default) writes the legacy one-file-per-field
  `.asc` layout; `outputFormat = "hdf5"` writes a single self-describing
  `solution.h5` per solution.
- `loglevel = "error" | "warn" | "info" (default) | "debug"` sets the
  verbosity of the progress/banner output.

### HDF5 (optional, for `outputFormat = "hdf5"`)

The HDF5 backend needs the HDF5 C library:

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

Analysis/validation tools live under `tools/` and are managed with `uv`:

```bash
uv sync --dev
uv run tools/smoke.py out/l1_from_scratch.toml
```

## What a fresh clone gets you

Everything needed to build, run and test works out of the box after installing
the packages above -- no curated data required:

- the full source tree, CMake presets, vendored `tomlc99`, tests and CI config;
- the from-scratch smoke config (`out/l1_from_scratch.toml`), its coarse-grid
  CI variant (`out/l1_smoke_ci.toml`, used by `.github/workflows/ci.yml` so the
  UMFPACK fallback smoke runs in seconds) and its HDF5
  variant (`out/l1_from_scratch_hdf5.toml`), plus the continuation config
  (`out/l1_from_initial_data.toml`);
- all catalogue/convergence/stability parameter templates and the published
  summary tables (`data/summaries/`, `data/params/`, `data/convergence/`,
  `data/paper/`);
- the derivation notebooks (`derivations/`) and all Python tooling.

Two things are **not** in the repo (gitignored; 4.7 GB, restored from the
backup drive -- see `data/golden/README.md` for provenance and
`data/golden/MANIFEST.sha256` for checksums):

- `data/golden/` -- the archived publication solutions (the §4c regression
  gate compares against them);
- `data/seeds/` -- the interpolation seeds referenced by
  `data/params/regeneration/*.toml` (which use paths relative to `out/`).

Without them you can still build, run CTest, solve from scratch, and use HDF5
output; you only cannot re-run the golden-regeneration validation chain.
`tools/smoke.py` and `tools/hdf5_roundtrip.py` work on freshly generated
solutions alone.

# Generating l=1 data

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

# TODO

* Explain $l \geq 2$.
* Explain interpolation as initial data.
* Explain the nonlinear solver.
* Redo everything in a friendlier language... 😂