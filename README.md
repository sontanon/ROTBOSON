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

### libconfig

Fedora: `sudo dnf install libconfig-devel`
Ubuntu/Debian: `sudo apt-get install libconfig-dev`

(For the MKL-free build via OpenBLAS + SuiteSparse/UMFPACK, see the
`ROTBOSON_SOLVER_BACKEND=umfpack` option in `CMakeLists.txt`.)

## Compilation (CMake)

```bash
source /opt/intel/oneapi/setvars.sh   # sets MKLROOT
cmake --preset release                # or: dev (asan/ubsan), asan-ubsan
cmake --build --preset release -j
```

This produces `build/release/ROTBOSON`. A legacy GNU Makefile is kept at the
repo root but is deprecated.

## Python tooling

Analysis/validation tools live under `tools/` and are managed with `uv`:

```bash
uv sync --dev
uv run tools/smoke.py out/l1_from_scratch.par
```

# Generating l=1 data

Two parameter files generate $l=1$ data in `out`. Run from `out/` (ROTBOSON
changes into the output directory it creates):

```bash
cd out
../build/release/ROTBOSON l1_from_scratch.par
```

This generates initial data for $l=1$, $m=1$, $\omega=0.95$ in a directory named
`l=1,w=9.50000E-01,dr=6.25000E-02,N=0256` (parameters unchanged).

Then use the other parameter file to generate many more solutions by
continuation from the previous "seed":

```bash
../build/release/ROTBOSON l1_from_initial_data.par
```

This runs for a while (up to $\omega = 0.675222$, where it stops because the
scalar field is too "spiky" for the grid resolution).

# TODO

* Explain $l \geq 2$.
* Explain interpolation as initial data.
* Explain the nonlinear solver.
* Redo everything in a friendlier language... 😂