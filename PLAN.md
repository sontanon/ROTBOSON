# ROTBOSON Modernization Plan

## 1. Context

ROTBOSON is a C code (originally ~2019–2022, graduate work at ICN UNAM) that generates
initial data for **rotating boson stars** in axisymmetry for numerical relativity.
It solves a system of six coupled, nonlinear elliptic PDEs (Einstein–Klein–Gordon,
3+1 decomposition, quasi-isotropic coordinates, axis regularization) plus the scalar
field frequency ω as an unknown, on a uniform 2D Cartesian `(r, z)` grid using
finite differences (orders 2/4), a global Newton method, and the sparse direct
solver PARDISO (Intel MKL).

The code produced the publication:

> Santiago Ontañón and Miguel Alcubierre, **"Rotating boson stars using finite
> differences and global Newton methods"**, Classical and Quantum Gravity **38**
> 154003 (2021). arXiv:2103.13993, DOI 10.1088/1361-6382/ac0b53.

A second paper (evolution/stability in axisymmetry) used the `RBS Stability` dataset
and a perturbation variant of this solver.

**Goal:** modernize the codebase in phases — without discarding the working C code —
into a robust, tested, well-documented numerical relativity code, with an eventual
Rust rewrite (Phase 6). The C implementation remains the physics oracle until parity
is proven. The old published data is the fidelity anchor for every refactor.

## 2. Key findings (decided during exploration, 2026-08)

### 2.1 Backup drive

External disk: `/run/media/santiago/Seagate Expansion Drive` (Seagate Expansion Drive).

| Drive folder | Content | Decision |
|---|---|---|
| `RBS Output/StandarizedOutput` | 66 solutions, l=1–6, all `error_code=0`, uniform naming, 70-file outputs, par files, Oct-2020 executables | **Golden reference dataset** → copy to `data/golden/` |
| `RBS Output/UniversalOutput` | 61 GB, 671 solutions l=1–6, reduced final quantities | Leave bulk on drive; scalars used via summaries |
| `RBS/Catalogue2` | 265 GB, 894 production solutions + tiny `l={1..6}.asc` summary tables | Leave on drive; copy summary tables → `data/summaries/` |
| `RBS/Catalogue`, `Catalogue3` | Launch par templates, PBS scripts, characterization notebooks | Copy par templates + PBS → `data/params/` |
| `RBS/Convergence`, `RBS/Regularized Convergence`, `RBS/Radial Convergence` | Resolution + boundary convergence studies | Copy par files/listings → `data/convergence/`; templates for Phase 3 tests |
| `RBS/HolyGrails` | First good l=2–6 sequence (2019, pre-regularization) | Historical; leave on drive |
| `RBS Output/Output*, 14–18` | Intermediate campaigns, some failed runs | Leave on drive |
| `RBS Stability/` | l=2 stability data, `ROTBOSON_PERTURBATION.exe`, `f_p0_i.asc`/`f_p1_i.asc` | Paper-2 data. **Perturbation solver exists only as uncommitted changes in the drive's git working tree** (`RBS Stability/ROTBOSON/ROTBOSON`). Decision: **deferred** — do not copy now; revisit before Phase 6. At-risk asset (only copy). |
| `RBS Mathematica Notebooks/` (7 `.nb`) + `RBS/*.nb` (3 derivation notebooks) | Symbolic derivations of the PDE system, Jacobian, regularization, Kerr matching | Copy → `derivations/mathematica/` |
| `RBS/*.ipynb` | Python-3 analysis + codegen notebooks (see §2.3) | Copy → `derivations/notebooks/` |

Local disk has 279 GB free; the curated copy is ~4.5 GB.

### 2.2 Git forensics

- Repo branches: `master`, `origin/Sweep`, `origin/Regularization`, `origin/Regularization-2`,
  `origin/Interpolation`, `origin/MemoryOptimization`, `origin/Constant-Omega`.
- `Regularization` and `Regularization-2` **are fully merged into master** (merge base of
  master and Regularization-2 is its last commit, `3817e02`). Master = Regularization-2 + Sweep + README.
- StandarizedOutput executables (2020-10-30) match the Sweep-era code.
  **Current master ≈ publication code.** No branch resurrection needed.
- Caveat: the "Kerr matching commented-out" commit (`b3885a7`, 2020-11-27) is slightly
  later than the StandarizedOutput binaries; Phase 0 verification will detect any drift.
- The drive's git repo (`RBS Stability/ROTBOSON/ROTBOSON`) has no unique *committed* code,
  but its dirty working tree holds the perturbation solver (see §2.1).

### 2.3 Mathematica / symbolic derivation status

- `.nb` files are plain-text Wolfram expressions; extractable without a license.
- The heavy code generation was already done in Python: `Mathematica CSR Code Generation.ipynb`
  (python3 kernel) contains the Jacobian entries as strings (generated in Mathematica and
  pasted by hand) and programmatically emitted `csr_vars.c`. All analysis notebooks are python3.
- Decision: re-derive in SymPy from the paper + `.nb` extraction, cross-check against the
  codegen notebook strings, and standardize in a single notebook under `uv`.
  No Mathematica license or Wolfram Engine required.

### 2.4 Tooling conventions

- **Python:** all Python tooling uses `uv` (single `pyproject.toml` at repo root, dev deps:
  numpy, h5py, sympy, nbformat, etc.).
- **Build:** Phase 1 introduces CMake + presets. Until then, the existing Makefile is the build.
- **Golden data naming:** the archive folder `StandarizedOutput` is renamed `data/golden/`;
  solution directories keep their self-describing names
  (`l=<l>,w=<w>,dr=<dr>,N=<N>`). A provenance `README.md` accompanies the data.

## 3. Phased roadmap

### Phase 0 — Curate, freeze, and establish the fidelity chain (first)

Tasks:
1. Copy curated items from the drive (see §2.1 table) into the project layout:
   - `data/golden/` (4.4 GB) with provenance `README.md`
   - `data/summaries/` (l={1..6}.asc tables from Catalogue2)
   - `data/params/` (Catalogue par templates + PBS scripts)
   - `data/convergence/` (par files + directory listings from the three convergence folders)
   - `derivations/mathematica/` (all 10 `.nb` files)
   - `derivations/notebooks/` (codegen, characterization, interpolator, convergence `.ipynb`)
2. Tag current master as `pre-refactor`; record toolchain (MKL 2024.1, gcc, libconfig) in `data/golden/README.md`.
3. Set up the `uv` project + a smoke-run script: build `ROTBOSON`, run a par file,
   extract ω, M_Komar, J_Komar, φ_max.
4. **Validation chain (acceptance criterion for Phase 0):**
   regenerate 2–3 golden solutions with current master from their archived par files
   → compare field/scalar outputs to the archived `.asc` files
   → compare to the `l={1..6}.asc` summary tables
   → compare to the paper's published tables (extract from arXiv:2103.13993).
   Any deviation is documented; if a specific commit reproduces StandarizedOutput
   exactly, that commit becomes the reference baseline snapshot.
5. If time permits, extract paper tables into CSV under `data/paper/`.

### Phase 1 — Build & environment modernization

- CMake + presets (`dev`, `release`, `asan-ubsan`), `compile_commands.json`,
  `-Wall -Wextra` (warnings promoted as errors over time), keep OpenMP.
- Dependency strategy: oneMKL via CMake when available (PARDISO + BLAS), otherwise
  OpenBLAS + SuiteSparse/UMFPACK fallback. Introduce a thin `solver_backend` interface
  so both coexist (low-rank update remains PARDISO-only).
- Remove `env.bash` in favor of presets + documented apt/conda/spack instructions.
- CI (GitHub Actions): matrix {MKL build, OSS-fallback build} + smoke test from Phase 0.
- oneMKL is now free (no license/serial); available via apt/conda/spack/docker.

### Phase 2 — Config overhaul + structural refactor

- libconfig → TOML (pure-C99 `tomlc99` preferred to keep the codebase C-only).
  Strict validation (unknown keys rejected, range checks). Port the `.par` files.
- Decompose `main.c` (804 lines) into: driver, Newton orchestration, sweep/continuation,
  analysis, I/O.
- Eliminate the `param.h` global-state pattern (`#ifdef MAIN_FILE` + `extern` soup):
  introduce a `rb_context` struct passed explicitly. Do this incrementally,
  re-running the golden tests after each step.
- Delete `deprecated/`, dead `regularization_coupling.h` code, unused ifdefs.

### Phase 3 — Testing strategy (Criterion + CTest)

Layered suite:
1. **Unit:** FD stencil weights vs Fornberg-generated coefficients; every derivative operator.
2. **Component:** manufactured-solution tests — source terms added to the PDE; assert the
   residual/Jacobian recovers the known solution with correct convergence order.
   Templates come from the `data/convergence/` campaigns.
3. **Integration:** full Newton solve on small grids vs the golden set.
4. **Regression:** end-to-end par runs vs Phase 0 goldens (ω, masses, profiles),
   run on every refactor commit.

### Phase 4 — SymPy code generation

- Re-derive the Einstein–Klein–Gordon system (axisymmetric, l≥1, regularization variable)
  in SymPy from the paper + `.nb` extraction.
- Cross-check SymPy expressions against the Jacobian strings in
  `Mathematica CSR Code Generation.ipynb` and the current `rhs_vars.c`/`csr_vars.c`.
- Produce one standardized, `uv`-managed notebook/script that regenerates the residual
  and Jacobian C code (plus stencil weights). Generated files are checked in; a test
  verifies "regenerated == checked-in".

### Phase 5 — Output format: HDF5

- HDF5 via CMake: one self-describing file per solution (fields, grids, attributes:
  params, git hash, solver settings). Keep a legacy `.asc` exporter during transition.
- Python readers (`h5py`) in the `uv` project; golden `.asc` data provides migration fixtures.

### Phase 6 — Rust migration (last, opportunistic)

- Prerequisites: validated golden suite, clean C reference, HDF5 I/O, stable config.
- Port incrementally: config → grid/derivatives → residual → Newton → solver backend,
  cross-validating against the C binary at each step.
- Candidate stack: `faer` or FFI to MKL/UMFPACK, `serde` + `toml`, `hdf5` crate, `ndarray`.
- Before starting, revisit the deferred perturbation-solver patch on the drive
  (archive it; it may inform the stability/evolution features).

## 4. Dependencies between phases

- Phase 0 must precede everything (freeze + fidelity chain).
- Phase 1 unblocks CI; Phase 2 and 3 interleave (refactor ↔ test).
- Phase 4 and 5 are independent after Phase 2.
- Phase 6 consumes all of the above.

## 5. Risks & notes

- Phase 0 verification may show master ≠ StandarizedOutput exactly (post-Oct-2020 commits).
  Mitigation: snapshot the matching commit as the reference baseline.
- Perturbation/stability solver: uncommitted, exists only on the drive (deferred copy).
  Drive failure before Phase 6 would lose it.
- The paper reports 4th-order (resolution) and 3rd-order (boundary) convergence;
  the convergence folders are the evidence — preserve their par files.
- Catalogue2 (265 GB) and UniversalOutput (61 GB) stay on the drive; only their scalar
  summaries enter the repo.
