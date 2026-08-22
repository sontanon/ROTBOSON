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

#### Phase 0 — OUTCOME (completed 2026-08-22)

Phase 0 is **closed**. Branch `phase0/curation` (commits `2b892aa`..`de2e9b5`)
was merged into `master` and tagged `phase0-complete`. Details in `VALIDATION.md`.

- Curated copy from the backup drive: `data/golden/` (4.6 GB, untracked, SHA-256
  manifest), `data/summaries/`, `data/params/`, `data/convergence/`, `data/seeds/`,
  `data/paper/`, `derivations/{mathematica,notebooks}/`.
- `uv`-managed Python tooling under `tools/` (smoke, compare_solutions,
  check_against_summary, reconstruct_grv, ladder_continue).
- Fidelity chain established and passed:
  - master regenerates the StandarizedOutput golden solutions (l=1,3,4,5,6) and the
    l=2 catalogue step to ~1e-13 (fields + all physical observables).
  - M_Komar / J_Komar match the published Catalogue2 tables to ~1e-14.
  - Critical points (M_max, J_max) match the paper's Table IX.1 to all published
    significant figures for l=1..6 (see `data/paper/table_ix1.csv`).
- One documented discrepancy (not a code bug): GRV2/GRV3 for the l=2 Catalogue2 copy
  differ ~1% due to a 2020-10-30 `analysis.c` change that postdates that run; the
  publication dataset is unaffected. `tools/reconstruct_grv.py` reproduces the C output.
- Build fix: `mkl_dcsrgemv` removed from MKL 2026 → replaced with inline CSR matvec
  in `src/bicubic_interpolation.c`.
- Deferred (see §5): perturbation-solver patch on the drive; l=2 golden points at
  w=7.29141/7.20859 require finer continuation (Phase 3 territory).

### Phase 1 — Build, environment & tooling modernization

- **Build:** CMake + presets (`dev`, `release`, `asan-ubsan`), `compile_commands.json`,
  `-Wall -Wextra` (promote to `-Werror` incrementally), keep OpenMP.
- **Dependencies:** oneMKL via CMake when available (PARDISO + BLAS), otherwise
  OpenBLAS + SuiteSparse/UMFPACK fallback. Introduce a thin `solver_backend` interface
  so both coexist (low-rank update remains PARDISO-only). Remove `env.bash` in favor of
  presets + documented `dnf`/`apt`/conda/spack instructions. oneMKL is now free
  (no license/serial); available via package managers/docker.
- **Linting / formatting / static analysis (new):**
  - C: `clang-format` (checked in `.clang-format`), `clang-tidy` + `cppcheck` as CMake
    `lint` target, compiler sanitizers (ASan/UBSan) in the `asan-ubsan` preset.
  - Python: `ruff` (lint + format) as a `uv` dev dependency and CI check.
  - `pre-commit` hooks (format + trailing-whitespace + the above) configured but optional
    to enable.
- **CI (GitHub Actions):** matrix {MKL build, OSS-fallback build} × {smoke test, lint},
  plus a Python `ruff` job. Smoke test from Phase 0 runs on every PR.

#### Phase 1 — OUTCOME (completed 2026-08-22)

Phase 1 is **closed** on branch `phase1/build-system`.

- **Build:** CMake + presets `release` / `dev` / `asan-ubsan` / `umfpack`;
  `compile_commands.json` symlinked to the repo root (clangd/editors now resolve
  MKL headers and `MKL_ILP64`). `env.bash` removed; README rewritten for
  dnf/apt (Fedora, Ubuntu/Debian) and macOS (Homebrew).
- **Solver backend interface** (`src/solver.h`) with two implementations:
  - PARDISO (oneMKL) — default, behavior unchanged.
  - UMFPACK (SuiteSparse, 64-bit `int64_t` indices) — MKL-free fallback, with a
    `src/compat/mkl.h` shim (`MKL_INT = long long`, inline level-1 CBLAS,
    thread-control no-ops). Low-rank update falls back to full refactorization.
  - **Validated:** the UMFPACK smoke test reproduces the MKL result bit-for-bit.
- **Tooling:** `.clang-format` (style target for Phase 2), `ruff` config (tools/
  lint-clean), `--binary` flag + wall-time reporting in `smoke.py`.
- **CI:** two build+smoke jobs (oneMKL and UMFPACK) + a ruff job.
- **Notable finding:** MKL's `cblas_idamax` is 0-based (verified empirically);
  the shim matches it.

**CI timing (measured):** the l=1 N=256 from-scratch smoke solve is ~1.5 min on
12 cores (~5 GB peak memory). On GitHub's 2-core Ubuntu runners expect ~3–8 min
for the solve plus ~1–2 min for the oneMKL apt install (~8–15 min/job). The
N=400 golden cases (~970k unknowns) exceed GitHub's 7 GB free-tier memory and
are **not** per-push CI — run them via a manual `workflow_dispatch` (to add in a
later phase) or locally.

**Forward-looking conclusion (deferred to Phase 6/Rust):** oneMKL is increasingly
not worth the dependency cost — the UMFPACK fallback is bit-for-bit equivalent at
the sizes we solve, oneMKL's CMake package was buggy (dropped in favor of
MKLROOT), and the Rust rewrite will likely target `faer` or SuiteSparse. MKL is
kept as the default only because it is the battle-tested, parallel (faster)
reference; revisit before Phase 6.

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

## 4b. Branching & working conventions

- One feature branch per phase, merged into `master` at completion and tagged:
  `phase0/curation` (done) → `phase1/build-system` → `phase2/config-refactor` → ...
- `master` is always green and reproducible; the golden suite must pass before a merge.
- Renaming per-phase branches is unnecessary — they are short-lived and deleted after merge.

## 5. Risks & notes

- ~~Phase 0 verification may show master ≠ StandarizedOutput exactly.~~ RESOLVED:
  master reproduces StandarizedOutput to ~1e-13 (see `VALIDATION.md`).
- GRV2/GRV3 are sensitive derived diagnostics whose historical values predate a
  2020-10-30 analysis fix; compare physical observables (M, J, fields) for regression,
  not GRV (see `VALIDATION.md` §4).
- Perturbation/stability solver: uncommitted, exists only on the drive (deferred copy).
  Drive failure before Phase 6 would lose it.
- The paper reports 4th-order (resolution) and 3rd-order (boundary) convergence;
  the convergence folders are the evidence — preserve their par files.
- Catalogue2 (265 GB) and UniversalOutput (61 GB) stay on the drive; only their scalar
  summaries enter the repo.
