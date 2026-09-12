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

#### Phase 2 — OUTCOME (completed 2026-08-22)

Phase 2 is **closed** on branch `phase2/config-refactor`. Full write-up of the
anti-patterns found and how each was fixed is in `docs/code-critique.md`.

- **Config:** libconfig removed; a vendored `tomlc99` (`third_party/tomlc99/`,
  MIT, commit `29076df`) parses flat TOML parameter files. `parser.c` shrunk
  ~672 → ~470 lines behind a known-key schema: unknown keys are **rejected**
  (libconfig silently ignored them — e.g. the dead `*BoundOrder`/`dirname` keys),
  wrong-typed values are hard errors, and range checks are unchanged. All
  tracked `.par` files ported to `.toml` via `tools/par_to_toml.py`.
- **Global state:** `param.h`'s `#ifdef MAIN_FILE` + `extern` globals are gone;
  an `rb_context` struct (`src/context.h`) is passed explicitly to `parser`,
  `rhs`, `csr_gen_jacobian`, `initial_guess`, `solver_diff_gen`, the norm/dot
  algebra, and the Newton/qn solver cores (callback typedefs `rb_rhs_fn` etc.).
  The global-capturing convenience macros (`diff1r/...`, `cart_to_pol`,
  `analysis`) were deleted in favour of explicit `ex_*` calls. `tools.h` gained
  the include guards it had always been missing.
- **Structure:** `main.c` (797 → ~760 lines, but now a thin driver) split into
  `print_banner` / `print_parameters` / `configure_openmp` / `run_newton` /
  `run_analysis` / `sweep_advance`.
- **Dead code:** `src/deprecated/`, `regularization_coupling.h`, the
  always-disabled `REGULARIZATION_COUPLING` blocks and `coupled_du`, `#ifdef WIN`
  branches, and the `#undef`'d `PRINT_HISTORY` / unused `NEXT_SCALE_JUMP` are gone.
- **Validation:** bit-for-bit unchanged on **both** backends. l=1 w=0.95 smoke
  solve reproduces `w=9.49999...E-01` and every Komar/phi observable to the last
  digit (PARDISO and UMFPACK). The l=1 N=400 golden regeneration matches
  `data/golden/` to ~1e-13 (same as Phase 0), all fields + observables PASS.

#### Deferred cleanup backlog (from the Phase 2 critique)

The items below were catalogued in `docs/code-critique.md` but deliberately not
fixed in Phase 2 (out of scope / risky to do without tests). They are queued
against the phase that will actually address them, so nothing is lost:

| # | Smell (critique §) | Target phase | Note |
|---|---|---|---|
| 1 | `pardiso_param.h` global state (`solver`/`pt[64]`/`iparm[64]`/`perm`/`diff`) | 6 | wrap in a `solver_backend` struct before the Rust port |
| 2 | `MKL_INT` pervades non-MKL code | 6 | de-MKL-ification of types |
| 3 | `tools.h` kitchen-sink header (MKL/OpenMP) | 6 (start in 5) | writers moved out in 5; CSR/mem/read helpers remain |
| 4 | magic numbers: `RESCALE`, `MIN/MAX/ABS`, `BASE=1`, `8,8` trial limits | 3 | **done**: `RESCALE` removed, trial limits → `MAX_TRIAL_{A,B}_ITERATIONS` |
| 5 | missing `const`-correctness on read-only pointers | 3 | **done** for the derivative operators |
| 6 | solver error/return-convention confusion (`err_code` out-param + ±k) | 3 | documented; convention left stable (golden-verified control flow) |
| 7 | generated code hygiene (unused params/vars in `csr_vars.c` etc.) | 4 | fixed by SymPy regeneration, not hand-editing |
| 8 | ASCII `.asc` I/O baked in everywhere; no reader/writer abstraction | 5 | **done**: `solution_writer` abstraction + ASCII/HDF5 backends |
| 9 | logging vs banner noise; `***` spam; no log level | 5 | **done**: level-gated `log.{h,c}` + `loglevel` key |
| 10 | dead initializers / naming (`double w = m;`, reused `i,j,k,counter_i`) | 3 | **done** for `w = m`; counter reuse left as harmless churn |
| 11 | `io.c` commented-out `system("cp …")` block | 5 | **done**: `io.c`/`io.h` deleted; writer is path-aware |

### Phase 3 — Testing strategy (CTest + plain-C asserts)

Decision: **CTest** (CMake's built-in runner, already installed) with plain-C
assertion helpers, not Criterion — Criterion would add a system dependency +
sudo for little benefit over the existing Python tooling (see `tests/`). The
core library is split into `rotboson_core` so tests link it directly.

Layered suite (only what has an independent oracle today):
1. **Unit:** FD stencil weights vs Fornberg — interior, one-sided edges, and
   symmetry reflection (done, `tests/test_derivatives.c`).
2. **Unit:** FD convergence order (interior + axis/equator + boundary) using
   parity-consistent functions.
3. **Unit:** remaining operators — polar `diff1th`/`diff1rr`, second-to-last and
   6th-order edges.
4. **Sanity:** trivial-vacuum residual — `rhs(Minkowski, psi=0) ≈ 0` to machine
   precision (no symbolic derivation needed).
5. **Integration:** full Newton solve on small grids vs the golden set.
6. **Regression:** end-to-end runs vs Phase 0 goldens (ω, masses, profiles); the
   smoke test runs in CI, the full golden regeneration is the §4c gate.

**Manufactured-solution (MMS) tests are deferred to Phase 4.** They require an
independent source term `S = L[u_man]`, which only exists once SymPy re-derives
`L`. Doing MMS now would be circular (validate the hand-written residual against
a source computed from that same residual) or a throwaway duplicate of Phase 4's
symbolic work. Phase 0 already validates the whole pipeline to ~1e-13 via the
golden data (strong but indirect); MMS is the direct residual-order check that
catches off-axis / mutually-cancelling bugs.

Phase 3 also retires backlog items #4–#6 and #10 above as the tests give a safe
net for the mechanical cleanups.

#### Phase 3 — OUTCOME (completed 2026-08-22)

Phase 3 is **closed** on branch `phase3/testing`.

- **Build:** `rotboson_core` static library (everything but `main.c`) + the
  executable driver, so tests link the core directly. `enable_testing()` +
  `tests/`; CI runs `ctest` in both build jobs before the smoke test.
- **Framework:** CTest (already present, no new dependency) with a ~50-line
  dependency-free harness (`tests/test.h`); Criterion rejected (would add a
  system dep + sudo for little gain over the Python tooling).
- **Unit — derivative operators** (`tests/test_derivatives.c`, 257 checks):
  interior stencil weights vs an independent Fornberg generator
  (`tests/fornberg.c`); one-sided edge weights (incl. second-to-last and
  6th-order) vs Fornberg on one-sided node sets; the mixed derivative vs the
  1D-weight outer product; the polar operators; ghost-zone symmetry reflection;
  and convergence order on parity-consistent functions.
- **Sanity** (`tests/test_residual.c`): the flat-Minkowski vacuum residual is
  exactly 0 to machine precision, exercising the full `rhs` assembly.
- **Findings (documented in `docs/code-critique.md` §16):**
  - interior stencils converge at exactly their design order (2/4/6);
  - boundary/axis points are ~3rd order (matches the paper's "3rd-order
    boundary" claim);
  - the 6th-order radial operator's axis stencils hard-code the even
    reflection, so ODD is unsupported (latent; production uses EVEN only).
- **Backlog retired:** #4 (magic numbers — `RESCALE` removed, `8,8` trial
  limits named), #5 (`const`-correctness on the derivative operators),
  #10 (`w = m` dead initializer). #6 (solver return convention) documented and
  left stable — golden-verified control flow, low value to change.
- **Integration/regression:** the l=1 w=0.95 smoke solve (CI) + the §4c golden
  regeneration gate serve as the end-to-end anchor; bit-for-bit unchanged on
  both PARDISO and UMFPACK after every Phase 3 step.

### Phase 4 — SymPy code generation

- Re-derive the Einstein–Klein–Gordon system (axisymmetric, l≥1, regularization variable)
  in SymPy from the paper + `.nb` extraction.
- Cross-check SymPy expressions against the Jacobian strings in
  `Mathematica CSR Code Generation.ipynb` and the current `rhs_vars.c`/`csr_vars.c`.
- Produce one standardized, `uv`-managed notebook/script that regenerates the residual
  and Jacobian C code (plus stencil weights). Generated files are checked in; a test
  verifies "regenerated == checked-in".

**MMS (manufactured-solution) tests land here:** with the independent SymPy `L`,
manufacture a solution, emit the source term `S = L[u_man]`, and assert the
checked-in residual/Jacobian recovers it at the correct convergence order
(4th-order interior / 3rd-order boundary). This is the direct, local validation
the Phase-3 component layer wanted, deferred here because it needs `L` to exist
first.

Phase 4 also retires backlog item #7 (generated-code hygiene), since the code is
regenerated rather than hand-cleaned.

#### Phase 4 — INPUTS & starting point (handover)

Everything needed is committed; start a `phase4/sympy` branch from `master` and
build/test as in §4c. The symbolic inputs and the code to regenerate are:

- **Derivations** (plain-text Wolfram `.nb`, no license needed):
  `derivations/mathematica/Rotating Boson Stars {Regularization,Vanilla Scalar
  Field,Decaying Scalar Field,...} Equations.nb` + `Full Jacobian Approach.nb`,
  `Jacobian Approach.nb`, `Kerr in Lewis-Papapetrou.nb`, `Asymptotic Kerr.nb`.
- **The existing codegen notebook** (python3 kernel; the Jacobian entries were
  generated in Mathematica, pasted as strings, then emitted to `csr_vars.c`):
  `derivations/notebooks/Mathematica CSR Code Generation.ipynb`.
- **The generated code to regenerate + cross-check against:**
  - residual kernel: `src/rhs_vars.c` + `src/rhs_vars.h`
  - Jacobian kernel: `src/csr_vars.c` + `src/csr_vars.h` (~4000 lines)
  - stencil assembly + boundary dispatch: `src/csr_grid_fill.c`,
    `src/csr_exp_decay.c`, `src/csr_robin.c`, `src/csr_symmetry.c`,
    `src/csr_omega_constraint.c`
  - FD stencil weights: `src/derivatives.c` (already Fornberg-validated in
    Phase 3; reuse `tests/fornberg.c` as the independent weight reference)
- **Tooling:** `uv`-managed `pyproject.toml` already has `sympy>=1.14` and
  `nbformat`; strip notebook outputs with `tools/strip_notebooks.py` before
  committing (tracked `.ipynb` are output-stripped).

Two things from earlier phases that shape Phase 4: (1) the MMS tests are
deferred here because the SymPy `L` is their independent oracle (§Phase 3);
(2) the order-6 radial operator's axis stencils hard-code the even reflection
(`docs/code-critique.md` §16) — the regeneration should emit parity-correct
stencils (or drop order 6).

Note: `data/golden/` and `data/seeds/` are gitignored (restore from the backup
drive, §4c) — not needed for the SymPy cross-check itself, but required for the
full regression gate and any MMS-against-binary checks.

#### Phase 4 — OUTCOME (completed 2026-08-22)

Phase 4 is **closed** on branch `phase4/sympy`.

- **SymPy re-derivation** (`tools/sympy_system.py`): the six EKG residuals
  `R_0..R_5` (lapse, shift, log h, log a, rescaled scalar field, and the
  regularization variable λ) are written symbolically from the equations in
  `rhs_vars.c`, including the `a2 → h2 + r²λ` regularization re-definition and
  the `Dr(α)/r`, `Dr(H)/r` auxiliary terms.  The Jacobian is the exact
  `d f_i / d(subtype)` (6×6 grid functions × {value, ∂r, ∂z, ∂rr, ∂zz} + ω),
  with the ω chain rule `dw_du(xi, m)` applied at run time.
- **Cross-check** (`tools/sympy_check.py`): the SymPy Jacobian is compared
  term-for-term against the Mathematica-pasted strings in
  `derivations/notebooks/Mathematica CSR Code Generation.ipynb` — **all 6×31
  entries agree** at hundreds of random points.  The residual is additionally
  pinned by the golden gate below.
- **Code generation** (`tools/generate_kernels.py`): regenerates
  `src/rhs_vars.c`, `src/rhs_vars.h`, `src/csr_vars.c`, `src/csr_vars.h`
  from the SymPy system.  The old hand-pasted Mathematica strings and the
  hand-written 2nd-order Jacobian are gone; all five stencil kernels
  (`cc`/`cs`/`sc`/`ss` 4th order + 2nd order `cc`) are emitted from the same
  SymPy Jacobian and the Fornberg stencil tables.  Backlog item #7 is retired
  (no unused parameters/variables; `-Wall -Wextra` clean for the kernels).
- **MMS test** (`tools/mms_test.py`): a manufactured even solution is fed
  through the residual with 4th-order Fornberg stencils and compared to the
  analytic source `S = L[u_man]`; the interior residual converges at the
  design order (observed ~3.3 → ~3.8 on 64→256 grids).
- **Reproducibility**: `tools/generate_kernels.py --check` asserts the
  regenerated kernels are byte-identical to the checked-in files (CI).
- **Formatting:** the checked-in `.clang-format` (LLVM base, 4-space, no tabs,
  `AlignTrailingComments: false`) was finally applied repo-wide (the Phase 1
  plan had deferred it to "Phase 2" but it never ran).  `generate_kernels.py`
  pipes its output through `clang-format` so the idempotency check survives the
  reformat.  Purely cosmetic (verified against the smoke/golden results).
- **Validation:** the regenerated `csr_vars.c` + `rhs_vars.c` rebuild cleanly
  and the §4c gate passes — l=1 w=0.9 N=400 golden regeneration matches
  `data/golden/` to ~1e-13 (all fields + observables), and the l=1 w=0.95
  from-scratch smoke reproduces the expected ω.

Not done (deferred, tracked for Phase 5/6): the order-6 radial operator's
parity-correct axis stencils (`docs/code-critique.md` §16) — the Jacobian
regeneration uses the same parity behaviour as the checked-in operators, so
the odd-input limitation is unchanged and remains latent (production uses
EVEN only).

### Phase 5 — Output format: HDF5

- HDF5 via CMake: one self-describing file per solution (fields, grids, attributes:
  params, git hash, solver settings). Keep a legacy `.asc` exporter during transition.
- Python readers (`h5py`) in the `uv` project; golden `.asc` data provides migration fixtures.

Phase 5 also retires backlog items #8, #9 and #11 (I/O abstraction, logging, and
the stale commented-out copy block in `io.c`).

#### Phase 5 — INPUTS & starting point (handover)

Everything is on `master` (tag `phase4-complete`); start a `phase5/hdf5` branch
and build/test as in §4c. The full golden suite (l=1..6) was re-run and PASSES
on master. The I/O you will refactor:

- **Writers live in `src/tools.c` (the kitchen-sink, see backlog #3), not in a
  dedicated module.** `write_single_file_1d` / `write_single_integer_file_1d` /
  `write_single_file_2d` / `write_single_file_2d_polar` (declared in
  `tools.h`). All emit the same ASCII format: `%9.18E`, tab-separated, one file
  per field. This is the §14 "I/O baked into the numerics" problem.
- **Callers of the writers:** `analysis.c` (all `M_*`, `J_*`, `GRV*`, `r99`,
  `rr_phi_max`, `phi_max`, `ergoregion_flag` and the `sph_*` interpolations),
  `initial.c` (`*_0.asc` initial data), `initial_interpolation.c` (`r_0.asc`,
  `z_0.asc`, `sph_*_0.asc`), `main.c` (`run_analysis` orchestrates), and
  `rhs.c` (only under `DERIVATIVE_DEBUG`, which is `#undef`'d — dead path).
- **`src/io.c`** is a single `io(dirname, parfile)` that `mkdir`s, prompts if
  the dir exists, and then `chdir(dirname)` — so *every* subsequent write is
  relative to the process cwd. That is the design to replace with an explicit
  `solution_writer`/path-aware API (no global cwd). The ~35-line commented-out
  `system("cp …")` block is backlog #11.
- **`.asc` readers already exist in Python:** `tools/rotboson_io.py`
  (`extract_scalars`, `find_solution_dirs`, `compare_fields`, `compare_scalars`)
  back `compare_solutions.py` / `smoke.py` / `check_against_summary.py`. They
  are the reference for what HDF5 must reproduce.
- **Tooling:** `h5py>=3.16.0` is already a `uv` dev dependency. CMake HDF5 is
  the only new build dependency (`find_package(HDF5)`); keep the UMFPACK and
  ASan presets working.
- **Backlog retired here:** #8 (I/O abstraction), #9 (logging vs `***` banner
  spam — introduce a log level / structured output alongside the new writer),
  #11 (dead `cp` block). Also carry over, if convenient, the order-6 radial
  operator parity fix (`docs/code-critique.md` §16), which Phase 4 explicitly
  deferred to "Phase 5/6".
- **Validation:** the golden `.asc` files in `data/golden/` are the migration
  fixtures. The transition should be two separable steps: (1) introduce the
  writer abstraction behind the *existing* `.asc` format and prove bit-parity
  with `compare_solutions.py` (the §4c gate), then (2) add HDF5 as a second
  backend and round-trip `h5py` → `.asc` → golden. Do not land HDF5 until the
  `.asc` refactor is golden-clean.

Note: `data/golden/` and `data/seeds/` are gitignored (restore from the backup
drive, §4c) — required for the full regression gate, not for the I/O refactor
itself.

#### Phase 5 — OUTCOME (completed 2026-08-24)

Phase 5 is **closed** on branch `phase5/hdf5`.

- **Writer abstraction (`src/output.{h,c}`, `src/output_internal.h`):** a
  path-aware `solution_writer` replaces the `write_single_file_*` writers that
  lived in `tools.c` and the global-`chdir` `io()`. Two backends:
  - **ASCII** (default): the legacy `%9.18E` tab-separated, one-file-per-field
    layout, byte-identical to the pre-Phase-5 output. `src/io.{c,h}` (including
    the dead `system("cp …")` block, backlog #11) is deleted.
  - **HDF5**: a single `solution.h5` per solution. Datasets are named
    `<field>.asc` (1:1 with the ASCII files) and parameters, solver settings,
    the build git hash (`ROTBOSON_GIT_HASH` from CMake) and the computed
    analysis results are stored as root-group attributes. The backend is
    optional (`ROTBOSON_HDF5` CMake option, `find_package(HDF5)`); without it
    `outputFormat = "hdf5"` fails with a clear runtime message.
- **Config:** new keys `outputFormat` (`"ascii"` | `"hdf5"`) and `loglevel`
  (`"error"`/`"warn"`/`"info"`/`"debug"`), both strictly validated by the
  parser. `outputFormat` defaults to ascii so the golden gate is untouched.
- **Logging (`src/log.{h,c}`):** level-gated `rb_log()`, default INFO to
  preserve pre-Phase-5 stdout. The `***` banner/status/warning blocks in
  `main.c` now go through it (WARN→stderr, INFO→stdout); the analysis report
  tables remain plain `printf` (they are results, not banner noise). Backlog
  #9 retired.
- **Callers refactored:** `main.c`, `analysis.c` (`ex_analysis`/
  `ex_phi_analysis` now take a writer instead of a `print` flag),
  `initial_interpolation.c`; the dead `DERIVATIVE_DEBUG`/`I_DEBUG`/`DEBUG`
  write blocks in `rhs.c`, `initial.c`, `initial_interpolation.c` and
  `low_rank.c` were removed. Backlog #8 retired.
- **Python tooling:** `rotboson_io.py` gained `read_hdf5`, `hdf5_to_asc`,
  `extract_scalars_from_hdf5`, `compare_hdf5_to_ascii`; new
  `tools/hdf5_roundtrip.py` exports `solution.h5` back to `.asc` and/or
  compares it against a legacy `.asc` directory.
- **Test:** `tests/test_output.c` (CTest) asserts the ASCII writer's byte
  output for 1D/2D/polar/int fields and its path-awareness (no chdir).
- **Validation:** (1) the `.asc` refactor passes the §4c golden gate — the
  l=1 N=400 regeneration matches `data/golden/` to ~1e-13 (all fields +
  observables), identical to Phase 0/4; (2) the HDF5 smoke run round-trips
  `solution.h5` → `.asc` → reference `.asc` with PASS on every field/scalar
  (`tools/hdf5_roundtrip.py`), and the exporter is lossless (bit-identical
  after `.asc` re-parse). All three presets (release, umfpack, asan-ubsan)
  build and pass CTest.
- **Deferred (unchanged):** the order-6 radial operator parity fix
  (`docs/code-critique.md` §16) — left for Phase 6 as "if convenient" was
  exercised and deprioritized in favour of the core I/O work.
- **Fresh-clone portability (follow-up to the outcome):** removing the
  process `chdir` made relative seed paths work, so the §4c regeneration
  configs (`data/params/regeneration/*.toml`) now reference
  `../data/seeds/...` / `../data/golden/...` instead of machine-specific
  absolute paths (verified: the l=1 golden gate + summary cross-check still
  PASS). The golden provenance README and SHA-256 manifest are now tracked
  (force-added past the `data/golden/` ignore) so a fresh clone carries the
  restore instructions and checksums even though the 4.7 GB of gitignored
  data does not travel with it. CI installs `libhdf5-dev` in both build jobs
  and the oneMKL job now runs CTest.

### Phase 6 — Rust migration (last, opportunistic)

- Prerequisites: validated golden suite, clean C reference, HDF5 I/O, stable config.
- Port incrementally: config → grid/derivatives → residual → Newton → solver backend,
  cross-validating against the C binary at each step.
- Candidate stack: `faer` or FFI to MKL/UMFPACK, `serde` + `toml`, `hdf5` crate, `ndarray`.
- Before starting, revisit the deferred perturbation-solver patch on the drive
  (archive it; it may inform the stability/evolution features).

Phase 6 also retires backlog items #1–#3 (the `solver_backend` struct, `MKL_INT`
de-MKL-ification, and the `tools.h` split).

## 4. Dependencies between phases

- Phase 0 must precede everything (freeze + fidelity chain).
- Phase 1 unblocks CI; Phase 2 and 3 interleave (refactor ↔ test).
- Phase 4 and 5 are independent after Phase 2.
- Phase 6 consumes all of the above.

## 4b. Branching & working conventions

- One feature branch per phase, merged into `master` at completion and tagged:
  `phase0/curation` → `phase1/build-system` → `phase2/config-refactor` →
  `phase3/testing` → `phase4/sympy` → `phase5/hdf5` (all but the last done).
- `master` is always green and reproducible; the golden suite must pass before a merge.
- Renaming per-phase branches is unnecessary — they are short-lived and deleted after merge.

## 4c. Regression gate (reference commands)

The golden suite is the acceptance gate for every refactor step (Phase 2+). It
**depends on untracked local data**: `data/golden/` (4.6 GB) and `data/seeds/`
(57 MB), which are gitignored and absent on a fresh clone. If missing, restore
them from the backup drive (see `data/golden/README.md` and `MANIFEST.sha256`).

```bash
# Build + smoke (l=1, w=0.95, N=256 from scratch)
source /opt/intel/oneapi/setvars.sh        # sets MKLROOT (pardiso backend only)
cmake --preset release && cmake --build --preset release -j
uv sync --dev
uv run tools/smoke.py out/l1_from_scratch.toml --skip-build

# Regenerate one golden solution and compare (config must run from out/).
# Regeneration files for l=1..6 live in data/params/regeneration/.
cd out && ../build/release/ROTBOSON ../data/params/regeneration/l=1,validate.toml
cd .. && uv run tools/compare_solutions.py \
    "data/golden/l=1,w=9.00000E-01,dr=8.00000E-02,N=0400" \
    "out/l=1,w=9.00000E-01,dr=8.00000E-02,N=0400"

# Cross-check M_Komar/J_Komar against the published tables
uv run tools/check_against_summary.py out/l=1,w=9.00000E-01,dr=8.00000E-02,N=0400
```

Notes: l=2's template is one step of a fixedPhi ladder (produces w=8.74062E-01,
validated against the drive's Catalogue2, not `data/golden/`); see `VALIDATION.md`.
The UMFPACK backend is interchangeable: `cmake --preset umfpack` + the same smoke test.

## 5. Risks & notes

- ~~Phase 0 verification may show master ≠ StandarizedOutput exactly.~~ RESOLVED:
  master reproduces StandarizedOutput to ~1e-13 (see `VALIDATION.md`).
- **2026-09 catalogue rebuild:** l=1..4 re-verified end-to-end with the
  modernized pipeline (driver + HDF5) against the paper's Table IX.1; session
  report: `docs/milestone-report-2026-09.md`, figures: `docs/figures/`.
- GRV2/GRV3 are sensitive derived diagnostics whose historical values predate a
  2020-10-30 analysis fix; compare physical observables (M, J, fields) for regression,
  not GRV (see `VALIDATION.md` §4).
- Perturbation/stability solver: uncommitted, exists only on the drive (deferred copy).
  Drive failure before Phase 6 would lose it.
- The paper reports 4th-order (resolution) and 3rd-order (boundary) convergence;
  the convergence folders are the evidence — preserve their par files.
- Catalogue2 (265 GB) and UniversalOutput (61 GB) stay on the drive; only their scalar
  summaries enter the repo.
