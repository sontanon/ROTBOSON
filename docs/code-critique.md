# Code critique: what the original ROTBOSON did wrong, and what changed

> **Historical document.** This review describes the 2019–2022 codebase as it
> was before the Phase 2 modernization; its "what changed" entries are a
> record of that refactor, not a description of current work. See `PLAN.md`
> for the roadmap and `README.md` for the current architecture.

This is an honest, itemized review of the anti-patterns in the 2019-2022
codebase, written as part of Phase 2 (`PLAN.md`). The original focus was the
physics and the numerics, so none of this is a judgment on the science - the code
is numerically correct and reproduced the published results to ~1e-13. These are
the *structural* habits that make a working code hard to test, audit, or
maintain, and that Phase 2 set out to remove.

Each item is marked **[fixed]** (done in Phase 2), **[partial]** (started), or
**[deferred]** (tracked, deliberately left for a later phase).

---

## 1. Global state: the `#ifdef MAIN_FILE` + `extern` soup  [fixed]

`param.h` (and, separately, `pardiso_param.h`) used the classic trick:

```c
#ifdef MAIN_FILE
double dr = 0.0625;   /* definition, only where MAIN_FILE is set */
#else
extern double dr;     /* declaration everywhere else */
#endif
```

Every `.c` file shared ~70 file-scope globals (`dr`, `dz`, `dim`, `l`, `m`,
`w_idx`, `Dr_u`, `M_KOMAR`, `sweep`, ...). Consequences:

- **Data flow is invisible.** You cannot tell from a signature which state a
  function reads/writes; `rhs(double *f, double *u)` touched ~25 globals.
- **Nothing is testable in isolation.** A unit test would have to set up a dozen
  globals before calling anything.
- **One mutable process, forever.** Global state forbids running two solves in
  one process, embedding the solver as a library, or parallelizing across
  configurations.

**Fix:** a single `rb_context` struct (`src/context.h`) owned by `main()` and
passed explicitly everywhere; callback typedefs (`rb_rhs_fn`, `rb_jacobian_fn`,
`rb_norm_fn`, `rb_dot_fn`, `rb_linear_solve_fn`) make the solver cores
self-documenting.

**Still open [deferred]:** `pardiso_param.h` is the *same* pattern for the
PARDISO backend's internal state (`solver`, `pt[64]`, `iparm[64]`, `perm`,
`diff`). Smaller and genuinely backend-internal, so left for now; it should
become a `solver_backend` struct before Phase 6.

---

## 2. Macros that capture ambient variables by name  [fixed]

Three families of "convenience" macros silently referenced variables assumed to
be in scope:

- `IDX(i, j)` = `(i) * NzTotal + (j)` - needs a variable literally named
  `NzTotal` to be in scope.
- `diff1r(u, ...)` etc. - expanded to `ex_diff1r(..., dr, NrTotal, NzTotal,
  ghost, order)`, i.e. captured five globals.
- `cart_to_pol(...)` / `analysis(...)` - captured `dr`, `dz`, `NrInterior`, `m`,
  `l`, `ghost`, `order`, and more.

This is a landmine: the macro only works in a scope that happens to have a
variable with exactly the right name. It is also why removing the globals broke
compilation in non-obvious ways.

**Fix:** deleted the `diff*`, `cart_to_pol`, and `analysis` macros; call sites
now call `ex_*` explicitly with the context fields. `IDX` remains (hundreds of
uses across the generated CSR code), but the driver routines that use it now
declare an explicit local `const MKL_INT NzTotal = ctx->NzTotal;` so the
dependency is stated where it is used.

---

## 3. No include guards  [fixed]

`tools.h` had no `#ifndef` guard. It compiled only because each translation unit
included it exactly once. The moment `context.h` (which includes `tools.h`) was
added, every file that included both failed with "redefinition of
`struct csr_matrices`". A classic latent bug that is invisible until it is not.

**Fix:** added the guard.

---

## 4. Duplicated definitions: `csr_matrix` and friends defined twice  [fixed]

The `csr_matrix` typedef and the `IDX`/`MIN`/`MAX`/`ABS`/`BASE`/`ARRAY_SUM`/
`SAFE_MALLOC` macros were copy-pasted into *both* `tools.h` and `tools.c`. Two
copies of a typedef or macro is a "which one is authoritative?" trap; a drift
between the copies compiles and then breaks at runtime.

**Fix:** `tools.h` is now the single source of truth (with guards, `tools.c` can
include it instead of redeclaring).

---

## 5. Kitchen-sink header  [partial]

`tools.h` pulled in `mkl.h`, OpenMP, `<libconfig.h>`, and the indexing macros -
so including `tools.h` dragged the whole world into every file, including the
generated Jacobian code that should be pure arithmetic.

**Fix:** the libconfig include is gone (TOML is included only by `parser.c`).
`tools.h` still includes MKL/OpenMP; fully splitting it is Phase 6 territory.

---

## 6. `MKL_INT` permeates the whole codebase  [deferred]

Every integer - grid sizes, loop counters, file dimensions - is `MKL_INT`
(i.e. ILP64 `long long`), because PARDISO wants 64-bit indices. This ties the
entire program to one vendor's type alias; Phase 1 needed a shim to build
without MKL at all.

**Fix:** deferred to Phase 6 (Rust rewrite). `context.h` documents which `MKL_INT`
fields are params vs state, which will make the eventual migration mechanical.

---

## 7. Magic numbers scattered around  [partial]

- `GNUM = 6` defined in `param.h` *and* re-`#define`d in `csr_grid_fill.c`.
- `w_idx = 135200` hard-coded as a "default" (it is derivable:
  `GNUM * NrTotal * NzTotal`).
- `BASE = 1` (PARDISO's 1-based CSR), `RESCALE = 1.0` (never used),
  `MIN/MAX/ABS` macros, and solver hard-codes (`8, 8` trial limits).

**Fix:** `GNUM` now lives in one place (`context.h`); `RESCALE` and the trial
limits are the next candidates for named constants / removal.

---

## 8. Silent, permissive configuration parsing  [fixed]

The libconfig parser:

- **silently ignored unknown keys** - the shipped `.par` files contained dead
  keys (`alphaBoundOrder`, `betaBoundOrder`, ..., `dirname`) the parser never
  read and never complained about;
- treated "key missing" and "key has the wrong type" identically
  (`CONFIG_FALSE` -> warn + keep default), so a typo'd or mistyped key silently
  fell back to a default;
- was ~672 lines of near-identical `config_lookup_*` blocks, one copy per
  parameter.

**Fix:** a schema of known keys; unknown keys are hard errors, wrong-typed values
are hard errors, and a small `lookup_*`/`check_*` helper layer replaces ~30
copy-paste blocks. `tools/par_to_toml.py` converted the legacy files and dropped
the dead keys.

---

## 9. Dead code  [fixed]

- `src/deprecated/old_csr_vars.c` - never built.
- `regularization_coupling.h` - `#undef REGULARIZATION_COUPLING` on line 2 meant
  the feature (and `coupled_du`, `REG_MU`) was always disabled.
- `#ifdef WIN` branches - `arch.h` is literally `#undef WIN`.
- `PRINT_HISTORY` - `#undef`'d before its only use.
- `NEXT_SCALE` / the `#else` branch of `NEXT_SCALE_JUMP` - one branch always taken.
- `io.c` had a ~35-line commented-out `system("cp ...")` block.

Dead code is not neutral: every future maintainer reads it, and it gives a false
sense of available features ("does this run on Windows?" - the `#ifdef WIN`
suggested maybe; it never did).

**Fix:** all of the above deleted.

---

## 10. A monolithic `main()`  [fixed]

`main.c` was ~800 lines doing banner printing, memory management, grid filling,
the whole sweep loop, Newton dispatch, file I/O, and analysis. No structure
visible at a glance; every concern entangled with every other.

**Fix:** `main()` is now a thin driver; the stages are named functions:
`print_banner`, `print_parameters`, `configure_openmp`, `run_newton`,
`run_analysis`, `sweep_advance`.

---

## 11. Inconsistent error handling  [partial]

- `exit(-1)`, `exit(1)`, and `exit(EXIT_FAILURE)` were all used for the same
  "bad input" case.
- `errCode = 1` was initialized to a value that reads as "already failed".
- Solver routines mix two conventions: positive return = converged index,
  negative return = failed index, plus a separate `err_code` out-param.

**Fix:** `parser.c` now has a single `die()` that always exits with
`EXIT_FAILURE`. The solver return convention is preserved (changing it would
touch the golden-verified control flow); document-and-migrate is Phase 3 work.

---

## 12. No const-correctness  [partial]

Functions that only read their inputs (`rhs`'s `u`, every analysis routine)
took non-`const` pointers, so a caller cannot tell what is mutated.

**Fix:** started in the refactor (norm/dot algebra takes `const rb_context *`,
`nnz_jacobian` takes `const rb_context *`). Full const propagation is a
mechanical follow-up.

---

## 13. Generated code with unused parameters  [fixed]

`csr_vars.c`, `csr_exp_decay.c`, `rhs_vars.c` etc. are machine-generated Jacobian
code and carry many unused parameters and unused variables (the compiler warns
about them today). They should be regenerated cleanly in Phase 4 (SymPy) rather
than hand-edited.

**Fix (Phase 4):** `csr_vars.c`/`rhs_vars.c` are now regenerated from the
SymPy system (`tools/generate_kernels.py`).  Unused grid-value parameters are
declared anonymous (no name), the redundant `NrTotal`/`u_aux`/`M`/`J`
parameters are `(void)`-cast, and the dead commented-out blocks (Kerr
matching, `ANALYTIC`, the `dRu4`/`dZu4` derivatives of log a, which the
residual never uses) are gone.  The generated kernels build `-Wall -Wextra`
clean.

---

## 14. I/O baked into the numerical code  [fixed]

The `.asc` ASCII format (`%9.18E`, tab-separated, one file per field) was
written directly from everywhere via `write_single_file_2d` with no
reader/writer abstraction, no metadata and no versioning, and the process
`chdir`'d into the output directory.

**Fix (Phase 5):** a path-aware `solution_writer` (`src/output.{h,c}`) replaces
all of that. The ASCII backend emits the legacy `.asc` layout byte-identically
(verified against the golden gate); the HDF5 backend emits one self-describing
`solution.h5` (datasets named `<field>.asc`, parameters/solver settings/git
hash as attributes). No global cwd juggling remains.

---

## 15. Naming and flow smells  [partial]

- `double w = m;` in `main()` initializes `w` from the mass, then immediately
  overwrites it - the initializer is dead.
- `k`, `i`, `j`, `counter_i` are reused for unrelated purposes across the file.
- The `***` banner spam mixes logging with the actual output; there is no log
  level or structured output.

**Fix:** the `w = m` dead initializer is now `double w = 0.0;` (Phase 3).
Phase 5 adds a level-gated logger (`src/log.{h,c}`, config key `loglevel`) and
routes the banner/status/warning blocks in `main.c` through it (INFO/WARN to
stdout/stderr, suppressed by `loglevel`). The analysis result tables remain as
plain `printf` — they are the report, not banner noise.

---

## 16. Derivative operators: boundary order and an unsupported symmetry  [documented]

Findings from the Phase 3 convergence tests (`tests/test_derivatives.c`), which
measure the empirical order of every operator on parity-consistent functions:

- **Interior stencils converge at exactly their design order** (2/4/6).
- **Boundary/axis points are ~3rd order** for the 4th-order operators (observed
  ~3.3–5.4 depending on derivative order and parity). This is consistent with
  the paper's "3rd-order boundary" claim, so it is the expected cost of the
  hand-derived reflected/one-sided stencils — not a bug, but worth knowing: the
  *local* boundary truncation error is the convergence bottleneck.
- **The 6th-order radial operator (`ex_diff1r`, order 6) only works for EVEN
  functions.** Its axis/equator stencils hard-code the even reflection (no `sym`
  term, unlike the order-2/4 operators), so an odd input gives a constant ~O(1)
  error (measured 0.002 observed order). This is latent rather than active:
  production only calls it with `order+2` on the *even* metric functions in the
  regularization auxiliaries. Phase 4's SymPy regeneration should fix it by
  generating the reflected stencils for both parities (or dropping order 6).

---

## Summary

The original code was a correct physics oracle that was, as software, a single
mutable global state machine with invisible data flow, silent configuration,
and a lot of dead weight. Phase 2 made the state explicit (`rb_context`), the
configuration strict (TOML + unknown-key rejection), and the structure legible
(named driver units), without changing a single floating-point result - verified
bit-for-bit on both solver backends and against the published golden data.

The remaining items (PARDISO's own globals, `MKL_INT` everywhere, the
`tools.h` kitchen sink, HDF5) are queued for Phases 5-6.  (Generated-code
hygiene, #13, was retired in Phase 4 by the SymPy regeneration.)
