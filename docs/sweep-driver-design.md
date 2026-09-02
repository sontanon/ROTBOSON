# Design doc: Python sweep driver with adaptive stepping

**Linear:** [SAN-7](https://linear.app/sontanon/issue/SAN-7/design-doc-python-sweep-driver-with-adaptive-stepping) · status: **IMPLEMENTED** (all design slices shipped; see §8 rollout table — C contract strip, core driver, adaptive layer, and the Revision 2 fixed-step workhorse all live in `tools/sweep_driver.py`; remaining work is golden-sequence verification, tracked in Linear as SAN-13).
**Successors:** SAN-10 (strip C), SAN-17 (core driver), SAN-14 (adaptive logic), SAN-13 (verification)
**Author:** drafted 2026-08-29 from SAN-7 requirements + paper §VI/§IX + audit of `sweep_advance` (`src/main.c:274`) and the fixedPhi ladder machinery.

---

## 1. Problem

The C binary currently does everything: single solve, ω-sweep, fixedPhi ladder. This makes
the loop awkward to control and impossible to steer mid-run — the three failure modes that
actually bit during the original catalogue runs are:

1. **Boundary drift** — as amplitude grows the field's support approaches the outer boundary;
   the sweep then dies opaquely (`rr_phi_max > rr_phi_max_maximum`).
2. **Under-resolution** — the field develops a sharp maximum near the rotation axis
   (paper §IX: "the ρ coordinate location of the field's maximum becomes smaller and
   smaller"); the sweep dies when `hwl_res < hwl_min`.
3. **The minimum-frequency turning point** — stepping in ω, ω stops responding as the
   branch turns; the old sweep either stalls or overshoots past the turning point.

None of these can be fixed from inside C without either a big interactive loop or
hard-coded heuristics. The plan (approved in SAN-7 discussion): **C solves one solution;
Python drives.**

### Key design decision: step in ψ₀, not ω

The paper (§VI, Eq. VI.1) already gives us the answer for failure mode 3. The eigenvalue
problem is closed by constraining the field at the origin, ψ₀ ≡ ψ(0,0); *for each ψ₀ (and
each rotation number l) there exists a unique ω for the ground-state — the converse is not
true*. ω(ψ₀) is single-valued through the turning point, so:

- **ψ₀ is the continuation parameter.** Each driver step bumps the fixedPhi constraint by
  Δψ₀ and lets Newton solve for ω as an unknown eigenvalue (exactly what the C code already
  does when `fixedPhi = 1`: ω is part of the unknown vector, ψ(0,0) = ψ₀ is the closing
  constraint). The branch turns naturally at ω_min; no special logic is needed to *cross*
  it — only to *detect* it (§6) so we can stop or record it.
- **ω-parameterized stepping (`fixedOmega = 1`) is kept as a fallback mode** for targeted
  runs at a specific frequency (e.g. reproducing a golden point), but it is never the
  campaign mode.

This matches how the catalogue was produced (paper §IX/Summary: "the entire branch is
computed by varying the scalar field's value at the origin ψ₀"), so golden-sequence
verification stays meaningful.

---

## 2. C binary contract (single-solution mode)

One invocation = one Newton solve. Everything else is Python's job.

### CLI

```
ROTBOSON <params.toml>        # unchanged: file name in argv[1]
```

### Exit codes (new, strict)

| code | meaning                                                              |
|------|----------------------------------------------------------------------|
| 0    | converged (`error_code = 0`), solution written                       |
| 1    | Newton did not converge within `maxNewtonIter`                       |
| 2    | solver error (PARDISO/UMFPACK failure)                               |
| 3    | configuration/parse error (bad or missing TOML keys, bad seed paths) |
| 4    | I/O error (cannot open output dir, HDF5 failure)                     |

Today most of these funnel into `EXIT_FAILURE` or a hang-then-die; the driver needs them
distinguishable to apply the decision table (§5). The existing `error_code.asc` convention
stays for in-solution diagnostics.

### Output

- HDF5 primary (`outputFormat = "hdf5"`), ASCII kept for golden-compat tooling.
  The driver reads HDF5 via `tools/rotboson_io.py` (`read_hdf5`, `extract_scalars_from_hdf5`).
- Provenance written into the solution: git hash, params echo, backend, timestamp.
  (Small extension of `solution_writer` metadata; Phase 5 already stores the param file.)

### What C keeps

parser, grid/CSR assembly, NLEQ-ERR Newton (`nleq_err.c`), ω-as-unknown machinery
(`csr_omega_constraint.c`, `omega_calc.c`), the fixedPhi closing constraint, initial-data
read **and interpolation** (`initial_interpolation.c`, `readInitialData = 2`), analysis
(`rr_phi_max`, `hwl_resolution`, Komar masses, `r99`, GRV, ergoregion flag), `solution_writer`.

### What C drops (SAN-10)

- `sweep_advance()` and the `sweep > 0` driver loop in `main.c`.
- Ladder/scale machinery knobs that only existed for the in-C loop: `scale_next`,
  `w_step` advancement, `w_min`/`w_max` sweep bounds.
- **Kept but reinterpreted:** `scale_u4` (initial-data scaling) stays — it is a legitimate
  seed-shaping knob; the *decision* of when/how much to scale moves to Python.
- Sweep-control thresholds (`rr_phi_max_minimum/maximum`, `hwl_min/max`) leave the C
  contract; they become driver decision-table inputs (§5). The binary solves or fails; it
  no longer judges whether the sweep should continue.

---

## 3. Driver architecture

`tools/sweep_driver.py`, run with `uv run`. No new dependencies beyond what `uv.lock`
already has (numpy, h5py).

### 3.1 Campaign spec (TOML)

```toml
[campaign]
l                = 1
target           = "psi0"          # what we step in: "psi0" (primary) | "omega" (fallback)
psi0_start       = 1.0e-2          # ψ₀ of the seed solution
psi0_target      = 2.0             # "up": stop when ψ₀ ≥ this; "down": stop when ψ₀ ≤ this
omega_target     = 0.90            # optional, either direction: stop when ω crosses this
                                   # ("up": ω ≤ omega_target near the minimum; "down": ω ≥ it)
direction        = "up"            # "up" (amplitude grows, ω → ω_min) | "down" (amplitude → 0, ω → m)
max_steps        = 200

[seed]
policy           = "solution"      # "solution" | "from_scratch" | "golden"
source           = "out/<seed solution dir>"

[grid]
dr               = 8.0e-2          # initial grid; N fixed by grid.N
dr_max           = 3.2e-1          # regrid cap: domain growth budget (dr ×2 never beyond this)
N                = 400
order            = 4

[solver]                           # passed through to the C binary
solverType       = 1
epsilon          = 2.0e-13
maxNewtonIter    = 50

[adaptivity]                       # §5 thresholds; sane defaults so this block is optional
hwl_min          = 8
hwl_max          = 40
support_fraction = 0.85            # r99 / r_bdy above this → widen domain
psi0_step        = 3.0e-2          # per-step ratio ψ₀·(1±psi0_step) (default mode)
psi0_step_mode   = "relative"      # golden-ladder-like, scale-free across decades;
                                   # "absolute" = fixed Δψ₀
max_retries      = 3               # shrink-and-retry attempts on Newton failure

[output]
root             = "out/campaigns/<name>/"   # solution dirs + state.json
```

Unknown keys rejected (same philosophy as the C parser's vendored tomlc99).

### 3.2 Resumable state

One `state.json` per campaign, written **atomically after every step** (tmp + rename):

```json
{
  "campaign": "l1-up",
  "spec_hash": "…",
  "steps": [
    {"i": 0, "psi0": 1.0e-2, "omega": 0.90, "dr": 8.0e-2, "N": 400,
     "sol_dir": "out/campaigns/l1-up/step0000", "exit_code": 0,
     "newton_iters": 7, "lambda_min": 0.32, "rr_phi_max": 1.8,
     "hwl": 21, "r99": 41.2, "M_Komar": 0.113, "J_Komar": 0.031}
  ],
  "status": "running"            // running | done | stopped:turning_point | failed
}
```

Interrupted campaigns resume from the last completed step; the driver re-reads the spec
and validates `spec_hash` (a changed spec aborts resume with a clear error).

### 3.3 Step loop

```
for step in campaign:
    psi0_next = next_target(state)                          # §5; shrunk on retry
    seed      = initial_guess(state, psi0_next)             # §4
    par       = render_params(spec, state, psi0_next, seed) # TOML → out/campaigns/.../step.toml
    run C binary (subprocess, timeout); map exit code
    if exit_code == 1: shrink the step (÷2) and retry, up to max_retries (core
                       implements this subset of the §5 table; regrid/actions
                       beyond shrinking are SAN-14)
    if exit_code in {2,3,4}: stop (not retryable by stepping)
    read HDF5 → diagnostics → append to state.json
    if turning-point test (§6) fires: record & stop (or continue per spec)
```

### 3.4 Initial guess for the next step

Base (implemented in the SAN-17 core): **previous solution + exact ψ rescale** —
`scale_u4 = ψ₀_target / ψ_prev(fixedPhi point)`, the archived ladder template's trick.
Linear extrapolation in ψ₀ across the last two solutions
(`u_guess = u_k + (u_k − u_{k−1}) · Δψ₀_k/Δψ₀_{k−1}`, design §3.4) activates only once
**both** predecessors are fixedPhi continuation steps: extrapolating across the fixedOmega
seed solve was verified A/B to produce guesses Newton cannot recover from (λ → λ_min
stagnation), while the rescale-only guess converges in 2–5 iterations on the test branch.
Both guesses are only seeds; the fixedPhi constraint lands ψ₀ exactly on target.

---

## 4. Regrid strategy (ladder across grids)

Both failure modes are regrid events. The C side already has everything needed:
`initial_interpolator()` interpolates a solution from grid (dr₀, N₀) onto (dr₁, N₁) with
`readInitialData = 2` — the paper did exactly this "seamlessly".

**The knob is dr with N fixed.** Domain size is `(N + 2·ghost)·dr`, so:

- **Finer + smaller domain** (field localizing, spiky near axis): `dr ← dr/2`
  → same interior points, half the physical extent, doubled resolution of the spike.
- **Coarser + larger domain** (field spreading, support → boundary): `dr ← dr·2`
  → same interior points, doubled physical extent.

Halving/doubling keeps the interpolation exact on common grid points (staggered half-cell
offsets preserved), which is why parity verification (§7) is tractable.

Regrid event flow: (1) re-solve the **same ψ₀** on the new grid from the interpolated seed;
(2) compare global parameters (M_Komar, J_Komar, ω) old-vs-new grid — accept when the
difference is below the truncation-error proxy (paper §IX criterion, §7.3); (3) only then
continue stepping on the new grid. If the re-solve fails to converge, halve the move
(intermediate dr) or fall back to stepping on the old grid with a smaller Δψ₀.

---

## 5. Adaptive decision table

Diagnostics read from each solution (all already computed by C's analysis):

- `psi0`, `omega`, `M_Komar`, `J_Komar` — branch coordinates
- `rr_phi_max` — radial location of field max (axis drift vs boundary drift)
- `hwl_resolution` — points across the field's half-wavelength (resolution)
- `r99` — support radius (boundary proximity), as fraction of `r_bdy`
- Newton health: iteration count, `lambda_min`, final `norm_f`

| # | Symptom (from diagnostics)                                    | Action                                                                 |
|---|---------------------------------------------------------------|------------------------------------------------------------------------|
| 1 | Newton converged in ≤ ~8 iters, `lambda_min` healthy          | Grow Δψ₀ (×1.25, capped)                                               |
| 2 | Newton iterations high, or `lambda_min` collapsed (< 1e-3)    | Shrink Δψ₀ (×0.5); retry the step from the last good solution          |
| 3 | Newton failed (exit 1)                                        | Retry with Δψ₀ ×0.25; two consecutive failures → stop, flag for human  |
| 4 | Solver error (exit 2)                                         | Retry once; if persistent → stop (backend/environment problem)          |
| 5 | `r99 / r_bdy > support_fraction` (support → boundary)         | **Regrid**: dr ×2 (coarser, larger domain), re-solve same ψ₀ (§4) — only while `dr < dr_max`; otherwise **stop** (`stopped:domain_budget`) |
| 6 | `hwl < hwl_min` (under-resolved spike)                        | **Regrid**: dr ÷2 (finer, smaller domain), re-solve same ψ₀ (§4)        |
| 7 | `hwl > hwl_max` (over-resolved, wasteful)                     | Optional: dr ×2 to save time (never blocks the campaign)               |
| 8 | `rr_phi_max` below floor (max spike hugging the axis, l ≥ 2)  | Same as 6 — this is the paper's high-amplitude limit case; also raise the resolution floor requirement |
| 9 | ω stopped decreasing across k steps while ψ₀ grew             | Turning point: switch to small Δψ₀ (fine sampling), record ω_min, stop or continue past per spec (§6) |
| 10| ψ₀ ≥ psi0_target or max_steps                                 | Campaign done                                                          |

Defaults for thresholds come from the campaign spec, initialized to the historical
values (`rr_phi_max_minimum = 0.5`, `hwl_min = 8`, …) so behaviour starts conservative
and familiar.

---

## 6. Exit conditions & turning-point handling

### 6.1 Exit conditions (both directions)

A campaign stops when any of these fires; the reason is recorded in `state.json`:

| condition | `up` direction | `down` direction |
|---|---|---|
| ψ₀ target reached | ψ₀ ≥ `psi0_target` | ψ₀ ≤ `psi0_target` (floor, e.g. the paper's 1e-8) |
| ω target crossed | ω ≤ `omega_target` (past the minimum) | ω ≥ `omega_target` (Newtonian asymptote; catalogue used ω = 0.9) |
| domain budget exhausted | n/a (domain shrinks) | `dr` reached `dr_max` and support still hits the boundary → `stopped:domain_budget` |
| step budget | `max_steps` | `max_steps` |
| turning point | §6 detection | n/a (ω monotone rising toward m) |
| failure stall | rules 3/4 | rules 3/4 |

The domain-budget stop is the honest answer to "regrids can grow forever": with N fixed,
growing dr costs no memory but destroys the truncation-error budget and eventually solves
a physically trivial weak-field configuration. `dr_max` defaults to 4× the seed dr (two
regrids); going beyond it requires an explicit spec override.

### 6.2 Turning point (minimum ω) handling

- **Detection:** track `dω/dψ₀` across the last 3 steps (central differences). When it
  approaches 0 and changes sign, mark `ω_min ≈ min(ω)` between the bracketing steps.
- **Refinement:** near the detected minimum, shrink Δψ₀ automatically (rule 9) to sample
  the bottom of the branch densely.
- **Localization:** for the reported turning-point frequency, fit a low-order polynomial
  (the paper used a 4th-degree spline on ω(ψ₀) or ω(M)) to the sampled points and take the
  extremum. This is a post-processing mode of the driver (`--summarize`), not in-loop logic.
- **Stopping:** `stop_at_turning_point = true` (default) halts the campaign with
  `status = stopped:turning_point`; `false` continues stepping in ψ₀ onto the
  high-ω/high-amplitude side of the branch, which ψ₀-stepping handles without any special
  logic — this is the whole point of the design decision in §1.

---

## 7. Verification ("what it means to trust the driver") — feeds SAN-13

### 7.1 Golden ω-sweep sequence
Regenerate an l=1 sequence spanning several catalogue points from `data/golden/`
(dr=8.0e-2, N=400) with the driver, and match every catalogue solution in the span
against `tools/check_against_summary.py` / `compare_solutions.py` at the §4c tolerances
(rtol = 1e-10, atol = 1e-12; historically ~1e-13).

### 7.2 fixedPhi ladder
Reproduce the archived l=2 ladder step (golden template, `scale_u4 = 1.125`,
w = 8.74062E-01 vs Catalogue2) using `tools/ladder_continue.py` semantics ported into the
driver, including a multi-step run and a resume-after-interruption check of `state.json`.

### 7.3 Regrid parity
Take a golden solution; interpolate it to dr ÷2 and dr ×2 (same N), re-solve the same ψ₀,
and verify: (a) both regridded solves converge; (b) M_Komar, J_Komar, ω agree with the
original within the truncation-error proxy (paper §IX: "changes in resolution have been
done seamlessly"); (c) interpolation round-trip error measured directly
(`hdf5_roundtrip.py` machinery).

### 7.4 Turning point + low-amplitude tail
Run an up-direction l=1 campaign toward the known minimum-ω region; check that the
driver detects, samples, and reports ω_min consistent with the published turning-point
values (Table IX.1 / Catalogue2). Then run a down-direction l=1 campaign from a
mid-branch seed (ω ≈ 0.87): verify it regrows the domain (dr ×2 at most, per `dr_max`),
stops cleanly at `omega_target` (0.9), and that every solution along the tail passes
`check_against_summary.py` at the §4c tolerances.

Gate: none of SAN-10/SAN-17/SAN-14 is "done" until its slice of §7.1–7.4 passes on both
backends (release/MKL and umfpack).

---

## 8. Rollout & out of scope

| Slice | Issue  | Content                                             |
|-------|--------|-----------------------------------------------------|
| 1     | SAN-10 | C contract: exit codes, drop sweep, provenance       |
| 2     | SAN-17 | Core driver: spec, state, ψ₀ stepping, resume, §7.1–7.2 |
| 3     | SAN-14 | Decision table + regrid ladder, §7.3                 |
| 4     | SAN-13 | Full verification pass, §7.4, both backends          |

**Out of scope:** parallel/multi-campaign orchestration, M4 multigrid/non-uniform grids,
perturbation/stability solver, paper figure generation (driver only emits solutions +
state; plotting stays ad hoc).

---

## Revision 2 (SAN-21) — the fixed-step workhorse and one refinement rule

Revision 1's full adaptive apparatus was validated end-to-end on the l=1
branch (from scratch through the turning point to ψ₀ = 0.5; see SAN-14).
It worked — and the run showed most of the policy layer was not worth its
cost: the fixed 3% relative ladder crossed the fold without a single
failure (176 solves, 1.6 h) while the adaptive run needed 2173 solves and
12.6 h, with six distinct policy findings. Revision 2 keeps what earned
its keep and deletes the rest.

### What stays

- ψ₀ continuation, plain-rescale seeds, fixedPhi/interpolated restarts,
  state.json resume, strict exit codes, timeout/physics-limit/signal
  handling, curve tooling and `--summarize`.
- **Fixed relative Δψ₀ stepping** (default 3%) — no growth, no
  damping-based shrinking. The λ history carries a trailing 0.0 sentinel
  on healthy solves ([1.0, 0.0]), so any tail statistic over it is
  meaningless; the λ-based "grudging convergence" metric is retired.
- **ψ₀-stepping crosses the fold naturally** — the minimum is a
  measurement target, not a resolution event.

### The one refinement rule (verify-then-commit)

When the field's peak is under-resolved — hwl < `hwl_min` (8) **or** the
peak sits closer to the axis than half its own width
(`rr_phi_max < (hwl/2)·dr`, the grid-relative form of the old
rr_phi_max-floor rule; within a half-width of the origin the φ ∝ r^l
power law dominates and the peak-location fit is axis-biased) — refine
**once**: dr ÷2 (domain shrinks, N fixed), interpolated re-solve at the
same ψ₀, committed immediately, and the next continuation step doubles
as verification. If verification fails, the refinement is not committed:
restore the previous grid, keep the fine solution as a measurement
(`fold_fine_grid_measurement`), and disable further refinements. One
decision, permanent, no oscillation. `max_refinements` (default 2)
bounds the count per campaign.

### What is deleted

Rules 1 (growth), 2 (damping-based shrink), 5 (support widening — weak-field
boundary error dominates and widening cannot fix it; down campaigns stop at
`stopped:boundary` instead, with `omega_target = 0.9` as the practical
paper-convention end), 7 (coarsening), the trend windows/cooldowns/
blacklists/auto-revert machinery, the domain budget, and the fine-sampling
mode. Resolution changes are irreversible factors of 2.

### Evidence (l=1, dr = 0.125/N = 128, both campaigns on the same grid)

| | fixed 3% (SAN-17) | adaptive v1 (SAN-14) |
|---|---|---|
| wall clock | 1.6 h | 12.6 h |
| solves | 176 | 2173 |
| branch curve | golden-anchored | matches v1 reference to 1.7e-5 |
| fold measurement | ω 0.64708 | ω 0.64597 on dr ÷2 (paper: 0.64561) |

The v2 acceptance run is the fixed ladder's predictability plus the fold
measurement: one uninterrupted campaign, ≤ 3 h, reproducing the branch and
refining where the spikes demand it.
