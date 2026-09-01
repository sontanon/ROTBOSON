"""Python sweep driver with adaptive stepping — SAN-17 (core) + SAN-14 (adaptive).

C solves one solution; Python drives. The continuation parameter is **ψ₀**
(the field value at the fixedPhi grid point): each step renders a parameter
file that seeds from the previous solution(s) and constrains
ψ(fixedPhi point) = ψ₀_target, so Newton solves ω as an eigenvalue. This keeps
ω(ψ₀) single-valued through the minimum-frequency turning point (design §1).

Usage:
    uv run tools/sweep_driver.py <campaign.toml> [--fresh] [--dry-run] [--summarize]

A campaign writes `state.json` (atomically, after every step) plus one
solution directory per step under `[output] root`. An interrupted campaign
resumes automatically from the last completed step; a changed spec aborts
resume (delete state.json or pass --fresh).

Adaptive layer (SAN-14, design §4–6):
  * step-size control in Δψ₀ (grow on fast healthy convergence, shrink on
    trouble, persistent factor in state.json);
  * regrid ladder — dr ×2 / ÷2 at fixed N via the C interpolator
    (readInitialData = 3), re-solving the same ψ₀ and accepting only when
    ω/M_Komar/J_Komar agree within the truncation-error proxy;
  * domain-growth budget (`dr_max`) → `stopped:domain_budget`;
  * turning-point detection (dω/dψ₀ sign change) with fine sampling or a
    clean `stopped:turning_point` stop, plus `--summarize` post-processing
    that localizes ω_min with a low-order polynomial fit.
The decision logic lives in pure functions (`decide_action`,
`detect_turning_point`, `turning_point_estimate`) unit-tested in
`tests/test_driver_decisions.py`. Golden-sequence verification is SAN-13.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from rotboson_io import (
    extract_scalars,
    extract_scalars_from_hdf5,
    find_solution_dirs,
    is_solution_dir,
    read_hdf5,
)

REPO = Path(__file__).resolve().parent.parent

# Binary location: same search order as tools/smoke.py.
BUILD_PRESETS = ("release", "umfpack", "dev", "asan-ubsan")

# 2D field datasets used to build the next step's seed (final fields only).
SEED_FIELDS = [
    "log_alpha_f.asc",
    "beta_f.asc",
    "log_h_f.asc",
    "log_a_f.asc",
    "psi_f.asc",
    "lambda_f.asc",
]

SEED_PARAM_KEYS = {
    "log_alpha_i": "log_alpha_f.asc",
    "beta_i": "beta_f.asc",
    "log_h_i": "log_h_f.asc",
    "log_a_i": "log_a_f.asc",
    "psi_i": "psi_f.asc",
    "lambda_i": "lambda_f.asc",
}


class SpecError(ValueError):
    """Invalid campaign spec."""


# ---------------------------------------------------------------------------
# Spec loading / validation
# ---------------------------------------------------------------------------

CAMPAIGN_KEYS = {
    "l",
    "m",
    "direction",
    "psi0_target",
    "omega_target",
    "psi0_step",
    "psi0_step_mode",
    "max_retries",
    "max_steps",
    "fixedPhiR",
    "fixedPhiZ",
    "stop_at_turning_point",
}
SEED_KEYS = {"policy", "source", "w0", "psi0", "sigmaR", "sigmaZ", "rExt"}
GRID_KEYS = {"dr", "dr_max", "N", "order"}
SOLVER_KEYS = {
    "solverType",
    "localSolver",
    "epsilon",
    "maxNewtonIter",
    "lambda0",
    "lambdaMin",
    "useLowRank",
}
OUTPUT_KEYS = {"root", "format"}
ADAPTIVITY_KEYS = {
    "hwl_min",
    "hwl_max",
    "support_fraction",
    "rr_phi_max_min",
    "grow_factor",
    "shrink_factor",
    "factor_max",
    "newton_fast_iters",
    "lambda_min_floor",
    "regrid_rtol",
}
KNOWN = {
    "campaign": CAMPAIGN_KEYS,
    "seed": SEED_KEYS,
    "grid": GRID_KEYS,
    "solver": SOLVER_KEYS,
    "output": OUTPUT_KEYS,
    "adaptivity": ADAPTIVITY_KEYS,
}

DEFAULTS = {
    "campaign": {
        "m": 1.0,
        "psi0_step_mode": "relative",
        "max_retries": 3,
        "max_steps": 50,
        "fixedPhiR": 2,
        "fixedPhiZ": 2,
        "stop_at_turning_point": True,
    },
    "seed": {"psi0": 0.01, "sigmaR": 4.0, "sigmaZ": 4.0, "rExt": 12.0},
    "grid": {"order": 4},
    "solver": {
        "solverType": 1,
        "localSolver": 1,
        "epsilon": 1.0e-8,
        "maxNewtonIter": 50,
        "lambda0": 1.0e-3,
        "lambdaMin": 1.0e-5,
        "useLowRank": 0,
    },
    "output": {"format": "hdf5"},
    # Design §5: defaults start conservative at the historical C sweep
    # thresholds (hwl_min/max, rr_phi_max floor) so early behaviour matches
    # what the old in-C sweep tolerated.
    "adaptivity": {
        "hwl_min": 8,
        "hwl_max": 40,
        "support_fraction": 0.85,
        "rr_phi_max_min": 0.5,
        "grow_factor": 1.25,
        "shrink_factor": 0.5,
        "factor_max": 4.0,
        "newton_fast_iters": 8,
        "lambda_min_floor": 1.0e-3,
        "regrid_rtol": 2.0e-2,
    },
}


def load_spec(path: Path) -> dict:
    """Load and validate a campaign spec; returns the fully-defaulted dict."""
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode())

    unknown_tables = set(raw) - set(KNOWN)
    if unknown_tables:
        raise SpecError(f"unknown spec tables: {sorted(unknown_tables)}")
    for table, allowed in KNOWN.items():
        if table in raw:
            extra = set(raw[table]) - allowed
            if extra:
                raise SpecError(f"[{table}] unknown keys: {sorted(extra)}")

    spec: dict = {}
    for table in KNOWN:
        merged = dict(DEFAULTS.get(table, {}))
        merged.update(raw.get(table, {}))
        spec[table] = merged
    spec["campaign"].update(raw.get("campaign", {}))

    c = spec["campaign"]
    for key in ("l", "direction", "psi0_target", "psi0_step"):
        if key not in c:
            raise SpecError(f"[campaign] missing required key '{key}'")
    if c["direction"] not in ("up", "down"):
        raise SpecError('[campaign] direction must be "up" or "down"')
    if not c["psi0_step"] > 0:
        raise SpecError("[campaign] psi0_step must be > 0")
    if c["psi0_step_mode"] not in ("absolute", "relative"):
        raise SpecError('[campaign] psi0_step_mode must be "absolute" or "relative"')
    if c["max_retries"] < 0:
        raise SpecError("[campaign] max_retries must be >= 0")

    seed = spec["seed"]
    if "policy" not in seed:
        raise SpecError("[seed] missing required key 'policy'")
    if seed["policy"] not in ("from_scratch", "solution"):
        raise SpecError('[seed] policy must be "from_scratch" or "solution"')
    if seed["policy"] == "solution":
        if "source" not in seed:
            raise SpecError("[seed] policy=solution requires 'source'")
        if not Path(seed["source"]).is_absolute():
            seed["source"] = str((path.parent / seed["source"]).resolve())
        if not is_solution_dir(Path(seed["source"]).name) or not Path(seed["source"]).is_dir():
            raise SpecError(f"[seed] source is not a solution directory: {seed['source']}")
    if seed["policy"] == "from_scratch" and "w0" not in seed:
        raise SpecError("[seed] policy=from_scratch requires 'w0' (fixedOmega seed solve)")

    grid = spec["grid"]
    if "dr" not in grid or "N" not in grid:
        raise SpecError("[grid] missing required keys 'dr' and/or 'N'")
    # Domain-growth budget (design §6.1): defaults to 4× the seed dr (two
    # coarsening regrids); going beyond requires an explicit override.
    grid.setdefault("dr_max", 4.0 * grid["dr"])
    if grid["dr_max"] < grid["dr"]:
        raise SpecError("[grid] dr_max must be >= dr")

    a = spec["adaptivity"]
    if a["hwl_min"] <= 0 or a["hwl_max"] <= a["hwl_min"]:
        raise SpecError("[adaptivity] hwl thresholds must satisfy 0 < hwl_min < hwl_max")
    if not 0.0 < a["support_fraction"] <= 1.0:
        raise SpecError("[adaptivity] support_fraction must be in (0, 1]")
    if a["rr_phi_max_min"] <= 0:
        raise SpecError("[adaptivity] rr_phi_max_min must be > 0")
    if a["grow_factor"] <= 1.0:
        raise SpecError("[adaptivity] grow_factor must be > 1")
    if not 0.0 < a["shrink_factor"] < 1.0:
        raise SpecError("[adaptivity] shrink_factor must be in (0, 1)")
    if a["factor_max"] < 1.0:
        raise SpecError("[adaptivity] factor_max must be >= 1")
    if a["newton_fast_iters"] < 1:
        raise SpecError("[adaptivity] newton_fast_iters must be >= 1")
    if a["lambda_min_floor"] <= 0:
        raise SpecError("[adaptivity] lambda_min_floor must be > 0")
    if a["regrid_rtol"] <= 0:
        raise SpecError("[adaptivity] regrid_rtol must be > 0")
    if not isinstance(c["stop_at_turning_point"], bool):
        raise SpecError("[campaign] stop_at_turning_point must be a boolean")

    out = spec["output"]
    if "root" not in out:
        name = path.stem
        out["root"] = str(REPO / "out" / "campaigns" / name)
    if not Path(out["root"]).is_absolute():
        out["root"] = str((path.parent / out["root"]).resolve())
    if out["format"] not in ("hdf5", "ascii"):
        raise SpecError('[output] format must be "hdf5" or "ascii"')

    # Spec hash: content hash with runtime control keys (max_steps,
    # max_retries) excluded — raising a limit must not invalidate the physics
    # state of a running campaign; any physics-affecting change does.
    control = re.compile(r"^\s*(max_steps|max_retries)\s*=.*$", re.MULTILINE)
    canon = control.sub("", raw_bytes.decode()).encode()
    spec["_spec_hash"] = hashlib.sha256(canon).hexdigest()
    return spec


def find_binary() -> Path:
    for preset in BUILD_PRESETS:
        candidate = REPO / "build" / preset / "ROTBOSON"
        if candidate.exists():
            return candidate
    raise SystemExit(
        "ROTBOSON binary not found; build with cmake --preset release && "
        "cmake --build --preset release"
    )


# ---------------------------------------------------------------------------
# State (design §3.2): atomic writes, spec-hash-guarded resume
# ---------------------------------------------------------------------------


def state_path(spec: dict) -> Path:
    return Path(spec["output"]["root"]) / "state.json"


def load_state(spec: dict) -> dict | None:
    p = state_path(spec)
    if not p.exists():
        return None
    state = json.loads(p.read_text())
    if state.get("spec_hash") != spec["_spec_hash"]:
        raise SystemExit(
            f"state.json at {p} was written for a different spec version; "
            "pass --fresh to discard it."
        )
    return state


def save_state(spec: dict, state: dict) -> None:
    p = state_path(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(p)  # atomic on POSIX


def fresh_state(spec: dict) -> dict:
    return {
        "spec_hash": spec["_spec_hash"],
        "spec_file": None,  # filled by main()
        "steps": [],
        "status": "running",
        "stop_reason": None,
        # Adaptive state (SAN-14): the grid can drift from the spec's initial
        # value via regrids; the step factor persists grow/shrink decisions.
        "grid": {"dr": spec["grid"]["dr"], "N": spec["grid"]["N"]},
        "step_factor": 1.0,
        "fine_sampling": False,
        "turning_point": None,
    }


def current_grid(state: dict, spec: dict) -> dict:
    """Current grid (dr may have moved from the spec's initial value).

    Backwards compatible with SAN-17 state.json files that predate regrids.
    """
    g = state.get("grid")
    if g is None:
        g = {"dr": spec["grid"]["dr"], "N": spec["grid"]["N"]}
        state["grid"] = g
    return g


# ---------------------------------------------------------------------------
# Solution reading (HDF5 or ASCII, both handled via rotboson_io)
# ---------------------------------------------------------------------------


def solution_scalars(sol_dir: Path) -> dict:
    if (sol_dir / "solution.h5").exists():
        return extract_scalars_from_hdf5(sol_dir)
    return extract_scalars(sol_dir)


def solution_fields(sol_dir: Path) -> dict[str, np.ndarray]:
    """Return the six final 2D fields of a solution (HDF5 preferred)."""
    h5 = sol_dir / "solution.h5"
    if h5.exists():
        datasets, _ = read_hdf5(h5)
        return {k: np.asarray(v) for k, v in datasets.items() if k in SEED_FIELDS}
    from rotboson_io import read_2d

    return {name: read_2d(sol_dir / name) for name in SEED_FIELDS}


def psi_at_fixed_point(psi: np.ndarray, spec: dict) -> float:
    """ψ at the fixedPhi grid point — the value the C constraint enforces."""
    c = spec["campaign"]
    return float(psi[c["fixedPhiR"], c["fixedPhiZ"]])


# ---------------------------------------------------------------------------
# Seed rendering (design §3.4): linear extrapolation in ψ₀ + exact rescale
# ---------------------------------------------------------------------------


def write_seed_field(path: Path, data: np.ndarray) -> None:
    # Whitespace-agnostic on the C side (fscanf %lE); use the ASCII layout.
    with path.open("w") as fh:
        for row in data:
            fh.write("\t".join(f"{v:9.18E}" for v in row) + "\n")


def render_seed(
    spec: dict, root: Path, prev: list[dict], psi0_target: float
) -> tuple[float, float]:
    """Build the seed files for the next step.

    Base guess = the previous converged solution with ψ rescaled so its value
    at the fixedPhi point is exactly ψ0_target — the golden fixedPhi-ladder
    semantics (scale_u4-only). Linear extrapolation across the last two
    *continuation* steps (design §3.4) kicks in once both predecessors are
    fixedPhi solves; mixing the fixedOmega seed solve into the extrapolation
    empirically produces guesses Newton cannot recover from.
    Returns (scale_u4, omega_guess).
    """
    seed_dir = root / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cur = prev[-1]
    fields = solution_fields(Path(cur["sol_dir"]))

    can_extrapolate = (
        len(prev) >= 2
        and prev[-1].get("mode") == "fixedPhi"
        and prev[-2].get("mode") == "fixedPhi"
        and prev[-1]["psi0"] != prev[-2]["psi0"]
    )
    if can_extrapolate:
        older = solution_fields(Path(prev[-2]["sol_dir"]))
        ratio = (psi0_target - prev[-1]["psi0"]) / (prev[-1]["psi0"] - prev[-2]["psi0"])
        for name in SEED_FIELDS:
            fields[name] = fields[name] + ratio * (fields[name] - older[name])
        w_guess = cur["omega"] + ratio * (cur["omega"] - prev[-2]["omega"])
    else:
        w_guess = cur["omega"]

    psi_fixed = psi_at_fixed_point(fields["psi_f.asc"], spec)
    if psi_fixed == 0:
        raise SystemExit("seed ψ at the fixedPhi point is zero; cannot rescale")
    scale_u4 = psi0_target / psi_fixed

    for name in SEED_FIELDS:
        write_seed_field(seed_dir / name, fields[name])
    # ω guess for the eigenvalue (C scales it by scale_u6, left at 1.0).
    (seed_dir / "w_f.asc").write_text(f"{w_guess:9.18E}\n")

    return scale_u4, w_guess


def spec_with_grid(spec: dict, dr: float) -> dict:
    """Shallow spec copy with the grid's dr replaced (regrid rendering)."""
    return {**spec, "grid": {**spec["grid"], "dr": dr}}


def render_params(
    spec: dict,
    root: Path,
    step: int,
    *,
    scale_u4: float | None = None,
    seed_dir: Path | None = None,
    initial_grid: dict | None = None,
) -> Path:
    """Render the per-step parameter file into the campaign root."""
    c, grid, solver = spec["campaign"], spec["grid"], spec["solver"]
    lines = [
        f"# Rendered by tools/sweep_driver.py — step {step}, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "# GRID",
        f"dr = {grid['dr']:.6E}",
        f"dz = {grid['dr']:.6E}",
        f"NrInterior = {grid['N']}",
        f"NzInterior = {grid['N']}",
        f"order = {grid['order']}",
        "",
        "# SCALAR FIELD PROPERTIES",
        f"l = {c['l']}",
        f"m = {c['m']}",
        "",
    ]

    seed = spec["seed"]
    if seed["policy"] == "from_scratch" and step == 0:
        lines += [
            "# INITIAL DATA (analytic guess, seed solve at fixed ω)",
            "readInitialData = 0",
            f"psi0 = {seed['psi0']:.6E}",
            f"sigmaR = {seed['sigmaR']:.6E}",
            f"sigmaZ = {seed['sigmaZ']:.6E}",
            f"rExt = {seed['rExt']:.6E}",
            "",
            "# INITIAL FREQUENCY (fixed for the seed solve)",
            f"w0 = {seed['w0']:.6E}",
            "",
            "fixedPhi = 0",
            "fixedOmega = 1",
        ]
    else:
        lines += [
            "# INITIAL DATA (extrapolated from previous converged solutions)",
            # readInitialData = 3: interpolated restart from a stated source
            # grid (design §4) — used for regrids; plain same-grid restarts
            # use mode 1.
            f"readInitialData = {3 if initial_grid else 1}",
        ]
        for param, field in SEED_PARAM_KEYS.items():
            lines.append(f'{param} = "{(seed_dir / field).resolve()}"')
        lines.append(f'w_i = "{(seed_dir / "w_f.asc").resolve()}"')
        if initial_grid:
            lines += [
                f"NrTotalInitial = {initial_grid['NrTotalInitial']}",
                f"NzTotalInitial = {initial_grid['NzTotalInitial']}",
                f"order_i = {initial_grid['order_i']}",
                f"ghost_i = {initial_grid['ghost_i']}",
                f"dr_i = {initial_grid['dr_i']:.6E}",
                f"dz_i = {initial_grid['dz_i']:.6E}",
            ]
        lines += [
            "",
            "# Scale the field so ψ at the fixedPhi point is exactly ψ₀_target",
            f"scale_u4 = {scale_u4:.10E}",
            "",
            "fixedPhi = 1",
            f"fixedPhiR = {c['fixedPhiR']}",
            f"fixedPhiZ = {c['fixedPhiZ']}",
            "fixedOmega = 0",
        ]

    lines += [
        "",
        "# SOLVER PARAMETERS",
        f"solverType = {solver['solverType']}",
        f"localSolver = {solver['localSolver']}",
        f"epsilon = {solver['epsilon']:.6E}",
        f"maxNewtonIter = {solver['maxNewtonIter']}",
        f"lambda0 = {solver['lambda0']:.6E}",
        f"lambdaMin = {solver['lambdaMin']:.6E}",
        f"useLowRank = {solver['useLowRank']}",
        "",
        f'outputFormat = "{spec["output"]["format"]}"',
    ]
    if spec["output"]["format"] == "ascii":
        lines.append('loglevel = "warn"')

    out = root / f"step{step:04d}.toml"
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Running one step
# ---------------------------------------------------------------------------


def run_binary(
    binary: Path, params: Path, root: Path, step: int, label: str | None = None
) -> tuple[int, Path]:
    """Run ROTBOSON from the campaign root; returns (exit code, log path)."""
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"step{step if label is None else label}.log"
    t0 = time.monotonic()
    with log.open("w") as lf:
        lf.write(
            f"# ROTBOSON {binary} {params}\n# start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        lf.flush()
        proc = subprocess.run(
            [str(binary), str(params)], cwd=root, stdout=lf, stderr=subprocess.STDOUT, timeout=3600
        )
    wall = time.monotonic() - t0
    with log.open("a") as lf:
        lf.write(f"# exit_code={proc.returncode} wall={wall:.1f}s\n")
    return proc.returncode, log


def initial_dirname(spec: dict) -> str:
    """The pre-rename output dir name the C binary writes (parser convention)."""
    c, grid = spec["campaign"], spec["grid"]
    return f"l={c['l']},w=X.XXXXXE-01,dr={grid['dr']:.5E},N={grid['N']:04d}"


def new_solution_dir(root: Path, before: set[Path], spec: dict, step: int) -> Path | None:
    """Locate the solution the run just produced.

    Normally the binary renames its output dir to the final w= name. If that
    name already exists (re-runs, repeated ω), the rename fails and the output
    stays under the X= initial name — recover it under a unique name instead
    of losing the step.
    """
    after = set(find_solution_dirs(root))
    new = sorted(after - before)
    if new:
        return new[-1]

    stale = root / initial_dirname(spec)
    if not stale.is_dir():
        return None
    unique = root / f"l=step{step:04d},w=rename-collision,dr=0,N=0000"
    if unique.exists():
        return stale  # give up disambiguating; read in place
    stale.rename(unique)
    return unique


def do_regrid(
    spec: dict,
    root: Path,
    state: dict,
    binary: Path,
    step_no: int,
    new_dr: float,
    base_psi0: float,
) -> tuple[bool, dict]:
    """Re-solve the *same* ψ₀ on a grid with dr → `new_dr` (design §4).

    Seeds through the C interpolator (readInitialData = 3) from the last good
    solution, constrains ψ(fixedPhi point) = ψ₀ as usual, and accepts only
    when ω / M_Komar / J_Komar agree with the source solution within the
    truncation-error proxy (`[adaptivity] regrid_rtol`).

    The C freezes the Newton update at the fixedPhi point, so the enforced ψ₀
    is the *interpolated seed's* value there — which can drift from the
    requested ψ₀ (the C bicubic overshoots near the axis on coarse seeds).
    The frozen value scales exactly linearly with scale_u4, so the first run
    measures the drift and a second run with a corrected scale lands exactly
    on ψ₀.

    Only an *accepted* re-grid is recorded as a step (mode "regrid") — it is
    the same branch point as the source, not a new one. Failed or rejected
    attempts go to state['rejected_regrids'] for provenance; the caller
    falls back (midpoint dr, then smaller steps on the old grid).

    Returns (accepted, step_record).
    """
    # Source = last good solution that is not a rejected regrid attempt:
    # seeding from a rejected attempt would chase its drifted branch point.
    src = None
    for s in reversed(state["steps"]):
        if s.get("psi0") is None or s.get("exit_code", 1) != 0:
            continue
        if s.get("mode") == "regrid" and not s.get("regrid", {}).get("accepted", False):
            continue
        src = s
        break
    if src is None:
        return False, {}

    # Re-solve the SOURCE's ψ₀ — not the last step's value, which a failed
    # attempt may have moved.
    base_psi0 = float(src["psi0"])
    src_dr, src_n = float(src["dr"]), int(src["N"])
    order = spec["grid"]["order"]

    # Finer regrids shrink the domain (N fixed): never amputate the field —
    # if the support would not fit in the new domain, the attempt is futile.
    if new_dr < src_dr and src.get("r99") is not None:
        new_domain = (src_n + 2 * ghost_of(order)) * new_dr
        if src["r99"] > spec["adaptivity"]["support_fraction"] * new_domain:
            print(
                f"[driver] regrid step {step_no}: skipped — support r99={src['r99']:.3g} "
                f"would not fit in the {new_domain:.3g} domain"
            )
            state.setdefault("rejected_regrids", []).append(
                {
                    "i": step_no,
                    "mode": "regrid-probe",
                    "exit_code": None,
                    "skipped": "support would not fit",
                    "from_dr": src_dr,
                    "to_dr": new_dr,
                }
            )
            save_state(spec, state)
            return False, {}

    # Coarsening loses accuracy: gate on the truncation-error proxy. Refining
    # only gains accuracy — a global-parameter difference there measures the
    # OLD grid's error (the reason we are refining), so it is recorded as
    # provenance but does not block acceptance; convergence + the exact ψ₀
    # landing are the guards.
    strict = new_dr > src_dr
    initial_grid = {
        "NrTotalInitial": src_n + 2 * ghost_of(order),
        "NzTotalInitial": src_n + 2 * ghost_of(order),
        "order_i": order,
        "ghost_i": ghost_of(order),
        "dr_i": src_dr,
        "dz_i": src_dr,
    }
    spec2 = spec_with_grid(spec, new_dr)
    rejected: list[dict] = []

    scale_u4, _ = render_seed(spec, root, [src], base_psi0)
    rec: dict = {}
    for attempt in range(2):
        params = render_params(
            spec2,
            root,
            step_no,
            scale_u4=scale_u4,
            seed_dir=root / "seed",
            initial_grid=initial_grid,
        )
        stale = root / initial_dirname(spec2)
        if stale.is_dir():
            import shutil

            shutil.rmtree(stale)
        before = set(find_solution_dirs(root))
        code, log = run_binary(
            binary,
            params,
            root,
            step_no,
            label=f"{step_no:04d}_probe{attempt}" if attempt else None,
        )
        sol = new_solution_dir(root, before, spec2, step_no)
        probe: dict = {
            "i": step_no,
            "mode": "regrid-probe",
            "exit_code": code,
            "scale_u4": scale_u4,
            "log": str(log),
        }
        if sol is not None:
            probe["sol_dir"] = str(sol)
            probe["scalars"] = solution_scalars(sol)
            try:
                fields = solution_fields(sol)
                probe["psi0"] = psi_at_fixed_point(fields["psi_f.asc"], spec2)
            except Exception:  # noqa: BLE001 — probe diagnostics only
                probe["psi0"] = None

        achieved = probe.get("psi0")
        if attempt == 0 and code == 0 and achieved is not None:
            drift = abs(achieved - base_psi0) / abs(base_psi0)
            if drift > 1.0e-9:
                # Frozen constraint value is linear in scale_u4: one
                # correction lands the re-solve exactly on ψ₀.
                rejected.append(probe)
                scale_u4 *= base_psi0 / achieved
                print(
                    f"[driver] regrid step {step_no}: interpolated constraint drifted "
                    f"{drift:.2e}; correcting scale_u4 → {scale_u4:.10E}"
                )
                continue
        rec = probe
        break

    state.setdefault("rejected_regrids", []).extend(rejected)

    accepted = False
    rec_step: dict = {}
    if rec and rec.get("exit_code") == 0 and rec.get("psi0") is not None:
        rtol = spec["adaptivity"]["regrid_rtol"]
        rel = {}
        for key, fname in (
            ("omega", "w_f.asc"),
            ("M_Komar", "M_Komar1.asc"),
            ("J_Komar", "J_Komar1.asc"),
        ):
            old, new = src.get(key), rec.get("scalars", {}).get(fname)
            if old is None or new is None or abs(old) == 0:
                rel = {}
                break
            rel[key] = abs(new - old) / abs(old)
        accepted = bool(rel) and (strict is False or all(v < rtol for v in rel.values()))

    if accepted:
        # Promote the accepted re-solve to a real branch-point step.
        sol = Path(rec["sol_dir"])
        record_step(state, spec2, step_no, sol, 0, mode="regrid")
        rec_step = state["steps"][-1]
        rec_step["regrid"] = {
            "from_dr": src_dr,
            "to_dr": new_dr,
            "source": src.get("sol_dir"),
            "rel_diff": rel,
            "accepted": True,
        }
        save_state(spec, state)
        diffs = ", ".join(f"{k}={v:.2e}" for k, v in rel.items())
        kind = "coarser (within truncation proxy)" if strict else "finer (old-grid error recorded)"
        print(
            f"[driver] regrid step {step_no}: dr {src_dr:.5E} → {new_dr:.5E} accepted: {kind} ({diffs})"
        )
    else:
        if rec:
            state.setdefault("rejected_regrids", []).append(rec)
        save_state(spec, state)
        print(
            f"[driver] regrid step {step_no}: dr {src_dr:.5E} → {new_dr:.5E} "
            f"REJECTED (exit {rec.get('exit_code') if rec else 'n/a'}, log: {rec.get('log') if rec else 'n/a'})"
        )
    return accepted, rec_step


def newton_health(sol_dir: Path, fmt: str) -> dict:
    """Newton health from the iteration histories (design §5 diagnostics).

    Returns {} when the histories are unavailable (failed step). `lambda_min`
    is the smallest damping factor over the tail of the λ history — a healthy
    NLEQ-ERR run walks λ up toward 1, so a λ still stuck near `lambdaMin` at
    the end flags a step that converged only grudgingly.
    """
    try:
        datasets = None
        if fmt == "hdf5" and (sol_dir / "solution.h5").exists():
            datasets, _ = read_hdf5(sol_dir / "solution.h5")
        health: dict = {}
        lam = None
        if datasets is not None and "lambda.asc" in datasets:
            lam = np.asarray(datasets["lambda.asc"], dtype=float)
        elif datasets is None:
            from rotboson_io import read_1d

            f = sol_dir / "lambda.asc"
            if f.exists():
                lam = read_1d(f)
        if lam is not None and lam.size:
            health["newton_iters"] = int(lam.size)
            health["lambda_min"] = float(np.min(lam[-3:]))
        norm = None
        if datasets is not None and "norm_f.asc" in datasets:
            norm = np.asarray(datasets["norm_f.asc"], dtype=float)
        elif datasets is None:
            from rotboson_io import read_1d

            f = sol_dir / "norm_f.asc"
            if f.exists():
                norm = read_1d(f)
        if norm is not None and norm.size:
            health["norm_f"] = float(norm[-1])
        return health
    except Exception:  # noqa: BLE001 — diagnostics only, never fail the step
        return {}


def record_step(
    state: dict, spec: dict, i: int, sol_dir: Path | None, exit_code: int, mode: str = "fixedPhi"
) -> None:
    step: dict = {"i": i, "exit_code": exit_code, "mode": mode}
    if sol_dir is not None:
        scalars = solution_scalars(sol_dir)
        entry = {
            "sol_dir": str(sol_dir),
            "dr": spec["grid"]["dr"],
            "N": spec["grid"]["N"],
            "omega": scalars.get("w_f.asc"),
            "M_Komar": scalars.get("M_Komar1.asc"),
            "J_Komar": scalars.get("J_Komar1.asc"),
            "rr_phi_max": scalars.get("rr_phi_max.asc"),
            "r99": scalars.get("r99.asc"),
            "hwl": scalars.get("hwl_resolution.asc"),
        }
        try:
            # ψ₀ from the field at the fixedPhi point (the constraint value).
            fields = solution_fields(sol_dir)
            entry["psi0"] = psi_at_fixed_point(fields["psi_f.asc"], spec)
        except Exception:  # noqa: BLE001 — a failed step may lack field data
            entry["psi0"] = None
        step.update(entry)
        step.update(newton_health(sol_dir, spec["output"]["format"]))
    state["steps"].append(step)


def psi0_of_last(state: dict) -> float:
    for step in reversed(state["steps"]):
        if step.get("psi0") is not None and step.get("exit_code", 1) == 0:
            return float(step["psi0"])
    raise SystemExit("no completed step records ψ₀; cannot continue")


def omega_of_last(state: dict) -> float:
    for step in reversed(state["steps"]):
        if step.get("omega") is not None and step.get("exit_code", 1) == 0:
            return float(step["omega"])
    raise SystemExit("no completed step records ω; cannot continue")


def finished(state: dict, spec: dict) -> str | None:
    """Fixed-grid subset of the §6.1 exit conditions. Returns stop reason."""
    c = spec["campaign"]
    steps = [s for s in state["steps"] if s.get("psi0") is not None]
    if not steps:
        return None
    psi0 = float(steps[-1]["psi0"])
    omega = omega_of_last(state)
    direction = c["direction"]

    # ψ₀ hits the constraint value exactly, up to last-ulp rounding of the
    # scale factor; compare with a tolerance so an exact landing stops the
    # campaign instead of re-solving the same point forever.
    tol = 1e-9 * max(1.0, abs(c["psi0_target"]))
    if direction == "up" and psi0 >= c["psi0_target"] - tol:
        return "done:psi0_target"
    if direction == "down" and psi0 <= c["psi0_target"] + tol:
        return "done:psi0_target"
    if "omega_target" in c and c["omega_target"] is not None:
        if direction == "up" and omega <= c["omega_target"]:
            return "done:omega_target"
        if direction == "down" and omega >= c["omega_target"]:
            return "done:omega_target"
    if len(state["steps"]) >= c["max_steps"]:
        return "done:max_steps"
    if (
        c.get("stop_at_turning_point", True)
        and direction == "up"
        and detect_turning_point(state["steps"])
    ):
        return "stopped:turning_point"
    if state["steps"][-1]["exit_code"] not in (0, None):
        code = state["steps"][-1]["exit_code"]
        if code < 0:
            # Killed by a signal (e.g. -11 = SIGSEGV). Rare, pre-existing C
            # backend flakiness (SAN-19); recorded distinctly from exit codes.
            return f"failed:sig{signal.Signals(-code).name.removeprefix('SIG').lower()}"
        reason = {1: "failed:newton", 2: "failed:solver", 3: "failed:config", 4: "failed:io"}
        return reason.get(code, f"failed:exit{code}")
    return None


# ---------------------------------------------------------------------------
# Adaptive decision layer (design §4–6) — pure functions, unit-tested
# ---------------------------------------------------------------------------


def ghost_of(order: int) -> int:
    """Ghost-cell count for a finite-difference order (C: order 2 → 1, 4 → 2)."""
    return order // 2


def decide_action(diag: dict) -> str:
    """Design §5 decision table for a *converged* step; returns an action.

    One of: "regrid_finer", "regrid_coarser", "regrid_coarser_optional",
    "stop:domain_budget", "shrink", "grow", "ok".

    `diag` bundles the step's diagnostics with the [adaptivity] thresholds:
    newton_iters, lambda_min, hwl, rr_phi_max, r99, r_bdy, dr, dr_max,
    hwl_min, hwl_max, support_fraction, rr_phi_max_min, newton_fast_iters,
    lambda_min_floor. None diagnostics are skipped (step lacked data).

    Precedence note: the design table lists solver-health rules first, but a
    grid about to be replaced makes step-size growth wasted work, so grid
    rules are evaluated first. Under-resolution (rule 6/8) wins over boundary
    proximity (rule 5): a regrid to finer+smaller fixes the spike and
    (by shrinking the domain) only improves the support fraction, while the
    converse would refine away from the failing solution. Failure stalls
    (rules 3–4) never reach here — the retry loop handles them.
    """
    hwl = diag.get("hwl")
    if hwl is not None and hwl < diag["hwl_min"]:
        return "regrid_finer"  # rules 6/8: under-resolved spike / axis hug

    rr = diag.get("rr_phi_max")
    if rr is not None and rr < diag["rr_phi_max_min"]:
        return "regrid_finer"  # rule 8: field maximum hugging the axis

    r99, r_bdy = diag.get("r99"), diag.get("r_bdy")
    if r99 is not None and r_bdy and r99 / r_bdy > diag["support_fraction"]:
        # rule 5: support → boundary; widen the domain while the budget lasts
        if diag["dr"] < diag["dr_max"]:
            return "regrid_coarser"
        return "stop:domain_budget"

    if hwl is not None and hwl > diag["hwl_max"] and diag["dr"] < diag["dr_max"]:
        return "regrid_coarser_optional"  # rule 7: over-resolved, save time

    iters, lam = diag.get("newton_iters"), diag.get("lambda_min")
    grudging = lam is not None and lam < diag["lambda_min_floor"]
    slow = iters is not None and iters > 2 * diag["newton_fast_iters"]
    if grudging or slow:
        return "shrink"  # rule 2: converged, but only just — ease off
    if (
        iters is not None
        and iters <= diag["newton_fast_iters"]
        and (lam is None or lam >= diag["lambda_min_floor"])
    ):
        return "grow"  # rule 1: fast, healthy convergence
    return "ok"


def detect_turning_point(steps: list[dict]) -> bool:
    """Design §6.2: has the branch's minimum-ω turning point been crossed?

    Looks at the last three fixedPhi steps with distinct increasing ψ₀ and
    checks that the central slope dω/dψ₀ changed sign (decreasing → strictly
    increasing). Only meaningful in the up direction; ω rises monotonically
    toward m on the down side.
    """
    pts = [
        (float(s["psi0"]), float(s["omega"]))
        for s in steps
        if s.get("mode") in ("fixedPhi", "seed")
        and s.get("psi0") is not None
        and s.get("omega") is not None
    ]
    if len(pts) < 3:
        return False
    (p1, w1), (p2, w2), (p3, w3) = pts[-3], pts[-2], pts[-1]
    if not (p2 > p1 and p3 > p2):
        return False
    s1 = (w2 - w1) / (p2 - p1)
    s2 = (w3 - w2) / (p3 - p2)
    return s1 < 0.0 < s2


def turning_point_estimate(psi0s, omegas, degree: int = 4) -> dict:
    """Design §6.2 localization: fit a low-order polynomial ω(ψ₀) over the
    samples bracketing the smallest sampled ω and take its extremum (the
    paper used a 4th-degree spline). Returns a report dict; falls back to
    the raw sample minimum when too few points exist for a fit.
    """
    pts = sorted(
        (float(p), float(w))
        for p, w in zip(psi0s, omegas, strict=False)
        if p is not None and w is not None
    )
    if not pts:
        return {}
    k = min(range(len(pts)), key=lambda i: pts[i][1])
    lo, hi = max(0, k - 3), min(len(pts), k + 4)
    xs = np.array([pts[i][0] for i in range(lo, hi)])
    ys = np.array([pts[i][1] for i in range(lo, hi)])
    report = {
        "psi0_sample_min": float(xs[int(np.argmin(ys))]),
        "omega_sample_min": float(ys.min()),
        "n_points": int(xs.size),
    }
    deg = min(degree, xs.size - 2)
    if deg < 2 or np.ptp(xs) == 0:
        report["method"] = "sample"
        return report
    coef = np.polyfit(xs, ys, deg)
    grid = np.linspace(xs[0], xs[-1], 4001)
    vals = np.polyval(coef, grid)
    j = int(np.argmin(vals))
    report.update({"psi0": float(grid[j]), "omega": float(vals[j]), "method": f"poly{deg}"})
    return report


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_campaign(spec: dict, fresh: bool, dry_run: bool) -> int:
    root = Path(spec["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    binary = find_binary()

    if fresh:
        import shutil

        for entry in root.iterdir():
            if entry.is_dir() or entry.name == "state.json":
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        state = fresh_state(spec)
    else:
        state = load_state(spec)
        if state is None:
            state = fresh_state(spec)
        else:
            # A resumed campaign is running again: clear any status left over
            # from the run that wrote this state (e.g. its own done/failed).
            state["status"] = "running"
            state["stop_reason"] = None
    state["spec_file"] = None

    step_no = len(state["steps"])
    print(f"[driver] campaign root: {root}")
    print(f"[driver] starting at step {step_no}, status={state['status']}")

    # ----- seed step -------------------------------------------------------
    if step_no == 0:
        seed = spec["seed"]
        if seed["policy"] == "solution":
            src = Path(seed["source"])
            scalars = solution_scalars(src)
            fields = solution_fields(src)
            entry = {
                "i": 0,
                "exit_code": 0,
                "sol_dir": str(src),
                "dr": spec["grid"]["dr"],
                "N": spec["grid"]["N"],
                "omega": scalars.get("w_f.asc"),
                "psi0": psi_at_fixed_point(fields["psi_f.asc"], spec),
                "M_Komar": scalars.get("M_Komar1.asc"),
                "J_Komar": scalars.get("J_Komar1.asc"),
                "rr_phi_max": scalars.get("rr_phi_max.asc"),
                "r99": scalars.get("r99.asc"),
                "hwl": scalars.get("hwl_resolution.asc"),
                "newton_iters": None,
                "mode": "seed",
            }
            state["steps"].append(entry)
            save_state(spec, state)
            print(
                f"[driver] step 0 (seed): ψ₀={entry['psi0']:.6E} ω={entry['omega']:.6E} from {src.name}"
            )
            step_no = 1
        elif dry_run:
            render_params(spec, root, 0)
            print("[dry-run] rendered step 0 params; stopping (seed solve not run)")
            return 0
        else:
            params = render_params(spec, root, 0)
            before = set(find_solution_dirs(root))
            code, _ = run_binary(binary, params, root, 0)
            sol = new_solution_dir(root, before, spec, 0)
            record_step(state, spec, 0, sol, code)
            state["stop_reason"] = finished(state, spec)
            save_state(spec, state)
            if code != 0 or sol is None:
                state["status"] = "failed"
                state["stop_reason"] = state["stop_reason"] or f"failed:seed_exit{code}"
                save_state(spec, state)
                print(
                    f"[driver] seed solve failed (exit {code}); see {root / 'logs' / 'step0000.log'}"
                )
                return 1
            print(
                f"[driver] step 0 (seed solve): ψ₀={state['steps'][-1]['psi0']:.6E} "
                f"ω={state['steps'][-1]['omega']:.6E}"
            )
            step_no = 1
    elif dry_run:
        print("[dry-run] campaign already has steps; nothing to render")
        return 0

    # ----- continuation steps ---------------------------------------------
    c = spec["campaign"]
    sign = 1 if c["direction"] == "up" else -1

    def next_target(base_psi0: float, factor: float = 1.0) -> float:
        """ψ₀ target for the next step, shrunk by `factor` after a retry.

        relative mode (default): fixed-ratio
        steps (target = ψ₀·(1 ± psi0_step) — the golden ladder's
        scale_u4 = 1.125 semantics), which stay scale-free as ψ₀ spans orders
        of magnitude and never overshoot the way a fixed absolute Δψ₀ does
        from a tiny seed. The target is clamped to land exactly on
        psi0_target.
        """
        if c["psi0_step_mode"] == "relative":
            target = base_psi0 * (1.0 + sign * factor * c["psi0_step"])
        else:
            target = base_psi0 + sign * factor * c["psi0_step"]
        if state.get("fine_sampling"):
            # Rule 9 (continuing past the turning point): sample the branch
            # bottom densely regardless of the stored step factor.
            factor = min(factor, 0.25)
        if c["direction"] == "up":
            return min(round(target, 15), c["psi0_target"])
        return max(round(target, 15), c["psi0_target"])

    while True:
        stop = finished(state, spec)
        if stop is not None:
            if stop.startswith("done"):
                state["status"] = "done"
            elif stop.startswith("stopped:"):
                state["status"] = "stopped"  # clean stop (turning point / budget)
            else:
                state["status"] = "failed"
            state["stop_reason"] = stop
            if stop == "stopped:turning_point":
                steps = state["steps"]
                state["turning_point"] = turning_point_estimate(
                    [s.get("psi0") for s in steps], [s.get("omega") for s in steps]
                )
            save_state(spec, state)
            print(f"[driver] stop: {stop}")
            tp = state.get("turning_point")
            if tp:
                w = tp.get("omega", tp.get("omega_sample_min"))
                p = tp.get("psi0", tp.get("psi0_sample_min"))
                print(f"[driver] ω_min ≈ {w:.6E} at ψ₀ ≈ {p:.6E} ({tp.get('method')})")
            return 0 if state["status"] != "failed" else 1

        # Retry loop (decision-table rules 2-3, core subset): on Newton
        # non-convergence (exit 1) shrink the step and retry from the last
        # good solution. A step killed by a signal (code < 0, e.g. SIGSEGV —
        # SAN-19 backend flakiness) is retried at the same step size: unlike
        # a Newton failure, the step size is not the cause. Solver/config/
        # I-O errors are not retryable.
        base_psi0 = psi0_of_last(state)
        factor = float(state.get("step_factor", 1.0))
        attempts = 0
        while True:
            psi0_target = next_target(base_psi0, factor)
            prev = [s for s in state["steps"] if s.get("psi0") is not None]
            scale_u4, w_guess = render_seed(spec, root, prev, psi0_target)
            params = render_params(spec, root, step_no, scale_u4=scale_u4, seed_dir=root / "seed")

            stale = root / initial_dirname(spec)
            if stale.is_dir():
                import shutil

                shutil.rmtree(stale)
            before = set(find_solution_dirs(root))
            code, log = run_binary(binary, params, root, step_no)
            sol = new_solution_dir(root, before, spec, step_no)
            record_step(state, spec, step_no, sol, code)
            last = state["steps"][-1]
            last["psi0_target"] = psi0_target
            last["step_factor"] = factor
            save_state(spec, state)

            if code == 0 and sol is not None and last.get("psi0") is not None:
                break

            retryable = code == 1 or code < 0 or (code == 2 and attempts == 0)
            exhausted = attempts >= c["max_retries"] or (
                code in (1, 2) and psi0_target == base_psi0
            )
            if retryable and not exhausted:
                attempts += 1
                if code == 1:
                    factor /= 2.0  # Newton failure: shrink the step
                if code == 2:
                    # Rule 4: a solver error gets exactly one retry (a smaller
                    # step also yields a closer, better-conditioned seed);
                    # persistent failure stops the campaign.
                    factor /= 2.0
                # Drop the failed attempt from the step history (keep the log).
                state["steps"].pop()
                save_state(spec, state)
                cause = (
                    "did not converge"
                    if code == 1
                    else f"killed by signal {signal.Signals(-code).name}"
                    if code < 0
                    else "solver error"
                )
                print(
                    f"[driver] step {step_no} attempt {attempts} {cause}; "
                    f"retrying (factor {factor:.3f}), log: {log}"
                )
                continue

            state["status"] = "failed"
            state["stop_reason"] = finished(state, spec) or f"failed:exit{code}"
            save_state(spec, state)
            print(f"[driver] step {step_no} FAILED (exit {code}); log: {log}")
            return 1

        note = f", {attempts} shrink-retry" if attempts else ""
        step_no += 1  # the continuation step consumed its slot

        # ----- adaptive decision (design §5) --------------------------------
        state["step_factor"] = factor
        a = spec["adaptivity"]
        g = current_grid(state, spec)
        dr, n = float(g["dr"]), int(g["N"])
        diag = {
            "newton_iters": last.get("newton_iters"),
            "lambda_min": last.get("lambda_min"),
            "hwl": last.get("hwl"),
            "rr_phi_max": last.get("rr_phi_max"),
            "r99": last.get("r99"),
            "r_bdy": (n + 2 * ghost_of(spec["grid"]["order"])) * dr,
            "dr": dr,
            "dr_max": spec["grid"]["dr_max"],
            "hwl_min": a["hwl_min"],
            "hwl_max": a["hwl_max"],
            "support_fraction": a["support_fraction"],
            "rr_phi_max_min": a["rr_phi_max_min"],
            "newton_fast_iters": a["newton_fast_iters"],
            "lambda_min_floor": a["lambda_min_floor"],
        }
        action = decide_action(diag)

        if action == "grow":  # rule 1: fast, healthy convergence → grow Δψ₀
            state["step_factor"] = min(factor * a["grow_factor"], a["factor_max"])
        elif action == "shrink":  # rule 2: converged, but only just — ease off
            state["step_factor"] = max(factor * a["shrink_factor"], 1.0 / 64.0)
        save_state(spec, state)

        if action == "stop:domain_budget":
            state["status"] = "stopped"
            state["stop_reason"] = "stopped:domain_budget"
            save_state(spec, state)
            print(
                f"[driver] step {step_no}: support fraction "
                f"{last['r99'] / diag['r_bdy']:.3f} > {a['support_fraction']} at "
                f"dr_max={dr:.5E}; stopping (design §6.1)"
            )
            return 0

        if action.startswith("regrid"):
            required = action != "regrid_coarser_optional"
            if action == "regrid_finer":
                new_dr = dr / 2.0
            else:
                new_dr = min(dr * 2.0, spec["grid"]["dr_max"])
            accepted = False
            tries = [new_dr]
            if required and new_dr != dr:
                mid = (dr + new_dr) / 2.0  # design §4 fallback: halve the move
                if dr < mid < new_dr:
                    tries.append(mid)
            for try_dr in tries:
                # The regrid consumes the next free step slot.
                ok, _ = do_regrid(spec, root, state, binary, step_no, try_dr, base_psi0)
                if ok:
                    accepted = True
                    current_grid(state, spec)["dr"] = try_dr
                    state["step_factor"] = 1.0  # fresh grid: restart step sizing
                    state["regrid_failures"] = 0
                    save_state(spec, state)
                    break
                if not required:
                    break  # rule 7 is optional: never blocks the campaign
                step_no += 1  # failed attempt consumed its slot; try the next dr
            if accepted:
                continue  # re-check exit conditions on the new grid
            # Regrid ladder exhausted: stay on the old grid, ease the step
            # (design §4: "fall back to stepping on the old grid with a
            # smaller Δψ₀").
            state["step_factor"] = max(factor * a["shrink_factor"], 1.0 / 64.0)
            if required:
                state["regrid_failures"] = state.get("regrid_failures", 0) + 1
            save_state(spec, state)
            print(f"[driver] regrid rejected; continuing on dr={dr:.5E} with smaller steps")
            if state.get("regrid_failures", 0) >= 3:
                # The support keeps violating the budget and every widening
                # regrid is rejected on accuracy (or fails): the domain
                # budget is exhausted in the §6.1 sense even though
                # dr < dr_max — wider grids cannot be trusted at this dr.
                state["status"] = "stopped"
                state["stop_reason"] = "stopped:domain_budget"
                save_state(spec, state)
                print(
                    f"[driver] step {step_no}: 3 consecutive widening regrids rejected; "
                    "stopping (design §6.1 domain budget)"
                )
                return 0
        else:
            state["regrid_failures"] = 0

        # Rule 9 with stop_at_turning_point = false: continue past the turning
        # point, sampling the branch bottom densely.
        if (
            not c.get("stop_at_turning_point", True)
            and c["direction"] == "up"
            and not state.get("fine_sampling")
            and detect_turning_point(state["steps"])
        ):
            state["fine_sampling"] = True
            save_state(spec, state)
            print("[driver] turning point crossed; switching to fine Δψ₀ sampling")

        print(
            f"[driver] step {step_no - 1}: ψ₀={last['psi0']:.6E} ω={last['omega']:.6E} "
            f"(guess ω≈{w_guess:.4f}, {last['newton_iters']} iters{note}) [{action}] "
            f"-> {Path(last['sol_dir']).name}"
        )


def summarize(spec: dict) -> int:
    """--summarize: localize ω_min from a finished campaign (design §6.2).

    Post-processing only: reads state.json, fits a low-order polynomial to
    ω(ψ₀) over the samples bracketing the smallest sampled ω, and reports the
    extremum (the paper used a 4th-degree spline). Writes <root>/summary.json.
    """
    state = load_state(spec)
    if state is None:
        print("[driver] no state.json to summarize", file=sys.stderr)
        return 1
    steps = state["steps"]
    psi0s = [s.get("psi0") for s in steps]
    omegas = [s.get("omega") for s in steps]
    if not any(w is not None for w in omegas):
        print("[driver] no completed steps record ω; nothing to summarize", file=sys.stderr)
        return 1
    report = turning_point_estimate(psi0s, omegas)
    report["campaign"] = str(spec["output"]["root"])
    report["status"] = state.get("status")
    report["stop_reason"] = state.get("stop_reason")
    out = Path(spec["output"]["root"]) / "summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[driver] summary: {json.dumps(report)}")
    print(f"[driver] wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign", type=Path, help="campaign spec TOML")
    ap.add_argument(
        "--fresh", action="store_true", help="discard any existing state.json and start over"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="render the seed-step parameter file and exit"
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="localize ω_min from the campaign's state.json (post-processing)",
    )
    args = ap.parse_args()

    try:
        spec = load_spec(args.campaign)
    except SpecError as e:
        print(f"[driver] invalid campaign spec: {e}", file=sys.stderr)
        return 3

    if args.summarize:
        return summarize(spec)
    return run_campaign(spec, fresh=args.fresh, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
