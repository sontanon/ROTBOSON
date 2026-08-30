"""Python sweep driver (core, non-adaptive) — SAN-17, design doc §3.

C solves one solution; Python drives. The continuation parameter is **ψ₀**
(the field value at the fixedPhi grid point): each step renders a parameter
file that seeds from the previous solution(s) and constrains
ψ(fixedPhi point) = ψ₀_target, so Newton solves ω as an eigenvalue. This keeps
ω(ψ₀) single-valued through the minimum-frequency turning point (design §1).

Usage:
    uv run tools/sweep_driver.py <campaign.toml> [--fresh] [--dry-run]

A campaign writes `state.json` (atomically, after every step) plus one
solution directory per step under `[output] root`. An interrupted campaign
resumes automatically from the last completed step; a changed spec aborts
resume (delete state.json or pass --fresh).

Non-adaptive scope (SAN-17): fixed Δψ₀ stepping, both directions, fixed grid,
exit conditions = fixed-grid subset of design §6.1 (ψ₀ target, ω target,
max steps, solver failure). Regrid/adaptive logic is SAN-14; golden-sequence
verification is SAN-13.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
}
SEED_KEYS = {"policy", "source", "w0", "psi0", "sigmaR", "sigmaZ", "rExt"}
GRID_KEYS = {"dr", "N", "order"}
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

KNOWN = {
    "campaign": CAMPAIGN_KEYS,
    "seed": SEED_KEYS,
    "grid": GRID_KEYS,
    "solver": SOLVER_KEYS,
    "output": OUTPUT_KEYS,
}

DEFAULTS = {
    "campaign": {
        "m": 1.0,
        "psi0_step_mode": "absolute",
        "max_retries": 3,
        "max_steps": 50,
        "fixedPhiR": 2,
        "fixedPhiZ": 2,
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
    }


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


def render_params(
    spec: dict,
    root: Path,
    step: int,
    *,
    scale_u4: float | None = None,
    seed_dir: Path | None = None,
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
            "readInitialData = 1",
        ]
        for param, field in SEED_PARAM_KEYS.items():
            lines.append(f'{param} = "{(seed_dir / field).resolve()}"')
        lines.append(f'w_i = "{(seed_dir / "w_f.asc").resolve()}"')
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


def run_binary(binary: Path, params: Path, root: Path, step: int) -> tuple[int, Path]:
    """Run ROTBOSON from the campaign root; returns (exit code, log path)."""
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"step{step:04d}.log"
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


def newton_iterations(sol_dir: Path, fmt: str) -> int | None:
    """Newton iteration count = length of the lambda history, when present."""
    try:
        if fmt == "hdf5" and (sol_dir / "solution.h5").exists():
            datasets, _ = read_hdf5(sol_dir / "solution.h5")
            if "lambda.asc" in datasets:
                return int(np.asarray(datasets["lambda.asc"]).size)
        else:
            from rotboson_io import read_1d

            f = sol_dir / "lambda.asc"
            if f.exists():
                return int(read_1d(f).size)
    except Exception:  # noqa: BLE001 — diagnostics only, never fail the step
        return None
    return None


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
            "hwl": None,
        }
        try:
            # ψ₀ from the field at the fixedPhi point (the constraint value).
            fields = solution_fields(sol_dir)
            entry["psi0"] = psi_at_fixed_point(fields["psi_f.asc"], spec)
        except Exception:  # noqa: BLE001 — a failed step may lack field data
            entry["psi0"] = None
        step.update(entry)
        step["newton_iters"] = newton_iterations(sol_dir, spec["output"]["format"])
    state["steps"].append(step)


def psi0_of_last(state: dict) -> float:
    for step in reversed(state["steps"]):
        if step.get("psi0") is not None:
            return float(step["psi0"])
    raise SystemExit("no completed step records ψ₀; cannot continue")


def omega_of_last(state: dict) -> float:
    for step in reversed(state["steps"]):
        if step.get("omega") is not None:
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
    if state["steps"][-1]["exit_code"] not in (0, None):
        code = state["steps"][-1]["exit_code"]
        reason = {1: "failed:newton", 2: "failed:solver", 3: "failed:config", 4: "failed:io"}
        return reason.get(code, f"failed:exit{code}")
    return None


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
                "hwl": None,
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

        absolute mode: fixed Δψ₀ per design §3.1. relative mode: fixed-ratio
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
        if c["direction"] == "up":
            return min(round(target, 15), c["psi0_target"])
        return max(round(target, 15), c["psi0_target"])

    while True:
        stop = finished(state, spec)
        if stop is not None:
            state["status"] = "done" if stop.startswith("done") else "failed"
            state["stop_reason"] = stop
            save_state(spec, state)
            print(f"[driver] stop: {stop}")
            return 0 if stop.startswith("done") else 1

        # Retry loop (decision-table rules 2-3, core subset): on Newton
        # non-convergence (exit 1) shrink the step and retry from the last
        # good solution. Solver/config/I-O errors are not retryable.
        base_psi0 = psi0_of_last(state)
        factor = 1.0
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

            if code == 1 and attempts < c["max_retries"] and psi0_target != base_psi0:
                attempts += 1
                factor /= 2.0
                # Drop the failed attempt from the step history (keep the log).
                state["steps"].pop()
                save_state(spec, state)
                print(
                    f"[driver] step {step_no} attempt {attempts} did not converge; "
                    f"shrinking step (factor {factor:.3f}), log: {log}"
                )
                continue

            state["status"] = "failed"
            state["stop_reason"] = finished(state, spec) or f"failed:exit{code}"
            save_state(spec, state)
            print(f"[driver] step {step_no} FAILED (exit {code}); log: {log}")
            return 1

        note = f", {attempts} shrink-retry" if attempts else ""
        print(
            f"[driver] step {step_no}: ψ₀={last['psi0']:.6E} ω={last['omega']:.6E} "
            f"(guess ω≈{w_guess:.4f}, {last['newton_iters']} iters{note}) "
            f"-> {Path(last['sol_dir']).name}"
        )
        step_no += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign", type=Path, help="campaign spec TOML")
    ap.add_argument(
        "--fresh", action="store_true", help="discard any existing state.json and start over"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="render the seed-step parameter file and exit"
    )
    args = ap.parse_args()

    try:
        spec = load_spec(args.campaign)
    except SpecError as e:
        print(f"[driver] invalid campaign spec: {e}", file=sys.stderr)
        return 3

    return run_campaign(spec, fresh=args.fresh, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
