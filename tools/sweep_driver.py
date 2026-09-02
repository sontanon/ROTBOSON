"""Python sweep driver with adaptive continuation stepping.

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

Adaptive layer (design §4–6):
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
`tests/test_driver_decisions.py`.
"""

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, Self, cast

import numpy as np
from logsetup import configure, get_logger
from rotboson_io import (
    extract_scalars,
    extract_scalars_from_hdf5,
    find_solution_dirs,
    is_solution_dir,
    read_hdf5,
)

REPO = Path(__file__).resolve().parent.parent

# Operational telemetry goes through logging (stderr); this module has no
# stdout report product (see logsetup's convention note).
logger = get_logger(__name__)

# Binary location: same search order as tools/smoke.py.
BUILD_PRESETS: Final[tuple[str, ...]] = ("release", "umfpack", "dev", "asan-ubsan")

# 2D field datasets used to build the next step's seed (final fields only).
SEED_FIELDS: Final[tuple[str, ...]] = (
    "log_alpha_f.asc",
    "beta_f.asc",
    "log_h_f.asc",
    "log_a_f.asc",
    "psi_f.asc",
    "lambda_f.asc",
)

SEED_PARAM_KEYS: Final[Mapping[str, str]] = {
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
# Closed vocabularies: the string sets that cross the JSON/TOML
# boundaries. Values match the legacy state.json / param-file spellings
# exactly — resume and rendering stay byte-compatible.
# ---------------------------------------------------------------------------


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"


class SeedPolicy(StrEnum):
    FROM_SCRATCH = "from_scratch"
    SOLUTION = "solution"


class Psi0StepMode(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class OutputFormat(StrEnum):
    HDF5 = "hdf5"
    ASCII = "ascii"


class StepMode(StrEnum):
    """How a step was produced (state.json `mode`)."""

    FIXED_PHI = "fixedPhi"
    SEED = "seed"
    REGGRID = "regrid"
    REGGRID_PROBE = "regrid-probe"


class Status(StrEnum):
    """Campaign lifecycle status (state.json `status`)."""

    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"
    FAILED = "failed"


class StopReason(StrEnum):
    """The closed stop reasons; dynamic ones come from the factories below.

    `stop_reason` is *stored* as a plain string in state.json because two
    reasons embed runtime values (signal name, raw exit code); the enum
    catalogues every fixed reason and keeps the dynamic spellings uniform.
    """

    DONE_PSI0_TARGET = "done:psi0_target"
    DONE_OMEGA_TARGET = "done:omega_target"
    DONE_MAX_STEPS = "done:max_steps"
    STOPPED_NEWTONIAN_LIMIT = "stopped:newtonian_limit"
    STOPPED_BOUNDARY = "stopped:boundary"
    STOPPED_TURNING_POINT = "stopped:turning_point"
    FAILED_TIMEOUT = "failed:timeout"
    FAILED_NEWTON = "failed:newton"
    FAILED_SOLVER = "failed:solver"
    FAILED_CONFIG = "failed:config"
    FAILED_IO = "failed:io"

    @staticmethod
    def signal(sig_name: str) -> str:
        return f"failed:sig{sig_name}"

    @staticmethod
    def exit_code(code: int) -> str:
        return f"failed:exit{code}"

    @staticmethod
    def seed_exit(code: int) -> str:
        return f"failed:seed_exit{code}"


def stop_family(reason: str) -> str:
    """'done' / 'stopped' / 'failed' — the component before the first ':'."""
    return reason.split(":", 1)[0]


class Action(StrEnum):
    """Adaptive decision layer output (`decide_action`)."""

    OK = "ok"
    REGGRID_FINER = "regrid_finer"


# ---------------------------------------------------------------------------
# Spec model: frozen dataclasses parsed + validated in one place.
# parse() classmethods are the single choke point raising SpecError; the
# messages match the previous dict-based validator verbatim. The spec hash
# is computed over the raw TOML text and is deliberately untouched
# by this model.
# ---------------------------------------------------------------------------


def _int(table: str, key: str, v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise SpecError(f"[{table}] '{key}' must be an integer")
    return v


def _num(table: str, key: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SpecError(f"[{table}] '{key}' must be a number")
    return float(v)


def _str(table: str, key: str, v: object) -> str:
    if not isinstance(v, str):
        raise SpecError(f"[{table}] '{key}' must be a string")
    return v


def _bool(table: str, key: str, v: object) -> bool:
    if not isinstance(v, bool):
        raise SpecError(f"[{table}] '{key}' must be a boolean")
    return v


# State-file coercions (from_dict): state.json is machine-written, but a
# corrupted/truncated file should fail with a clear message instead of a
# cryptic crash deep in a run.


def _st_float(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"state.json: expected a number, got {type(v).__name__}")
    return float(v)


def _st_int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"state.json: expected an integer, got {type(v).__name__}")
    return v


def _st_str(v: object) -> str:
    if not isinstance(v, str):
        raise ValueError(f"state.json: expected a string, got {type(v).__name__}")
    return v


def _st_opt_float(v: object) -> float | None:
    return None if v is None else _st_float(v)


def _st_opt_int(v: object) -> int | None:
    return None if v is None else _st_int(v)


def _st_opt_str(v: object) -> str | None:
    return None if v is None else _st_str(v)


def _st_scalar(v: object) -> float | int:
    """A JSON number that must keep its int/float identity on round-trip."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"state.json: expected a number, got {type(v).__name__}")
    return v


class _Tables:
    """TOML table names."""

    CAMPAIGN = "campaign"
    SEED = "seed"
    GRID = "grid"
    SOLVER = "solver"
    OUTPUT = "output"
    ADAPTIVITY = "adaptivity"


KNOWN: Final[Mapping[str, set[str]]] = {
    _Tables.CAMPAIGN: {
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
    },
    _Tables.SEED: {"policy", "source", "w0", "psi0", "sigmaR", "sigmaZ", "rExt"},
    _Tables.GRID: {"dr", "dr_max", "N", "order"},
    _Tables.SOLVER: {
        "solverType",
        "localSolver",
        "epsilon",
        "maxNewtonIter",
        "lambda0",
        "lambdaMin",
        "useLowRank",
    },
    _Tables.OUTPUT: {"root", "format"},
    _Tables.ADAPTIVITY: {
        "hwl_min",
        "max_refinements",
        "newtonian_delta",
        "boundary_fraction",
    },
}


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    l: int
    direction: Direction
    psi0_target: float
    psi0_step: float
    m: float = 1.0
    omega_target: float | None = None
    psi0_step_mode: Psi0StepMode = Psi0StepMode.RELATIVE
    max_retries: int = 3
    max_steps: int = 50
    fixedPhiR: int = 2
    fixedPhiZ: int = 2
    stop_at_turning_point: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Self:
        t = _Tables.CAMPAIGN
        for key in ("l", "direction", "psi0_target", "psi0_step"):
            if key not in raw:
                raise SpecError(f"[{t}] missing required key '{key}'")
        direction = raw["direction"]
        if direction not in (Direction.UP, Direction.DOWN):
            raise SpecError(f'[{t}] direction must be "up" or "down"')
        psi0_step = _num(t, "psi0_step", raw["psi0_step"])
        if not psi0_step > 0:
            raise SpecError(f"[{t}] psi0_step must be > 0")
        mode = raw.get("psi0_step_mode", Psi0StepMode.RELATIVE)
        if mode not in (Psi0StepMode.ABSOLUTE, Psi0StepMode.RELATIVE):
            raise SpecError(f'[{t}] psi0_step_mode must be "absolute" or "relative"')
        max_retries = _int(t, "max_retries", raw.get("max_retries", 3))
        if max_retries < 0:
            raise SpecError(f"[{t}] max_retries must be >= 0")
        stop_at_turning_point = _bool(
            t, "stop_at_turning_point", raw.get("stop_at_turning_point", True)
        )
        omega_target = raw.get("omega_target")
        return cls(
            l=_int(t, "l", raw["l"]),
            direction=Direction(direction),
            psi0_target=_num(t, "psi0_target", raw["psi0_target"]),
            psi0_step=psi0_step,
            m=_num(t, "m", raw.get("m", 1.0)),
            omega_target=None if omega_target is None else _num(t, "omega_target", omega_target),
            psi0_step_mode=Psi0StepMode(mode),
            max_retries=max_retries,
            max_steps=_int(t, "max_steps", raw.get("max_steps", 50)),
            fixedPhiR=_int(t, "fixedPhiR", raw.get("fixedPhiR", 2)),
            fixedPhiZ=_int(t, "fixedPhiZ", raw.get("fixedPhiZ", 2)),
            stop_at_turning_point=stop_at_turning_point,
        )


@dataclass(frozen=True, slots=True)
class SeedSpec:
    policy: SeedPolicy
    psi0: float = 0.01
    sigmaR: float = 4.0
    sigmaZ: float = 4.0
    rExt: float = 12.0
    source: str | None = None
    w0: float | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, object], base_dir: Path) -> Self:
        t = _Tables.SEED
        if "policy" not in raw:
            raise SpecError(f"[{t}] missing required key 'policy'")
        policy = raw["policy"]
        if policy not in (SeedPolicy.FROM_SCRATCH, SeedPolicy.SOLUTION):
            raise SpecError(f'[{t}] policy must be "from_scratch" or "solution"')
        source: str | None = None
        if policy == SeedPolicy.SOLUTION:
            if "source" not in raw:
                raise SpecError(f"[{t}] policy=solution requires 'source'")
            source = _str(t, "source", raw["source"])
            resolved = Path(source)
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve()
                source = str(resolved)
            if not is_solution_dir(resolved.name) or not resolved.is_dir():
                raise SpecError(f"[{t}] source is not a solution directory: {source}")
        w0 = raw.get("w0")
        if policy == SeedPolicy.FROM_SCRATCH and w0 is None:
            raise SpecError(f"[{t}] policy=from_scratch requires 'w0' (fixedOmega seed solve)")
        return cls(
            policy=SeedPolicy(policy),
            psi0=_num(t, "psi0", raw.get("psi0", 0.01)),
            sigmaR=_num(t, "sigmaR", raw.get("sigmaR", 4.0)),
            sigmaZ=_num(t, "sigmaZ", raw.get("sigmaZ", 4.0)),
            rExt=_num(t, "rExt", raw.get("rExt", 12.0)),
            source=source,
            w0=None if w0 is None else _num(t, "w0", w0),
        )


@dataclass(frozen=True, slots=True)
class GridSpec:
    dr: float
    N: int
    dr_max: float
    order: int = 4

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Self:
        t = _Tables.GRID
        if "dr" not in raw or "N" not in raw:
            raise SpecError(f"[{t}] missing required keys 'dr' and/or 'N'")
        dr = _num(t, "dr", raw["dr"])
        # Domain-growth budget (design §6.1): defaults to 4× the seed dr
        # (two coarsening regrids); going beyond requires an explicit override.
        dr_max = _num(t, "dr_max", raw.get("dr_max", 4.0 * dr))
        if dr_max < dr:
            raise SpecError(f"[{t}] dr_max must be >= dr")
        return cls(
            dr=dr,
            dr_max=dr_max,
            N=_int(t, "N", raw["N"]),
            order=_int(t, "order", raw.get("order", 4)),
        )


@dataclass(frozen=True, slots=True)
class SolverSpec:
    solverType: int = 1
    localSolver: int = 1
    epsilon: float = 1.0e-8
    maxNewtonIter: int = 50
    lambda0: float = 1.0e-3
    lambdaMin: float = 1.0e-5
    useLowRank: int = 0

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Self:
        t = _Tables.SOLVER
        return cls(
            solverType=_int(t, "solverType", raw.get("solverType", 1)),
            localSolver=_int(t, "localSolver", raw.get("localSolver", 1)),
            epsilon=_num(t, "epsilon", raw.get("epsilon", 1.0e-8)),
            maxNewtonIter=_int(t, "maxNewtonIter", raw.get("maxNewtonIter", 50)),
            lambda0=_num(t, "lambda0", raw.get("lambda0", 1.0e-3)),
            lambdaMin=_num(t, "lambdaMin", raw.get("lambdaMin", 1.0e-5)),
            useLowRank=_int(t, "useLowRank", raw.get("useLowRank", 0)),
        )


@dataclass(frozen=True, slots=True)
class AdaptivitySpec:
    # Design §5: defaults start conservative at the historical C sweep
    # thresholds (hwl_min/max, rr_phi_max floor) so early behaviour matches
    # what the old in-C sweep tolerated.
    hwl_min: float = 8
    max_refinements: int = 2
    newtonian_delta: float = 1.0e-2
    boundary_fraction: float = 0.95

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Self:
        t = _Tables.ADAPTIVITY
        hwl_min = _num(t, "hwl_min", raw.get("hwl_min", 8))
        if hwl_min <= 0:
            raise SpecError(f"[{t}] hwl_min must be > 0")
        max_refinements = _int(t, "max_refinements", raw.get("max_refinements", 2))
        if max_refinements < 0:
            raise SpecError(f"[{t}] max_refinements must be >= 0")
        newtonian_delta = _num(t, "newtonian_delta", raw.get("newtonian_delta", 1.0e-2))
        if newtonian_delta < 0:
            raise SpecError(f"[{t}] newtonian_delta must be >= 0")
        boundary_fraction = _num(t, "boundary_fraction", raw.get("boundary_fraction", 0.95))
        if not 0.0 < boundary_fraction <= 1.0:
            raise SpecError(f"[{t}] boundary_fraction must be in (0, 1]")
        return cls(
            hwl_min=hwl_min,
            max_refinements=max_refinements,
            newtonian_delta=newtonian_delta,
            boundary_fraction=boundary_fraction,
        )


@dataclass(frozen=True, slots=True)
class OutputSpec:
    root: Path
    format: OutputFormat = OutputFormat.HDF5

    @classmethod
    def parse(cls, raw: Mapping[str, object], spec_path: Path) -> Self:
        t = _Tables.OUTPUT
        fmt = raw.get("format", OutputFormat.HDF5)
        if fmt not in (OutputFormat.HDF5, OutputFormat.ASCII):
            raise SpecError(f'[{t}] format must be "hdf5" or "ascii"')
        root = raw.get("root")
        if root is None:
            resolved = REPO / "out" / "campaigns" / spec_path.stem
        else:
            resolved = Path(_str(t, "root", root))
            if not resolved.is_absolute():
                resolved = (spec_path.parent / resolved).resolve()
        return cls(root=resolved, format=OutputFormat(fmt))


@dataclass(frozen=True, slots=True)
class Spec:
    """A fully-validated campaign spec (the former `spec: dict`)."""

    campaign: CampaignSpec
    seed: SeedSpec
    grid: GridSpec
    solver: SolverSpec
    output: OutputSpec
    adaptivity: AdaptivitySpec
    spec_hash: str

    def with_grid(self, dr: float) -> Spec:
        """Copy with the grid's dr replaced (regrid rendering)."""
        return replace(self, grid=replace(self.grid, dr=dr))


def load_spec(path: Path) -> Spec:
    """Load and validate a campaign spec; returns the fully-defaulted model."""
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

    spec = Spec(
        campaign=CampaignSpec.parse(raw.get("campaign", {})),
        seed=SeedSpec.parse(raw.get("seed", {}), base_dir=path.parent),
        grid=GridSpec.parse(raw.get("grid", {})),
        solver=SolverSpec.parse(raw.get("solver", {})),
        output=OutputSpec.parse(raw.get("output", {}), spec_path=path),
        adaptivity=AdaptivitySpec.parse(raw.get("adaptivity", {})),
        spec_hash="",
    )

    # Spec hash: content hash with runtime control keys (max_steps,
    # max_retries) excluded — raising a limit must not invalidate the physics
    # state of a running campaign; any physics-affecting change does.
    # The hash ignores runtime control keys AND comment lines — commentary
    # edits must not invalidate a running campaign's resume.
    control = re.compile(r"^\s*(max_steps|max_retries)\s*=.*$", re.MULTILINE)
    comments = re.compile(r"^\s*#.*$", re.MULTILINE)
    canon = comments.sub("", control.sub("", raw_bytes.decode())).encode()
    return replace(spec, spec_hash=hashlib.sha256(canon).hexdigest())


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
#
# The state.json plumbing is frozen dataclasses. `to_dict` emits keys
# in exactly the order the previous dict-based writer produced them, so a
# state.json written by this code re-serializes byte-identically (round-trip
# unit-tested against an archived state.json). `from_dict` is deliberately
# lenient: legacy state.json files (early format, missing keys) load and
# resume. Resume is a one-way upgrade: a legacy file re-saved by this code
# gains the newer keys.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridPosition:
    """The campaign's current grid (dr drifts via refinements)."""

    dr: float
    N: int

    def to_dict(self) -> dict[str, object]:
        return {"dr": self.dr, "N": self.N}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        return cls(dr=_st_float(raw.get("dr", 0.0)), N=_st_int(raw.get("N", 0)))


@dataclass(frozen=True, slots=True)
class RegridInfo:
    """Accepted-regrid provenance attached to a StepRecord."""

    from_dr: float
    to_dr: float
    source: str | None
    rel_diff: dict[str, float]
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "from_dr": self.from_dr,
            "to_dr": self.to_dr,
            "source": self.source,
            "rel_diff": self.rel_diff,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        rel = raw.get("rel_diff", {})
        return cls(
            from_dr=_st_float(raw.get("from_dr", 0.0)),
            to_dr=_st_float(raw.get("to_dr", 0.0)),
            source=_st_opt_str(raw.get("source")),
            rel_diff={k: _st_float(v) for k, v in rel.items()} if isinstance(rel, Mapping) else {},
            accepted=bool(raw.get("accepted", False)),
        )


@dataclass(frozen=True, slots=True)
class RegridProbe:
    """A rejected/skipped regrid attempt (state.json `rejected_regrids`).

    Two legacy shapes share this model: a solved-but-rejected probe
    (scale_u4/log/sol_dir/scalars/psi0) and a support-fit skip (skipped/
    from_dr/to_dr). `to_dict` emits present fields in the legacy insertion
    order so both round-trip byte-compatibly.
    """

    i: int
    mode: StepMode
    exit_code: int | None
    scale_u4: float | None = None
    log: str | None = None
    sol_dir: str | None = None
    scalars: dict[str, float | int] | None = None
    psi0: float | None = None
    skipped: str | None = None
    from_dr: float | None = None
    to_dr: float | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"i": self.i, "mode": self.mode, "exit_code": self.exit_code}
        if self.sol_dir is not None:
            # psi0 is always written once a solution exists (null on read failure).
            out.update(
                scale_u4=self.scale_u4,
                log=self.log,
                sol_dir=self.sol_dir,
                scalars=self.scalars,
                psi0=self.psi0,
            )
        else:
            extras = {
                "scale_u4": self.scale_u4,
                "log": self.log,
                "skipped": self.skipped,
                "from_dr": self.from_dr,
                "to_dr": self.to_dr,
            }
            out.update({k: v for k, v in extras.items() if v is not None})
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        scalars = raw.get("scalars")
        return cls(
            i=_st_int(raw.get("i", 0)),
            mode=StepMode(_st_str(raw.get("mode", "regrid-probe"))),
            exit_code=_st_opt_int(raw.get("exit_code")),
            scale_u4=_st_opt_float(raw.get("scale_u4")),
            log=_st_opt_str(raw.get("log")),
            sol_dir=_st_opt_str(raw.get("sol_dir")),
            scalars={k: _st_scalar(v) for k, v in scalars.items()}
            if isinstance(scalars, Mapping)
            else None,
            psi0=_st_opt_float(raw.get("psi0")),
            skipped=_st_opt_str(raw.get("skipped")),
            from_dr=_st_opt_float(raw.get("from_dr")),
            to_dr=_st_opt_float(raw.get("to_dr")),
        )


@dataclass(frozen=True, slots=True)
class FoldMeasurement:
    """A kept measurement at a fold (state.json `fold_fine_grid_measurement`)."""

    psi0: float | None
    omega: float | None
    dr: float
    sol_dir: str
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "psi0": self.psi0,
            "omega": self.omega,
            "dr": self.dr,
            "sol_dir": self.sol_dir,
        }
        if self.note is not None:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        return cls(
            psi0=_st_opt_float(raw.get("psi0")),
            omega=_st_opt_float(raw.get("omega")),
            dr=_st_float(raw.get("dr", 0.0)),
            sol_dir=_st_str(raw.get("sol_dir", "")),
            note=_st_opt_str(raw.get("note")),
        )


@dataclass(frozen=True, slots=True)
class PendingRefinement:
    """An adopted-but-unverified refinement (state.json `pending_refinement`)."""

    from_dr: float
    regrid_step: int
    measurement: FoldMeasurement

    def to_dict(self) -> dict[str, object]:
        return {
            "from_dr": self.from_dr,
            "regrid_step": self.regrid_step,
            "measurement": self.measurement.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        measurement = raw.get("measurement")
        return cls(
            from_dr=_st_float(raw.get("from_dr", 0.0)),
            regrid_step=_st_int(raw.get("regrid_step", 0)),
            measurement=(
                FoldMeasurement.from_dict(measurement)
                if isinstance(measurement, Mapping)
                else FoldMeasurement(psi0=None, omega=None, dr=0.0, sol_dir="")
            ),
        )


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One campaign step (state.json `steps[i]`).

    Optional scalars are None exactly when the legacy writer omitted the key:
    a failed step records only (i, exit_code, mode [+ psi0_target]); a step
    with a solution directory always carries the scalar block (values may be
    null when a file was unreadable).
    """

    i: int
    exit_code: int
    mode: StepMode = StepMode.FIXED_PHI
    sol_dir: str | None = None
    dr: float | None = None
    N: int | None = None
    omega: float | None = None
    M_Komar: float | None = None
    J_Komar: float | None = None
    rr_phi_max: float | None = None
    r99: float | None = None
    hwl: float | None = None
    psi0: float | None = None
    newton_iters: int | None = None
    norm_f: float | None = None
    psi0_target: float | None = None
    regrid: RegridInfo | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"i": self.i, "exit_code": self.exit_code, "mode": self.mode}
        if self.sol_dir is not None:
            out.update(
                sol_dir=self.sol_dir,
                dr=self.dr,
                N=self.N,
                omega=self.omega,
                M_Komar=self.M_Komar,
                J_Komar=self.J_Komar,
                rr_phi_max=self.rr_phi_max,
                r99=self.r99,
                hwl=self.hwl,
                psi0=self.psi0,
            )
        if self.newton_iters is not None:
            out["newton_iters"] = self.newton_iters
        if self.norm_f is not None:
            out["norm_f"] = self.norm_f
        if self.psi0_target is not None:
            out["psi0_target"] = self.psi0_target
        if self.regrid is not None:
            out["regrid"] = self.regrid.to_dict()
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        regrid = raw.get("regrid")
        return cls(
            i=_st_int(raw.get("i", 0)),
            exit_code=_st_int(raw.get("exit_code", 0)),
            mode=StepMode(_st_str(raw.get("mode", "fixedPhi"))),
            sol_dir=_st_opt_str(raw.get("sol_dir")),
            dr=_st_opt_float(raw.get("dr")),
            N=_st_opt_int(raw.get("N")),
            omega=_st_opt_float(raw.get("omega")),
            M_Komar=_st_opt_float(raw.get("M_Komar")),
            J_Komar=_st_opt_float(raw.get("J_Komar")),
            rr_phi_max=_st_opt_float(raw.get("rr_phi_max")),
            r99=_st_opt_float(raw.get("r99")),
            hwl=_st_opt_float(raw.get("hwl")),
            psi0=_st_opt_float(raw.get("psi0")),
            newton_iters=_st_opt_int(raw.get("newton_iters")),
            norm_f=_st_opt_float(raw.get("norm_f")),
            psi0_target=_st_opt_float(raw.get("psi0_target")),
            regrid=RegridInfo.from_dict(regrid) if isinstance(regrid, Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Campaign resume state (state.json)."""

    spec_hash: str
    spec_file: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    status: Status = Status.RUNNING
    stop_reason: str | None = None
    grid: GridPosition | None = None
    refinements_left: int | None = None
    turning_point: dict[str, object] | None = None
    rejected_regrids: list[RegridProbe] | None = None
    pending_refinement: PendingRefinement | None = None
    fold_fine_grid_measurement: list[FoldMeasurement] | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "spec_hash": self.spec_hash,
            "spec_file": self.spec_file,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "stop_reason": self.stop_reason,
        }
        if self.grid is not None:
            out["grid"] = self.grid.to_dict()
        if self.refinements_left is not None:
            out["refinements_left"] = self.refinements_left
        out["turning_point"] = self.turning_point
        if self.rejected_regrids is not None:
            out["rejected_regrids"] = [p.to_dict() for p in self.rejected_regrids]
        out["pending_refinement"] = (
            self.pending_refinement.to_dict() if self.pending_refinement else None
        )
        if self.fold_fine_grid_measurement is not None:
            out["fold_fine_grid_measurement"] = [
                m.to_dict() for m in self.fold_fine_grid_measurement
            ]
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        # Legacy files (pre-regrid format) lack grid/refinements_left and the
        # provenance lists: they stay None and the effective values come
        # from the spec via current_grid().
        grid = raw.get("grid")
        rejected = raw.get("rejected_regrids")
        fold = raw.get("fold_fine_grid_measurement")
        pending = raw.get("pending_refinement")
        turning_point = raw.get("turning_point")
        steps = raw.get("steps", [])
        return cls(
            spec_hash=_st_str(raw.get("spec_hash", "")),
            spec_file=_st_opt_str(raw.get("spec_file")),
            steps=[StepRecord.from_dict(s) for s in cast("list[Mapping[str, object]]", steps)]
            if isinstance(steps, list)
            else [],
            status=Status(_st_str(raw.get("status", "running"))),
            stop_reason=_st_opt_str(raw.get("stop_reason")),
            grid=GridPosition.from_dict(grid) if isinstance(grid, Mapping) else None,
            refinements_left=_st_opt_int(raw.get("refinements_left")),
            turning_point=dict(cast("Mapping[str, object]", turning_point))
            if isinstance(turning_point, Mapping)
            else None,
            rejected_regrids=[
                RegridProbe.from_dict(p) for p in cast("list[Mapping[str, object]]", rejected)
            ]
            if isinstance(rejected, list)
            else None,
            pending_refinement=PendingRefinement.from_dict(pending)
            if isinstance(pending, Mapping)
            else None,
            fold_fine_grid_measurement=[
                FoldMeasurement.from_dict(m) for m in cast("list[Mapping[str, object]]", fold)
            ]
            if isinstance(fold, list)
            else None,
        )

    def with_rejected_probe(self, probe: RegridProbe) -> CampaignState:
        """Append to `rejected_regrids` (creating the list on first use)."""
        return replace(self, rejected_regrids=[*(self.rejected_regrids or []), probe])

    def with_fold_measurement(self, measurement: FoldMeasurement) -> CampaignState:
        """Append to `fold_fine_grid_measurement` (creating the list on first use)."""
        return replace(
            self, fold_fine_grid_measurement=[*(self.fold_fine_grid_measurement or []), measurement]
        )


def state_path(spec: Spec) -> Path:
    return spec.output.root / "state.json"


def load_state(spec: Spec) -> CampaignState | None:
    p = state_path(spec)
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"state.json at {p} is corrupt: top level is not a JSON object")
    if raw.get("spec_hash") != spec.spec_hash:
        raise SystemExit(
            f"state.json at {p} was written for a different spec version; "
            "pass --fresh to discard it."
        )
    return CampaignState.from_dict(raw)


def save_state(spec: Spec, state: CampaignState) -> None:
    p = state_path(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2) + "\n")
    tmp.replace(p)  # atomic on POSIX


def fresh_state(spec: Spec) -> CampaignState:
    return CampaignState(
        spec_hash=spec.spec_hash,
        # Adaptive state: the grid can drift from the spec's
        # initial value via refinements; refinements_left bounds them.
        grid=GridPosition(dr=spec.grid.dr, N=spec.grid.N),
        refinements_left=spec.adaptivity.max_refinements,
    )


def current_grid(state: CampaignState, spec: Spec) -> GridPosition:
    """Effective grid: the state's tracked value, else the spec's initial grid.

    Backwards compatible with state.json files that predate regrids
    (which lack the `grid` key entirely).
    """
    return state.grid or GridPosition(dr=spec.grid.dr, N=spec.grid.N)


# ---------------------------------------------------------------------------
# Solution reading (HDF5 or ASCII, both handled via rotboson_io)
# ---------------------------------------------------------------------------


def solution_scalars(sol_dir: Path) -> dict[str, float | int]:
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


def psi_at_fixed_point(psi: np.ndarray, spec: Spec) -> float:
    """ψ at the fixedPhi grid point — the value the C constraint enforces."""
    c = spec.campaign
    return float(psi[c.fixedPhiR, c.fixedPhiZ])


# ---------------------------------------------------------------------------
# Seed rendering (design §3.4): linear extrapolation in ψ₀ + exact rescale
# ---------------------------------------------------------------------------


def write_seed_field(path: Path, data: np.ndarray) -> None:
    # Whitespace-agnostic on the C side (fscanf %lE); use the ASCII layout.
    with path.open("w") as fh:
        for row in data:
            fh.write("\t".join(f"{v:9.18E}" for v in row) + "\n")


def render_seed(
    spec: Spec, root: Path, prev: list[StepRecord], psi0_target: float, extrapolate: bool = True
) -> tuple[float, float]:
    """Build the seed files for the next step.

    Base guess = the previous converged solution with ψ rescaled so its value
    at the fixedPhi point is exactly ψ0_target — the golden fixedPhi-ladder
    semantics (scale_u4-only). Linear extrapolation across the last two
    *continuation* steps (design §3.4) kicks in once both predecessors are
    fixedPhi solves; mixing the fixedOmega seed solve into the extrapolation
    empirically produces guesses Newton cannot recover from.

    `extrapolate=False` forces the plain rescale — used by the retry loop
    after an attempt failed: in marginal regions the extrapolated guess
    diverges Newton while the rescaled one converges (empirically the same
    target went exit 2 with extrapolation and exit 0 without).
    Returns (scale_u4, omega_guess).
    """
    seed_dir = root / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cur = prev[-1]
    assert cur.sol_dir is not None
    fields = solution_fields(Path(cur.sol_dir))

    can_extrapolate = (
        extrapolate
        and len(prev) >= 2
        and prev[-1].mode == StepMode.FIXED_PHI
        and prev[-2].mode == StepMode.FIXED_PHI
        and prev[-1].psi0 != prev[-2].psi0
    )
    if can_extrapolate:
        assert cur.psi0 is not None and cur.omega is not None
        assert prev[-2].psi0 is not None and prev[-2].omega is not None
        assert prev[-2].sol_dir is not None
        older = solution_fields(Path(prev[-2].sol_dir))
        ratio = (psi0_target - cur.psi0) / (cur.psi0 - prev[-2].psi0)
        for name in SEED_FIELDS:
            fields[name] = fields[name] + ratio * (fields[name] - older[name])
        w_guess = cur.omega + ratio * (cur.omega - prev[-2].omega)
    else:
        assert cur.omega is not None
        w_guess = cur.omega

    psi_fixed = psi_at_fixed_point(fields["psi_f.asc"], spec)
    if psi_fixed == 0:
        raise SystemExit("seed ψ at the fixedPhi point is zero; cannot rescale")
    scale_u4 = psi0_target / psi_fixed

    for name in SEED_FIELDS:
        write_seed_field(seed_dir / name, fields[name])
    # ω guess for the eigenvalue (C scales it by scale_u6, left at 1.0).
    (seed_dir / "w_f.asc").write_text(f"{w_guess:9.18E}\n")

    return scale_u4, w_guess


@dataclass(frozen=True, slots=True)
class InitialGrid:
    """Source-grid description for an interpolated regrid restart (C side)."""

    NrTotalInitial: int
    NzTotalInitial: int
    order_i: int
    ghost_i: int
    dr_i: float
    dz_i: float


def render_params(
    spec: Spec,
    root: Path,
    step: int,
    *,
    scale_u4: float | None = None,
    seed_dir: Path | None = None,
    initial_grid: InitialGrid | None = None,
) -> Path:
    """Render the per-step parameter file into the campaign root."""
    c, grid, solver = spec.campaign, spec.grid, spec.solver
    lines = [
        f"# Rendered by tools/sweep_driver.py — step {step}, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "# GRID",
        f"dr = {grid.dr:.6E}",
        f"dz = {grid.dr:.6E}",
        f"NrInterior = {grid.N}",
        f"NzInterior = {grid.N}",
        f"order = {grid.order}",
        "",
        "# SCALAR FIELD PROPERTIES",
        f"l = {c.l}",
        f"m = {c.m}",
        "",
    ]

    seed = spec.seed
    if seed.policy == SeedPolicy.FROM_SCRATCH and step == 0:
        lines += [
            "# INITIAL DATA (analytic guess, seed solve at fixed ω)",
            "readInitialData = 0",
            f"psi0 = {seed.psi0:.6E}",
            f"sigmaR = {seed.sigmaR:.6E}",
            f"sigmaZ = {seed.sigmaZ:.6E}",
            f"rExt = {seed.rExt:.6E}",
            "",
            "# INITIAL FREQUENCY (fixed for the seed solve)",
            f"w0 = {seed.w0:.6E}",
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
        # seed_dir is only read on this branch; the from_scratch step-0
        # branch renders analytic initial data instead.
        assert seed_dir is not None, "seed_dir required when seeding from files"
        for param, field in SEED_PARAM_KEYS.items():
            lines.append(f'{param} = "{(seed_dir / field).resolve()}"')
        lines.append(f'w_i = "{(seed_dir / "w_f.asc").resolve()}"')
        if initial_grid is not None:
            lines += [
                f"NrTotalInitial = {initial_grid.NrTotalInitial}",
                f"NzTotalInitial = {initial_grid.NzTotalInitial}",
                f"order_i = {initial_grid.order_i}",
                f"ghost_i = {initial_grid.ghost_i}",
                f"dr_i = {initial_grid.dr_i:.6E}",
                f"dz_i = {initial_grid.dz_i:.6E}",
            ]
        lines += [
            "",
            "# Scale the field so ψ at the fixedPhi point is exactly ψ₀_target",
            f"scale_u4 = {scale_u4:.10E}",
            "",
            "fixedPhi = 1",
            f"fixedPhiR = {c.fixedPhiR}",
            f"fixedPhiZ = {c.fixedPhiZ}",
            "fixedOmega = 0",
        ]

    lines += [
        "",
        "# SOLVER PARAMETERS",
        f"solverType = {solver.solverType}",
        f"localSolver = {solver.localSolver}",
        f"epsilon = {solver.epsilon:.6E}",
        f"maxNewtonIter = {solver.maxNewtonIter}",
        f"lambda0 = {solver.lambda0:.6E}",
        f"lambdaMin = {solver.lambdaMin:.6E}",
        f"useLowRank = {solver.useLowRank}",
        "",
        f'outputFormat = "{spec.output.format}"',
    ]
    if spec.output.format == OutputFormat.ASCII:
        lines.append('loglevel = "warn"')

    out = root / f"step{step:04d}.toml"
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Running one step
# ---------------------------------------------------------------------------

# Sentinel exit code for a step that hit run_binary's wall-clock timeout (not
# a real signal — chosen far from any errno/signal value).
TIMEOUT_EXIT = -999


def run_binary(
    binary: Path,
    params: Path,
    root: Path,
    step: int,
    label: str | None = None,
    timeout_s: float = 3600.0,
) -> tuple[int, Path]:
    """Run ROTBOSON from the campaign root; returns (exit code, log path).

    A wall-clock timeout returns TIMEOUT_EXIT instead of raising, so a hung
    solver is recorded like any other failure.
    """
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"step{step if label is None else label}.log"
    t0 = time.monotonic()
    with log.open("w") as lf:
        lf.write(
            f"# ROTBOSON {binary} {params}\n# start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        lf.flush()
        try:
            proc = subprocess.run(
                [str(binary), str(params)],
                cwd=root,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            wall = time.monotonic() - t0
            lf.write(f"# TIMEOUT after {timeout_s:.0f}s (wall={wall:.1f}s)\n")
            return TIMEOUT_EXIT, log
    wall = time.monotonic() - t0
    with log.open("a") as lf:
        lf.write(f"# exit_code={proc.returncode} wall={wall:.1f}s\n")
    return proc.returncode, log


def initial_dirname(spec: Spec) -> str:
    """The pre-rename output dir name the C binary writes (parser convention)."""
    c, grid = spec.campaign, spec.grid
    return f"l={c.l},w=X.XXXXXE-01,dr={grid.dr:.5E},N={grid.N:04d}"


def new_solution_dir(root: Path, before: set[Path], spec: Spec, step: int) -> Path | None:
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
    spec: Spec,
    root: Path,
    state: CampaignState,
    binary: Path,
    step_no: int,
    new_dr: float,
) -> tuple[bool, StepRecord | None, CampaignState]:
    """Re-solve the *same* ψ₀ on a grid with dr → `new_dr` (design §4).

    Seeds through the C interpolator (readInitialData = 3) from the last good
    solution and constrains ψ(fixedPhi point) = ψ₀ as usual. v2 only refines
    (dr ÷2): acceptance is convergence + the exact ψ₀ landing, and the
    recorded ω / M_Komar / J_Komar differences are the old grid's error.

    The C freezes the Newton update at the fixedPhi point, so the enforced ψ₀
    is the *interpolated seed's* value there — which can drift from the
    requested ψ₀ (the C bicubic overshoots near the axis on coarse seeds).
    The frozen value scales exactly linearly with scale_u4, so the first run
    measures the drift and a second run with a corrected scale lands exactly
    on ψ₀.

    Only an *accepted* re-grid is recorded as a step (mode "regrid") — it is
    the same branch point as the source, not a new one. Failed or rejected
    attempts go to state['rejected_regrids'] for provenance; the caller
    stays on the current grid and counts the attempt against
    max_refinements.

    Returns (accepted, step_record, updated_state).
    """
    # Source = last good solution that is not a rejected regrid attempt:
    # seeding from a rejected attempt would chase its drifted branch point.
    src: StepRecord | None = None
    for s in reversed(state.steps):
        if s.psi0 is None or s.exit_code != 0:
            continue
        if s.mode == StepMode.REGGRID and (s.regrid is None or not s.regrid.accepted):
            continue
        src = s
        break
    if src is None:
        return False, None, state

    # Re-solve the SOURCE's ψ₀ — not the last step's value, which a failed
    # attempt may have moved.
    assert src.psi0 is not None and src.dr is not None and src.N is not None
    base_psi0 = float(src.psi0)
    src_dr, src_n = float(src.dr), int(src.N)
    order = spec.grid.order

    # Finer regrids shrink the domain (N fixed): never amputate the field —
    # if the support would not fit in the new domain, the attempt is futile.
    if new_dr < src_dr and src.r99 is not None:
        new_domain = (src_n + 2 * ghost_of(order)) * new_dr
        if src.r99 > spec.adaptivity.boundary_fraction * new_domain:
            logger.warning(
                "regrid step %d: skipped — support r99=%.3g would not fit in the %.3g domain",
                step_no,
                src.r99,
                new_domain,
            )
            state = state.with_rejected_probe(
                RegridProbe(
                    i=step_no,
                    mode=StepMode.REGGRID_PROBE,
                    exit_code=None,
                    skipped="support would not fit",
                    from_dr=src_dr,
                    to_dr=new_dr,
                )
            )
            save_state(spec, state)
            return False, None, state

    initial_grid = InitialGrid(
        NrTotalInitial=src_n + 2 * ghost_of(order),
        NzTotalInitial=src_n + 2 * ghost_of(order),
        order_i=order,
        ghost_i=ghost_of(order),
        dr_i=src_dr,
        dz_i=src_dr,
    )
    spec2 = spec.with_grid(new_dr)
    rejected: list[RegridProbe] = []

    scale_u4, _ = render_seed(spec, root, [src], base_psi0)
    rec: RegridProbe | None = None
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
        probe = RegridProbe(
            i=step_no,
            mode=StepMode.REGGRID_PROBE,
            exit_code=code,
            scale_u4=scale_u4,
            log=str(log),
        )
        if sol is not None:
            probe = replace(
                probe,
                sol_dir=str(sol),
                scalars=solution_scalars(sol),
                psi0=_probe_psi0(sol, spec2),
            )

        achieved = probe.psi0
        if attempt == 0 and code == 0 and achieved is not None:
            drift = abs(achieved - base_psi0) / abs(base_psi0)
            if drift > 1.0e-9:
                # Frozen constraint value is linear in scale_u4: one
                # correction lands the re-solve exactly on ψ₀.
                rejected.append(probe)
                scale_u4 *= base_psi0 / achieved
                logger.warning(
                    "regrid step %d: interpolated constraint drifted %.2e; "
                    "correcting scale_u4 → %.10E",
                    step_no,
                    drift,
                    scale_u4,
                )
                continue
        rec = probe
        break

    for p in rejected:
        state = state.with_rejected_probe(p)

    accepted = False
    rel: dict[str, float] = {}
    rec_step: StepRecord | None = None
    if rec is not None and rec.exit_code == 0 and rec.psi0 is not None and rec.sol_dir is not None:
        for key, fname in (
            ("omega", "w_f.asc"),
            ("M_Komar", "M_Komar1.asc"),
            ("J_Komar", "J_Komar1.asc"),
        ):
            old, new = getattr(src, key), (rec.scalars or {}).get(fname)
            if old is None or new is None or abs(old) == 0:
                rel = {}
                break
            rel[key] = abs(new - old) / abs(old)
        accepted = bool(rel)

    if accepted and rec is not None:
        # Promote the accepted re-solve to a real branch-point step.
        assert rec.sol_dir is not None
        record_step(state, spec2, step_no, Path(rec.sol_dir), 0, mode=StepMode.REGGRID)
        rec_step = state.steps[-1]
        state = replace(
            state,
            steps=[
                *state.steps[:-1],
                replace(
                    state.steps[-1],
                    regrid=RegridInfo(
                        from_dr=src_dr,
                        to_dr=new_dr,
                        source=src.sol_dir,
                        rel_diff=rel,
                        accepted=True,
                    ),
                ),
            ],
        )
        save_state(spec, state)
        diffs = ", ".join(f"{k}={v:.2e}" for k, v in rel.items())
        logger.info(
            "regrid step %d: dr %.5E → %.5E accepted: finer (old-grid error recorded) (%s)",
            step_no,
            src_dr,
            new_dr,
            diffs,
        )
    else:
        if rec is not None:
            state = state.with_rejected_probe(rec)
        save_state(spec, state)
        logger.warning(
            "regrid step %d: dr %.5E → %.5E REJECTED (exit %s, log: %s)",
            step_no,
            src_dr,
            new_dr,
            rec.exit_code if rec else "n/a",
            rec.log if rec else "n/a",
        )
    return accepted, rec_step, state


def _probe_psi0(sol: Path, spec: Spec) -> float | None:
    """ψ₀ achieved by a regrid probe (None when fields are unreadable)."""
    try:
        fields = solution_fields(sol)
        return psi_at_fixed_point(fields["psi_f.asc"], spec)
    except Exception:  # noqa: BLE001 — probe diagnostics only
        return None


def newton_health(sol_dir: Path, fmt: OutputFormat) -> dict[str, float | int]:
    """Newton health from the iteration histories (design §5 diagnostics).

    Returns {} when the histories are unavailable (failed step). `lambda_min`
    is the smallest damping factor over the tail of the λ history — a healthy
    NLEQ-ERR run walks λ up toward 1, so a λ still stuck near `lambdaMin` at
    the end flags a step that converged only grudgingly.
    """
    try:
        datasets: dict[str, np.ndarray] | None = None
        if fmt == OutputFormat.HDF5 and (sol_dir / "solution.h5").exists():
            datasets, _ = read_hdf5(sol_dir / "solution.h5")
        health: dict[str, float | int] = {}
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
    state: CampaignState,
    spec: Spec,
    i: int,
    sol_dir: Path | None,
    exit_code: int,
    mode: StepMode = StepMode.FIXED_PHI,
    psi0_target: float | None = None,
) -> StepRecord:
    step = StepRecord(i=i, exit_code=exit_code, mode=mode, psi0_target=psi0_target)
    if sol_dir is not None:
        scalars = solution_scalars(sol_dir)
        health = newton_health(sol_dir, spec.output.format)
        psi0: float | None
        try:
            # ψ₀ from the field at the fixedPhi point (the constraint value).
            fields = solution_fields(sol_dir)
            psi0 = psi_at_fixed_point(fields["psi_f.asc"], spec)
        except Exception:  # noqa: BLE001 — a failed step may lack field data
            psi0 = None
        step = replace(
            step,
            sol_dir=str(sol_dir),
            dr=spec.grid.dr,
            N=spec.grid.N,
            omega=scalars.get("w_f.asc"),
            M_Komar=scalars.get("M_Komar1.asc"),
            J_Komar=scalars.get("J_Komar1.asc"),
            rr_phi_max=scalars.get("rr_phi_max.asc"),
            r99=scalars.get("r99.asc"),
            hwl=scalars.get("hwl_resolution.asc"),
            psi0=psi0,
            newton_iters=(int(health["newton_iters"]) if "newton_iters" in health else None),
            norm_f=health.get("norm_f"),
        )
    state.steps.append(step)
    return step


def psi0_of_last(state: CampaignState) -> float:
    for step in reversed(state.steps):
        if step.psi0 is not None and step.exit_code == 0:
            return float(step.psi0)
    raise SystemExit("no completed step records ψ₀; cannot continue")


def omega_of_last(state: CampaignState) -> float:
    for step in reversed(state.steps):
        if step.omega is not None and step.exit_code == 0:
            return float(step.omega)
    raise SystemExit("no completed step records ω; cannot continue")


def newtonian_limit_stop(state: CampaignState, spec: Spec) -> str | None:
    """Is a failed step the branch's physical Newtonian end?

    In the down direction ω → m as ψ₀ → 0; the field extends without bound
    and the Jacobian becomes singular, so the solver exits 2 near ω = m. That
    is the branch's physical termination, not a backend failure: return the
    `stopped:newtonian_limit` stop reason when the last step failed with exit
    2 and the last *completed* ω sits within `[adaptivity] newtonian_delta`
    of m. Returns None otherwise (including the up direction, whose M_max
    end has no clean ω-based signature).
    """
    c = spec.campaign
    if c.direction != Direction.DOWN:
        return None
    if not state.steps or state.steps[-1].exit_code != 2:
        return None
    omega = next(
        (float(s.omega) for s in reversed(state.steps) if s.omega is not None and s.exit_code == 0),
        None,
    )
    if omega is None:
        return None
    delta = spec.adaptivity.newtonian_delta
    if omega >= c.m - delta:
        return StopReason.STOPPED_NEWTONIAN_LIMIT
    return None


def finished(state: CampaignState, spec: Spec) -> str | None:
    """Fixed-grid subset of the §6.1 exit conditions. Returns stop reason."""
    c = spec.campaign
    steps = [s for s in state.steps if s.psi0 is not None]
    if not steps:
        return None
    psi0 = float(steps[-1].psi0) if steps[-1].psi0 is not None else 0.0
    omega = omega_of_last(state)
    direction = c.direction

    # ψ₀ hits the constraint value exactly, up to last-ulp rounding of the
    # scale factor; compare with a tolerance so an exact landing stops the
    # campaign instead of re-solving the same point forever.
    tol = 1e-9 * max(1.0, abs(c.psi0_target))
    if direction == Direction.UP and psi0 >= c.psi0_target - tol:
        return StopReason.DONE_PSI0_TARGET
    if direction == Direction.DOWN and psi0 <= c.psi0_target + tol:
        return StopReason.DONE_PSI0_TARGET
    if c.omega_target is not None:
        if direction == Direction.UP and omega <= c.omega_target:
            return StopReason.DONE_OMEGA_TARGET
        if direction == Direction.DOWN and omega >= c.omega_target:
            return StopReason.DONE_OMEGA_TARGET
    if len(state.steps) >= c.max_steps:
        return StopReason.DONE_MAX_STEPS
    if direction == Direction.DOWN:
        # Weak-field boundary stop (v2): near ω → m the tail reaches the
        # outer boundary and boundary error dominates — widening cannot fix
        # it, so stop instead of regridding. The practical default is to end
        # down campaigns at omega_target = 0.9 first (paper convention).
        last = steps[-1]
        if last.r99 is not None and last.dr and last.N:
            r_bdy = (int(last.N) + 2 * ghost_of(spec.grid.order)) * float(last.dr)
            if last.r99 / r_bdy > spec.adaptivity.boundary_fraction:
                return StopReason.STOPPED_BOUNDARY
    if c.stop_at_turning_point and direction == Direction.UP and detect_turning_point(state.steps):
        return StopReason.STOPPED_TURNING_POINT
    if state.steps[-1].exit_code != 0:
        code = state.steps[-1].exit_code
        if code == 2:
            # A solver error near ω → m is the branch's physical end.
            nl = newtonian_limit_stop(state, spec)
            if nl is not None:
                return nl
        if code == TIMEOUT_EXIT:
            return StopReason.FAILED_TIMEOUT
        if code < 0:
            # Killed by a signal (e.g. -11 = SIGSEGV). Rare, pre-existing C
            # backend flakiness); recorded distinctly from exit codes.
            return StopReason.signal(signal.Signals(-code).name.removeprefix("SIG").lower())
        reason = {
            1: StopReason.FAILED_NEWTON,
            2: StopReason.FAILED_SOLVER,
            3: StopReason.FAILED_CONFIG,
            4: StopReason.FAILED_IO,
        }
        return reason.get(code, StopReason.exit_code(code))
    return None


# ---------------------------------------------------------------------------
# Adaptive decision layer (design §4–6) — pure functions, unit-tested
# ---------------------------------------------------------------------------


def ghost_of(order: int) -> int:
    """Ghost-cell count for a finite-difference order (C: order 2 → 1, 4 → 2)."""
    return order // 2


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Diagnostics feeding `decide_action` (design §5, formerly a plain dict).

    `hwl`/`rr_phi_max` are None when the step's scalars were unreadable —
    an under-resolved peak cannot be proven, so None means "do not refine".
    """

    hwl: float | None
    rr_phi_max: float | None
    dr: float
    refinements_left: int
    hwl_min: float


def decide_action(diag: StepDiagnostics) -> Action:
    """Decision policy: exactly one active rule — refine when the field's
    peak is under-resolved.

    Under-resolution means either the half-max width of the field drops
    below `hwl_min` points (spikes), or the peak sits closer to the axis
    than half its own width (`rr_phi_max < (hwl/2)·dr`) — a grid-relative
    form of the old axis-hug rule: within a half-width of the origin the
    φ ∝ r^l power law dominates the profile and the peak-location fit
    becomes axis-biased. The criterion is self-consistent under refinement
    (hwl grows as dr shrinks, so it converges to a statement about the
    continuum profile, not the grid).

    Everything else is fixed relative stepping: no growth, no damping-based
    shrinking (the λ history carries a trailing 0.0 convergence sentinel
    that made any tail statistic meaningless), no boundary regrids
    (weak-field boundary error dominates and widening cannot fix it — the
    campaign stops instead, see `finished`). `refinements_left` bounds the
    number of dr ÷2 refinements per campaign (irreversible,
    verify-then-commit: a refinement is only adopted after the next step
    converges on it).

    Returns "regrid_finer" or "ok".
    """
    if diag.refinements_left <= 0:
        return Action.OK
    if diag.hwl is None:
        return Action.OK
    if diag.hwl < diag.hwl_min:
        return Action.REGGRID_FINER
    if diag.rr_phi_max is not None and diag.rr_phi_max < (diag.hwl / 2.0) * diag.dr:
        return Action.REGGRID_FINER
    return Action.OK


def detect_turning_point(steps: list[StepRecord]) -> bool:
    """Design §6.2: has the branch's minimum-ω turning point been crossed?

    Looks at the last three fixedPhi steps with distinct increasing ψ₀ and
    checks that the central slope dω/dψ₀ changed sign (decreasing → strictly
    increasing). Only meaningful in the up direction; ω rises monotonically
    toward m on the down side.
    """
    pts = [
        (float(s.psi0), float(s.omega))
        for s in steps
        if s.mode in (StepMode.FIXED_PHI, StepMode.SEED)
        and s.psi0 is not None
        and s.omega is not None
    ]
    if len(pts) < 3:
        return False
    (p1, w1), (p2, w2), (p3, w3) = pts[-3], pts[-2], pts[-1]
    if not (p2 > p1 and p3 > p2):
        return False
    s1 = (w2 - w1) / (p2 - p1)
    s2 = (w3 - w2) / (p3 - p2)
    return s1 < 0.0 < s2


def turning_point_estimate(
    psi0s: Sequence[float | None], omegas: Sequence[float | None], degree: int = 4
) -> dict[str, object]:
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


def run_campaign(spec: Spec, fresh: bool, dry_run: bool) -> int:
    root = spec.output.root
    root.mkdir(parents=True, exist_ok=True)
    binary = find_binary()

    if fresh:
        import shutil

        for entry in root.iterdir():
            if entry.is_dir() or entry.name == "state.json":
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        state = fresh_state(spec)
    else:
        loaded = load_state(spec)
        if loaded is None:
            state = fresh_state(spec)
        else:
            # A resumed campaign is running again: clear any status left over
            # from the run that wrote this state (e.g. its own done/failed).
            state = replace(loaded, status=Status.RUNNING, stop_reason=None)

    step_no = len(state.steps)
    logger.info("campaign root: %s", root)
    logger.info("starting at step %d, status=%s", step_no, state.status)

    # ----- seed step -------------------------------------------------------
    if step_no == 0:
        seed = spec.seed
        if seed.policy == SeedPolicy.SOLUTION:
            src = Path(seed.source or "")
            scalars = solution_scalars(src)
            fields = solution_fields(src)
            entry = StepRecord(
                i=0,
                exit_code=0,
                mode=StepMode.SEED,
                sol_dir=str(src),
                dr=spec.grid.dr,
                N=spec.grid.N,
                omega=scalars.get("w_f.asc"),
                psi0=psi_at_fixed_point(fields["psi_f.asc"], spec),
                M_Komar=scalars.get("M_Komar1.asc"),
                J_Komar=scalars.get("J_Komar1.asc"),
                rr_phi_max=scalars.get("rr_phi_max.asc"),
                r99=scalars.get("r99.asc"),
                hwl=scalars.get("hwl_resolution.asc"),
            )
            state.steps.append(entry)
            save_state(spec, state)
            logger.info(
                "step 0 (seed): ψ₀=%.6E ω=%.6E from %s",
                entry.psi0,
                entry.omega,
                src.name,
            )
            step_no = 1
        elif dry_run:
            render_params(spec, root, 0)
            logger.info("dry-run: rendered step 0 params; stopping (seed solve not run)")
            return 0
        else:
            params = render_params(spec, root, 0)
            before = set(find_solution_dirs(root))
            code, _ = run_binary(binary, params, root, 0)
            sol = new_solution_dir(root, before, spec, 0)
            record_step(state, spec, 0, sol, code)
            state = replace(state, stop_reason=finished(state, spec))
            save_state(spec, state)
            if code != 0 or sol is None:
                state = replace(
                    state,
                    status=Status.FAILED,
                    stop_reason=state.stop_reason or StopReason.seed_exit(code),
                )
                save_state(spec, state)
                logger.error(
                    "seed solve failed (exit %d); see %s", code, root / "logs" / "step0000.log"
                )
                return 1
            logger.info(
                "step 0 (seed solve): ψ₀=%.6E ω=%.6E",
                state.steps[-1].psi0,
                state.steps[-1].omega,
            )
            step_no = 1
    elif dry_run:
        logger.info("dry-run: campaign already has steps; nothing to render")
        return 0

    # ----- continuation steps ---------------------------------------------
    c = spec.campaign
    sign = 1 if c.direction == Direction.UP else -1

    def next_target(base_psi0: float) -> float:
        """ψ₀ target for the next step: fixed relative stepping (v2) —
        target = ψ₀·(1 ± psi0_step), the golden-ladder semantics, clamped to
        land exactly on psi0_target.
        """
        if c.psi0_step_mode == Psi0StepMode.RELATIVE:
            target = base_psi0 * (1.0 + sign * c.psi0_step)
        else:
            target = base_psi0 + sign * c.psi0_step
        if c.direction == Direction.UP:
            return min(round(target, 15), c.psi0_target)
        return max(round(target, 15), c.psi0_target)

    while True:
        stop = finished(state, spec)
        if stop is not None:
            family = stop_family(stop)
            if family == "done":
                new_status = Status.DONE
            elif family == "stopped":
                new_status = Status.STOPPED  # clean stop (turning point / budget)
            else:
                new_status = Status.FAILED
            state = replace(state, status=new_status, stop_reason=stop)
            if stop == StopReason.STOPPED_TURNING_POINT:
                steps = state.steps
                state = replace(
                    state,
                    turning_point=turning_point_estimate(
                        [s.psi0 for s in steps], [s.omega for s in steps]
                    ),
                )
            save_state(spec, state)
            logger.info("stop: %s", stop)
            tp = state.turning_point
            if tp:
                w = tp.get("omega", tp.get("omega_sample_min"))
                p = tp.get("psi0", tp.get("psi0_sample_min"))
                logger.info("ω_min ≈ %.6E at ψ₀ ≈ %.6E (%s)", w, p, tp.get("method"))
            return 0 if state.status != Status.FAILED else 1

        # Retry loop (decision-table rules 2-3, core subset): on Newton
        # non-convergence (exit 1) shrink the step and retry from the last
        # good solution. A step killed by a signal (code < 0, e.g. SIGSEGV —
        # rare backend flakiness) is retried at the same step size: unlike
        # a Newton failure, the step size is not the cause. Solver/config/
        # I-O errors are not retryable.
        base_psi0 = psi0_of_last(state)
        attempts = 0
        reverted = False
        while True:
            psi0_target = next_target(base_psi0)
            prev = [s for s in state.steps if s.psi0 is not None]
            # After a failed attempt drop the linear extrapolation: in
            # marginal regions it diverges Newton while the plain rescale
            # from the last good solution converges.
            scale_u4, w_guess = render_seed(
                spec, root, prev, psi0_target, extrapolate=attempts == 0
            )
            params = render_params(spec, root, step_no, scale_u4=scale_u4, seed_dir=root / "seed")

            stale = root / initial_dirname(spec)
            if stale.is_dir():
                import shutil

                shutil.rmtree(stale)
            before = set(find_solution_dirs(root))
            code, log = run_binary(binary, params, root, step_no)
            sol = new_solution_dir(root, before, spec, step_no)
            last = record_step(state, spec, step_no, sol, code, psi0_target=psi0_target)
            save_state(spec, state)

            if code == 0 and sol is not None and last.psi0 is not None:
                break

            # A solver error near ω → m is the branch's physical end: do not
            # waste a retry chasing it.
            near_limit = code == 2 and newtonian_limit_stop(state, spec) is not None
            retryable = (code in (1, 2, TIMEOUT_EXIT) or code < 0) and not near_limit
            if retryable and attempts < c.max_retries:
                # v2: fixed Δψ₀ — retries keep the same target and drop the
                # extrapolated seed (the rescale converges
                # where the extrapolation diverges).
                attempts += 1
                state.steps.pop()
                save_state(spec, state)
                cause = (
                    "did not converge"
                    if code == 1
                    else "timed out"
                    if code == TIMEOUT_EXIT
                    else f"killed by signal {signal.Signals(-code).name}"
                    if code < 0
                    else "solver error"
                )
                logger.warning(
                    "step %d attempt %d %s; retrying with rescaled seed, log: %s",
                    step_no,
                    attempts,
                    cause,
                    log,
                )
                continue

            # Verify-then-commit (v2): a pending (adopted-but-unverified)
            # refinement whose verification step failed is not committed —
            # restore the previous grid, keep the finer-grid solution as a
            # measurement, and disable further refinements. One decision,
            # permanent, no oscillation.
            pending = state.pending_refinement
            if pending is not None:
                measurement = replace(
                    pending.measurement,
                    note=(
                        "refined-grid re-solve at the fold; its verification step failed so the "
                        "refinement was not committed — kept as the fold measurement"
                    ),
                )
                rg_step = pending.regrid_step
                kept_steps = [
                    s
                    for s in state.steps
                    if not (s.mode == StepMode.REGGRID and s.i == rg_step) and s.exit_code == 0
                ]
                state = replace(
                    state,
                    steps=kept_steps,
                ).with_fold_measurement(measurement)
                state = replace(
                    state,
                    grid=replace(current_grid(state, spec), dr=pending.from_dr),
                    refinements_left=0,
                    pending_refinement=None,
                )
                save_state(spec, state)
                logger.warning(
                    "step %d: verification failed on the refined grid; "
                    "reverted to dr=%.5E, refinements disabled",
                    step_no,
                    pending.from_dr,
                )
                reverted = True
                break

            state = replace(
                state,
                status=Status.FAILED,
                stop_reason=finished(state, spec) or StopReason.exit_code(code),
            )
            save_state(spec, state)
            logger.error("step %d FAILED (exit %d); log: %s", step_no, code, log)
            return 1

        note = f", {attempts} retry" if attempts else ""
        step_no += 1  # the continuation step consumed its slot
        if reverted:
            continue  # refinement reverted: re-check exit conditions

        # ----- refinement decision (design §5, v2) --------------------------
        a = spec.adaptivity
        dr = current_grid(state, spec).dr
        diag = StepDiagnostics(
            hwl=last.hwl,
            rr_phi_max=last.rr_phi_max,
            dr=dr,
            refinements_left=(
                state.refinements_left if state.refinements_left is not None else a.max_refinements
            ),
            hwl_min=a.hwl_min,
        )
        action = decide_action(diag)

        if action == Action.REGGRID_FINER:
            # Irreversible refinement (dr ÷2, domain shrinks, N fixed): the
            # interpolated re-solve at the same ψ₀ is committed immediately,
            # and the next continuation step doubles as its verification —
            # verify-then-commit.
            new_dr = dr / 2.0
            ok, _, state = do_regrid(spec, root, state, binary, step_no, new_dr)
            if ok:
                state = replace(
                    state,
                    grid=replace(current_grid(state, spec), dr=new_dr),
                    refinements_left=diag.refinements_left - 1,
                    pending_refinement=PendingRefinement(
                        from_dr=dr,
                        regrid_step=step_no,
                        measurement=FoldMeasurement(
                            psi0=state.steps[-1].psi0,
                            omega=state.steps[-1].omega,
                            dr=new_dr,
                            sol_dir=str(Path(state.steps[-1].sol_dir or "")),
                        ),
                    ),
                )
                step_no += 1
                save_state(spec, state)
                continue  # re-check exit conditions on the new grid
            # The finer re-solve itself failed: count the attempt and stay
            # on this grid (a failed refinement is not retried blindly).
            state = replace(state, refinements_left=diag.refinements_left - 1)
            save_state(spec, state)
            logger.warning("refinement rejected; continuing on dr=%.5E", dr)

        logger.info(
            "step %d: ψ₀=%.6E ω=%.6E (guess ω≈%.4f, %s iters%s) [%s] -> %s",
            step_no - 1,
            last.psi0,
            last.omega,
            w_guess,
            last.newton_iters,
            note,
            action,
            Path(last.sol_dir or "").name,
        )

    # pragma: no cover — the campaign loop only exits via return


def summarize(spec: Spec) -> int:
    """--summarize: localize ω_min from a finished campaign (design §6.2).

    Post-processing only: reads state.json, fits a low-order polynomial to
    ω(ψ₀) over the samples bracketing the smallest sampled ω, and reports the
    extremum (the paper used a 4th-degree spline). Writes <root>/summary.json.
    """
    state = load_state(spec)
    if state is None:
        logger.error("no state.json to summarize")
        return 1
    omegas = [s.omega for s in state.steps]
    if not any(w is not None for w in omegas):
        logger.error("no completed steps record ω; nothing to summarize")
        return 1
    report = turning_point_estimate(
        [s.psi0 for s in state.steps],
        omegas,
    )
    report["campaign"] = str(spec.output.root)
    report["status"] = state.status
    report["stop_reason"] = state.stop_reason
    out = spec.output.root / "summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    # The summary payload is the tool's stdout product (machine-consumed):
    # printed, not logged.
    print(f"[driver] summary: {json.dumps(report)}")
    logger.info("wrote %s", out)
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
    verbosity = ap.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose diagnostics (DEBUG logging)",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only warnings and errors (WARNING logging)",
    )
    ap.add_argument(
        "--log-json",
        action="store_true",
        help="emit logs as one JSON object per line on stderr (campaign telemetry)",
    )
    args = ap.parse_args()

    configure(
        "DEBUG" if args.verbose else "WARNING" if args.quiet else None,
        json_logs=args.log_json,
    )

    try:
        spec = load_spec(args.campaign)
    except SpecError as e:
        logger.error("invalid campaign spec: %s", e)
        return 3

    if args.summarize:
        return summarize(spec)
    return run_campaign(spec, fresh=args.fresh, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
