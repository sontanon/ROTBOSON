"""Unit tests for the v2 adaptive decision layer (SAN-21; design §4–6 rev 2).

The decision logic lives in pure functions in tools/sweep_driver.py; these
tests feed them synthetic diagnostics. v2 policy: fixed relative stepping,
one refinement rule (under-resolved peak → dr ÷2, verify-then-commit),
physics-limit/boundary stops, timeout handling, turning-point detection.
"""


import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from sweep_driver import (  # noqa: E402
    TIMEOUT_EXIT,
    decide_action,
    detect_turning_point,
    finished,
    ghost_of,
    newtonian_limit_stop,
    turning_point_estimate,
)


def diag(**overrides) -> dict:
    """A well-resolved step's diagnostics (defaults → 'ok' in v2)."""
    base = {
        "hwl": 21,
        "rr_phi_max": 6.9,
        "dr": 0.125,
        "refinements_left": 2,
        "hwl_min": 8,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# decide_action — v2: the single refinement rule
# ---------------------------------------------------------------------------


class TestDecideActionV2:
    def test_well_resolved_step_is_ok(self):
        assert decide_action(diag()) == "ok"

    def test_spikes_trigger_refinement(self):
        # hwl below the floor: the field's half-max width is under-sampled
        assert decide_action(diag(hwl=7)) == "regrid_finer"

    def test_axis_hug_triggers_refinement(self):
        # peak closer to the axis than half its own width:
        # (hwl/2)·dr = 2.5·0.125 = 0.3125 > rr_phi_max
        d = diag(hwl=5, rr_phi_max=0.30)
        assert decide_action(d) == "regrid_finer"

    def test_axis_criterion_is_grid_relative(self):
        # the SAME physical configuration passes on a finer grid:
        # (hwl/2)·dr = 5.5·0.0625 = 0.344 <= rr_phi_max
        d = diag(hwl=11, rr_phi_max=0.394, dr=0.0625)
        assert decide_action(d) == "ok"

    def test_axis_clear_and_resolved_is_ok(self):
        # hwl 9 (≥ hwl_min) and the max clears (hwl/2)·dr = 0.5625:
        assert decide_action(diag(hwl=9, rr_phi_max=0.6)) == "ok"

    def test_refinements_exhausted_means_ok(self):
        # the campaign-wide cap bounds refinements; under-resolution then
        # falls through (matching the archived campaign's hwl ≈ 5 tail)
        d = diag(hwl=5, rr_phi_max=0.30, refinements_left=0)
        assert decide_action(d) == "ok"

    def test_missing_diagnostics_are_skipped(self):
        assert decide_action(diag(hwl=None, rr_phi_max=None)) == "ok"


# ---------------------------------------------------------------------------
# finished() — v2 stops: boundary (down), physics limit, timeout
# ---------------------------------------------------------------------------


def spec_down(**adaptivity) -> dict:
    base = {
        "campaign": {"direction": "down", "max_steps": 100, "psi0_target": 1e-5, "m": 1.0},
        "grid": {"order": 4},
    }
    base["adaptivity"] = {"newtonian_delta": 1.0e-2, "boundary_fraction": 0.95, **adaptivity}
    return base


def state_failed(code: int, omega: float | None = 0.995) -> dict:
    # failed step LAST (most recent); a prior completed step provides ω
    steps = []
    if omega is not None:
        steps.append({"exit_code": 0, "psi0": 1e-3, "omega": omega, "mode": "fixedPhi"})
    steps.append({"exit_code": code, "psi0": 1e-4, "mode": "fixedPhi"})
    return {"steps": steps}


class TestNewtonianLimit:
    def test_exit2_near_m_is_physical_limit(self):
        # down direction, exit 2 with last completed ω within δ of m = 1
        assert newtonian_limit_stop(state_failed(2, 0.995), spec_down()) == (
            "stopped:newtonian_limit"
        )
        assert finished(state_failed(2, 0.995), spec_down()) == "stopped:newtonian_limit"

    def test_boundary_of_delta_counts(self):
        assert newtonian_limit_stop(state_failed(2, 0.99), spec_down()) == (
            "stopped:newtonian_limit"
        )

    def test_exit2_far_from_m_is_solver_failure(self):
        assert newtonian_limit_stop(state_failed(2, 0.95), spec_down()) is None
        assert finished(state_failed(2, 0.95), spec_down()) == "failed:solver"

    def test_up_direction_never_classifies(self):
        s = spec_down()
        s["campaign"]["direction"] = "up"
        s["campaign"]["psi0_target"] = 1.0
        assert newtonian_limit_stop(state_failed(2, 0.995), s) is None
        assert finished(state_failed(2, 0.995), s) == "failed:solver"

    def test_only_exit2_triggers(self):
        assert newtonian_limit_stop(state_failed(1, 0.995), spec_down()) is None

    def test_no_completed_omega_is_unclassifiable(self):
        assert newtonian_limit_stop(state_failed(2, omega=None), spec_down()) is None

    def test_delta_knob(self):
        assert newtonian_limit_stop(state_failed(2, 0.95), spec_down(newtonian_delta=0.1)) == (
            "stopped:newtonian_limit"
        )


class TestBoundaryStop:
    def state_at(self, r99: float, dr: float = 0.125, n: int = 128) -> dict:
        return {
            "steps": [{"exit_code": 0, "psi0": 1e-4, "omega": 0.995, "r99": r99, "dr": dr, "N": n}]
        }

    def test_boundary_grazing_tail_stops_down_campaign(self):
        # r99/domain = 16/16.5 = 0.97 > 0.95
        assert finished(self.state_at(16.0), spec_down()) == "stopped:boundary"

    def test_comfortable_support_does_not_stop(self):
        # r99/domain = 10/16.5 = 0.61
        assert finished(self.state_at(10.0), spec_down()) is None

    def test_up_direction_ignores_boundary(self):
        # the up direction's boundary-grazing weak tail is transient
        s = spec_down()
        s["campaign"]["direction"] = "up"
        s["campaign"]["psi0_target"] = 1.0
        assert finished(self.state_at(16.0), s) is None

    def test_boundary_fraction_knob(self):
        assert finished(self.state_at(16.0), spec_down(boundary_fraction=0.99)) is None


class TestTimeout:
    def test_finished_maps_timeout(self):
        assert finished(state_failed(TIMEOUT_EXIT, 0.95), spec_down()) == "failed:timeout"

    def test_timeout_sentinel_not_treated_as_signal(self):
        # TIMEOUT_EXIT is negative; it must not route through the signal name
        # mapping (Signals(-(-999)) would raise ValueError)
        import signal

        with pytest.raises(ValueError):
            signal.Signals(-TIMEOUT_EXIT)
        assert finished(state_failed(TIMEOUT_EXIT, 0.95), spec_down()) == "failed:timeout"


# ---------------------------------------------------------------------------
# detect_turning_point — design §6.2
# ---------------------------------------------------------------------------


def step(psi0: float, omega: float, mode: str = "fixedPhi") -> dict:
    return {"mode": mode, "psi0": psi0, "omega": omega, "exit_code": 0}


class TestDetectTurningPoint:
    def test_needs_three_branch_points(self):
        assert not detect_turning_point([])
        assert not detect_turning_point([step(0.1, 0.9)])
        assert not detect_turning_point([step(0.1, 0.9), step(0.2, 0.8)])
        # a seed step IS a branch point and counts toward the three
        assert detect_turning_point([step(0.1, 0.9, mode="seed"), step(0.2, 0.8), step(0.3, 0.81)])

    def test_decreasing_then_increasing_fires(self):
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.66)]
        assert detect_turning_point(steps)

    def test_monotone_decreasing_does_not_fire(self):
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.64)]
        assert not detect_turning_point(steps)

    def test_flat_slope_does_not_fire(self):
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.65)]
        assert not detect_turning_point(steps)

    def test_repeated_psi0_does_not_fire(self):
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.11, 0.66), step(0.13, 0.66)]
        assert not detect_turning_point(steps)

    def test_design_semantics_last_three_steps_only(self):
        steps = [
            step(0.10, 0.70),
            step(0.11, 0.64),
            step(0.12, 0.65),
            step(0.13, 0.66),
            step(0.14, 0.67),
        ]
        assert not detect_turning_point(steps)
        # in a live campaign detection runs every step, so the crossing is
        # caught at the FIRST post-minimum sample (last three straddle it)
        assert detect_turning_point(steps[:3])


# ---------------------------------------------------------------------------
# turning_point_estimate — §6.2 localization
# ---------------------------------------------------------------------------


class TestTurningPointEstimate:
    def test_quadratic_bowl_localization(self):
        psi0s = np.linspace(0.40, 0.52, 7)
        omegas = 0.65 + 50.0 * (psi0s - 0.46) ** 2
        rep = turning_point_estimate(list(psi0s), list(omegas))
        assert rep["method"].startswith("poly")
        assert rep["omega"] == pytest.approx(0.65, abs=1e-8)
        assert rep["psi0"] == pytest.approx(0.46, abs=1e-6)

    def test_few_points_falls_back_to_sample(self):
        rep = turning_point_estimate([0.4, 0.5], [0.7, 0.6])
        assert rep["method"] == "sample"
        assert rep["omega_sample_min"] == 0.6

    def test_empty_is_empty(self):
        assert turning_point_estimate([], []) == {}

    def test_catalogue_like_minimum(self):
        psi0s = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
        omegas = [0.6485, 0.6478, 0.64745, 0.64742, 0.64750, 0.6479, 0.6486]
        rep = turning_point_estimate(psi0s, omegas)
        assert rep["omega"] == pytest.approx(min(omegas), abs=2e-4)
        assert 0.44 <= rep["psi0"] <= 0.46


# ---------------------------------------------------------------------------
# ghost_of — grid bookkeeping used by the refinement path
# ---------------------------------------------------------------------------


def test_ghost_of():
    assert ghost_of(2) == 1
    assert ghost_of(4) == 2
