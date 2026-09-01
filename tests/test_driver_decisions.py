"""Unit tests for the SAN-14 adaptive decision layer (design §4–6).

The decision logic lives in pure functions in tools/sweep_driver.py; these
tests feed them synthetic diagnostics and check every branch of the §5
decision table plus turning-point detection and ω_min localization (§6.2).
"""

from __future__ import annotations

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
    """A healthy, comfortable step's diagnostics (defaults → 'grow')."""
    base = {
        "newton_iters": 5,
        "lambda_min": 0.9,
        "hwl": 21,
        "rr_phi_max": 6.9,
        "r99": 10.0,
        "r_bdy": 17.0,
        "dr": 0.25,
        "dr_max": 1.0,
        "hwl_min": 8,
        "hwl_max": 40,
        "support_fraction": 0.85,
        "rr_phi_max_min": 0.5,
        "newton_fast_iters": 8,
        "lambda_min_floor": 1.0e-3,
        "direction": "down",  # rule 5's conservative branch
        "support_window": None,
        "optional_coarsening": True,  # rule-7 tests opt in explicitly
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# decide_action — design §5 rules
# ---------------------------------------------------------------------------


class TestRules1And2:
    def test_fast_healthy_step_grows(self):
        assert decide_action(diag()) == "grow"

    def test_slow_step_shrinks(self):
        # rule 2: "Newton iterations high" — above 2× newton_fast_iters
        assert decide_action(diag(newton_iters=17)) == "shrink"

    def test_boundary_slow_step_is_ok(self):
        # exactly at the 2× threshold is not "high"
        assert decide_action(diag(newton_iters=16)) == "ok"

    def test_collapsed_lambda_shrinks(self):
        # rule 2: lambda_min collapsed below the floor
        assert decide_action(diag(lambda_min=5.0e-4)) == "shrink"

    def test_lambda_at_floor_is_healthy(self):
        assert decide_action(diag(lambda_min=1.0e-3)) == "grow"


class TestGridRules:
    def test_underresolved_spike_regrids_finer(self):
        # rule 6: hwl below the floor wins over everything else
        assert decide_action(diag(hwl=7)) == "regrid_finer"

    def test_axis_hug_regrids_finer(self):
        # rule 8: field maximum drifting onto the axis (l >= 2 limit case)
        assert decide_action(diag(rr_phi_max=0.4)) == "regrid_finer"

    def test_support_hitting_boundary_regrids_coarser(self):
        # rule 5: r99/r_bdy above support_fraction
        assert decide_action(diag(r99=15.6, r_bdy=17.0)) == "regrid_coarser"

    def test_support_at_domain_budget_stops(self):
        # rule 5 + §6.1: same symptom but dr already at the cap
        d = diag(r99=15.6, r_bdy=17.0, dr=1.0, dr_max=1.0)
        assert decide_action(d) == "stop:domain_budget"

    def test_overresolved_regrid_is_optional(self):
        # rule 7: wastefully fine — optional coarsening, never a stop
        assert decide_action(diag(hwl=41)) == "regrid_coarser_optional"
        # ...but if the support also hits the boundary, rule 5 wins (required)
        assert decide_action(diag(hwl=41, r99=15.6, r_bdy=17.0)) == "regrid_coarser"
        # at the domain cap the optional regrid is skipped entirely
        d2 = diag(hwl=41, dr=1.0, dr_max=1.0)
        assert decide_action(d2) == "grow"

    def test_down_support_rising_regrids_coarser(self):
        # down direction: the conservative single-step trend check applies
        d = diag(r99=15.6, r_bdy=17.0, direction="down", support_window=[0.80])
        assert decide_action(d) == "regrid_coarser"
        # unknown trend (first step) keeps the conservative behaviour
        assert decide_action(diag(r99=15.6, r_bdy=17.0, direction="down")) == "regrid_coarser"

    def test_up_stagnant_support_does_not_fire(self):
        # SAN-20 sweep finding: the weak seed's tail grazes the boundary with
        # a support fraction stuck at ~0.949, wobbling in the 4th decimal.
        # A single-step trend check fired on every uptick and burned the
        # campaign on futile widening regrids; the up direction now needs the
        # fraction to rise across the WHOLE 3-step window.
        d = diag(r99=15.6, r_bdy=17.0, direction="up", support_window=[0.9489, 0.9490, 0.9491])
        assert decide_action(d) == "grow"

    def test_up_persistently_rising_still_fires(self):
        d = diag(r99=15.6, r_bdy=17.0, direction="up", support_window=[0.70, 0.75, 0.80])
        assert decide_action(d) == "regrid_coarser"

    def test_finer_blacklist_suppresses_rules_6_and_8(self):
        # SAN-20: when the finer grid rejects stepping, the driver blacklists
        # finer regrids for the campaign — under-resolution then falls
        # through to the ordinary step-size rules
        d = diag(hwl=5, rr_phi_max=0.4)
        assert decide_action(d) == "regrid_finer"
        d["finer_blacklisted"] = True
        assert decide_action(d) == "grow"

    def test_rule7_disabled_by_default(self):
        # SAN-20 sweep finding: an accepted coarsening can stall the
        # campaign, so rule 7 runs only when explicitly enabled
        d = diag(hwl=41)
        d["optional_coarsening"] = False
        assert decide_action(d) == "grow"
        d["optional_coarsening"] = True
        assert decide_action(d) == "regrid_coarser_optional"

    def test_up_unknown_window_does_not_fire(self):
        # no same-grid history yet (first steps): the up direction does not
        # treat boundary-grazing as an emergency
        assert decide_action(diag(r99=15.6, r_bdy=17.0, direction="up")) == "grow"

    def test_domain_budget_still_stops_when_spreading_at_cap(self):
        d = diag(r99=15.6, r_bdy=17.0, direction="down", support_window=[0.80], dr=1.0, dr_max=1.0)
        assert decide_action(d) == "stop:domain_budget"
        # ...but a shrinking field at the cap does not stop
        d2 = diag(r99=15.6, r_bdy=17.0, direction="down", support_window=[0.92], dr=1.0, dr_max=1.0)
        assert decide_action(d2) == "grow"

    def test_finer_wins_over_coarser_when_both_fire(self):
        # contradictory needs: under-resolution takes precedence (docstring)
        d = diag(hwl=7, r99=15.6, r_bdy=17.0)
        assert decide_action(d) == "regrid_finer"

    def test_missing_diagnostics_are_skipped(self):
        # a step with no analysis data must not crash the decision
        d = diag()
        for k in ("hwl", "rr_phi_max", "r99"):
            d[k] = None
        assert decide_action(d) == "grow"


class TestPrecedence:
    def test_regrid_beats_growth(self):
        # fast healthy step on a grid that needs replacing: regrid first
        assert decide_action(diag(hwl=7)) == "regrid_finer"


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
        # s2 == 0 is not a *crossed* turning point
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.12, 0.65), step(0.13, 0.65)]
        assert not detect_turning_point(steps)

    def test_repeated_psi0_does_not_fire(self):
        # identical ψ₀ (e.g. a regrid step landing twice) must not divide by zero
        steps = [step(0.10, 0.70), step(0.11, 0.66), step(0.11, 0.66), step(0.13, 0.66)]
        assert not detect_turning_point(steps)

    def test_design_semantics_last_three_steps_only(self):
        # The design tracks slopes across the LAST three steps; a minimum
        # several steps back is not re-detected (in a live campaign detection
        # runs every step, so the crossing is caught the moment it happens).
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
        # exact parabola: the fit should recover the vertex to fit precision
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
        # samples straddling a real-ish l=1 minimum: the fit should localize
        # the extremum at the bottom of the branch (within fit noise of the
        # smallest sample) between the bracketing ψ₀ values
        psi0s = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
        omegas = [0.6485, 0.6478, 0.64745, 0.64742, 0.64750, 0.6479, 0.6486]
        rep = turning_point_estimate(psi0s, omegas)
        assert rep["omega"] == pytest.approx(min(omegas), abs=2e-4)
        assert 0.44 <= rep["psi0"] <= 0.46


# ---------------------------------------------------------------------------
# SAN-20 item 1: physics-limit vs solver-failure classification
# ---------------------------------------------------------------------------


def spec_down(**adaptivity) -> dict:
    base = {"campaign": {"direction": "down", "max_steps": 100, "psi0_target": 1e-5, "m": 1.0}}
    base["adaptivity"] = {"newtonian_delta": 1.0e-2, **adaptivity}
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
        # ω = m − δ exactly is within the limit region
        assert newtonian_limit_stop(state_failed(2, 0.99), spec_down()) == (
            "stopped:newtonian_limit"
        )

    def test_exit2_far_from_m_is_solver_failure(self):
        assert newtonian_limit_stop(state_failed(2, 0.95), spec_down()) is None
        assert finished(state_failed(2, 0.95), spec_down()) == "failed:solver"

    def test_up_direction_never_classifies(self):
        # the M_max end has no clean ω signature yet (SAN-20 item 5)
        s = spec_down()
        s["campaign"]["direction"] = "up"
        s["campaign"]["psi0_target"] = 1.0  # keep the ψ₀ exit condition quiet
        assert newtonian_limit_stop(state_failed(2, 0.995), s) is None
        assert finished(state_failed(2, 0.995), s) == "failed:solver"

    def test_only_exit2_triggers(self):
        assert newtonian_limit_stop(state_failed(1, 0.995), spec_down()) is None

    def test_no_completed_omega_is_unclassifiable(self):
        s = state_failed(2, omega=None)
        assert newtonian_limit_stop(s, spec_down()) is None

    def test_delta_knob(self):
        assert newtonian_limit_stop(state_failed(2, 0.95), spec_down(newtonian_delta=0.1)) == (
            "stopped:newtonian_limit"
        )


# ---------------------------------------------------------------------------
# SAN-20 item 2: timeout handling
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_run_binary_returns_timeout_sentinel(self, tmp_path):
        import textwrap

        from sweep_driver import run_binary

        hung = tmp_path / "hung.sh"
        hung.write_text(
            textwrap.dedent("""\
            #!/bin/sh
            sleep 30
        """)
        )
        hung.chmod(0o755)
        (tmp_path / "p.toml").write_text("# params")
        code, log = run_binary(hung, tmp_path / "p.toml", tmp_path, 0, timeout_s=1.0)
        assert code == TIMEOUT_EXIT
        assert "TIMEOUT" in log.read_text()

    def test_finished_maps_timeout(self):
        assert finished(state_failed(TIMEOUT_EXIT, 0.95), spec_down()) == "failed:timeout"

    def test_timeout_sentinel_not_treated_as_signal(self):
        # TIMEOUT_EXIT is negative; it must not route through the signal name
        # mapping (Signals(-(-999)) would raise ValueError)
        import signal

        with pytest.raises(ValueError):
            signal.Signals(-TIMEOUT_EXIT)  # guards the sentinel's distinctness
        assert finished(state_failed(TIMEOUT_EXIT, 0.95), spec_down()) == "failed:timeout"


# ---------------------------------------------------------------------------
# ghost_of — grid bookkeeping used by the regrid path
# ---------------------------------------------------------------------------


def test_ghost_of():
    assert ghost_of(2) == 1
    assert ghost_of(4) == 2
