"""Unit tests for the v2 adaptive decision layer (design §4–6 rev 2).

The decision logic lives in pure functions in tools/sweep_driver.py; these
tests feed them frozen-dataclass diagnostics. v2 policy: fixed
relative stepping, one refinement rule (under-resolved peak → dr ÷2,
verify-then-commit), physics-limit/boundary stops, timeout handling,
turning-point detection.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from sweep_driver import (  # noqa: E402
    TIMEOUT_EXIT,
    Action,
    AdaptivitySpec,
    CampaignSpec,
    CampaignState,
    Direction,
    GridSpec,
    OutputFormat,
    OutputSpec,
    RegridInfo,
    SeedPolicy,
    SeedSpec,
    SolverSpec,
    Spec,
    SpecError,
    StepDiagnostics,
    StepMode,
    StepRecord,
    boundary_failure_stop,
    coarsening_acceptable,
    decide_action,
    decide_coarsening,
    detect_turning_point,
    finished,
    ghost_of,
    newtonian_limit_stop,
    turning_point_estimate,
    would_amputate,
)


def diag(**overrides) -> StepDiagnostics:
    """A well-resolved step's diagnostics (defaults → 'ok' in v2)."""
    base = {
        "hwl": 21.0,
        "rr_phi_max": 6.9,
        "dr": 0.125,
        "refinements_left": 2,
        "hwl_min": 8.0,
    }
    base.update(overrides)
    return StepDiagnostics(**base)


# ---------------------------------------------------------------------------
# decide_action — v2: the single refinement rule
# ---------------------------------------------------------------------------


class TestDecideActionV2:
    def test_well_resolved_step_is_ok(self):
        assert decide_action(diag()) == Action.OK

    def test_spikes_trigger_refinement(self):
        # hwl below the floor: the field's half-max width is under-sampled
        assert decide_action(diag(hwl=7.0)) == Action.REGGRID_FINER

    def test_axis_hug_triggers_refinement(self):
        # peak closer to the axis than half its own width:
        # (hwl/2)·dr = 2.5·0.125 = 0.3125 > rr_phi_max
        d = diag(hwl=5.0, rr_phi_max=0.30)
        assert decide_action(d) == Action.REGGRID_FINER

    def test_axis_criterion_is_grid_relative(self):
        # the SAME physical configuration passes on a finer grid:
        # (hwl/2)·dr = 5.5·0.0625 = 0.344 <= rr_phi_max
        d = diag(hwl=11.0, rr_phi_max=0.394, dr=0.0625)
        assert decide_action(d) == Action.OK

    def test_axis_clear_and_resolved_is_ok(self):
        assert decide_action(diag(hwl=21.0, rr_phi_max=6.9)) == Action.OK

    def test_refinements_exhausted_means_ok(self):
        # falls through (matching the archived campaign's hwl ≈ 5 tail)
        d = diag(hwl=5.0, rr_phi_max=0.30, refinements_left=0)
        assert decide_action(d) == Action.OK

    def test_missing_diagnostics_are_skipped(self):
        assert decide_action(diag(hwl=None, rr_phi_max=None)) == Action.OK


# ---------------------------------------------------------------------------
# finished() — v2 stops: boundary (down), physics limit, timeout
# ---------------------------------------------------------------------------


def spec_down(direction: str = "down", **adaptivity) -> Spec:
    adapt: dict[str, float | int | bool] = {
        "hwl_min": 8,
        "max_refinements": 2,
        "newtonian_delta": 1.0e-2,
        "boundary_fraction": 0.95,
    }
    adapt.update(adaptivity)
    return Spec(
        campaign=CampaignSpec(
            l=1,
            direction=Direction(direction),
            psi0_target=1e-5 if direction == "down" else 1.0,
            psi0_step=0.1,
            m=1.0,
            max_steps=100,
        ),
        seed=SeedSpec(policy=SeedPolicy.FROM_SCRATCH, w0=0.9),
        grid=GridSpec(dr=0.125, N=128, dr_max=0.5, order=4),
        solver=SolverSpec(),
        output=OutputSpec(root=Path("/tmp/spec-down-test"), format=OutputFormat.HDF5),
        adaptivity=AdaptivitySpec(
            hwl_min=float(adapt["hwl_min"]),
            max_refinements=int(adapt["max_refinements"]),
            newtonian_delta=float(adapt["newtonian_delta"]),
            boundary_fraction=float(adapt["boundary_fraction"]),
            refine_keeps_domain=bool(adapt.get("refine_keeps_domain", False)),
            regrid_rtol=float(adapt.get("regrid_rtol", 2.0e-2)),
            support_fraction=float(adapt.get("support_fraction", 0.85)),
            max_widenings=int(adapt.get("max_widenings", 2)),
        ),
        spec_hash="test",
    )


def state_failed(code: int, omega: float | None = 0.995) -> CampaignState:
    # failed step LAST (most recent); a prior completed step provides ω
    steps = []
    if omega is not None:
        steps.append(StepRecord(i=0, exit_code=0, psi0=1e-3, omega=omega, mode=StepMode.FIXED_PHI))
    steps.append(StepRecord(i=1, exit_code=code, psi0=1e-4, mode=StepMode.FIXED_PHI))
    return CampaignState(spec_hash="test", steps=steps)


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
        s = spec_down(direction="up")
        assert newtonian_limit_stop(state_failed(2, 0.995), s) is None
        assert finished(state_failed(2, 0.995), s) == "failed:solver"

    def test_only_exit2_triggers(self):
        assert newtonian_limit_stop(state_failed(1, 0.995), spec_down()) is None

    def test_no_completed_omega_is_unclassifiable(self):
        assert newtonian_limit_stop(state_failed(2, omega=None), spec_down()) is None

    def test_delta_knob(self):
        assert (
            newtonian_limit_stop(state_failed(2, 0.95), spec_down(newtonian_delta=0.1))
            == "stopped:newtonian_limit"
        )


class TestBoundaryStop:
    def state_at(self, r99: float, dr: float = 0.125, n: int = 128) -> CampaignState:
        return CampaignState(
            spec_hash="test",
            steps=[
                StepRecord(
                    i=0,
                    exit_code=0,
                    psi0=1e-4,
                    omega=0.995,
                    r99=r99,
                    dr=dr,
                    N=n,
                )
            ],
        )

    def test_boundary_grazing_tail_stops_down_campaign(self):
        # r99/domain = 16/16.5 = 0.97 > 0.95 — but the guard now YIELDS to
        # available coarsening (dr×2=0.25 ≤ dr_max, budget left): widening
        # is tried first (SAN-30). It stops only when widening is spent.
        assert finished(self.state_at(16.0), spec_down()) is None
        state = self.state_at(16.0)
        state = replace(state, widenings_left=0)
        assert finished(state, spec_down()) == "stopped:boundary"
        # ... or when the coarseness floor blocks the widening outright
        spec = replace(spec_down(), grid=replace(spec_down().grid, dr_max=0.2))
        assert finished(self.state_at(16.0), spec) == "stopped:boundary"

    def test_coarsening_budget_roundtrip(self):
        # the budget is honored from state (a resumed campaign's remaining
        # widenings), not re-defaulted
        state = replace(self.state_at(16.0), widenings_left=1)
        assert finished(state, spec_down()) is None

    def test_comfortable_support_does_not_stop(self):
        # r99/domain = 10/16.5 = 0.61
        assert finished(self.state_at(10.0), spec_down()) is None

    def test_up_direction_ignores_boundary(self):
        # the up direction's boundary-grazing weak tail is transient
        assert finished(self.state_at(16.0), spec_down(direction="up")) is None

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


def step(psi0: float, omega: float, mode: StepMode = StepMode.FIXED_PHI) -> StepRecord:
    return StepRecord(i=0, exit_code=0, mode=mode, psi0=psi0, omega=omega)


class TestDetectTurningPoint:
    def test_needs_three_branch_points(self):
        assert not detect_turning_point([])
        assert not detect_turning_point([step(0.1, 0.9)])
        assert not detect_turning_point([step(0.1, 0.9), step(0.2, 0.8)])
        # a seed step IS a branch point and counts toward the three
        assert detect_turning_point(
            [step(0.1, 0.9, StepMode.SEED), step(0.2, 0.8), step(0.3, 0.81)]
        )

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
        assert str(rep["method"]).startswith("poly")
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
        assert isinstance(rep["psi0"], float)
        assert 0.44 <= rep["psi0"] <= 0.46


# ---------------------------------------------------------------------------
# ghost_of — grid bookkeeping used by the refinement path
# ---------------------------------------------------------------------------


def test_ghost_of():
    assert ghost_of(2) == 1
    assert ghost_of(4) == 2


# ---------------------------------------------------------------------------
# SAN-30 driver controls: domain-keeping refinement, coarsening + budgets,
# boundary-class failure classification
# ---------------------------------------------------------------------------
class TestAdaptivityNewKnobs:
    def test_defaults(self):
        a = AdaptivitySpec.parse({})
        assert a.refine_keeps_domain is False
        assert a.regrid_rtol == pytest.approx(2.0e-2)
        assert a.support_fraction == pytest.approx(0.85)
        assert a.max_widenings == 2

    def test_parse(self):
        a = AdaptivitySpec.parse(
            {
                "refine_keeps_domain": True,
                "regrid_rtol": 1e-3,
                "support_fraction": 0.8,
                "max_widenings": 3,
            }
        )
        assert a.refine_keeps_domain is True
        assert a.regrid_rtol == pytest.approx(1e-3)
        assert a.support_fraction == pytest.approx(0.8)
        assert a.max_widenings == 3

    def test_support_fraction_must_precede_boundary_stop(self):
        with pytest.raises(SpecError, match="support_fraction"):
            AdaptivitySpec.parse({"support_fraction": 0.95, "boundary_fraction": 0.95})
        with pytest.raises(SpecError, match="support_fraction"):
            AdaptivitySpec.parse({"support_fraction": 0.99})

    def test_regrid_rtol_validation(self):
        with pytest.raises(SpecError, match="regrid_rtol"):
            AdaptivitySpec.parse({"regrid_rtol": 0.0})


class TestWouldAmputate:
    """The refinement-support guard: only domain-shrinking regrids."""

    def test_legacy_shrink_amputates(self):
        # source: dr=0.125, N=128, r99=15.6 (94.5% of the 16.5 boundary,
        # the driver's (N+2·ghost)·dr convention); legacy refinement
        # dr=0.0625 at N=128 → boundary 8.25, far inside the support.
        assert would_amputate(15.6, 0.125, 128, 0.0625, 128, 4, 0.95)

    def test_keep_domain_never_amputates(self):
        # N×2 with dr÷2 keeps the interior domain (16 = 16); the ghost-incl.
        # boundary moves 16.5 → 16.25 but the support fits outright.
        assert not would_amputate(15.6, 0.125, 128, 0.0625, 256, 4, 0.95)
        # even the seed's 96%-of-boundary support (r99=15.685 of 16.25)
        assert not would_amputate(15.685, 0.125, 128, 0.0625, 256, 4, 0.95)

    def test_fit_shrink_ok(self):
        # a compact field fits the smaller domain
        assert not would_amputate(4.0, 0.125, 128, 0.0625, 128, 4, 0.95)

    def test_unknown_support_is_safe(self):
        assert not would_amputate(None, 0.125, 128, 0.0625, 128, 4, 0.95)


class TestCoarseningAcceptable:
    def test_within_rtol(self):
        assert coarsening_acceptable({"omega": 1e-4, "M_Komar": 2e-3}, 2e-2)

    def test_beyond_rtol(self):
        assert not coarsening_acceptable({"omega": 5e-2}, 2e-2)

    def test_empty_is_reject(self):
        assert not coarsening_acceptable({}, 2e-2)


class TestDecideCoarsening:
    def test_support_below_trigger(self):
        assert decide_coarsening(0.5, 0.85, 0.125, 0.5, 2) == "ok"
        assert decide_coarsening(None, 0.85, 0.125, 0.5, 2) == "ok"

    def test_support_over_trigger_widens(self):
        assert decide_coarsening(0.9, 0.85, 0.125, 0.5, 2) == "regrid_coarser"

    def test_dr_floor_stops(self):
        # dr×2 would exceed the coarseness floor
        assert decide_coarsening(0.9, 0.85, 0.4, 0.5, 2) == "stop_budget"
        assert decide_coarsening(0.9, 0.85, 0.25, 0.5, 2) == "regrid_coarser"

    def test_budget_exhausted_stops(self):
        assert decide_coarsening(0.9, 0.85, 0.125, 0.5, 0) == "stop_budget"


class TestBoundaryFailureStop:
    def test_support_fills_domain(self):
        # r99=15.9 of the 16.5 boundary (N=128, dr=0.125, order 4): 96% > 95%
        assert boundary_failure_stop(15.9, 0.125, 128, 4, 0.95)

    def test_support_fits(self):
        assert not boundary_failure_stop(13.1, 0.125, 128, 4, 0.95)

    def test_unknown_support(self):
        assert not boundary_failure_stop(None, 0.125, 128, 4, 0.95)
        assert not boundary_failure_stop(15.6, None, 128, 4, 0.95)
        assert not boundary_failure_stop(15.6, 0.125, None, 4, 0.95)


class TestNewStateFieldsRoundTrip:
    def test_regrid_to_n_roundtrip(self):
        info = RegridInfo(
            from_dr=0.125,
            to_dr=0.0625,
            source="src",
            rel_diff={"omega": 1e-4},
            accepted=True,
            to_n=256,
        )
        d = info.to_dict()
        assert d["to_n"] == 256
        assert RegridInfo.from_dict(d).to_n == 256

    def test_regrid_to_n_omitted_for_legacy(self):
        info = RegridInfo(from_dr=0.125, to_dr=0.0625, source=None, rel_diff={}, accepted=True)
        assert "to_n" not in info.to_dict()
        assert RegridInfo.from_dict(info.to_dict()).to_n is None

    def test_widenings_left_omitted_when_unset(self):
        state = CampaignState(spec_hash="t")
        assert "widenings_left" not in state.to_dict()
        state2 = CampaignState(spec_hash="t", widenings_left=3)
        d = state2.to_dict()
        assert d["widenings_left"] == 3
        assert CampaignState.from_dict(d).widenings_left == 3
