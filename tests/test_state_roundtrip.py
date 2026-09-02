"""state.json round-trip tests.

The frozen-dataclass state model must be compatible with in-flight campaigns:

* the CURRENT writer's shape must round-trip **byte-identically**
  — resuming must not churn state.json formats mid-campaign;
* legacy v1 state.json files (minimal top-level keys, per-step
  extras like `step_factor`) must load and resume; a re-save upgrades the
  file, dropping keys the v2 model no longer tracks.
"""

import json
import sys
from pathlib import Path
from typing import cast

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from sweep_driver import CampaignState, StepMode, StepRecord  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> tuple[str, dict[str, object]]:
    text = (FIXTURES / name).read_text()
    return text, cast("dict[str, object]", json.loads(text))


def raw_steps(raw: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", raw["steps"])


def raw_probes(raw: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", raw["rejected_regrids"])


def raw_fold(raw: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", raw["fold_fine_grid_measurement"])


class TestCurrentWriterByteRoundTrip:
    """The v2 (current) state.json shape re-serializes byte-identically."""

    def test_byte_identical_round_trip(self):
        text, raw = load_fixture("state_v2_current.json")
        state = CampaignState.from_dict(raw)
        assert json.dumps(state.to_dict(), indent=2) + "\n" == text

    def test_double_round_trip_is_stable(self):
        text, raw = load_fixture("state_v2_current.json")
        once = CampaignState.from_dict(raw).to_dict()
        twice = CampaignState.from_dict(once).to_dict()
        assert json.dumps(twice, indent=2) + "\n" == text

    def test_data_preserved(self):
        _, raw = load_fixture("state_v2_current.json")
        state = CampaignState.from_dict(raw)
        assert state.spec_hash == raw["spec_hash"]
        assert len(state.steps) == len(raw_steps(raw))
        assert [s.i for s in state.steps] == [r["i"] for r in raw_steps(raw)]
        assert [s.omega for s in state.steps] == [r["omega"] for r in raw_steps(raw)]
        assert [s.psi0 for s in state.steps] == [r["psi0"] for r in raw_steps(raw)]
        assert state.refinements_left == raw["refinements_left"]
        probes = state.rejected_regrids
        assert probes is not None and len(probes) == len(raw_probes(raw))
        assert [p.psi0 for p in probes] == [r.get("psi0") for r in raw_probes(raw)]
        assert probes[0].scalars == raw_probes(raw)[0]["scalars"]
        fold = state.fold_fine_grid_measurement
        assert fold is not None
        assert [m.sol_dir for m in fold] == [m["sol_dir"] for m in raw_fold(raw)]
        assert fold[0].note == raw_fold(raw)[0]["note"]

    def test_int_scalars_keep_identity(self):
        # int-valued scalars (e.g. error_code) must not become floats
        _, raw = load_fixture("state_v2_current.json")
        state = CampaignState.from_dict(raw)
        assert state.rejected_regrids is not None
        probe = state.rejected_regrids[0]
        assert probe.scalars is not None
        orig_scalars = cast("dict[str, object]", raw_probes(raw)[0]["scalars"])
        for key, value in probe.scalars.items():
            assert type(value) is type(orig_scalars[key])


class TestLegacyCompat:
    """Legacy v1 state.json files load; the model keeps the modeled data."""

    @pytest.fixture
    def state(self):
        _, raw = load_fixture("state_legacy_v1.json")
        return CampaignState.from_dict(raw), raw

    def test_legacy_loads(self, state):
        s, raw = state
        assert s.status.value == raw["status"]
        assert s.stop_reason == raw["stop_reason"]
        assert len(s.steps) == len(raw["steps"])
        # legacy files predate the adaptive keys
        assert s.grid is None
        assert s.refinements_left is None

    def test_legacy_modeled_data_survives(self, state):
        s, raw = state
        for raw_step in raw_steps(raw):
            modeled = StepRecord.from_dict(raw_step).to_dict()
            # every key the current writer emits matches the original value
            for key, value in modeled.items():
                if key == "mode":
                    assert value == StepMode(raw_step[key])
                else:
                    assert value == raw_step[key]

    def test_legacy_semantic_round_trip(self, state):
        # to_dict(from_dict(x)) == from_dict(x) (normalized form is stable)
        s, _ = state
        once = s.to_dict()
        twice = CampaignState.from_dict(once).to_dict()
        assert once == twice


class TestSyntheticRecords:
    """Failed records and regrid bookkeeping round-trip."""

    def test_failed_step_record(self):
        rec = StepRecord(i=3, exit_code=2, psi0_target=0.42)
        assert CampaignState.from_dict(rec.to_dict()).steps == []
        parsed = StepRecord.from_dict(rec.to_dict())
        assert parsed == rec

    def test_regrid_record(self):
        from sweep_driver import RegridInfo

        rec = StepRecord(
            i=7,
            exit_code=0,
            mode=StepMode.REGGRID,
            sol_dir="/tmp/sol",
            dr=0.0625,
            N=128,
            omega=0.91,
            psi0=0.31,
            regrid=RegridInfo(
                from_dr=0.125,
                to_dr=0.0625,
                source="/tmp/src",
                rel_diff={"omega": 1e-4},
                accepted=True,
            ),
        )
        assert StepRecord.from_dict(rec.to_dict()) == rec

    def test_minimal_state(self):
        state = CampaignState(spec_hash="abc")
        assert state.to_dict() == {
            "spec_hash": "abc",
            "spec_file": None,
            "steps": [],
            "status": "running",
            "stop_reason": None,
            "turning_point": None,
            "pending_refinement": None,
        }
        assert CampaignState.from_dict(state.to_dict()) == state
