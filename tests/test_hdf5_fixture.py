"""Regression fixture tests: tests/fixtures/l1_smoke_n64.h5.

The repo's only tracked numerics (M2 definition of done): a small HDF5
solution (l=1, w=0.95, from-scratch ansatz, N=64/dr=0.25 — the CI smoke's
grid) that pins the HDF5 output schema and provides the roundtrip gate.
Regenerate via configs/l1_smoke_n64_hdf5.toml (see its header).

These tests run in CI's python-checks job; they require only h5py — no
golden data, no solver build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from rotboson_io import hdf5_to_asc  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "l1_smoke_n64.h5"

# Scalar observables pinned from the fixture itself (the solver's values at
# the N=64 CI grid). A regenerated fixture may differ in the last digits
# (OpenMP reduction noise ~1e-12); the pins guard the schema and physics,
# not bit-identity of a re-run.
PINNED_SCALARS: dict[str, float] = {
    "error_code.asc": 0.0,
    "w_f.asc": 0.95,
    "M_ADM.asc": 8.399858878e-01,
    "M_Komar1.asc": 7.882675516e-01,
    "M_Komar2.asc": 7.882660517e-01,
    "J_Komar1.asc": 7.900647626e-01,
    "J_Komar2.asc": 7.900645193e-01,
    "phi_max.asc": 1.235309468e-02,
    "r99.asc": 1.582979393e01,
    "GRV2.asc": 1.369286286e-03,
    "GRV3.asc": 9.169771211e-01,
    "hwl_resolution.asc": 43.0,
    "ergoregion_flag.asc": 0.0,
}


@pytest.fixture(scope="module")
def h5():
    with h5py.File(FIXTURE, "r") as f:
        yield f


def test_fixture_schema(h5: h5py.File) -> None:
    """70 field/scalar datasets + 62 provenance attributes (writer contract)."""
    datasets = set(h5.keys())
    assert len(datasets) == 70
    for name in PINNED_SCALARS:
        assert name in datasets
    assert len(h5.attrs) == 62
    for attr in ("git_hash", "parfile", "output_backend", "created", "format_version"):
        assert attr in h5.attrs, f"missing provenance attr {attr}"
    assert bytes(h5.attrs["output_backend"]) == b"hdf5"
    assert int(h5.attrs["NrInterior"]) == 64
    assert float(h5.attrs["dr"]) == 0.25


def test_fixture_scalars(h5: h5py.File) -> None:
    for name, expected in PINNED_SCALARS.items():
        got = float(np.ravel(np.array(h5[name]))[-1])
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), name


def test_fixture_self_consistency(h5: h5py.File) -> None:
    """The converged-solve identity: M_Komar1 = M_Komar2 (robust criterion)."""
    m1 = float(np.ravel(np.array(h5["M_Komar1.asc"]))[-1])
    m2 = float(np.ravel(np.array(h5["M_Komar2.asc"]))[-1])
    assert m1 == pytest.approx(m2, rel=2e-6)
    assert m1 > 0.5  # a boson-star solution, not the trivial attractor
    assert float(np.ravel(np.array(h5["error_code.asc"]))[-1]) == 0.0


def test_fixture_roundtrip_lossless(h5: h5py.File, tmp_path: Path) -> None:
    """The documented lossless property: solution.h5 -> .asc -> re-parse is
    bit-exact (the %9.18E format round-trips doubles exactly)."""
    out_dir = tmp_path / "asc"
    written = hdf5_to_asc(FIXTURE, out_dir)
    assert written
    datasets = {p.name for p in written}
    for name in sorted(datasets):
        asc = np.loadtxt(out_dir / name, ndmin=1)
        ref = np.array(h5[name])
        assert asc.shape == ref.shape, name
        assert np.array_equal(asc, ref), f"roundtrip not bit-exact: {name}"
