"""psi0_origin_estimate: grid-independent ψ₀ label (even quadratic fit)."""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from sweep_driver import psi0_origin_estimate  # noqa: E402


def _full_grid(core):
    """Embed a 3x3 (or larger) interior patch in a ghosted grid."""
    n = core.shape[0] + 4
    g = np.zeros((n, n))
    g[2 : 2 + core.shape[0], 2 : 2 + core.shape[1]] = core
    return g


def test_exact_for_quadratic_profile():
    # ψ = 1 - 0.3u² - 0.2v² on interior nodes u,v = 0.5,1.5,2.5 — the fit
    # model is exact, so the origin value must be recovered exactly.
    u = np.arange(2, 5) - 1.5
    U, V = np.meshgrid(u, u, indexing="ij")
    core = 1.0 - 0.3 * U**2 - 0.2 * V**2
    assert psi0_origin_estimate(_full_grid(core)) == pytest.approx(1.0)


def test_grid_independence_for_gaussian():
    # The same smooth field sampled on two grid spacings must give the same
    # origin estimate (this is what kills the +0.85% regrid label jump).
    for n in (3, 6, 9):
        k = np.arange(2, 2 + n) - 1.5
        U, V = np.meshgrid(k * 0.125, k * 0.125, indexing="ij")
        core = np.exp(-(U**2 + V**2) * 0.01)
        val = psi0_origin_estimate(_full_grid(core))
        assert abs(val - 1.0) < 5e-4, val
