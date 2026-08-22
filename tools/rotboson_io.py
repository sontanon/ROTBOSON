"""Shared I/O helpers for ROTBOSON tooling.

Legacy ROTBOSON output formats:
- 1D files: one %9.18E value per line (scalars, grids, histories).
- 2D files: NrTotal rows x NzTotal tab-separated columns.
- integer files: one %lld per line (e.g. error_code.asc).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SOLUTION_DIR_RE = re.compile(r"^l=\d+,w=[\d.Ee+-]+,dr=[\d.Ee+-]+,N=\d+$")

SCALAR_FILES = [
    "w_f.asc",
    "GRV2.asc",
    "GRV3.asc",
    "r99.asc",
    "rr_phi_max.asc",
    "phi_max.asc",
    "ergoregion_flag.asc",
]

# Radial-profile files: the physically meaningful value is the LAST entry
# (evaluated at the outer boundary rr_inf).
PROFILE_FILES = [
    "M_ADM.asc",
    "M_Komar1.asc",
    "M_Komar2.asc",
    "M_Schwarz.asc",
    "J_Komar1.asc",
    "J_Komar2.asc",
]

FIELD_FILES = [
    "log_alpha_f.asc",
    "beta_f.asc",
    "log_h_f.asc",
    "log_a_f.asc",
    "psi_f.asc",
    "lambda_f.asc",
    "sph_log_alpha_f.asc",
    "sph_beta_f.asc",
    "sph_log_h_f.asc",
    "sph_log_a_f.asc",
    "sph_psi_f.asc",
    "sph_lambda_f.asc",
]


def read_1d(path: str | Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path))


def read_2d(path: str | Path) -> np.ndarray:
    return np.loadtxt(path)


def read_scalar(path: str | Path) -> float:
    data = read_1d(path)
    return float(np.ravel(data)[0])


def is_solution_dir(name: str) -> bool:
    return bool(SOLUTION_DIR_RE.match(name))


def find_solution_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and is_solution_dir(p.name))


def extract_scalars(sol_dir: str | Path) -> dict[str, float | int]:
    """Read all scalar observables from a solution directory.

    M_*/J_* files are radial profiles; their last entry (outer boundary)
    is extracted as the scalar value.
    """
    sol_dir = Path(sol_dir)
    out: dict[str, float | int] = {}
    for fname in SCALAR_FILES:
        f = sol_dir / fname
        if f.exists():
            out[fname] = read_scalar(f)
    for fname in PROFILE_FILES:
        f = sol_dir / fname
        if f.exists():
            out[fname] = float(read_1d(f)[-1])
    f = sol_dir / "error_code.asc"
    if f.exists():
        out["error_code.asc"] = int(read_1d(f)[0])
    return out


def compare_scalars(
    ref: dict, new: dict, rtol: float = 1e-10, atol: float = 1e-12
) -> tuple[bool, list[str]]:
    """Compare scalar dicts; return (ok, report lines).

    A value passes if it agrees relatively (rtol) OR absolutely (atol);
    the latter matters for near-zero quantities where relative error is
    meaningless.
    """
    ok = True
    lines = []
    keys = sorted(set(ref) & set(new))
    for key in keys:
        r, n = ref[key], new[key]
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            diff = abs(float(r) - float(n))
            rel = diff / max(abs(float(r)), 1e-300)
            status = "PASS" if (diff <= atol or rel <= rtol) else "FAIL"
            ok &= status == "PASS"
            lines.append(f"  {key:22s} ref={float(r):+.16e} new={float(n):+.16e} "
                         f"abs_diff={diff:.3e} rel_diff={rel:.3e} {status}")
        else:
            match = "PASS" if r == n else "FAIL"
            ok &= match == "PASS"
            lines.append(f"  {key:22s} ref={r} new={n} {match}")
    missing = set(ref) - set(new)
    for key in sorted(missing):
        lines.append(f"  {key:22s} MISSING in new")
        ok = False
    return ok, lines


def compare_fields(ref_dir: str | Path, new_dir: str | Path, rtol: float = 1e-10, atol: float = 1e-12) -> tuple[bool, list[str]]:
    """Compare 2D field files between two solution dirs; return (ok, report)."""
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    ok = True
    lines = []
    for fname in FIELD_FILES:
        r, n = ref_dir / fname, new_dir / fname
        if not r.exists() or not n.exists():
            lines.append(f"  {fname:22s} {'missing ref' if not r.exists() else 'missing new'}")
            ok = False
            continue
        a, b = read_2d(r), read_2d(n)
        if a.shape != b.shape:
            lines.append(f"  {fname:22s} SHAPE MISMATCH ref={a.shape} new={b.shape}")
            ok = False
            continue
        denom = np.maximum(np.abs(a), 1e-300)
        diff = np.abs(a - b)
        rel = diff / denom
        max_rel = float(rel.max())
        max_abs = float(diff.max())
        status = "PASS" if (max_abs <= atol or max_rel <= rtol) else "FAIL"
        ok &= status == "PASS"
        lines.append(f"  {fname:22s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} {status}")
    return ok, lines
