"""Shared I/O helpers for ROTBOSON tooling.

Legacy ROTBOSON output formats:
- 1D files: one %9.18E value per line (scalars, grids, histories).
- 2D files: NrTotal rows x NzTotal tab-separated columns.
- integer files: one %lld per line (e.g. error_code.asc).
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

SOLUTION_DIR_RE: Final = re.compile(r"^l=\d+,w=[\d.Ee+-]+,dr=[\d.Ee+-]+,N=\d+$")

SCALAR_FILES: Final[tuple[str, ...]] = (
    "w_f.asc",
    "GRV2.asc",
    "GRV3.asc",
    "r99.asc",
    "rr_phi_max.asc",
    "phi_max.asc",
    "hwl_resolution.asc",
    "ergoregion_flag.asc",
)

# Radial-profile files: the physically meaningful value is the LAST entry
# (evaluated at the outer boundary rr_inf).
PROFILE_FILES: Final[tuple[str, ...]] = (
    "M_ADM.asc",
    "M_Komar1.asc",
    "M_Komar2.asc",
    "M_Schwarz.asc",
    "J_Komar1.asc",
    "J_Komar2.asc",
)

FIELD_FILES: Final[tuple[str, ...]] = (
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
)


def read_1d(path: str | Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path))


def read_2d(path: str | Path) -> np.ndarray:
    return np.loadtxt(path)


def read_scalar(path: str | Path) -> float:
    data = read_1d(path)
    return float(np.ravel(data)[0])


# ---------------------------------------------------------------------------
# HDF5 (Phase 5). The single-file backend stores one dataset per legacy
# ".asc" file, named "<field>.asc", plus scalar attributes (params, solver
# settings, git hash, analysis results).
# ---------------------------------------------------------------------------

HDF5_FILENAME = "solution.h5"


def read_hdf5(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read a solution.h5 into (datasets, attributes).

    Dataset keys are the legacy file names (e.g. "w_f.asc"); attribute values
    are decoded from bytes to str where applicable.
    """
    import h5py

    path = Path(path)
    datasets: dict[str, np.ndarray] = {}
    attrs: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            datasets[key] = np.asarray(f[key][...])
        for key, val in f.attrs.items():
            attrs[key] = val.decode() if isinstance(val, bytes) else val
    return datasets, attrs


def hdf5_to_asc(h5_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Export a solution.h5 back to the legacy .asc file layout.

    Returns the list of written files. Doubles are written with the same
    %9.18E format the C ASCII backend uses (round-trips exactly); integer
    fields as one %lld per line.
    """
    datasets, _ = read_hdf5(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, arr in datasets.items():
        arr = np.asarray(arr)
        path = out_dir / name
        if arr.dtype.kind in "iu":
            with path.open("w") as fh:
                fh.write("\n".join(str(int(v)) for v in arr.reshape(-1)) + "\n")
        else:
            kwargs = {"fmt": "%9.18E"}
            if arr.ndim > 1:
                kwargs["delimiter"] = "\t"
            np.savetxt(path, arr, **kwargs)
        written.append(path)
    return written


def extract_scalars_from_hdf5(sol_dir: str | Path) -> dict[str, float | int]:
    """Scalar observables from a solution.h5, mirroring extract_scalars()."""
    datasets, _ = read_hdf5(Path(sol_dir) / HDF5_FILENAME)
    out: dict[str, float | int] = {}
    for fname in SCALAR_FILES:
        if fname in datasets:
            out[fname] = float(np.ravel(datasets[fname])[0])
    for fname in PROFILE_FILES:
        if fname in datasets:
            out[fname] = float(np.ravel(datasets[fname])[-1])
    if "error_code.asc" in datasets:
        out["error_code.asc"] = int(np.ravel(datasets["error_code.asc"])[0])
    return out


def compare_hdf5_to_ascii(
    hdf5_dir: str | Path, ascii_dir: str | Path, rtol: float = 1e-10, atol: float = 1e-12
) -> tuple[bool, list[str]]:
    """Compare a solution.h5 against a legacy .asc directory (field + scalars)."""
    hdf5_dir, ascii_dir = Path(hdf5_dir), Path(ascii_dir)
    datasets, _ = read_hdf5(hdf5_dir / HDF5_FILENAME)
    ok = True
    lines: list[str] = []

    for fname in SCALAR_FILES + PROFILE_FILES + FIELD_FILES:
        asc = ascii_dir / fname
        if not asc.exists():
            lines.append(f"  {fname:22s} missing .asc reference")
            ok = False
            continue
        ref = read_1d(asc) if fname in SCALAR_FILES + PROFILE_FILES else read_2d(asc)
        if fname not in datasets:
            lines.append(f"  {fname:22s} missing in solution.h5")
            ok = False
            continue
        new = np.asarray(datasets[fname])
        if fname in PROFILE_FILES:
            ref = np.atleast_1d(ref)[-1]
            new = np.ravel(new)[-1]
        if np.shape(ref) != np.shape(new):
            lines.append(f"  {fname:22s} SHAPE MISMATCH ref={np.shape(ref)} new={np.shape(new)}")
            ok = False
            continue
        diff = np.abs(ref - new)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float((diff / np.maximum(np.abs(ref), 1e-300)).max()) if diff.size else 0.0
        status = "PASS" if (max_abs <= atol or max_rel <= rtol) else "FAIL"
        ok &= status == "PASS"
        lines.append(f"  {fname:22s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} {status}")

    # error_code integer field.
    ec = ascii_dir / "error_code.asc"
    if ec.exists() and "error_code.asc" in datasets:
        ref = int(np.ravel(read_1d(ec))[0])
        new = int(np.ravel(datasets["error_code.asc"])[0])
        status = "PASS" if ref == new else "FAIL"
        ok &= status == "PASS"
        lines.append(f"  error_code.asc      ref={ref} new={new} {status}")

    return ok, lines


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
    ref: Mapping[str, object], new: Mapping[str, object], rtol: float = 1e-10, atol: float = 1e-12
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
        if (
            isinstance(r, (int, float))
            and not isinstance(r, bool)
            and isinstance(n, (int, float))
            and not isinstance(n, bool)
        ):
            diff = abs(float(r) - float(n))
            rel = diff / max(abs(float(r)), 1e-300)
            status = "PASS" if (diff <= atol or rel <= rtol) else "FAIL"
            ok &= status == "PASS"
            lines.append(
                f"  {key:22s} ref={float(r):+.16e} new={float(n):+.16e} "
                f"abs_diff={diff:.3e} rel_diff={rel:.3e} {status}"
            )
        else:
            match = "PASS" if r == n else "FAIL"
            ok &= match == "PASS"
            lines.append(f"  {key:22s} ref={r} new={n} {match}")
    missing = set(ref) - set(new)
    for key in sorted(missing):
        lines.append(f"  {key:22s} MISSING in new")
        ok = False
    return ok, lines


def compare_fields(
    ref_dir: str | Path, new_dir: str | Path, rtol: float = 1e-10, atol: float = 1e-12
) -> tuple[bool, list[str]]:
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
