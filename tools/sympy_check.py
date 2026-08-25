"""Cross-check the SymPy re-derivation against the codegen notebook.

Parses the 6x31 Jacobian strings stored in
``derivations/notebooks/Mathematica CSR Code Generation.ipynb`` (produced in
Mathematica and pasted by hand) and compares them numerically against the
independent SymPy Jacobian from :mod:`tools.sympy_system`, at a few hundred
random sample points.  Exit code 0 iff every entry agrees.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import nbformat
import numpy as np
import sympy as sp
import sympy_system as ss

NB_PATH = str(
    Path(__file__).resolve().parent.parent
    / "derivations/notebooks/Mathematica CSR Code Generation.ipynb"
)


def load_notebook_jacobian(path: str = NB_PATH) -> list[list[str]]:
    """Return the 6x31 raw string Jacobian from the codegen notebook."""
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.lstrip().startswith("jacobian = ["):
            ns: dict = {}
            exec(cell.source, ns)
            return ns["jacobian"]
    raise RuntimeError("jacobian list not found in notebook")


def notebook_symbols() -> dict[str, sp.Symbol]:
    names = []
    for var in ("dRu", "dZu", "dRRu", "dZZu"):
        names += [f"{var}{k}" for k in range(1, 7)]
    names += [
        "ri",
        "r",
        "r2",
        "dr",
        "dr2",
        "dzodr",
        "drodz",
        "l",
        "m",
        "m2",
        "w",
        "rl",
        "rlm1",
        "Q1",
        "Q2",
        "alpha2",
        "h2",
        "a2",
        "a2_r",
        "phi",
        "phior",
        "phi2",
        "phi2or2",
        "wplOmega",
        "wplOmega2",
        "lam",
        "psi",
    ]
    syms = {n: sp.symbols(n, real=True) for n in names}
    syms["pi"] = sp.pi
    return syms


def parse_nb(expr_str: str, ns: dict[str, sp.Expr]) -> sp.Expr:
    code = expr_str.replace("M_PI", "pi").replace("lambda", "lam")
    return sp.sympify(code, locals=ns)


def sample_point(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    vals: dict[str, float] = {}
    for var in ("dRu", "dZu", "dRRu", "dZZu"):
        for k in range(1, 7):
            vals[f"{var}{k}"] = rng.uniform(-2.0, 2.0)
    vals.update(
        {
            "u1": rng.uniform(-1.0, 1.0),
            "u2": rng.uniform(-0.5, 0.5),
            "u3": rng.uniform(-1.0, 1.0),
            "u4": rng.uniform(-1.0, 1.0),
            "u5": rng.uniform(0.0, 2.0),
            "u6": rng.uniform(-0.5, 0.5),
            "ri": rng.uniform(0.5, 20.0),
            "l": float(rng.randint(1, 6)),
            "m": rng.uniform(0.5, 2.0),
            "m2": 0.0,  # filled below
            "w": rng.uniform(0.1, 0.9),
            "Q1": 1.0,
            "Q2": 1.0,
            "dr": rng.uniform(0.02, 0.2),
            "dz": rng.uniform(0.02, 0.2),
        }
    )
    vals["m2"] = vals["m"] ** 2
    dr, dz = vals["dr"], vals["dz"]
    vals["dr2"] = dr * dr
    vals["dz2"] = dz * dz
    vals["dzodr"] = dz / dr
    vals["drodz"] = dr / dz
    vals["r"] = vals["ri"] * dr
    vals["r2"] = vals["r"] ** 2
    vals["rlm1"] = 1.0 if vals["l"] == 1 else vals["r"] ** (vals["l"] - 1)
    vals["rl"] = vals["rlm1"] * vals["r"]
    return vals


def derived(vals: dict[str, float]) -> dict[str, float]:
    d = dict(vals)
    d["alpha2"] = np.exp(2 * vals["u1"])
    d["h2"] = np.exp(2 * vals["u3"])
    d["a2"] = np.exp(2 * vals["u4"])
    d["lam"] = vals["u6"]
    d["psi"] = vals["u5"]
    d["wplOmega"] = vals["w"] + vals["l"] * vals["u2"]
    d["wplOmega2"] = d["wplOmega"] ** 2
    d["phi"] = vals["rl"] * vals["u5"]
    d["phior"] = vals["rlm1"] * vals["u5"]
    d["phi2"] = d["phi"] ** 2
    d["phi2or2"] = d["phior"] ** 2
    d["a2_r"] = d["h2"] + vals["r2"] * vals["u6"]
    return d


def check() -> int:
    s, R, f, J = ss.build()
    nb_jac = load_notebook_jacobian()
    ns = notebook_symbols()

    # Lambdify my Jacobian over the full base symbol set.
    base_syms = [s[n] for n in ["u1", "u2", "u3", "u4", "u5", "u6"]]
    for var in ("dRu", "dZu", "dRRu", "dZZu"):
        base_syms += [s[f"{var}{k}"] for k in range(1, 7)]
    base_syms += [
        s[n]
        for n in (
            "ri",
            "r",
            "r2",
            "dr",
            "dz",
            "dr2",
            "dz2",
            "dzodr",
            "drodz",
            "l",
            "m",
            "m2",
            "w",
            "rl",
            "rlm1",
            "Q1",
            "Q2",
        )
    ]

    ns_syms = [
        ns[n]
        for n in (
            "dRu1",
            "dRu2",
            "dRu3",
            "dRu4",
            "dRu5",
            "dRu6",
            "dZu1",
            "dZu2",
            "dZu3",
            "dZu4",
            "dZu5",
            "dZu6",
            "dRRu1",
            "dRRu2",
            "dRRu3",
            "dRRu4",
            "dRRu5",
            "dRRu6",
            "dZZu1",
            "dZZu2",
            "dZZu3",
            "dZZu4",
            "dZZu5",
            "dZZu6",
            "ri",
            "r",
            "r2",
            "dr",
            "dr2",
            "dzodr",
            "drodz",
            "l",
            "m",
            "m2",
            "w",
            "rl",
            "rlm1",
            "Q1",
            "Q2",
            "alpha2",
            "h2",
            "a2",
            "a2_r",
            "phi",
            "phior",
            "phi2",
            "phi2or2",
            "wplOmega",
            "wplOmega2",
            "lam",
            "psi",
        )
    ]

    # Prepare my expressions via substitution to the notebook namespace so a
    # single numeric evaluation path can be reused.
    # (simpler: lambdify separately and compare)
    my_fn = [[sp.lambdify(base_syms, J[i][j], "numpy") for j in range(31)] for i in range(6)]
    nb_fn = [
        [sp.lambdify(ns_syms, parse_nb(nb_jac[i][j], ns), "numpy") for j in range(31)]
        for i in range(6)
    ]

    nbad = 0
    npts = 300
    for seed in range(npts):
        vals = sample_point(seed)
        dvals = derived(vals)
        base_args = [vals[n] for n in ("u1", "u2", "u3", "u4", "u5", "u6")]
        for var in ("dRu", "dZu", "dRRu", "dZZu"):
            base_args += [vals[f"{var}{k}"] for k in range(1, 7)]
        base_args += [
            vals[n]
            for n in (
                "ri",
                "r",
                "r2",
                "dr",
                "dz",
                "dr2",
                "dz2",
                "dzodr",
                "drodz",
                "l",
                "m",
                "m2",
                "w",
                "rl",
                "rlm1",
                "Q1",
                "Q2",
            )
        ]

        nb_args = [
            dvals[n]
            for n in (
                "dRu1",
                "dRu2",
                "dRu3",
                "dRu4",
                "dRu5",
                "dRu6",
                "dZu1",
                "dZu2",
                "dZu3",
                "dZu4",
                "dZu5",
                "dZu6",
                "dRRu1",
                "dRRu2",
                "dRRu3",
                "dRRu4",
                "dRRu5",
                "dRRu6",
                "dZZu1",
                "dZZu2",
                "dZZu3",
                "dZZu4",
                "dZZu5",
                "dZZu6",
                "ri",
                "r",
                "r2",
                "dr",
                "dr2",
                "dzodr",
                "drodz",
                "l",
                "m",
                "m2",
                "w",
                "rl",
                "rlm1",
                "Q1",
                "Q2",
                "alpha2",
                "h2",
                "a2",
                "a2_r",
                "phi",
                "phior",
                "phi2",
                "phi2or2",
                "wplOmega",
                "wplOmega2",
                "lam",
                "psi",
            )
        ]

        for i in range(6):
            for j in range(31):
                a = float(my_fn[i][j](*base_args))
                b = float(nb_fn[i][j](*nb_args))
                denom = max(abs(a), abs(b), 1.0)
                if not np.isclose(a, b, rtol=1e-9, atol=1e-11 * denom):
                    nbad += 1
                    if nbad <= 20:
                        print(
                            f"MISMATCH seed={seed} row={i} col={j}: sympy={a:.6e} notebook={b:.6e}"
                        )
    if nbad:
        print(f"FAIL: {nbad} mismatches across {npts} samples")
        return 1
    print(f"OK: SymPy Jacobian matches notebook at {npts} random points (6x31 entries)")
    return 0


if __name__ == "__main__":
    sys.exit(check())
