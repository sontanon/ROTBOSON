"""SymPy re-derivation of the ROTBOSON Einstein--Klein--Gordon system.

This module is the single source of truth for the 6 coupled PDE residuals
and their Jacobian, as an independent re-derivation from the equations in
`rhs_vars.c` (the checked-in residual kernel) and cross-checked against the
Jacobian strings in `Mathematica CSR Code Generation.ipynb`.

Conventions (identical to the codegen notebook so expressions can be compared
term-for-term):

* The six grid unknowns are ``u1..u6``:
    u1 = log(alpha)  (lapse),   u2 = beta  (shift),
    u3 = log(h)      (metric),   u4 = log(a)  (metric),
    u5 = psi         (scalar field rescaled),  u6 = lambda  (regularization).
* For each unknown we track five finite-difference "subtype" variables
  (already step-scaled, so the Jacobian is a plain derivative):
    ``u{k}`` (value), ``dRu{k} = dr*D_r u_k``, ``dZu{k} = dz*D_z u_k``,
    ``dRRu{k} = dr^2*D_rr u_k``, ``dZZu{k} = dz^2*D_zz u_k``.
* The residual handed to the Newton solver is ``f_i = dr^2 * dzodr * R_i``
  where ``R_i`` is the "physics" bracket below (``rescale`` is a common
  factor set to +1 here; the driver passes -1).
* ``omega`` is the physical frequency; it enters via
  ``wplOmega = w + l*beta``.  The code stores ``xi`` (``omega_calc``),
  so the runtime Jacobian w.r.t. ``xi`` is ``dw_du(xi, m) * df_i/dw``.

The lambda equation re-defines ``a2 -> a2_r = h2 + r2*lambda`` (the
regularization variable), and pulls in the auxiliaries
``Dr(alpha)/r`` and ``Dr(H)/r`` whose analytic radial derivatives are
``Dr_u6_aux`` / ``Dr_u7_aux``.
"""

from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


def build_symbols() -> dict[str, sp.Expr]:
    """Return a dict of the symbols shared by residual + Jacobian."""
    s: dict[str, sp.Expr] = {}

    # Grid-function values (u1..u6).
    for name in ("u1", "u2", "u3", "u4", "u5", "u6"):
        s[name] = sp.symbols(name, real=True)

    # Step-scaled first/second derivatives (dr*Dr, dz*Dz, dr^2*Drr, dz^2*Dzz).
    for var in ("dRu", "dZu", "dRRu", "dZZu"):
        for k in range(1, 7):
            s[f"{var}{k}"] = sp.symbols(f"{var}{k}", real=True)

    # Geometry / steps.
    for name in ("ri", "r", "r2", "dr", "dz", "dr2", "dz2", "dzodr", "drodz"):
        s[name] = sp.symbols(name, positive=True)

    # Physics.
    for name in ("l", "m", "m2", "w", "rlm1", "rl", "Q1", "Q2"):
        s[name] = sp.symbols(name, real=True)

    s["pi"] = sp.pi

    # Derived physical variables (as used verbatim by the notebook strings).
    s["alpha2"] = sp.exp(2 * s["u1"])
    s["h2"] = sp.exp(2 * s["u3"])
    s["a2"] = sp.exp(2 * s["u4"])
    s["wplOmega"] = s["w"] + s["l"] * s["u2"]
    s["wplOmega2"] = s["wplOmega"] ** 2
    s["lambda"] = s["u6"]
    s["psi"] = s["u5"]
    s["phi"] = s["rl"] * s["psi"]
    s["phior"] = s["rlm1"] * s["psi"]
    s["phi2"] = s["phi"] ** 2
    s["phi2or2"] = s["phior"] ** 2
    s["a2_r"] = s["h2"] + s["r2"] * s["lambda"]
    # Auxiliary radial derivatives (analytic form, see rhs.c).
    s["Dr_u6_aux"] = (
        sp.exp(s["u1"])
        * (s["dRRu1"] / s["dr2"] - (s["dRu1"] / s["dr"]) / s["r"] + (s["dRu1"] / s["dr"]) ** 2)
        / s["r"]
    )
    s["Dr_u7_aux"] = (
        2
        * sp.exp(2 * s["u3"])
        * (s["dRRu3"] / s["dr2"] - (s["dRu3"] / s["dr"]) / s["r"] + 2 * (s["dRu3"] / s["dr"]) ** 2)
        / s["r"]
    )

    return s


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def residuals(s: dict[str, sp.Expr]) -> list[sp.Expr]:
    """Return the six residual brackets ``R_0..R_5`` (before ``dr2*dzodr``)."""

    def Dr(k: int) -> sp.Expr:
        return s[f"dRu{k}"] / s["dr"]

    def Dz(k: int) -> sp.Expr:
        return s[f"dZu{k}"] / s["dz"]

    def Drr(k: int) -> sp.Expr:
        return s[f"dRRu{k}"] / s["dr2"]

    def Dzz(k: int) -> sp.Expr:
        return s[f"dZZu{k}"] / s["dz2"]

    r = s["r"]
    r2 = s["r2"]
    alpha2 = s["alpha2"]
    h2 = s["h2"]
    a2 = s["a2"]
    l = s["l"]
    m2 = s["m2"]
    wplOmega2 = s["wplOmega2"]
    phi2 = s["phi2"]
    phi2or2 = s["phi2or2"]
    psi = s["psi"]
    lam = s["lambda"]
    a2_r = s["a2_r"]
    pi = s["pi"]
    rlm1 = s["rlm1"]
    Q1 = s["Q1"]
    Q2 = s["Q2"]
    wplOmega = s["wplOmega"]

    R0 = (
        Drr(1)
        + Dzz(1)
        + Dr(1) / r
        + (Dr(1) ** 2 + Dz(1) ** 2)
        + (Dr(1) * Dr(3) + Dz(1) * Dz(3))
        - sp.Rational(1, 2) * (r2 * h2 / alpha2) * (Dr(2) ** 2 + Dz(2) ** 2)
        + 4 * pi * a2 * (m2 - 2 * wplOmega2 / alpha2) * phi2
    )

    R1 = (
        Drr(2)
        + Dzz(2)
        + 3 * Dr(2) / r
        - (Dr(1) * Dr(2) + Dz(1) * Dz(2))
        + 3 * (Dr(2) * Dr(3) + Dz(2) * Dz(3))
        - 16 * pi * a2 * l * wplOmega * phi2or2 / h2
    )

    R2 = (
        Drr(3)
        + Dzz(3)
        + 2 * Dr(3) / r
        + (Dr(3) ** 2 + Dz(3) ** 2)
        + (Dr(1) * Dr(3) + Dz(1) * Dz(3))
        + sp.Rational(1, 2) * (r2 * h2 / alpha2) * (Dr(2) ** 2 + Dz(2) ** 2)
        + Dr(1) / r
        + 4 * pi * a2 * (r2 * m2 + 2 * l * l / h2) * phi2or2
    )

    R3 = (
        Drr(4)
        + Dzz(4)
        - (Dr(1) * Dr(3) + Dz(1) * Dz(3))
        - sp.Rational(1, 4) * (r2 * h2 / alpha2) * (Dr(2) ** 2 + Dz(2) ** 2)
        - Dr(1) / r
        + 4
        * pi
        * (
            (l * l * (1 - a2 / h2) + a2 * r2 * wplOmega2 / alpha2) * phi2or2
            + (2 * l * r * psi * Dr(5) + r2 * (Dr(5) ** 2 + Dz(5) ** 2)) * rlm1 * rlm1
        )
    )

    R4 = (
        Drr(5)
        + Dzz(5)
        + (2 * l + 1) * Dr(5) / r
        + (Dr(1) * Dr(5) + Dz(1) * Dz(5))
        + (Dr(5) * Dr(3) + Dz(5) * Dz(3))
        + l * (Dr(1) / r + Dr(3) / r) * psi
        + a2 * (wplOmega2 / alpha2 - m2) * psi
        - l * l * lam * psi / h2
    )

    R5 = (
        Drr(6)
        + Dzz(6)
        + 3 * Dr(6) / r
        + (Dz(3) * Dz(6) - Dr(3) * Dr(6))
        - 4 * (h2 / a2_r) * (Dr(6) * Dr(3) + Dz(6) * Dz(3))
        - (r2 / a2_r) * (Dr(6) ** 2 + Dz(6) ** 2)
        + (Dz(1) * Dz(6) - Dr(1) * Dr(6))
        - 4 * (lam * lam / a2_r)
        - 4 * (lam / a2_r) * (r * Dr(6))
        - 2 * (Dr(1) / r) * lam
        + 2 * (Dr(3) / r) * (-4 * (h2 / a2_r) + 1) * lam
        - 4 * h2 * (Dr(3) / r) ** 2 * ((h2 / a2_r) + sp.Rational(1, 2))
        + 4 * (h2 / a2_r) * Dz(3) ** 2 * lam
        - 2 * Dr(3) ** 2 * lam
        - 4 * h2 * (Dr(3) / r) * (Dr(1) / r)
        - (h2 * h2 / alpha2) * (Dr(2) ** 2 + Dz(2) ** 2)
        - (a2_r * h2 / alpha2) * (Dr(2) ** 2)
        + 2 * lam * (Drr(1) + Dr(1) ** 2)
        + 2 * lam * (Drr(3) + 2 * Dr(3) ** 2)
        + Q1 * ((2 * h2 / sp.exp(s["u1"])) * (s["Dr_u6_aux"] / r))
        + Q2 * (s["Dr_u7_aux"] / r)
        + 8
        * pi
        * a2_r
        * (m2 * lam * phi2 + 2 * rlm1 * rlm1 * (Dr(5) / r) * (2 * l * psi + r * Dr(5)))
    )

    return [R0, R1, R2, R3, R4, R5]


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def jacobian(s: dict[str, sp.Expr], f: list[sp.Expr]) -> list[list[sp.Expr]]:
    """Return the 6x31 Jacobian ``df_i / d(subtype_j)`` (31 = 6*5 + omega).

    Row ``i`` is the residual ``f_i = dr2*dzodr*R_i``.  Column layout is
    ``[u, dRu, dZu, dRRu, dZZu]`` for each of ``u1..u6``, then ``w``.
    """
    dr2 = s["dr2"]
    dzodr = s["dzodr"]
    # The scalar residual handed to the solver (rescale = +1).
    F = [dr2 * dzodr * R for R in f]

    cols: list[sp.Expr] = []
    for k in range(1, 7):
        cols += [s[f"u{k}"], s[f"dRu{k}"], s[f"dZu{k}"], s[f"dRRu{k}"], s[f"dZZu{k}"]]
    cols += [s["w"]]

    J = [[sp.diff(Fi, c) for c in cols] for Fi in F]
    return J


# ---------------------------------------------------------------------------
# Conversion to the C codegen namespace (identical to the notebook strings)
# ---------------------------------------------------------------------------


def c_namespace_symbols() -> dict[str, sp.Symbol]:
    """Symbols used verbatim as C identifiers in the generated kernels."""
    return {
        n: sp.Symbol(n)
        for n in (
            "alpha2",
            "h2",
            "a2",
            "psi",
            "lambda",
            "wplOmega",
            "wplOmega2",
            "phi",
            "phior",
            "phi2",
            "phi2or2",
            "dr2",
            "drodz",
        )
    }


def to_c(expr: sp.Expr, s: dict[str, sp.Expr]) -> sp.Expr:
    """Rewrite a residual/Jacobian expression into the C kernel namespace.

    The result is a polynomial-style expression over exactly the symbols the
    generated C code defines: ``dRu/dZu/dRRu/dZZu{k}``, ``ri``, ``dr2``,
    ``dzodr``, ``drodz``, ``l``, ``m2``, ``w``, ``rl``, ``rlm1``, ``Q1``,
    ``Q2``, ``alpha2``, ``h2``, ``a2``, ``phi``, ``phior``, ``phi2``,
    ``phi2or2``, ``wplOmega``, ``lambda``, ``psi``.  The step symbols
    ``dr``/``dz``/``dz2`` are eliminated (they collapse into ``dzodr`` /
    ``drodz`` / ``dr2``).
    """
    C = c_namespace_symbols()
    dr, dzodr = s["dr"], s["dzodr"]

    e = expr.subs(
        {
            sp.exp(2 * s["u1"]): C["alpha2"],
            sp.exp(2 * s["u3"]): C["h2"],
            sp.exp(2 * s["u4"]): C["a2"],
            s["u5"]: C["psi"],
            s["u6"]: C["lambda"],
        }
    )
    # beta only ever enters via wplOmega = w + l*beta.
    e = sp.expand(e.subs(s["u2"], (C["wplOmega"] - s["w"]) / s["l"]))
    # Scalar-field products: squares before linear forms.
    e = e.subs(
        {s["rl"] ** 2 * C["psi"] ** 2: C["phi2"], s["rlm1"] ** 2 * C["psi"] ** 2: C["phi2or2"]}
    )
    e = e.subs({s["rl"] * C["psi"]: C["phi"], s["rlm1"] * C["psi"]: C["phior"]})
    # Fold the step algebra: express every step factor in terms of dr, which
    # cancels (the rescaled residual is step-count independent), leaving only
    # dr2 / dzodr / drodz.
    e = e.subs(
        {
            s["dz2"]: dzodr**2 * dr**2,
            s["dz"]: dzodr * dr,
            s["r"]: s["ri"] * dr,
            s["r2"]: s["ri"] ** 2 * dr**2,
            s["dr2"]: dr**2,
        }
    )
    # Cancel common step factors (e.g. the extra 1/dr from the omega/aux
    # chain rule against the regularized a2_r denominator), leaving only
    # even powers of dr.
    e = sp.cancel(e)
    e = sp.expand(e)
    e = e.subs({dr**2: C["dr2"]})
    # Convert z-derivative factors 1/dzodr^n -> drodz^n, leaving positive
    # dzodr powers (r-derivative factors) alone.
    e = e.replace(
        lambda t: t.is_Pow and t.base == dzodr and t.exp.is_negative,
        lambda t: C["drodz"] ** (-t.exp),
    )
    return e


# ---------------------------------------------------------------------------
# Convenience: build everything once
# ---------------------------------------------------------------------------


def rhs_residuals():
    """The six residual brackets expressed in *physical* derivatives.

    This is the form used by ``src/rhs_vars.c``: the derivatives are the
    already-computed ``Dr_u[k]`` / ``Dz_u[k]`` / ``Drr_u[k]`` / ``Dzz_u[k]``
    (with their ``1/r`` factors), and the regularization auxiliaries
    ``Dr_u6`` / ``Dr_u7`` are plain symbols (filled from the auxiliary arrays).
    """
    s: dict[str, sp.Expr] = {}
    vals = ("l_alpha", "beta", "l_h", "l_a", "psi", "lambda")
    for n in vals:
        s[n] = sp.symbols(n, real=True)
    for var in ("Dr_u", "Dz_u", "Drr_u", "Dzz_u"):
        for k in range(6):
            s[f"{var}{k}"] = sp.symbols(f"{var}{k}", real=True)
    for n in ("r", "r2", "rlm1", "rl", "l", "m", "m2", "w", "Q1", "Q2"):
        s[n] = sp.symbols(n, real=True)
    s["pi"] = sp.pi
    # Regularization auxiliaries (Dr(alpha)/r and Dr(H)/r radial derivatives).
    s["Dr_u6"] = sp.symbols("Dr_u6", real=True)
    s["Dr_u7"] = sp.symbols("Dr_u7", real=True)

    la, beta, lh, lA, psi, lam = (s[n] for n in vals)
    # Derived physical variables as plain symbols (the residual is emitted
    # directly in these C names, so no substitution is needed at codegen).
    for n in (
        "alpha",
        "alpha2",
        "h2",
        "a2",
        "a2_r",
        "wplOmega",
        "wplOmega2",
        "phi",
        "phior",
        "phi2",
        "phi2or2",
    ):
        s[n] = sp.symbols(n, real=True)

    def Dr(k):
        return s[f"Dr_u{k}"]

    def Dz(k):
        return s[f"Dz_u{k}"]

    def Drr(k):
        return s[f"Drr_u{k}"]

    def Dzz(k):
        return s[f"Dzz_u{k}"]

    r, r2, alpha2, h2, a2, l = (s["r"], s["r2"], s["alpha2"], s["h2"], s["a2"], s["l"])
    m2, wplOmega = s["m2"], s["wplOmega"]
    wplOmega2 = wplOmega**2
    phi2, phi2or2 = s["phi2"], s["phi2or2"]
    rlm1, Q1, Q2 = s["rlm1"], s["Q1"], s["Q2"]
    pi = s["pi"]

    R0 = (
        Drr(0)
        + Dzz(0)
        + Dr(0) / r
        + (Dr(0) ** 2 + Dz(0) ** 2)
        + (Dr(0) * Dr(2) + Dz(0) * Dz(2))
        - sp.Rational(1, 2) * (r2 * h2 / alpha2) * (Dr(1) ** 2 + Dz(1) ** 2)
        + 4 * pi * a2 * (m2 - 2 * wplOmega2 / alpha2) * phi2
    )

    R1 = (
        Drr(1)
        + Dzz(1)
        + 3 * Dr(1) / r
        - (Dr(0) * Dr(1) + Dz(0) * Dz(1))
        + 3 * (Dr(1) * Dr(2) + Dz(1) * Dz(2))
        - 16 * pi * a2 * l * wplOmega * phi2or2 / h2
    )

    R2 = (
        Drr(2)
        + Dzz(2)
        + 2 * Dr(2) / r
        + (Dr(2) ** 2 + Dz(2) ** 2)
        + (Dr(0) * Dr(2) + Dz(0) * Dz(2))
        + sp.Rational(1, 2) * (r2 * h2 / alpha2) * (Dr(1) ** 2 + Dz(1) ** 2)
        + Dr(0) / r
        + 4 * pi * a2 * (r2 * m2 + 2 * l * l / h2) * phi2or2
    )

    R3 = (
        Drr(3)
        + Dzz(3)
        - (Dr(0) * Dr(2) + Dz(0) * Dz(2))
        - sp.Rational(1, 4) * (r2 * h2 / alpha2) * (Dr(1) ** 2 + Dz(1) ** 2)
        - Dr(0) / r
        + 4
        * pi
        * (
            (l * l * (1 - a2 / h2) + a2 * r2 * wplOmega2 / alpha2) * phi2or2
            + (2 * l * r * psi * Dr(4) + r2 * (Dr(4) ** 2 + Dz(4) ** 2)) * rlm1 * rlm1
        )
    )

    R4 = (
        Drr(4)
        + Dzz(4)
        + (2 * l + 1) * Dr(4) / r
        + (Dr(0) * Dr(4) + Dz(0) * Dz(4))
        + (Dr(4) * Dr(2) + Dz(4) * Dz(2))
        + l * (Dr(0) / r + Dr(2) / r) * psi
        + a2 * (wplOmega2 / alpha2 - m2) * psi
        - l * l * lam * psi / h2
    )

    a2_r = s["a2_r"]
    Dr_u6, Dr_u7 = s["Dr_u6"], s["Dr_u7"]
    R5 = (
        Drr(5)
        + Dzz(5)
        + 3 * Dr(5) / r
        + (Dz(2) * Dz(5) - Dr(2) * Dr(5))
        - 4 * (h2 / a2_r) * (Dr(5) * Dr(2) + Dz(5) * Dz(2))
        - (r2 / a2_r) * (Dr(5) ** 2 + Dz(5) ** 2)
        + (Dz(0) * Dz(5) - Dr(0) * Dr(5))
        - 4 * (lam * lam / a2_r)
        - 4 * (lam / a2_r) * (r * Dr(5))
        - 2 * (Dr(0) / r) * lam
        + 2 * (Dr(2) / r) * (-4 * (h2 / a2_r) + 1) * lam
        - 4 * h2 * (Dr(2) / r) ** 2 * ((h2 / a2_r) + sp.Rational(1, 2))
        + 4 * (h2 / a2_r) * Dz(2) ** 2 * lam
        - 2 * Dr(2) ** 2 * lam
        - 4 * h2 * (Dr(2) / r) * (Dr(0) / r)
        - (h2 * h2 / alpha2) * (Dr(1) ** 2 + Dz(1) ** 2)
        - (a2_r * h2 / alpha2) * (Dr(1) ** 2)
        + 2 * lam * (Drr(0) + Dr(0) ** 2)
        + 2 * lam * (Drr(2) + 2 * Dr(2) ** 2)
        + Q1 * ((2 * h2 / s["alpha"]) * (Dr_u6 / r))
        + Q2 * (Dr_u7 / r)
        + 8
        * pi
        * a2_r
        * (m2 * lam * phi2 + 2 * rlm1 * rlm1 * (Dr(4) / r) * (2 * l * psi + r * Dr(4)))
    )

    return s, [R0, R1, R2, R3, R4, R5]


def build():
    s = build_symbols()
    R = residuals(s)
    f = [s["dr2"] * s["dzodr"] * Ri for Ri in R]
    J = jacobian(s, R)
    return s, R, f, J


def build_c():
    """Build the system and return the 6x31 Jacobian in the C namespace."""
    s, R, f, J = build()
    Jc = [[to_c(J[i][j], s) for j in range(31)] for i in range(6)]
    return s, R, f, Jc
