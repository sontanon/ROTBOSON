"""Generate the RBS-NG Rust residual + Jacobian kernels from SymPy (SAN-56).

This is the Rust twin of ``tools/generate_kernels.py`` (the C generator,
which stays untouched as the oracle's chain).  Both emit from the same
single source of truth, :mod:`sympy_system`; the checked-in C kernels
(``src/rhs_vars.c`` / ``src/csr_vars.c``) remain the structural
cross-reference for the Rust output:

* ``ng/rbs-elliptic/src/generated/kernels.rs`` -- the interior residual
  (physical-derivative form, mirroring ``rhs_vars``), the Robin/decay
  boundary rows (mirroring ``rhs_bdry``), and the step-scaled 6x31
  subtype Jacobian (mirroring the ``jacobian_submatrix_*`` assignments of
  the ``csr_vars.c`` stencil kernels);
* ``ng/rbs-elliptic/src/generated/mod.rs`` -- the small hand-shaped module
  preamble (lint allowances for machine-printed numerics);
* ``ng/rbs-elliptic/tests/fixtures/kernel_parity.txt`` -- a seeded
  random-state parity fixture: SymPy-evaluated reference values for the
  residual, Jacobian, and boundary rows, asserted by ``cargo test`` in
  ``rbs-elliptic`` (``src/kernel_parity.rs``), so numeric parity holds
  even where no Python is installed.

The generated Rust must compile as part of ``rbs-elliptic`` and match the
fixture to the tolerance recorded in it (a few ULP of evaluation-order
noise; see the fixture header).

Usage:
    uv run tools/generate_rust_kernels.py [--check] [--out-dir DIR]
                                          [--validate]
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import sympy as sp
import sympy.printing.rust  # noqa: F401  (loads RustCodePrinter into sp.printing)
import sympy_system as ss

# ---------------------------------------------------------------------------
# Rust printing: sympy's Rust printer, tuned for the kernel style.
# ---------------------------------------------------------------------------

_CAMEL_RUN = re.compile(r"[A-Z]+")


def rust_name(name: str) -> str:
    """Map a C-namespace symbol to its Rust local (dRu3 -> d_ru3,
    wplOmega -> wpl_omega, Q1 -> q1; runs of capitals become one _run)."""
    lead = re.match(r"^[A-Z]+", name)
    if lead is not None:
        head = lead.group(0)
        rest = _CAMEL_RUN.sub(lambda m: "_" + m.group(0).lower(), name[len(head) :])
        return head.lower() + rest
    return _CAMEL_RUN.sub(lambda m: "_" + m.group(0).lower(), name)


class RbsRustPrinter(sp.printing.rust.RustCodePrinter):
    """Rust printer with kernel conventions.

    * Rationals print as shortest float literals (``0.5``, not
      ``1_f64/2.0``) -- exact for the dyadic rationals that dominate the
      system, correctly rounded otherwise;
    * symbols print through :func:`rust_name`.
    """

    def _print_Rational(self, expr: sp.Rational) -> str:
        v = float(expr)
        return repr(v).replace("e+", "e")

    def _print_Symbol(self, expr: sp.Symbol) -> str:
        return rust_name(expr.name)


def rust_expr(expr: sp.Expr) -> str:
    """Print one expression with the kernel printer.

    The expression is expanded first: sympy 1.14's Rust printer mangles
    ``Mul(coeff, Add(...))`` structures (it distributes over the Add but
    keeps the leading factor only on the first distributed term -- e.g.
    ``4*pi*a*(x + y)`` prints as ``4.0*PI*a*x + y``), and expansion removes
    all Add factors, exactly like the C generator's ``to_c`` does.
    """
    code = RbsRustPrinter().doprint(sp.expand(expr))
    assert isinstance(code, str)
    return code


# ---------------------------------------------------------------------------
# Declaration blocks, mirroring the C kernels' local-variable blocks.
#
# Each entry: (namespace symbol, Rust declaration, dependencies).  Only the
# referenced closure is emitted (like the C generator's used-params rule),
# in this fixed order, which matches the C blocks for reviewability.
# ---------------------------------------------------------------------------

ResDecl = tuple[str, str, list[str]]


def _arr_decls(prefix: str, field: str, count: int = 6, base: int = 0) -> list[ResDecl]:
    """Decls for a namespace family: symbols ``<prefix><base..base+count-1>``
    map onto ``p.<field>[0..count-1]`` (the Jacobian namespace is 1-based
    like the C kernels, the residual namespace 0-based)."""
    return [
        (
            f"{prefix}{k}",
            f"let {rust_name(prefix)}{k} = p.{field}[{k - base}];",
            [],
        )
        for k in range(base, base + count)
    ]


RESIDUAL_DECLS: list[ResDecl] = [
    ("l_alpha", "let l_alpha = p.u[0];", []),
    ("beta", "let beta = p.u[1];", []),
    ("l_h", "let l_h = p.u[2];", []),
    ("l_a", "let l_a = p.u[3];", []),
    ("psi", "let psi = p.u[4];", []),
    ("lambda", "let lambda = p.u[5];", []),
    *(
        _arr_decls("Dr_u", "dr_u")
        + _arr_decls("Dz_u", "dz_u")
        + _arr_decls("Drr_u", "drr_u")
        + _arr_decls("Dzz_u", "dzz_u")
    ),
    ("Dr_u6", "let dr_u6 = p.dr_aux[0];", []),
    ("Dr_u7", "let dr_u7 = p.dr_aux[1];", []),
    ("l", "let l = p.l;", []),
    ("m", "let m = p.m;", []),
    ("w", "let w = p.w;", []),
    ("r", "let r = p.r;", []),
    ("dr", "let dr = p.dr;", []),
    ("dz", "let dz = p.dz;", []),
    ("dzodr", "let dzodr = dz / dr;", ["dr", "dz"]),
    ("alpha", "let alpha = l_alpha.exp();", ["l_alpha"]),
    ("alpha2", "let alpha2 = alpha * alpha;", ["alpha"]),
    ("h", "let h = l_h.exp();", ["l_h"]),
    ("h2", "let h2 = h * h;", ["h"]),
    ("a", "let a = l_a.exp();", ["l_a"]),
    ("a2", "let a2 = a * a;", ["a"]),
    ("m2", "let m2 = m * m;", ["m"]),
    ("wplOmega", "let wpl_omega = w + l * beta;", ["w", "l", "beta"]),
    ("r2", "let r2 = r * r;", ["r"]),
    ("rlm1", "let rlm1 = r.powf(l - 1.0);", ["r", "l"]),
    ("rl", "let rl = rlm1 * r;", ["rlm1", "r"]),
    ("phi", "let phi = rl * psi;", ["rl", "psi"]),
    ("phior", "let phior = rlm1 * psi;", ["rlm1", "psi"]),
    ("phi2", "let phi2 = phi * phi;", ["phi"]),
    ("phi2or2", "let phi2or2 = phior * phior;", ["phior"]),
    ("a2_r", "let a2_r = h2 + r2 * lambda;", ["h2", "r2", "lambda"]),
    ("Q1", "let q1 = p.q1;", []),
    ("Q2", "let q2 = p.q2;", []),
]

JACOBIAN_DECLS: list[ResDecl] = [
    *[(f"u{k}", f"let u{k} = p.u[{k - 1}];", []) for k in range(1, 7)],
    *(
        _arr_decls("dRu", "d_ru", base=1)
        + _arr_decls("dZu", "d_zu", base=1)
        + _arr_decls("dRRu", "d_rru", base=1)
        + _arr_decls("dZZu", "d_zzu", base=1)
    ),
    ("ri", "let ri = p.ri;", []),
    ("dr", "let dr = p.dr;", []),
    ("dz", "let dz = p.dz;", []),
    ("l", "let l = p.l;", []),
    ("m", "let m = p.m;", []),
    ("w", "let w = p.w;", []),
    ("r", "let r = ri * dr;", ["ri", "dr"]),
    ("dr2", "let dr2 = dr * dr;", ["dr"]),
    ("dzodr", "let dzodr = dz / dr;", ["dr", "dz"]),
    ("drodz", "let drodz = dr / dz;", ["dr", "dz"]),
    ("alpha", "let alpha = u1.exp();", ["u1"]),
    ("alpha2", "let alpha2 = alpha * alpha;", ["alpha"]),
    ("h", "let h = u3.exp();", ["u3"]),
    ("h2", "let h2 = h * h;", ["h"]),
    ("a", "let a = u4.exp();", ["u4"]),
    ("a2", "let a2 = a * a;", ["a"]),
    ("m2", "let m2 = m * m;", ["m"]),
    ("psi", "let psi = u5;", ["u5"]),
    ("lambda", "let lambda = u6;", ["u6"]),
    ("wplOmega", "let wpl_omega = w + l * u2;", ["w", "l", "u2"]),
    ("rlm1", "let rlm1 = r.powf(l - 1.0);", ["r", "l"]),
    ("rl", "let rl = rlm1 * r;", ["rlm1", "r"]),
    ("phior", "let phior = rlm1 * psi;", ["rlm1", "psi"]),
    ("phi", "let phi = r * phior;", ["r", "phior"]),
    ("phi2", "let phi2 = phi * phi;", ["phi"]),
    ("phi2or2", "let phi2or2 = phior * phior;", ["phior"]),
    ("Q1", "let q1 = p.q1;", []),
    ("Q2", "let q2 = p.q2;", []),
]


def referenced_closure(
    exprs: list[sp.Expr],
    decls: list[ResDecl],
    force: set[str] | None = None,
) -> list[ResDecl]:
    """The subset of `decls` needed to evaluate `exprs` (plus any `force`
    names used by emitted boilerplate), in table order."""
    table = {name: (line, deps) for name, line, deps in decls}
    needed = {str(sym) for e in exprs for sym in e.free_symbols}
    needed.update(force or set())
    # Expand through dependencies until stable.
    changed = True
    while changed:
        changed = False
        for name in list(needed):
            for dep in table.get(name, (None, []))[1]:
                if dep not in needed:
                    needed.add(dep)
                    changed = True
    return [(n, *table[n]) for n, *_ in decls if n in needed]


# ---------------------------------------------------------------------------
# File emitters.
# ---------------------------------------------------------------------------

_HEADER = """\
//! Automatically generated by `tools/generate_rust_kernels.py` (SAN-56) from
//! `tools/sympy_system.py`.  Do not edit by hand; edit the `SymPy` source
//! and regenerate.  The structural cross-reference is the checked-in C kernels
//! (`src/rhs_vars.c` for the residual and boundary rows, `src/csr_vars.c`
//! for the subtype Jacobian): same equations, term for term.
//!
//! * `residual_interior` mirrors `rhs_vars()`: physical derivatives in,
//!   step-scaled residual rows `f_k = rescale * (dr*dr*dzodr * R_k)` out
//!   (the C driver passes `rescale = -1`).
//! * `residual_boundary` mirrors `rhs_bdry()`: Robin / exponential-decay
//!   rows at the outer boundary (the C kernel's unused `M`/`J` arguments
//!   are dropped here).
//! * `jacobian_subtypes` mirrors the `jacobian_submatrix_*` assignments of
//!   the `csr_vars.c` stencil kernels: the step-scaled 6x31 Jacobian of
//!   `f = (dr^2 * dzodr) * R` w.r.t. `[u, dRu, dZu, dRRu, dZZu]` per field,
//!   with the omega column pre-multiplied by `dw/dxi` (the C kernel's
//!   `dw_du(xi, m) * (dF/dw)`).  The CSR scatter itself
//!   (`aa[col] = sum_k coeff_k * jacobian[row][5*(g-1)+k]` per stencil
//!   point) is assembly-side code (`rbs-elliptic::assembly`).
//!
//! Numeric parity is pinned by `tests/fixtures/kernel_parity.txt`
//! (`SymPy`-evaluated reference states; `cargo test -p rbs-elliptic`).
"""


def _emit_decl_block(decls: list[ResDecl], comment: str) -> str:
    lines = [f"    // {comment}"]
    lines += [f"    {line}" for _, line, _ in decls]
    return "\n".join(lines)


def emit_kernels_rs() -> str:
    """Generate the full kernels.rs text."""
    # --- interior residual (physical-derivative namespace) ---
    _s, res = ss.rhs_residuals()
    # dzodr is not in the brackets; it is used by the emitted step-scale
    # line, so force its (and its deps') declaration.
    res_decls = referenced_closure(res, RESIDUAL_DECLS, force={"dzodr"})
    res_strs = [rust_expr(e) for e in res]

    # --- subtype Jacobian (C namespace, step-scaled) ---
    _sj, _r, _f, jac = ss.build_c()
    jac_decls = referenced_closure([e for row in jac for e in row], JACOBIAN_DECLS)
    jac_strs: list[list[str]] = []
    for row in jac:
        entries: list[str] = []
        for j, e in enumerate(row):
            if e == 0:
                entries.append("0.0")
            elif j == 6 * 5:  # omega column, pre-multiplied like the C kernel
                entries.append(f"p.dw_dxi * ({rust_expr(e)})")
            else:
                entries.append(rust_expr(e))
        jac_strs.append(entries)

    parts: list[str] = []
    parts.append(_HEADER)
    parts.append("use std::f64::consts::PI;\n")
    parts.append(
        """\
/// Number of grid functions (u1..u6 = log alpha, beta, log h, log a, psi,
/// lambda); `GNUM` in `src/csr_vars.c`.
pub const GNUM: usize = 6;

/// Subtypes per field: `[u, dRu, dZu, dRRu, dZZu]` (value + step-scaled
/// derivatives `dr*Dr`, `dz*Dz`, `dr^2*Drr`, `dz^2*Dzz`).
pub const SUBTYPES_PER_FIELD: usize = 5;

/// Jacobian columns: `5` subtypes for each of `GNUM` fields, plus omega.
pub const JAC_COLS: usize = GNUM * SUBTYPES_PER_FIELD + 1;
"""
    )
    parts.append(
        """\
/// Point-local inputs of [`residual_interior`] (physical-derivative form,
/// mirroring the `rhs_vars` signature in `src/rhs_vars.c`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualPoint {
    /// Grid values `[l_alpha, beta, l_h, l_a, psi, lambda]`.
    pub u: [f64; 6],
    /// Physical radial derivatives `Dr_u0..Dr_u5`.
    pub dr_u: [f64; 6],
    /// Physical axial derivatives `Dz_u0..Dz_u5`.
    pub dz_u: [f64; 6],
    /// Physical second radial derivatives `Drr_u0..Drr_u5`.
    pub drr_u: [f64; 6],
    /// Physical second axial derivatives `Dzz_u0..Dzz_u5`.
    pub dzz_u: [f64; 6],
    /// Regularization auxiliaries `[Dr_u6, Dr_u7]` (the lambda equation's
    /// `Dr(alpha)/r` and `Dr(H)/r` chains, precomputed by the assembly).
    pub dr_aux: [f64; 2],
    /// Radial step.
    pub dr: f64,
    /// Axial step.
    pub dz: f64,
    /// Radial coordinate.
    pub r: f64,
    /// Azimuthal number.
    pub l: f64,
    /// Scalar-field mass.
    pub m: f64,
    /// Physical frequency (`omega_calc(xi, m)`; the C kernel computes it
    /// inside from `xi`).
    pub w: f64,
    /// Regularization coefficient (the C kernels hard-code `Q1 = 1.0`).
    pub q1: f64,
    /// Regularization coefficient (the C kernels hard-code `Q2 = 1.0`).
    pub q2: f64,
    /// Residual sign; the C driver passes `-1`.
    pub rescale: f64,
}
"""
    )
    res_body = [
        "/// Evaluate the six interior residual rows at one grid point.",
        "///",
        "/// Mirrors `rhs_vars()` in `src/rhs_vars.c`: returns",
        "/// `f_k = rescale * (dr * dr * dzodr * R_k)` where `R_k` is the",
        "/// physics bracket of `tools/sympy_system.py::rhs_residuals()`.",
        "#[must_use]",
        "pub fn residual_interior(p: &ResidualPoint) -> [f64; 6] {",
        _emit_decl_block(res_decls, "Point-local inputs and derived locals"),
        "    [",
    ]
    for expr in res_strs:
        res_body.append(f"        p.rescale * (p.dr * p.dr * dzodr * ({expr})),")
    res_body.append("    ]")
    res_body.append("}")
    parts.append("\n".join(res_body))
    parts.append(
        """\
/// Point-local inputs of [`residual_boundary`] (mirroring the `rhs_bdry`
/// signature in `src/rhs_vars.c`; the unused `M`/`J` are dropped).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryPoint {
    /// Grid values `[l_alpha, beta, l_h, l_a, psi, lambda]`.
    pub u: [f64; 6],
    /// Physical radial derivatives `Dr_u0..Dr_u5`.
    pub dr_u: [f64; 6],
    /// Physical axial derivatives `Dz_u0..Dz_u5`.
    pub dz_u: [f64; 6],
    /// Radial coordinate on the outer boundary.
    pub r: f64,
    /// Axial coordinate on the outer boundary.
    pub z: f64,
    /// Radial step.
    pub dr: f64,
    /// Axial step.
    pub dz: f64,
    /// Azimuthal number.
    pub l: f64,
    /// Scalar-field mass.
    pub m: f64,
    /// Physical frequency.
    pub w: f64,
    /// Residual sign; the C driver passes `-1`.
    pub rescale: f64,
}
"""
    )
    parts.append(
        """\
/// Evaluate the six Robin / exponential-decay boundary rows at one outer
/// boundary point; mirrors `rhs_bdry()` in `src/rhs_vars.c`.
#[must_use]
pub fn residual_boundary(p: &BoundaryPoint) -> [f64; 6] {
    let r = p.r;
    let z = p.z;
    let rr2 = r * r + z * z;
    let rr = rr2.sqrt();
    let scale = p.dr * p.dz / rr2;
    let m2 = p.m * p.m;
    let chi = (m2 - p.w * p.w).sqrt();
    [
        p.rescale * scale * (r * p.dr_u[0] + z * p.dz_u[0] + p.u[0]),
        p.rescale * scale * (r * p.dr_u[1] + z * p.dz_u[1] + 3.0 * p.u[1]),
        p.rescale * scale * (r * p.dr_u[2] + z * p.dz_u[2] + p.u[2]),
        p.rescale * scale * (r * p.dr_u[3] + z * p.dz_u[3] + p.u[3]),
        p.rescale * scale * (r * p.dr_u[4] + z * p.dz_u[4] + (rr * chi + p.l + 1.0) * p.u[4]),
        p.rescale * scale * (r * p.dr_u[5] + z * p.dz_u[5] + 4.0 * p.u[5]),
    ]
}
"""
    )
    parts.append(
        """\
/// Point-local inputs of [`jacobian_subtypes`] (step-scaled subtype form,
/// mirroring the declaration block of the `csr_vars.c` stencil kernels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JacobianPoint {
    /// Grid values at the stencil centre, `u1..u6`.
    pub u: [f64; 6],
    /// Step-scaled radial derivatives `dr * Dr u1..u6`.
    pub d_ru: [f64; 6],
    /// Step-scaled axial derivatives `dz * Dz u1..u6`.
    pub d_zu: [f64; 6],
    /// Step-scaled second radial derivatives `dr^2 * Drr u1..u6`.
    pub d_rru: [f64; 6],
    /// Step-scaled second axial derivatives `dz^2 * Dzz u1..u6`.
    pub d_zzu: [f64; 6],
    /// Radial cell index `ri = i + 0.5 - ghost`.
    pub ri: f64,
    /// Radial step.
    pub dr: f64,
    /// Axial step.
    pub dz: f64,
    /// Azimuthal number.
    pub l: f64,
    /// Scalar-field mass.
    pub m: f64,
    /// Physical frequency (the C kernel computes `omega_calc(xi, m)`
    /// internally; the Rust caller owns that map).
    pub w: f64,
    /// `dw/dxi` (the C kernel's `dw_du(xi, m)`), pre-multiplying the omega
    /// column.
    pub dw_dxi: f64,
    /// Regularization coefficient (the C kernels hard-code `Q1 = 1.0`).
    pub q1: f64,
    /// Regularization coefficient (the C kernels hard-code `Q2 = 1.0`).
    pub q2: f64,
}
"""
    )
    jac_body = [
        "/// Evaluate the step-scaled subtype Jacobian at one stencil centre:",
        "/// row `i` is the residual of field `i + 1`, column `5*(g-1) + k`",
        "/// is `df_i / d(u_g subtype_k)` for `k in [u, dRu, dZu, dRRu, dZZu]`,",
        "/// and column `30` is the omega chain-rule entry `dw/dxi * df/dw`.",
        "/// Mirrors the `jacobian_submatrix_*` assignments of `csr_vars.c`.",
        "#[must_use]",
        "pub fn jacobian_subtypes(p: &JacobianPoint) -> [[f64; JAC_COLS]; GNUM] {",
        _emit_decl_block(jac_decls, "Point-local inputs and derived locals"),
        "    [",
    ]
    for row in jac_strs:
        jac_body.append("        [" + ", ".join(row) + "],")
    jac_body += ["    ]", "}"]
    parts.append("\n".join(jac_body))
    return "\n\n".join(parts) + "\n"


def emit_mod_rs() -> str:
    """The small module preamble around the generated kernels."""
    return """\
//! The generated elliptic kernels: the Rust twin of `src/rhs_vars.c` +
//! `src/csr_vars.c`, emitted by `tools/generate_rust_kernels.py` (SAN-56).
//!
//! The lint allowances below exist because the child module is
//! machine-printed numerics: its shape is chosen to mirror the C kernels
//! term-for-term, not to satisfy hand-code style.
#![allow(clippy::identity_op)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]

pub mod kernels;

pub use kernels::{
    jacobian_subtypes, residual_boundary, residual_interior, BoundaryPoint,
    JacobianPoint, ResidualPoint, GNUM, JAC_COLS, SUBTYPES_PER_FIELD,
};
"""


# ---------------------------------------------------------------------------
# Parity fixture: seeded random states + SymPy-evaluated references.
# ---------------------------------------------------------------------------

TOLERANCE = 1e-12
SEED = 20261119
N_RESIDUAL = 12
N_JACOBIAN = 6
N_BOUNDARY = 6


def _in_symbols(names: list[str]) -> list[sp.Symbol]:
    syms = sp.symbols(" ".join(names), real=True)
    out: list[sp.Symbol] = []
    for s in syms:
        assert isinstance(s, sp.Symbol)
        out.append(s)
    return out


def _subs_by_name(exprs: list[sp.Expr], mapping: dict[str, object]) -> list[sp.Expr]:
    """Substitute into `exprs` keyed by *symbol name*, using the symbols the
    expressions actually carry (their assumptions vary between the
    namespaces, so string-keyed `subs` would silently miss)."""
    key: dict[sp.Basic | complex, sp.Expr | complex] = {}
    for e in exprs:
        for sym in e.free_symbols:
            if isinstance(sym, sp.Symbol) and sym.name in mapping:
                sub = sp.sympify(mapping[sym.name])
                assert isinstance(sub, sp.Expr)
                key[sym] = sub
    return [sp.sympify(e.subs(key)) for e in exprs]


def _residual_reference() -> tuple[list[sp.Symbol], Callable[..., list[float]]]:
    """Lambdified residual rows over the struct-field inputs."""
    u = _in_symbols([f"u{i}" for i in range(6)])
    dru = _in_symbols([f"dru{i}" for i in range(6)])
    dzu = _in_symbols([f"dzu{i}" for i in range(6)])
    drru = _in_symbols([f"drru{i}" for i in range(6)])
    dzzu = _in_symbols([f"dzzu{i}" for i in range(6)])
    aux = _in_symbols(["aux0", "aux1"])
    dr, dz, r, l, m, w, q1, q2, rescale = _in_symbols(
        ["dr", "dz", "r", "l", "m", "w", "q1", "q2", "rescale"]
    )
    args = [*u, *dru, *dzu, *drru, *dzzu, *aux, dr, dz, r, l, m, w, q1, q2, rescale]

    _s, res = ss.rhs_residuals()
    # Definitions mirroring the emitted Rust declaration block.
    alpha, h, a = sp.exp(u[0]), sp.exp(u[2]), sp.exp(u[3])
    rlm1 = r ** (l - 1)
    rl = rlm1 * r
    phi = rl * u[4]
    phior = rlm1 * u[4]
    subs = {
        "l_alpha": u[0],
        "beta": u[1],
        "l_h": u[2],
        "l_a": u[3],
        "psi": u[4],
        "lambda": u[5],
        "alpha": alpha,
        "alpha2": alpha * alpha,
        "h2": h * h,
        "a2": a * a,
        "m2": m * m,
        "wplOmega": w + l * u[1],
        "r2": r * r,
        "rlm1": rlm1,
        "rl": rl,
        "phi": phi,
        "phior": phior,
        "phi2": phi * phi,
        "phi2or2": phior * phior,
        "a2_r": h * h + r * r * u[5],
        "Q1": q1,
        "Q2": q2,
        "Dr_u6": aux[0],
        "Dr_u7": aux[1],
    }
    subs.update({f"Dr_u{k}": dru[k] for k in range(6)})
    subs.update({f"Dz_u{k}": dzu[k] for k in range(6)})
    subs.update({f"Drr_u{k}": drru[k] for k in range(6)})
    subs.update({f"Dzz_u{k}": dzzu[k] for k in range(6)})
    res_sub = _subs_by_name(res, subs)
    rows = [rescale * (dr * dr * (dz / dr) * r_k) for r_k in res_sub]
    fn: Callable[..., list[float]] = sp.lambdify(args, rows, modules="math")
    return args, fn


def _jacobian_reference() -> tuple[list[sp.Symbol], Callable[..., list[list[float]]]]:
    """Lambdified 6x31 subtype Jacobian over the struct-field inputs."""
    u = _in_symbols([f"u{k}" for k in range(1, 7)])
    dru = _in_symbols([f"dru{k}" for k in range(1, 7)])
    dzu = _in_symbols([f"dzu{k}" for k in range(1, 7)])
    drru = _in_symbols([f"drru{k}" for k in range(1, 7)])
    dzzu = _in_symbols([f"dzzu{k}" for k in range(1, 7)])
    ri, dr, dz, l, m, w, dw_dxi, q1, q2 = _in_symbols(
        ["ri", "dr", "dz", "l", "m", "w", "dw_dxi", "q1", "q2"]
    )
    args = [*u, *dru, *dzu, *drru, *dzzu, ri, dr, dz, l, m, w, dw_dxi, q1, q2]

    _s, _r, _f, jac = ss.build_c()
    # Definitions mirroring the emitted Rust declaration block.
    r_ = ri * dr
    rlm1 = r_ ** (l - 1)
    rl = rlm1 * r_
    phior = rlm1 * u[4]
    phi = r_ * phior
    alpha, h, a = sp.exp(u[0]), sp.exp(u[2]), sp.exp(u[3])
    subs = {
        "ri": ri,
        "dr2": dr * dr,
        "dzodr": dz / dr,
        "drodz": dr / dz,
        "m2": m * m,
        "w": w,
        "l": l,
        "alpha2": alpha * alpha,
        "h2": h * h,
        "a2": a * a,
        "psi": u[4],
        "lambda": u[5],
        "wplOmega": w + l * u[1],
        "rlm1": rlm1,
        "rl": rl,
        "phior": phior,
        "phi": phi,
        "phi2": phi * phi,
        "phi2or2": phior * phior,
        "Q1": q1,
        "Q2": q2,
    }
    subs.update({f"dRu{k}": dru[k - 1] for k in range(1, 7)})
    subs.update({f"dZu{k}": dzu[k - 1] for k in range(1, 7)})
    subs.update({f"dRRu{k}": drru[k - 1] for k in range(1, 7)})
    subs.update({f"dZZu{k}": dzzu[k - 1] for k in range(1, 7)})
    flat = _subs_by_name([e for row in jac for e in row], subs)
    rows: list[list[sp.Expr]] = []
    for i in range(len(jac)):
        vals = flat[i * len(jac[0]) : (i + 1) * len(jac[0])]
        vals[6 * 5] = dw_dxi * vals[6 * 5]  # omega column chain rule
        rows.append(list(vals))
    fn: Callable[..., list[list[float]]] = sp.lambdify(args, rows, modules="math")
    return args, fn


def _boundary_reference(
    u: list[float],
    dru: list[float],
    dzu: list[float],
    r: float,
    z: float,
    dr: float,
    dz: float,
    l: float,
    m: float,
    w: float,
    rescale: float,
) -> list[float]:
    """The rhs_bdry rows, evaluated in Python (mirroring the emitted Rust)."""
    rr2 = r * r + z * z
    rr = rr2**0.5
    scale = dr * dz / rr2
    m2 = m * m
    chi = (m2 - w * w) ** 0.5
    return [
        rescale * scale * (r * dru[0] + z * dzu[0] + u[0]),
        rescale * scale * (r * dru[1] + z * dzu[1] + 3.0 * u[1]),
        rescale * scale * (r * dru[2] + z * dzu[2] + u[2]),
        rescale * scale * (r * dru[3] + z * dzu[3] + u[3]),
        rescale * scale * (r * dru[4] + z * dzu[4] + (rr * chi + l + 1.0) * u[4]),
        rescale * scale * (r * dru[5] + z * dzu[5] + 4.0 * u[5]),
    ]


def _fmt(vals: list[float]) -> str:
    return " ".join(repr(v) for v in vals)


def emit_fixture() -> str:
    """Generate the parity fixture text."""
    rng = random.Random(SEED)
    ls = [1.0, 2.0, 3.0, 4.0]
    _rargs, res_fn = _residual_reference()
    _jargs, jac_fn = _jacobian_reference()

    lines = [
        "# RBS-NG kernel parity fixture.",
        "# Generated by tools/generate_rust_kernels.py (seeded RNG + SymPy",
        "# evaluation of the same expressions the generated kernels",
        "# evaluate).  Do not edit by hand; regenerate with:",
        "#   uv run tools/generate_rust_kernels.py",
        f"tolerance {TOLERANCE!r}",
    ]

    # --- residual states ---
    res_states: list[tuple[list[float], list[float]]] = []
    while len(res_states) < N_RESIDUAL:
        u = [rng.uniform(-0.8, 0.8) for _ in range(6)]
        dru = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        dzu = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        drru = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        dzzu = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        aux = [rng.uniform(-1.0, 1.0) for _ in range(2)]
        dr = rng.uniform(0.05, 0.2)
        dz = rng.uniform(0.05, 0.2)
        r = rng.uniform(0.05, 8.0)
        l = ls[len(res_states) % 4]
        m = rng.uniform(0.5, 1.5)
        w = m * rng.uniform(0.2, 0.8)
        q1 = q2 = 1.0
        rescale = 1.0 if len(res_states) % 4 == 0 else -1.0
        expect = res_fn(*u, *dru, *dzu, *drru, *dzzu, *aux, dr, dz, r, l, m, w, q1, q2, rescale)
        if all(v is not None and v == v and abs(v) != float("inf") for v in expect):
            res_states.append(
                (
                    [
                        *u,
                        *dru,
                        *dzu,
                        *drru,
                        *dzzu,
                        *aux,
                        dr,
                        dz,
                        r,
                        l,
                        m,
                        w,
                        q1,
                        q2,
                        rescale,
                    ],
                    list(expect),
                )
            )
    lines.append(f"residual {N_RESIDUAL}")
    for inputs, expect in res_states:
        for key, chunk in zip(
            ["u", "dr_u", "dz_u", "drr_u", "dzz_u"],
            [inputs[0:6], inputs[6:12], inputs[12:18], inputs[18:24], inputs[24:30]],
            strict=True,
        ):
            lines.append(f"{key} " + _fmt(chunk))
        lines.append("dr_aux " + _fmt(inputs[30:32]))
        for key, v in zip(
            ["dr", "dz", "r", "l", "m", "w", "q1", "q2", "rescale"],
            inputs[32:41],
            strict=True,
        ):
            lines.append(f"{key} {v!r}")
        lines.append("expect " + _fmt(expect))

    # --- jacobian states ---
    jac_states: list[tuple[list[float], list[float]]] = []
    while len(jac_states) < N_JACOBIAN:
        u = [rng.uniform(-0.8, 0.8) for _ in range(6)]
        dru = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        dzu = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        drru = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        dzzu = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        ri = rng.uniform(2.0, 40.0)
        dr = rng.uniform(0.05, 0.2)
        dz = rng.uniform(0.05, 0.2)
        l = ls[len(jac_states) % 4]
        m = rng.uniform(0.5, 1.5)
        w = m * rng.uniform(0.2, 0.8)
        dw_dxi = 0.5 * m * rng.uniform(0.1, 1.0)
        q1 = q2 = 1.0
        mat = jac_fn(*u, *dru, *dzu, *drru, *dzzu, ri, dr, dz, l, m, w, dw_dxi, q1, q2)
        flat = [float(v) for row in mat for v in row]
        if all(v == v and abs(v) != float("inf") for v in flat):
            jac_states.append(
                (
                    [*u, *dru, *dzu, *drru, *dzzu, ri, dr, dz, l, m, w, dw_dxi, q1, q2],
                    flat,
                )
            )
    lines.append(f"jacobian {N_JACOBIAN}")
    for inputs, expect in jac_states:
        for key, chunk in zip(
            ["u", "d_ru", "d_zu", "d_rru", "d_zzu"],
            [inputs[0:6], inputs[6:12], inputs[12:18], inputs[18:24], inputs[24:30]],
            strict=True,
        ):
            lines.append(f"{key} " + _fmt(chunk))
        for key, v in zip(
            ["ri", "dr", "dz", "l", "m", "w", "dw_dxi", "q1", "q2"],
            inputs[30:39],
            strict=True,
        ):
            lines.append(f"{key} {v!r}")
        lines.append("expect " + _fmt(expect))

    # --- boundary states ---
    bdr_states: list[tuple[list[float], list[float]]] = []
    while len(bdr_states) < N_BOUNDARY:
        u = [rng.uniform(-0.8, 0.8) for _ in range(6)]
        dru = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        dzu = [rng.uniform(-1.0, 1.0) for _ in range(6)]
        radius = rng.uniform(4.0, 16.0)
        theta = rng.uniform(0.05, 1.35)
        r, z = radius * float(sp.cos(theta)), radius * float(sp.sin(theta))
        dr = rng.uniform(0.05, 0.2)
        dz = rng.uniform(0.05, 0.2)
        l = float(1 + len(bdr_states) % 3)
        m = rng.uniform(0.5, 1.5)
        w = m * rng.uniform(0.2, 0.55)
        rescale = -1.0
        expect = _boundary_reference(u, dru, dzu, r, z, dr, dz, l, m, w, rescale)
        if all(v == v and abs(v) != float("inf") for v in expect):
            bdr_states.append(([*u, *dru, *dzu, r, z, dr, dz, l, m, w, rescale], expect))
    lines.append(f"boundary {N_BOUNDARY}")
    for inputs, expect in bdr_states:
        for key, chunk in zip(
            ["u", "dr_u", "dz_u"],
            [inputs[0:6], inputs[6:12], inputs[12:18]],
            strict=True,
        ):
            lines.append(f"{key} " + _fmt(chunk))
        for key, v in zip(
            ["r", "z", "dr", "dz", "l", "m", "w", "rescale"],
            inputs[18:26],
            strict=True,
        ):
            lines.append(f"{key} {v!r}")
        lines.append("expect " + _fmt(expect))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if regenerated files differ from the checked-in ones",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write into DIR instead of the repo (default: repo paths)",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="regenerate + run the cargo parity test (rbs-elliptic)",
    )
    args = ap.parse_args()

    base = (args.out_dir / "ng/rbs-elliptic") if args.out_dir else (repo / "ng" / "rbs-elliptic")
    files = {
        base / "src" / "generated" / "kernels.rs": emit_kernels_rs(),
        base / "src" / "generated" / "mod.rs": emit_mod_rs(),
        base / "tests" / "fixtures" / "kernel_parity.txt": emit_fixture(),
    }

    changed = False
    for target, content in files.items():
        if args.check or args.validate:
            if not target.exists() or target.read_text() != content:
                print(f"DIFFERS: {target.relative_to(repo)}")
                changed = True
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            print(f"wrote {target.relative_to(repo)} ({len(content)} bytes)")
    if args.check:
        print("[generate_rust_kernels] check " + ("FAILED" if changed else "OK"))
        return 1 if changed else 0

    if args.validate:
        if changed:
            print(
                "[generate_rust_kernels] regenerated files differ from "
                "checked-in; refusing to validate.  Run without --check first."
            )
            return 1
        proc = subprocess.run(
            ["cargo", "test", "-p", "rbs-elliptic", "--", "--nocapture"],
            cwd=repo / "ng",
        )
        return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
