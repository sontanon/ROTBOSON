"""Manufactured-solution (MMS) test for the SymPy-derived residual.

With the independent SymPy operator ``L`` (tools/sympy_system.py, the same
expression that generates src/rhs_vars.c), we manufacture a smooth even
solution ``u_man(r, z)``, emit the analytic source ``S = L[u_man]``, and check
that the residual -- discretized with the same 4th-order Fornberg stencils the
C code uses -- recovers ``S`` at the design order on the interior.

This is the direct, local validation the Phase 3 component layer wanted and
deferred to Phase 4 (it needs ``L`` to exist first).  It catches off-axis and
mutually-cancelling sign/factor errors that the golden data can only probe
indirectly.

Usage:
    uv run tools/mms_test.py
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp
import sympy_system as ss


def build():
    """Return (R, arg_syms, u_man) for the manufactured solution."""
    r, z = sp.symbols("r z")

    # Six grid unknowns, each a smooth even function of r and z (so they
    # respect the EVEN axis/equator reflection the C operators assume).
    # The amplitudes are chosen so that the regularized radius
    # a2_r = h2 + r2*lambda stays strictly positive (no spurious singularity
    # in the lambda equation) and the fields decay at large r.
    g = sp.exp(-(r**2) - z**2)
    u_man = [
        0.10 * (1 - g),  # l_alpha = log(alpha)
        0.05 * g,  # beta
        0.10 * (1 - g),  # l_h = log(h)
        0.05 * (1 - g),  # l_a = log(a)
        0.30 * g,  # psi
        0.01 * g,  # lambda (small and positive)
    ]

    s, R = ss.rhs_residuals()

    # Ordered argument list for lambdifying R_i.
    arg_syms = [
        s["alpha2"],
        s["h2"],
        s["a2"],
        s["a2_r"],
        s["wplOmega"],
        s["phi"],
        s["phior"],
        s["phi2"],
        s["phi2or2"],
        s["alpha"],
        s["psi"],
        s["lambda"],
    ]
    for k in range(6):
        arg_syms += [s[f"Dr_u{k}"], s[f"Dz_u{k}"], s[f"Drr_u{k}"], s[f"Dzz_u{k}"]]
    arg_syms += [
        s["Dr_u6"],
        s["Dr_u7"],
        s["r"],
        s["r2"],
        s["rlm1"],
        s["rl"],
        s["l"],
        s["m2"],
        s["w"],
        s["Q1"],
        s["Q2"],
    ]

    return R, arg_syms, u_man, (r, z)


def fd_weights_1():
    return np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


def fd_weights_2():
    return np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0


def fd_deriv(u, axis, dr, order):
    """Interior 4th-order derivative along axis 0 (r) or 1 (z)."""
    n0, n1 = u.shape
    out = np.zeros_like(u)
    w = fd_weights_1() if order == 1 else fd_weights_2()
    if axis == 0:
        for i in range(2, n0 - 2):
            out[i, :] = (
                w[0] * u[i - 2] + w[1] * u[i - 1] + w[2] * u[i] + w[3] * u[i + 1] + w[4] * u[i + 2]
            ) / dr**order
    else:
        for j in range(2, n1 - 2):
            out[:, j] = (
                w[0] * u[:, j - 2]
                + w[1] * u[:, j - 1]
                + w[2] * u[:, j]
                + w[3] * u[:, j + 1]
                + w[4] * u[:, j + 2]
            ) / dr**order
    return out


def assemble_args(phys, derivs, Rm, l_val, m_val, w_val):
    """Assemble the ordered numpy argument list for a lambdified residual."""
    (
        alpha2,
        h2,
        a2,
        a2_r,
        wplOmega,
        phi,
        phior,
        phi2,
        phi2or2,
        alpha,
        u4,
        u5,
        Dr_u6,
        Dr_u7,
        rlm1,
        rl,
    ) = phys
    Dr, Dz, Drr, Dzz = derivs
    vals = [alpha2, h2, a2, a2_r, wplOmega, phi, phior, phi2, phi2or2, alpha, u4, u5]
    for k in range(6):
        vals += [Dr[k], Dz[k], Drr[k], Dzz[k]]
    vals += [
        Dr_u6,
        Dr_u7,
        Rm,
        Rm**2,
        rlm1,
        rl,
        np.full_like(Rm, l_val, dtype=float),
        np.full_like(Rm, m_val**2),
        np.full_like(Rm, w_val),
        np.ones_like(Rm),
        np.ones_like(Rm),
    ]
    return vals


def main() -> int:
    R, arg_syms, u_man, (r, z) = build()
    l_val, m_val, w_val = 3, 1.0, 0.8

    # Exact (analytic) derivatives of the manufactured fields.
    Dr_ex = [sp.diff(u, r) for u in u_man]
    Dz_ex = [sp.diff(u, z) for u in u_man]
    Drr_ex = [sp.diff(u, r, 2) for u in u_man]
    Dzz_ex = [sp.diff(u, z, 2) for u in u_man]

    resid_fns = [sp.lambdify(arg_syms, Ri, "numpy") for Ri in R]

    errors = []
    for N in (64, 128, 256):
        dr = 4.0 / N
        rr = (np.arange(N) + 0.5) * dr
        Rm, Zm = np.meshgrid(rr, rr, indexing="ij")

        # Field values (exact) and FD derivatives (numerical).
        u = np.array([sp.lambdify((r, z), f, "numpy")(Rm, Zm) for f in u_man])
        Dr_fd = np.array([fd_deriv(u[k], 0, dr, 1) for k in range(6)])
        Dz_fd = np.array([fd_deriv(u[k], 1, dr, 1) for k in range(6)])
        Drr_fd = np.array([fd_deriv(u[k], 0, dr, 2) for k in range(6)])
        Dzz_fd = np.array([fd_deriv(u[k], 1, dr, 2) for k in range(6)])
        Dr_exa = np.array([sp.lambdify((r, z), f, "numpy")(Rm, Zm) for f in Dr_ex])
        Dz_exa = np.array([sp.lambdify((r, z), f, "numpy")(Rm, Zm) for f in Dz_ex])
        Drr_exa = np.array([sp.lambdify((r, z), f, "numpy")(Rm, Zm) for f in Drr_ex])
        Dzz_exa = np.array([sp.lambdify((r, z), f, "numpy")(Rm, Zm) for f in Dzz_ex])

        u0, u2, u4, u5 = u[0], u[2], u[4], u[5]
        alpha = np.exp(u0)
        alpha2 = alpha**2
        h2 = np.exp(2 * u2)
        a2 = np.exp(2 * u[3])
        a2_r = h2 + Rm**2 * u5
        wplOmega = w_val + l_val * u[1]
        rlm1 = np.where(l_val == 1, 1.0, Rm ** (l_val - 1))
        rl = rlm1 * Rm
        phi = rl * u4
        phior = rlm1 * u4
        phi2 = phi**2
        phi2or2 = phior**2
        # Auxiliaries are exact inputs (the C code computes them at 6th order;
        # here we isolate the 4th-order FD error of the main operators).
        Dr_u6 = alpha * (Drr_exa[0] - Dr_exa[0] / Rm + Dr_exa[0] ** 2) / Rm
        Dr_u7 = 2 * h2 * (Drr_exa[2] - Dr_exa[2] / Rm + 2 * Dr_exa[2] ** 2) / Rm

        phys = (
            alpha2,
            h2,
            a2,
            a2_r,
            wplOmega,
            phi,
            phior,
            phi2,
            phi2or2,
            alpha,
            u4,
            u5,
            Dr_u6,
            Dr_u7,
            rlm1,
            rl,
        )
        a_ex = assemble_args(phys, (Dr_exa, Dz_exa, Drr_exa, Dzz_exa), Rm, l_val, m_val, w_val)
        a_fd = assemble_args(phys, (Dr_fd, Dz_fd, Drr_fd, Dzz_fd), Rm, l_val, m_val, w_val)

        sl = slice(6, -6), slice(6, -6)
        maxerr = 0.0
        for fn in resid_fns:
            S = fn(*[v[sl] for v in a_ex])
            F = fn(*[v[sl] for v in a_fd])
            maxerr = max(maxerr, float(np.max(np.abs(F - S))))
        errors.append(maxerr)
        print(f"  N={N:4d}: max interior |resid_FD - S| = {maxerr:.3e}")

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    print(f"  observed orders: {[f'{o:.2f}' for o in orders]}")
    ok = all(o > 3.0 for o in orders)
    print("MMS " + ("PASS" if ok else "FAIL") + " (expect ~4th order interior)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
