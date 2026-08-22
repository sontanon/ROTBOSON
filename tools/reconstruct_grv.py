"""Faithful Python reconstruction of the GRV2/GRV3 virial identities.

Reimplements ex_analysis()'s GRV section (src/analysis.c) from the saved
spherical outputs of a ROTBOSON solution directory:

- diff1rr / diff1th : 4th-order stencils (src/derivatives.c), EVEN symmetry
- simps             : Simpson rule (src/simpson.c)
- GRV2/GRV3 virial integrals + Kerr extrapolation corrections

Usage:
    uv run tools/reconstruct_grv.py <solution_dir>

Prints virial, correction, total, and the values stored in GRV2.asc/GRV3.asc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rotboson_io import read_1d, read_2d, read_scalar

M_PI = np.pi
EVEN = 1


def diff1rr(var: np.ndarray, drr: float, symrr: int) -> np.ndarray:
    nrr, nth = var.shape
    dvar = np.zeros_like(var)
    idrr = 1.0 / drr
    twelfth = 1.0 / 12.0
    # origin row
    dvar[0, :] = twelfth * idrr * (-var[2, :] + 8.0 * var[1, :]) * (1.0 - symrr)
    # row 1
    dvar[1, :] = twelfth * idrr * (-var[3, :] + 8.0 * var[2, :] - 8.0 * var[0, :] + symrr * var[1, :])
    # interior
    for i in range(2, nrr - 2):
        dvar[i, :] = twelfth * idrr * (-(var[i + 2, :] - var[i - 2, :]) + 8.0 * (var[i + 1, :] - var[i - 1, :]))
    # last two rows
    dvar[nrr - 2, :] = twelfth * idrr * (
        3.0 * var[nrr - 1, :] + 10.0 * var[nrr - 2, :] - 18.0 * var[nrr - 3, :]
        + 6.0 * var[nrr - 4, :] - var[nrr - 5, :])
    dvar[nrr - 1, :] = twelfth * idrr * (
        25.0 * var[nrr - 1, :] - 48.0 * var[nrr - 2, :] + 36.0 * var[nrr - 3, :]
        - 16.0 * var[nrr - 4, :] + 3.0 * var[nrr - 5, :])
    return dvar


def diff1th(var: np.ndarray, dth: float, symr: int, symz: int) -> np.ndarray:
    nrr, nth = var.shape
    dvar = np.zeros_like(var)
    idth = 1.0 / dth
    twelfth = 1.0 / 12.0
    # axial symmetry rows
    dvar[:, 0] = twelfth * idth * (-var[:, 2] + 8.0 * var[:, 1]) * (1.0 - symr)
    dvar[:, 1] = twelfth * idth * (-8.0 * var[:, 0] + symr * var[:, 1] + 8.0 * var[:, 2] - var[:, 3])
    # interior
    for j in range(2, nth - 2):
        dvar[:, j] = twelfth * idth * (-(var[:, j + 2] - var[:, j - 2]) + 8.0 * (var[:, j + 1] - var[:, j - 1]))
    # equatorial symmetry rows
    dvar[:, nth - 2] = -twelfth * idth * (
        -8.0 * var[:, nth - 1] + symz * var[:, nth - 2] + 8.0 * var[:, nth - 3] - var[:, nth - 4])
    dvar[:, nth - 1] = -twelfth * idth * (-var[:, nth - 3] + 8.0 * var[:, nth - 2]) * (1.0 - symz)
    return dvar


def simps(y: np.ndarray, dx: float) -> float:
    n = y.shape[0] - 1
    dx_o_3 = dx / 3.0
    result = 0.0
    if n % 2 == 0:
        for k in range(1, n // 2):
            result += dx_o_3 * 2.0 * (2.0 * y[2 * k - 1] + y[2 * k])
        result += dx_o_3 * (y[0] + 4.0 * y[n - 1] + y[n])
    else:
        for k in range(1, (n + 1) // 2):
            result += dx_o_3 * 2.0 * (2.0 * y[2 * k - 1] + y[2 * k])
        result += dx_o_3 * (y[0] + 5.0 * y[1] + 5.0 * y[n] + y[n - 1])
        result *= 3.0 / 8.0
    return result


def grv_reconstruct(sol_dir: Path):
    sph_log_alpha = read_2d(sol_dir / "sph_log_alpha_f.asc")
    sph_beta = read_2d(sol_dir / "sph_beta_f.asc")
    sph_log_h = read_2d(sol_dir / "sph_log_h_f.asc")
    sph_log_a = read_2d(sol_dir / "sph_log_a_f.asc")
    sph_psi = read_2d(sol_dir / "sph_psi_f.asc")
    sph_rr = read_2d(sol_dir / "sph_rr.asc")
    sph_th = read_2d(sol_dir / "sph_th.asc")

    w = read_scalar(sol_dir / "w_f.asc")
    m = 1.0
    l = int(sol_dir.name.split(",")[0].split("=")[1])
    nr, nth = sph_psi.shape
    drr = float(np.diff(sph_rr[:, 0]).mean())
    dth = float(np.diff(sph_th[0, :]).mean())

    d_log_alpha_r = diff1rr(sph_log_alpha, drr, EVEN)
    d_beta_r = diff1rr(sph_beta, drr, EVEN)
    d_log_h_r = diff1rr(sph_log_h, drr, EVEN)
    d_log_a_r = diff1rr(sph_log_a, drr, EVEN)
    d_psi_r = diff1rr(sph_psi, drr, EVEN)
    d_log_alpha_th = diff1th(sph_log_alpha, dth, EVEN, EVEN)
    d_beta_th = diff1th(sph_beta, dth, EVEN, EVEN)
    d_log_h_th = diff1th(sph_log_h, dth, EVEN, EVEN)
    d_log_a_th = diff1th(sph_log_a, dth, EVEN, EVEN)
    d_psi_th = diff1th(sph_psi, dth, EVEN, EVEN)

    i0 = np.zeros_like(sph_rr)
    i1 = np.zeros_like(sph_rr)
    i2 = np.zeros_like(sph_rr)
    i3 = np.zeros_like(sph_rr)

    # GRV2 virial (analysis.c lines 264-315)
    for k in range(nth, nr * nth):
        rr = sph_rr.ravel()[k]
        th = sph_th.ravel()[k]
        r = rr * np.sin(th)
        rlm1 = 1.0 if l == 1 else r ** (l - 1)
        alpha2 = np.exp(2.0 * sph_log_alpha.ravel()[k])
        beta = sph_beta.ravel()[k]
        a2 = np.exp(2.0 * sph_log_a.ravel()[k])
        h2 = np.exp(2.0 * sph_log_h.ravel()[k])
        phi_o_r = sph_psi.ravel()[k] * rlm1
        phi2_o_r2 = phi_o_r * phi_o_r
        i0.ravel()[k] = 4.0 * M_PI * rr * (a2 * (((w + l * beta) * (w + l * beta) / alpha2 - m * m) * r * r + l * l / h2) * phi2_o_r2
            - (l * l * phi2_o_r2 + rlm1 * rlm1 * np.sin(th) ** 2 * ((rr * d_psi_r.ravel()[k]) ** 2 + d_psi_th.ravel()[k] ** 2)
               + 2.0 * l * phi_o_r * rlm1 * np.sin(th) * (np.sin(th) * (rr * d_psi_r.ravel()[k]) + np.cos(th) * d_psi_th.ravel()[k])))
        i1.ravel()[k] = 0.75 * h2 * rr * np.sin(th) ** 2 * ((rr * d_beta_r.ravel()[k]) ** 2 + d_beta_th.ravel()[k] ** 2) / alpha2
        i2.ravel()[k] = -(d_log_alpha_r.ravel()[k] * (rr * d_log_alpha_r.ravel()[k]) + d_log_alpha_th.ravel()[k] * (d_log_alpha_th.ravel()[k] / rr))
        i3.ravel()[k] = i0.ravel()[k] + i1.ravel()[k] + i2.ravel()[k]
    I3 = np.zeros(nr)
    for k in range(1, nr):
        I3[k] = 2.0 * simps(i3[k, :], dth)
    grv2_virial = simps(I3, drr)

    # GRV3 virial (analysis.c lines 317-370)
    for k in range(nth, nr * nth):
        rr = sph_rr.ravel()[k]
        th = sph_th.ravel()[k]
        r = rr * np.sin(th)
        rlm1 = 1.0 if l == 1 else r ** (l - 1)
        alpha2 = np.exp(2.0 * sph_log_alpha.ravel()[k])
        beta = sph_beta.ravel()[k]
        a2 = np.exp(2.0 * sph_log_a.ravel()[k])
        h2 = np.exp(2.0 * sph_log_h.ravel()[k])
        phi_o_r = sph_psi.ravel()[k] * rlm1
        phi2_o_r2 = phi_o_r * phi_o_r
        i0.ravel()[k] = 4.0 * M_PI * np.exp(sph_log_h.ravel()[k]) * rr * rr * np.sin(th) * (
            a2 * (1.5 * ((w + l * beta) * (w + l * beta) / alpha2 - m * m) * r * r - 0.5 * l * l / h2) * phi2_o_r2
            - 0.5 * (l * l * phi2_o_r2 + rlm1 * rlm1 * np.sin(th) ** 2 * ((rr * d_psi_r.ravel()[k]) ** 2 + d_psi_th.ravel()[k] ** 2)
                     + 2.0 * l * phi_o_r * rlm1 * np.sin(th) * (np.sin(th) * (rr * d_psi_r.ravel()[k]) + np.cos(th) * d_psi_th.ravel()[k])))
        i1.ravel()[k] = 0.375 * np.exp(3.0 * sph_log_h.ravel()[k]) * np.sin(th) ** 3 * rr * rr * ((rr * d_beta_r.ravel()[k]) ** 2 + d_beta_th.ravel()[k] ** 2) / alpha2
        i2.ravel()[k] = -np.sin(th) * np.exp(sph_log_h.ravel()[k]) * (((rr * d_log_alpha_r.ravel()[k]) ** 2 + d_log_alpha_th.ravel()[k] ** 2)
            - 0.5 * ((rr * d_log_h_r.ravel()[k]) * (rr * d_log_a_r.ravel()[k]) + d_log_h_th.ravel()[k] * d_log_a_th.ravel()[k])) \
            + 0.5 * (h2 - a2) * (np.sin(th) * (rr * d_log_a_r.ravel()[k]) + np.cos(th) * d_log_a_th.ravel()[k]
                                  - 0.5 * (np.sin(th) * (rr * d_log_h_r.ravel()[k]) + np.cos(th) * d_log_h_th.ravel()[k])) / np.exp(sph_log_h.ravel()[k])
        i3.ravel()[k] = i0.ravel()[k] + i1.ravel()[k] + i2.ravel()[k]
    I3 = np.zeros(nr)
    for k in range(1, nr):
        I3[k] = 4.0 * M_PI * simps(i3[k, :], dth)
    grv3_virial = simps(I3, drr)

    # Komar quantities (from profile files, last = outer boundary).
    M = 0.5 * (read_1d(sol_dir / "M_Komar1.asc")[-1] + read_1d(sol_dir / "M_Komar2.asc")[-1])
    J = 0.5 * (read_1d(sol_dir / "J_Komar1.asc")[-1] + read_1d(sol_dir / "J_Komar2.asc")[-1])
    rr_inf = drr * nr  # reconstructed spherical outer radius
    a = J / M
    x = M / rr_inf
    grv2_c = -M_PI * x * x * (0.5 + x * ((4.0 / 3.0) + x * (3.0 - (33.0 / 8.0) * a * a / (M * M) + x * ((32.0 / 5.0) - (31.0 / 5.0) * a * a / (M * M)))))
    grv3_c = M_PI * M * x * (4.0 + x * (8.0 + x * (8.0 * (86.0 - 15.0 * a * a / (M * M)) / 45.0
        + x * (2.0 * (1526.0 - 379.0 * a * a / (M * M)) / 105.0
        + x * (4.0 * (21576.0 - 9256.0 * a * a / (M * M) + 1365.0 * (a ** 4) / (M ** 4)) / 1575.0)))))

    stored2 = read_scalar(sol_dir / "GRV2.asc")
    stored3 = read_scalar(sol_dir / "GRV3.asc")
    print(f"solution: {sol_dir.name} (l={l}, w={w:.6e})")
    print(f"GRV2 virial={grv2_virial:+.10e}  correction={grv2_c:+.10e}  total={grv2_virial+grv2_c:+.10e}  stored={stored2:+.10e}")
    print(f"GRV3 virial={grv3_virial:+.10e}  correction={grv3_c:+.10e}  total={grv3_virial+grv3_c:+.10e}  stored={stored3:+.10e}")
    return grv2_virial, grv3_virial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sol_dir", type=Path)
    args = parser.parse_args()
    grv_reconstruct(args.sol_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
