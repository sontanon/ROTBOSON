//! Physical finite-difference operators with parity-embedded axis stencils
//! — the port of `src/derivatives.c` (`ex_diff1r`, `ex_diff2r`, `ex_diff1z`,
//! `ex_diff2z`).
//!
//! The C derivatives sweep the full grid (ghosts included): centered rules
//! in the interior, parity-reflected one-sided rules at the axis/equator
//! (the reflection is folded into the stencil weights, so ghost values of
//! the *derivative* are never read from the ghost zone), fully one-sided
//! rules at the outer boundary, and an explicit odd/even fill of the ghost
//! derivative values (`dvar[ghost - 1 - i] = ±dvar[ghost + i]`).
//!
//! Only the orders the elliptic solve exercises are ported: `4` (the
//! configured order of the N=64 vertical slice) and `6` (the
//! `order + 2` regularization-derivative order of `src/rhs.c`). Second
//! order is SAN-58 (grid completeness) and panics here.

use crate::grid::Grid2d;

/// Parity of a grid function under reflection across the axis/equator
/// (`EVEN = 1`, `ODD = -1` in `src/rhs.c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    /// The function is symmetric: `u(-x) = u(x)`.
    Even,
    /// The function is antisymmetric: `u(-x) = -u(x)`.
    Odd,
}

impl Parity {
    /// The C sign convention: `+1.0` for even, `-1.0` for odd.
    #[must_use]
    pub fn sign(self) -> f64 {
        match self {
            Self::Even => 1.0,
            Self::Odd => -1.0,
        }
    }
}

/// First radial derivative over the full grid (`ex_diff1r`), physical
/// scaling: `dvar = du/dr`.
///
/// # Panics
///
/// Panics on orders other than 4 and 6 (SAN-58 covers the rest).
#[allow(clippy::missing_panics_doc)]
pub fn diff1r(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff1r output length");
    assert_eq!(v.len(), grid.dim(), "diff1r input length");
    let idr = grid.dr().recip();
    let s = sym.sign();
    match order {
        4 => {
            for j in 0..nz {
                // Axis: parity-embedded (ghost reflections folded in).
                out[grid.idx(g, j)] = idr / 12.0
                    * (-v[grid.idx(g + 2, j)] + (8.0 + s) * v[grid.idx(g + 1, j)]
                        - 8.0 * s * v[grid.idx(g, j)]);
                out[grid.idx(g + 1, j)] = idr / 12.0
                    * (-v[grid.idx(g + 3, j)] + 8.0 * v[grid.idx(g + 2, j)]
                        + (s - 8.0) * v[grid.idx(g, j)]);
                // Main interior points.
                for i in g + 2..nr - 2 {
                    out[grid.idx(i, j)] = idr / 12.0
                        * (8.0 * (v[grid.idx(i + 1, j)] - v[grid.idx(i - 1, j)])
                            - (v[grid.idx(i + 2, j)] - v[grid.idx(i - 2, j)]));
                }
                // Outer boundary: semi-onesided then one-sided.
                out[grid.idx(nr - 2, j)] = idr / 12.0
                    * (3.0 * v[grid.idx(nr - 1, j)] + 10.0 * v[grid.idx(nr - 2, j)]
                        - 18.0 * v[grid.idx(nr - 3, j)] + 6.0 * v[grid.idx(nr - 4, j)]
                        - v[grid.idx(nr - 5, j)]);
                out[grid.idx(nr - 1, j)] = idr / 12.0
                    * (25.0 * v[grid.idx(nr - 1, j)] - 48.0 * v[grid.idx(nr - 2, j)]
                        + 36.0 * v[grid.idx(nr - 3, j)] - 16.0 * v[grid.idx(nr - 4, j)]
                        + 3.0 * v[grid.idx(nr - 5, j)]);
            }
        }
        6 => {
            for j in 0..nz {
                // Axis: parity-embedded 6th-order rules.
                out[grid.idx(g, j)] = idr / 60.0
                    * (-(v[grid.idx(g + 2, j)] - v[grid.idx(g + 3, j)])
                        + 9.0 * (v[grid.idx(g + 1, j)] - v[grid.idx(g + 2, j)])
                        - 45.0 * (v[grid.idx(g, j)] - v[grid.idx(g + 1, j)]));
                out[grid.idx(g + 1, j)] = idr / 60.0
                    * (-(v[grid.idx(g + 1, j)] - v[grid.idx(g + 4, j)])
                        + 9.0 * (v[grid.idx(g, j)] - v[grid.idx(g + 3, j)])
                        - 45.0 * (v[grid.idx(g, j)] - v[grid.idx(g + 2, j)]));
                out[grid.idx(g + 2, j)] = idr / 60.0
                    * (-(v[grid.idx(g, j)] - v[grid.idx(g + 5, j)])
                        + 9.0 * (v[grid.idx(g, j)] - v[grid.idx(g + 4, j)])
                        - 45.0 * (v[grid.idx(g + 1, j)] - v[grid.idx(g + 3, j)]));
                // Main interior points.
                for i in g + 3..nr - 3 {
                    out[grid.idx(i, j)] = idr / 60.0
                        * (-(v[grid.idx(i - 3, j)] - v[grid.idx(i + 3, j)])
                            + 9.0 * (v[grid.idx(i - 2, j)] - v[grid.idx(i + 2, j)])
                            - 45.0 * (v[grid.idx(i - 1, j)] - v[grid.idx(i + 1, j)]));
                }
                // Outer boundary: three one-sided rules.
                out[grid.idx(nr - 3, j)] = idr / 60.0
                    * (v[grid.idx(nr - 7, j)] - 8.0 * v[grid.idx(nr - 6, j)]
                        + 30.0 * v[grid.idx(nr - 5, j)] - 80.0 * v[grid.idx(nr - 4, j)]
                        + 35.0 * v[grid.idx(nr - 3, j)] + 24.0 * v[grid.idx(nr - 2, j)]
                        - 2.0 * v[grid.idx(nr - 1, j)]);
                out[grid.idx(nr - 2, j)] = idr / 60.0
                    * (-2.0 * v[grid.idx(nr - 7, j)] + 15.0 * v[grid.idx(nr - 6, j)]
                        - 50.0 * v[grid.idx(nr - 5, j)] + 100.0 * v[grid.idx(nr - 4, j)]
                        - 150.0 * v[grid.idx(nr - 3, j)] + 77.0 * v[grid.idx(nr - 2, j)]
                        + 10.0 * v[grid.idx(nr - 1, j)]);
                out[grid.idx(nr - 1, j)] = idr / 60.0
                    * (10.0 * v[grid.idx(nr - 7, j)] - 72.0 * v[grid.idx(nr - 6, j)]
                        + 225.0 * v[grid.idx(nr - 5, j)] - 400.0 * v[grid.idx(nr - 4, j)]
                        + 450.0 * v[grid.idx(nr - 3, j)] - 360.0 * v[grid.idx(nr - 2, j)]
                        + 147.0 * v[grid.idx(nr - 1, j)]);
            }
        }
        other => unimplemented!("diff1r order {other} (SAN-58 grid completeness)"),
    }
    // Symmetry fill of the ghost derivative values (first derivatives are
    // odd under the reflection, for both parities of the underlying field).
    for j in 0..nz {
        for i in 0..g {
            out[grid.idx(g - 1 - i, j)] = -s * out[grid.idx(g + i, j)];
        }
    }
}

/// First axial derivative over the full grid (`ex_diff1z`), physical
/// scaling: `dvar = du/dz`.
///
/// # Panics
///
/// Panics on orders other than 4 and 6 (SAN-58 covers the rest).
#[allow(clippy::missing_panics_doc)]
pub fn diff1z(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff1z output length");
    assert_eq!(v.len(), grid.dim(), "diff1z input length");
    let idz = grid.dz().recip();
    let s = sym.sign();
    match order {
        4 => {
            for i in 0..nr {
                // Equator: parity-embedded.
                out[grid.idx(i, g)] = idz / 12.0
                    * (-v[grid.idx(i, g + 2)] + (8.0 + s) * v[grid.idx(i, g + 1)]
                        - 8.0 * s * v[grid.idx(i, g)]);
                out[grid.idx(i, g + 1)] = idz / 12.0
                    * (-v[grid.idx(i, g + 3)] + 8.0 * v[grid.idx(i, g + 2)]
                        + (s - 8.0) * v[grid.idx(i, g)]);
                // Main interior points.
                for j in g + 2..nz - 2 {
                    out[grid.idx(i, j)] = idz / 12.0
                        * (8.0 * (v[grid.idx(i, j + 1)] - v[grid.idx(i, j - 1)])
                            - (v[grid.idx(i, j + 2)] - v[grid.idx(i, j - 2)]));
                }
                // Outer boundary.
                out[grid.idx(i, nz - 2)] = idz / 12.0
                    * (3.0 * v[grid.idx(i, nz - 1)] + 10.0 * v[grid.idx(i, nz - 2)]
                        - 18.0 * v[grid.idx(i, nz - 3)] + 6.0 * v[grid.idx(i, nz - 4)]
                        - v[grid.idx(i, nz - 5)]);
                out[grid.idx(i, nz - 1)] = idz / 12.0
                    * (25.0 * v[grid.idx(i, nz - 1)] - 48.0 * v[grid.idx(i, nz - 2)]
                        + 36.0 * v[grid.idx(i, nz - 3)] - 16.0 * v[grid.idx(i, nz - 4)]
                        + 3.0 * v[grid.idx(i, nz - 5)]);
            }
        }
        other => unimplemented!("diff1z order {other} (SAN-58 grid completeness)"),
    }
    for i in 0..nr {
        for j in 0..g {
            out[grid.idx(i, g - 1 - j)] = -s * out[grid.idx(i, g + j)];
        }
    }
}

/// Second radial derivative over the full grid (`ex_diff2r`), physical
/// scaling: `dvar = d2u/dr2`.
///
/// # Panics
///
/// Panics on orders other than 4 (SAN-58 covers the rest).
#[allow(clippy::missing_panics_doc)]
pub fn diff2r(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff2r output length");
    assert_eq!(v.len(), grid.dim(), "diff2r input length");
    let idr2 = grid.dr().recip().powi(2);
    let s = sym.sign();
    match order {
        4 => {
            for j in 0..nz {
                // Axis: parity-embedded.
                out[grid.idx(g, j)] = idr2 / 12.0
                    * (-v[grid.idx(g + 2, j)] + (16.0 - s) * v[grid.idx(g + 1, j)]
                        + (16.0 * s - 30.0) * v[grid.idx(g, j)]);
                out[grid.idx(g + 1, j)] = idr2 / 12.0
                    * (-v[grid.idx(g + 3, j)] + 16.0 * v[grid.idx(g + 2, j)]
                        - 30.0 * v[grid.idx(g + 1, j)] + (16.0 - s) * v[grid.idx(g, j)]);
                // Main interior points.
                for i in g + 2..nr - 2 {
                    out[grid.idx(i, j)] = -idr2 / 12.0
                        * (30.0 * v[grid.idx(i, j)]
                            - 16.0 * (v[grid.idx(i + 1, j)] + v[grid.idx(i - 1, j)])
                            + v[grid.idx(i + 2, j)] + v[grid.idx(i - 2, j)]);
                }
                // Outer boundary.
                out[grid.idx(nr - 2, j)] = idr2 / 12.0
                    * (10.0 * v[grid.idx(nr - 1, j)] - 15.0 * v[grid.idx(nr - 2, j)]
                        - 4.0 * v[grid.idx(nr - 3, j)] + 14.0 * v[grid.idx(nr - 4, j)]
                        - 6.0 * v[grid.idx(nr - 5, j)] + v[grid.idx(nr - 6, j)]);
                out[grid.idx(nr - 1, j)] = idr2 / 12.0
                    * (45.0 * v[grid.idx(nr - 1, j)] - 154.0 * v[grid.idx(nr - 2, j)]
                        + 214.0 * v[grid.idx(nr - 3, j)] - 156.0 * v[grid.idx(nr - 4, j)]
                        + 61.0 * v[grid.idx(nr - 5, j)] - 10.0 * v[grid.idx(nr - 6, j)]);
            }
        }
        other => unimplemented!("diff2r order {other} (SAN-58 grid completeness)"),
    }
    // Second derivatives are even under the reflection.
    for j in 0..nz {
        for i in 0..g {
            out[grid.idx(g - 1 - i, j)] = s * out[grid.idx(g + i, j)];
        }
    }
}

/// Second axial derivative over the full grid (`ex_diff2z`), physical
/// scaling: `dvar = d2u/dz2`.
///
/// # Panics
///
/// Panics on orders other than 4 (SAN-58 covers the rest).
#[allow(clippy::missing_panics_doc)]
pub fn diff2z(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff2z output length");
    assert_eq!(v.len(), grid.dim(), "diff2z input length");
    let idz2 = grid.dz().recip().powi(2);
    let s = sym.sign();
    match order {
        4 => {
            for i in 0..nr {
                // Equator: parity-embedded.
                out[grid.idx(i, g)] = idz2 / 12.0
                    * (-v[grid.idx(i, g + 2)] + (16.0 - s) * v[grid.idx(i, g + 1)]
                        + (16.0 * s - 30.0) * v[grid.idx(i, g)]);
                out[grid.idx(i, g + 1)] = idz2 / 12.0
                    * (-v[grid.idx(i, g + 3)] + 16.0 * v[grid.idx(i, g + 2)]
                        - 30.0 * v[grid.idx(i, g + 1)] + (16.0 - s) * v[grid.idx(i, g)]);
                // Main interior points.
                for j in g + 2..nz - 2 {
                    out[grid.idx(i, j)] = -idz2 / 12.0
                        * (30.0 * v[grid.idx(i, j)]
                            - 16.0 * (v[grid.idx(i, j + 1)] + v[grid.idx(i, j - 1)])
                            + v[grid.idx(i, j + 2)] + v[grid.idx(i, j - 2)]);
                }
                // Outer boundary.
                out[grid.idx(i, nz - 2)] = idz2 / 12.0
                    * (10.0 * v[grid.idx(i, nz - 1)] - 15.0 * v[grid.idx(i, nz - 2)]
                        - 4.0 * v[grid.idx(i, nz - 3)] + 14.0 * v[grid.idx(i, nz - 4)]
                        - 6.0 * v[grid.idx(i, nz - 5)] + v[grid.idx(i, nz - 6)]);
                out[grid.idx(i, nz - 1)] = idz2 / 12.0
                    * (45.0 * v[grid.idx(i, nz - 1)] - 154.0 * v[grid.idx(i, nz - 2)]
                        + 214.0 * v[grid.idx(i, nz - 3)] - 156.0 * v[grid.idx(i, nz - 4)]
                        + 61.0 * v[grid.idx(i, nz - 5)] - 10.0 * v[grid.idx(i, nz - 6)]);
            }
        }
        other => unimplemented!("diff2z order {other} (SAN-58 grid completeness)"),
    }
    for i in 0..nr {
        for j in 0..g {
            out[grid.idx(i, g - 1 - j)] = s * out[grid.idx(i, g + j)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The axis rules equal the centered rule applied to the
    /// parity-reflected grid (the reflection is folded into the weights).
    #[test]
    fn axis_rules_match_reflected_centered_rules() {
        // A smooth even function and a smooth odd function.
        fn f_even(x: f64) -> f64 {
            1.0 + 0.3 * x * x + 0.05 * x.powi(4)
        }
        fn df_even(x: f64) -> f64 {
            0.6 * x + 0.2 * x.powi(3)
        }
        fn ddf_even(x: f64) -> f64 {
            0.6 + 0.6 * x * x
        }
        fn f_odd(x: f64) -> f64 {
            0.7 * x - 0.02 * x.powi(3)
        }
        fn df_odd(x: f64) -> f64 {
            0.7 - 0.06 * x * x
        }
        fn ddf_odd(x: f64) -> f64 {
            -0.12 * x
        }
        let g = Grid2d::new(14, 9, 2, 0.25, 0.5);
        let dim = g.dim();
        for (f, df, ddf, sym) in [
            (
                f_even as fn(f64) -> f64,
                df_even as fn(f64) -> f64,
                ddf_even as fn(f64) -> f64,
                Parity::Even,
            ),
            (
                f_odd as fn(f64) -> f64,
                df_odd as fn(f64) -> f64,
                ddf_odd as fn(f64) -> f64,
                Parity::Odd,
            ),
        ] {
            let mut v = vec![0.0; dim];
            for i in 0..g.nr_total() {
                let x = g.r(i);
                for j in 0..g.nz_total() {
                    v[g.idx(i, j)] = f(x);
                }
            }
            // Fold the reflection into the ghost values, like the solver
            // state always is.
            for i in 0..g.ghost() {
                for j in 0..g.nz_total() {
                    v[g.idx(i, j)] = sym.sign() * v[g.idx(2 * g.ghost() - 1 - i, j)];
                }
            }
            let mut d1 = vec![0.0; dim];
            diff1r(&g, &mut d1, &v, sym, 4);
            let mut d2 = vec![0.0; dim];
            diff2r(&g, &mut d2, &v, sym, 4);
            for i in g.ghost()..g.nr_total() {
                let x = g.r(i);
                let idx = g.idx(i, g.ghost());
                assert!(
                    (d1[idx] - df(x)).abs() < 1e-11,
                    "d1 axis-ish row {i}: {} vs {}",
                    d1[idx],
                    df(x)
                );
                assert!(
                    (d2[idx] - ddf(x)).abs() < 1e-9,
                    "d2 row {i}: {} vs {}",
                    d2[idx],
                    ddf(x)
                );
            }
        }
    }

    /// Ghost derivative values follow the parity sign rules of the C
    /// (`d1` odd, `d2` even under the reflection).
    #[test]
    fn ghost_derivative_fills_follow_parity() {
        let g = Grid2d::new(12, 12, 2, 0.25, 0.25);
        let dim = g.dim();
        let v: Vec<f64> = (0..dim).map(|i| (i % 17) as f64 * 0.01).collect();
        let mut d1 = vec![0.0; dim];
        diff1r(&g, &mut d1, &v, Parity::Even, 4);
        for j in 0..g.nz_total() {
            assert!((d1[g.idx(1, j)] + d1[g.idx(2, j)]).abs() < 1e-14);
            assert!((d1[g.idx(0, j)] + d1[g.idx(3, j)]).abs() < 1e-14);
        }
        let mut d2 = vec![0.0; dim];
        diff2r(&g, &mut d2, &v, Parity::Even, 4);
        for j in 0..g.nz_total() {
            assert!((d2[g.idx(1, j)] - d2[g.idx(2, j)]).abs() < 1e-14);
            assert!((d2[g.idx(0, j)] - d2[g.idx(3, j)]).abs() < 1e-14);
        }
    }

    /// The 6th-order operator (the `order + 2` regularization order) is
    /// exact on even sextics, axis rows included.
    #[test]
    fn sixth_order_axis_rules_are_exact_on_sextics() {
        fn f(x: f64) -> f64 {
            2.0 + 0.7 * x * x - 0.4 * x.powi(4) + 0.02 * x.powi(6)
        }
        fn df(x: f64) -> f64 {
            1.4 * x - 1.6 * x.powi(3) + 0.12 * x.powi(5)
        }
        let g = Grid2d::new(16, 5, 2, 0.125, 0.25);
        let dim = g.dim();
        let mut v = vec![0.0; dim];
        for i in 0..g.nr_total() {
            let x = g.r(i);
            for j in 0..g.nz_total() {
                v[g.idx(i, j)] = f(x);
            }
        }
        for i in 0..g.ghost() {
            for j in 0..g.nz_total() {
                v[g.idx(i, j)] = v[g.idx(2 * g.ghost() - 1 - i, j)];
            }
        }
        let mut d1 = vec![0.0; dim];
        diff1r(&g, &mut d1, &v, Parity::Even, 6);
        for i in g.ghost()..g.nr_total() {
            let x = g.r(i);
            assert!(
                (d1[g.idx(i, g.ghost())] - df(x)).abs() < 1e-9,
                "row {i}: {} vs {}",
                d1[g.idx(i, g.ghost())],
                df(x)
            );
        }
    }
}
