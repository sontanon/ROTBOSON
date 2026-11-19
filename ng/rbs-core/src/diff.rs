//! Physical finite-difference operators with parity-embedded axis stencils
//! — the port of `src/derivatives.c` (`ex_diff1r`, `ex_diff2r`, `ex_diff1z`,
//! `ex_diff2z`, `ex_diff2rz`).
//!
//! The C derivatives sweep the full grid (ghosts included): centered rules
//! in the interior, parity-reflected one-sided rules at the axis/equator
//! (the reflection is folded into the stencil weights, so ghost values of
//! the *derivative* are never read from the ghost zone), fully one-sided
//! rules at the outer boundary, and an explicit odd/even fill of the ghost
//! derivative values (`dvar[ghost - 1 - i] = ±dvar[ghost + i]`).
//!
//! The mixed operator [`diff2rz`] is the odd one out (`ex_diff2rz`): its
//! sweep reads the RAW ghost values of the input (no parity folding —
//! the state convention keeps the ghosts mirrored), and it fills its own
//! ghost OUTPUT rows by reflection at the end.
//!
//! Orders: `2` (grid completeness, SAN-58), `4` (the configured order of
//! the N=64 vertical slice) and `6` (the `order + 2`
//! regularization-derivative order of `src/rhs.c`); `diff2rz` supports
//! the orders its C namesake does (`2` and `4`).

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
/// Panics on orders other than 2, 4 and 6.
#[allow(clippy::missing_panics_doc)]
pub fn diff1r(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff1r output length");
    assert_eq!(v.len(), grid.dim(), "diff1r input length");
    let idr = grid.dr().recip();
    let s = sym.sign();
    match order {
        2 => {
            for j in 0..nz {
                // Axis: parity-embedded (the reflected ghost value is
                // folded into the center weight).
                out[grid.idx(g, j)] = idr * 0.5 * (v[grid.idx(g + 1, j)] - s * v[grid.idx(g, j)]);
                // Main interior points (centered; the second-to-last
                // row is plain centered, no special rule).
                for i in g + 1..nr - 1 {
                    out[grid.idx(i, j)] = idr * 0.5 * (v[grid.idx(i + 1, j)] - v[grid.idx(i - 1, j)]);
                }
                // Outer boundary: one-sided (3, -4, 1) / 2h.
                out[grid.idx(nr - 1, j)] = idr * 0.5
                    * (3.0 * v[grid.idx(nr - 1, j)] - 4.0 * v[grid.idx(nr - 2, j)]
                        + v[grid.idx(nr - 3, j)]);
            }
        }
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
        other => unimplemented!("diff1r order {other}"),
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
/// Panics on orders other than 2 and 4.
#[allow(clippy::missing_panics_doc)]
pub fn diff1z(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff1z output length");
    assert_eq!(v.len(), grid.dim(), "diff1z input length");
    let idz = grid.dz().recip();
    let s = sym.sign();
    match order {
        2 => {
            for i in 0..nr {
                // Equator: parity-embedded.
                out[grid.idx(i, g)] = idz * 0.5 * (v[grid.idx(i, g + 1)] - s * v[grid.idx(i, g)]);
                // Main interior points.
                for j in g + 1..nz - 1 {
                    out[grid.idx(i, j)] =
                        idz * 0.5 * (v[grid.idx(i, j + 1)] - v[grid.idx(i, j - 1)]);
                }
                // Outer boundary: one-sided (3, -4, 1) / 2h.
                out[grid.idx(i, nz - 1)] = idz * 0.5
                    * (3.0 * v[grid.idx(i, nz - 1)] - 4.0 * v[grid.idx(i, nz - 2)]
                        + v[grid.idx(i, nz - 3)]);
            }
        }
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
        other => unimplemented!("diff1z order {other}"),
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
/// Panics on orders other than 2 and 4.
#[allow(clippy::missing_panics_doc)]
pub fn diff2r(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff2r output length");
    assert_eq!(v.len(), grid.dim(), "diff2r input length");
    let idr2 = grid.dr().recip().powi(2);
    let s = sym.sign();
    match order {
        2 => {
            for j in 0..nz {
                // Axis: parity-embedded centered rule (the reflected
                // ghost value folds into the center weight: EVEN keeps
                // `v[g+1] - v[g]`, ODD gives `v[g+1] - 3 v[g]`).
                out[grid.idx(g, j)] =
                    idr2 * (v[grid.idx(g + 1, j)] + (s - 2.0) * v[grid.idx(g, j)]);
                // Main interior points.
                for i in g + 1..nr - 1 {
                    out[grid.idx(i, j)] =
                        idr2 * (v[grid.idx(i + 1, j)] - 2.0 * v[grid.idx(i, j)] + v[grid.idx(i - 1, j)]);
                }
                // Outer boundary: one-sided (2, -5, 4, -1) / h^2.
                out[grid.idx(nr - 1, j)] = idr2
                    * (2.0 * v[grid.idx(nr - 1, j)] - 5.0 * v[grid.idx(nr - 2, j)]
                        + 4.0 * v[grid.idx(nr - 3, j)] - v[grid.idx(nr - 4, j)]);
            }
        }
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
        other => unimplemented!("diff2r order {other}"),
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
/// Panics on orders other than 2 and 4.
#[allow(clippy::missing_panics_doc)]
pub fn diff2z(grid: &Grid2d, out: &mut [f64], v: &[f64], sym: Parity, order: u32) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff2z output length");
    assert_eq!(v.len(), grid.dim(), "diff2z input length");
    let idz2 = grid.dz().recip().powi(2);
    let s = sym.sign();
    match order {
        2 => {
            for i in 0..nr {
                // Equator: parity-embedded centered rule.
                out[grid.idx(i, g)] =
                    idz2 * (v[grid.idx(i, g + 1)] + (s - 2.0) * v[grid.idx(i, g)]);
                // Main interior points.
                for j in g + 1..nz - 1 {
                    out[grid.idx(i, j)] =
                        idz2 * (v[grid.idx(i, j + 1)] - 2.0 * v[grid.idx(i, j)] + v[grid.idx(i, j - 1)]);
                }
                // Outer boundary: one-sided (2, -5, 4, -1) / h^2.
                out[grid.idx(i, nz - 1)] = idz2
                    * (2.0 * v[grid.idx(i, nz - 1)] - 5.0 * v[grid.idx(i, nz - 2)]
                        + 4.0 * v[grid.idx(i, nz - 3)] - v[grid.idx(i, nz - 4)]);
            }
        }
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
        other => unimplemented!("diff2z order {other}"),
    }
    for i in 0..nr {
        for j in 0..g {
            out[grid.idx(i, g - 1 - j)] = s * out[grid.idx(i, g + j)];
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed derivative (ex_diff2rz).
// ---------------------------------------------------------------------------

/// One 1D first-derivative rule of the mixed-operator product form:
/// node offsets in cells and the weight WITHOUT the step scaling
/// (the `1 / (dr * dz)` factor of the cross is applied once).
type Rule1d = &'static [(i32, f64)];

/// Centered rule, 2nd order: `(u[i+1] - u[i-1]) / 2`.
const C2: Rule1d = &[(-1, -0.5), (1, 0.5)];

/// Centered rule, 4th order: `(u[i-2] - 8 u[i-1] + 8 u[i+1] - u[i+2]) / 12`.
const C4: Rule1d = &[(-2, 1.0 / 12.0), (-1, -8.0 / 12.0), (1, 8.0 / 12.0), (2, -1.0 / 12.0)];

/// One-sided rule at the LAST node, 2nd order: `(3 u[i] - 4 u[i-1] + u[i-2]) / 2`.
const LAST2: Rule1d = &[(0, 1.5), (-1, -2.0), (-2, 0.5)];

/// One-sided rule at the LAST node, 4th order:
/// `(25 u[i] - 48 u[i-1] + 36 u[i-2] - 16 u[i-3] + 3 u[i-4]) / 12`.
const LAST4: Rule1d = &[
    (0, 25.0 / 12.0),
    (-1, -48.0 / 12.0),
    (-2, 36.0 / 12.0),
    (-3, -16.0 / 12.0),
    (-4, 3.0 / 12.0),
];

/// One-sided rule at the SECOND-TO-LAST node, 4th order (nodes at
/// offsets -3..+1): `(3 u[i+1] + 10 u[i] - 18 u[i-1] + 6 u[i-2] - u[i-3]) / 12`.
const SND4: Rule1d = &[
    (1, 3.0 / 12.0),
    (0, 10.0 / 12.0),
    (-1, -18.0 / 12.0),
    (-2, 6.0 / 12.0),
    (-3, -1.0 / 12.0),
];

/// Mixed derivative over the full grid (`ex_diff2rz`), physical scaling:
/// `dvar = d2u/dr dz`.
///
/// Unlike the pure operators, the sweep READS the raw ghost values of the
/// input (the state convention keeps them mirrored; no parity folding is
/// folded into the weights), and the ghost OUTPUT rows are filled by
/// reflection afterwards (`dvar[i, g-1-k] = -symz * dvar[i, g+k]`,
/// `dvar[g-1-k, j] = -symr * dvar[g+k, j]`, corners
/// `symr * symz`), exactly the closing loops of `src/derivatives.c`.
///
/// Every boundary rule is the outer product of the 1D first-derivative
/// rules above: one-sided x centered on the last column/row, one-sided
/// x one-sided at the corners. The C writes the same products as flat
/// `1/48`- and `1/144`-scaled coefficient lists (`src/derivatives.c`);
/// the association differs, the values agree to roundoff. The
/// INTERIOR rule is the one exception: at order 4 the C uses the
/// 8-point diagonal stencil `(16*Δ11 - Δ22) / 48` (NOT the tensor
/// product of the centered rules, which would carry (±1, ±2) cross
/// terms), transcribed flat; at order 2 the diagonal form and the
/// tensor product coincide.
///
/// Not used by the elliptic solve (`Drz_u` feeds analysis and
/// interpolation); supported orders are `2` and `4`, matching the C.
///
/// Verified against the C's own operator dumps (a scratch harness
/// calling `ex_diff2rz` on deterministic states, all four parity
/// combinations, both orders): order 2 agrees to 3.7e-15, order 4 to
/// 2.2e-14 relative — the association-order class of difference, the
/// boundary products against the C's flat `1/48`/`1/144` lists.
///
/// # Panics
///
/// Panics on orders other than 2 and 4. As in the C, the grid must be
/// large enough for the rules (at least 7 total nodes per axis at
/// order 4).
#[allow(clippy::missing_panics_doc)]
pub fn diff2rz(
    grid: &Grid2d,
    out: &mut [f64],
    v: &[f64],
    sym_r: Parity,
    sym_z: Parity,
    order: u32,
) {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    assert_eq!(out.len(), grid.dim(), "diff2rz output length");
    assert_eq!(v.len(), grid.dim(), "diff2rz input length");
    let idrz = grid.dr().recip() * grid.dz().recip();
    let (sr, sz) = (sym_r.sign(), sym_z.sign());

    // The outer product of an r-rule and a z-rule at one output point.
    let cross = |rr: Rule1d, zr: Rule1d, i: i32, j: i32| -> f64 {
        let mut acc = 0.0;
        for &(di, rw) in rr {
            for &(dj, zw) in zr {
                acc += rw * zw * v[grid.idx((i + di) as usize, (j + dj) as usize)];
            }
        }
        acc
    };

    match order {
        2 => {
            // Interior block, then the last-z column per row, the
            // last-r row, and the corner — the C's loop structure.
            for i in g..nr - 1 {
                for j in g..nz - 1 {
                    out[grid.idx(i, j)] = idrz * cross(C2, C2, i as i32, j as i32);
                }
                out[grid.idx(i, nz - 1)] = idrz * cross(C2, LAST2, i as i32, (nz - 1) as i32);
            }
            for j in g..nz - 1 {
                out[grid.idx(nr - 1, j)] = idrz * cross(LAST2, C2, (nr - 1) as i32, j as i32);
            }
            out[grid.idx(nr - 1, nz - 1)] =
                idrz * cross(LAST2, LAST2, (nr - 1) as i32, (nz - 1) as i32);
        }
        4 => {
            // Interior block up to the second-to-last rows: the
            // 8-point diagonal stencil
            // `(16 Δ11 - Δ22) / 48` — NOT the tensor product of the
            // two centered 4th-order rules (which would carry
            // (±1, ±2) cross terms); the C's flat form, transcribed.
            for i in g..nr - 2 {
                for j in g..nz - 2 {
                    out[grid.idx(i, j)] = idrz / 48.0
                        * (16.0
                            * (v[grid.idx(i + 1, j + 1)] + v[grid.idx(i - 1, j - 1)]
                                - v[grid.idx(i + 1, j - 1)]
                                - v[grid.idx(i - 1, j + 1)])
                            - (v[grid.idx(i + 2, j + 2)] + v[grid.idx(i - 2, j - 2)]
                                - v[grid.idx(i + 2, j - 2)]
                                - v[grid.idx(i - 2, j + 2)]));
                }
                // The two z-boundary rows per interior i (products:
                // centered-r x one-sided-z).
                out[grid.idx(i, nz - 2)] = idrz * cross(C4, SND4, i as i32, (nz - 2) as i32);
                out[grid.idx(i, nz - 1)] = idrz * cross(C4, LAST4, i as i32, (nz - 1) as i32);
            }
            for j in g..nz - 2 {
                out[grid.idx(nr - 2, j)] = idrz * cross(SND4, C4, (nr - 2) as i32, j as i32);
                out[grid.idx(nr - 1, j)] = idrz * cross(LAST4, C4, (nr - 1) as i32, j as i32);
            }
            out[grid.idx(nr - 2, nz - 2)] =
                idrz * cross(SND4, SND4, (nr - 2) as i32, (nz - 2) as i32);
            out[grid.idx(nr - 2, nz - 1)] =
                idrz * cross(SND4, LAST4, (nr - 2) as i32, (nz - 1) as i32);
            out[grid.idx(nr - 1, nz - 2)] =
                idrz * cross(LAST4, SND4, (nr - 1) as i32, (nz - 2) as i32);
            out[grid.idx(nr - 1, nz - 1)] =
                idrz * cross(LAST4, LAST4, (nr - 1) as i32, (nz - 1) as i32);
        }
        other => unimplemented!("diff2rz order {other}"),
    }

    // Symmetries on axis and equator (the closing loops of the C: each
    // k reads only sweep-computed rows, so the fill is order-free).
    for k in 0..g {
        for i in g - k..nr {
            out[grid.idx(i, g - 1 - k)] = -sz * out[grid.idx(i, g + k)];
        }
        for j in g - k..nz {
            out[grid.idx(g - 1 - k, j)] = -sr * out[grid.idx(g + k, j)];
        }
        out[grid.idx(g - 1 - k, g - 1 - k)] = sr * sz * out[grid.idx(g + k, g + k)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The axis rules equal the centered rule applied to the
    /// parity-reflected grid (the reflection is folded into the weights).
    #[test]
    fn axis_rules_match_reflected_centered_rules() {
        let g = Grid2d::new(14, 9, 2, 0.25, 0.5);
        let dim = g.dim();
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
        let g = Grid2d::new(16, 5, 2, 0.125, 0.25);
        let dim = g.dim();
        fn f(x: f64) -> f64 {
            2.0 + 0.7 * x * x - 0.4 * x.powi(4) + 0.02 * x.powi(6)
        }
        fn df(x: f64) -> f64 {
            1.4 * x - 1.6 * x.powi(3) + 0.12 * x.powi(5)
        }
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

    // -----------------------------------------------------------------------
    // SAN-58: the order-2 operators and ex_diff2rz.
    // -----------------------------------------------------------------------

    /// The order-2 rules are exact on quadratics — the axis rule IS
    /// the centered rule applied to the parity-reflected grid, and
    /// the outer boundary rows are the standard one-sided
    /// `(3, -4, 1) / 2h` and `(2, -5, 4, -1) / h^2` rules. Checked
    /// row by row over the FULL grid (axis, interior, boundary, ghost
    /// fill) at `ghost = 1`.
    #[test]
    fn second_order_rules_are_exact_on_quadratics() {
        let g = Grid2d::new(11, 13, 1, 0.25, 0.5);
        let dim = g.dim();
        // Even quadratic and odd linear test functions.
        let f_even = |x: f64| 3.0 + 1.7 * x * x;
        let df_even = |x: f64| 3.4 * x;
        let ddf_even = |_x: f64| 3.4;
        let f_odd = |x: f64| 0.9 * x;
        let df_odd = |_x: f64| 0.9;
        let ddf_odd = |_x: f64| 0.0;
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
            // Radial sweep: v varies with r only.
            let mut v = vec![0.0; dim];
            for i in 0..g.nr_total() {
                for j in 0..g.nz_total() {
                    v[g.idx(i, j)] = f(g.r(i));
                }
            }
            let mut d1 = vec![0.0; dim];
            diff1r(&g, &mut d1, &v, sym, 2);
            let mut d2 = vec![0.0; dim];
            diff2r(&g, &mut d2, &v, sym, 2);
            for i in 0..g.nr_total() {
                let x = g.r(i);
                for j in 0..g.nz_total() {
                    assert!(
                        (d1[g.idx(i, j)] - df(x)).abs() < 1e-11,
                        "d1r row {i}: {} vs {}",
                        d1[g.idx(i, j)],
                        df(x)
                    );
                    assert!(
                        (d2[g.idx(i, j)] - ddf(x)).abs() < 1e-9,
                        "d2r row {i}: {} vs {}",
                        d2[g.idx(i, j)],
                        ddf(x)
                    );
                }
            }
            // Axial sweep: v varies with z only.
            let mut v = vec![0.0; dim];
            for i in 0..g.nr_total() {
                for j in 0..g.nz_total() {
                    v[g.idx(i, j)] = f(g.z(j));
                }
            }
            let mut d1 = vec![0.0; dim];
            diff1z(&g, &mut d1, &v, sym, 2);
            let mut d2 = vec![0.0; dim];
            diff2z(&g, &mut d2, &v, sym, 2);
            for i in 0..g.nr_total() {
                for j in 0..g.nz_total() {
                    let x = g.z(j);
                    assert!(
                        (d1[g.idx(i, j)] - df(x)).abs() < 1e-11,
                        "d1z col {j}: {} vs {}",
                        d1[g.idx(i, j)],
                        df(x)
                    );
                    assert!(
                        (d2[g.idx(i, j)] - ddf(x)).abs() < 1e-9,
                        "d2z col {j}: {} vs {}",
                        d2[g.idx(i, j)],
                        ddf(x)
                    );
                }
            }
        }
    }

    /// The order-2 operators converge at their design order on a
    /// parity-consistent smooth function — interior (centered) and
    /// global (axis + one-sided boundary) error both drop like `h^2`
    /// (the C harness's `check_conv_2d` gate, order-2 rows).
    #[test]
    fn second_order_convergence_is_second_order() {
        let k = 1.1_f64;
        let (f, df, ddf, sym) = (
            |x: f64| (k * x).cos(),
            |x: f64| -k * (k * x).sin(),
            |x: f64| -k * k * (k * x).cos(),
            Parity::Even,
        );
        let measure = |h: f64| {
            let n = 96;
            let g = Grid2d::new(n, n, 1, h, h);
            let dim = g.dim();
            let mut v = vec![0.0; dim];
            for i in 0..n {
                for j in 0..n {
                    v[g.idx(i, j)] = f(g.r(i));
                }
            }
            let mut d1 = vec![0.0; dim];
            diff1r(&g, &mut d1, &v, sym, 2);
            let mut d2 = vec![0.0; dim];
            diff2r(&g, &mut d2, &v, sym, 2);
            let mut ei1 = 0.0_f64;
            let mut eg1 = 0.0_f64;
            let mut ei2 = 0.0_f64;
            let mut eg2 = 0.0_f64;
            for i in 1..n - 1 {
                let x = g.r(i);
                let e1 = (d1[g.idx(i, n / 2)] - df(x)).abs();
                let e2 = (d2[g.idx(i, n / 2)] - ddf(x)).abs();
                eg1 = eg1.max(e1);
                eg2 = eg2.max(e2);
                if i >= 4 && i <= n - 5 {
                    ei1 = ei1.max(e1);
                    ei2 = ei2.max(e2);
                }
            }
            (ei1, eg1, ei2, eg2)
        };
        let (ei1a, eg1a, ei2a, eg2a) = measure(0.1);
        let (ei1b, eg1b, ei2b, eg2b) = measure(0.05);
        let orders = [
            (ei1a / ei1b).log2(),
            (eg1a / eg1b).log2(),
            (ei2a / ei2b).log2(),
            (eg2a / eg2b).log2(),
        ];
        for (name, o) in [
            ("d1r interior", orders[0]),
            ("d1r global", orders[1]),
            ("d2r interior", orders[2]),
            ("d2r global", orders[3]),
        ] {
            eprintln!("{name}: order {o:.3}");
            assert!((o - 2.0).abs() < 0.35, "{name}: measured order {o:.3}");
        }
    }

    /// Ghost derivative values at `ghost = 1` follow the parity sign
    /// rules (`d1` odd, `d2` even under the reflection).
    #[test]
    fn second_order_ghost_fills_follow_parity() {
        let g = Grid2d::new(9, 9, 1, 0.25, 0.25);
        let dim = g.dim();
        let v: Vec<f64> = (0..dim).map(|i| (i % 13) as f64 * 0.03).collect();
        let mut d1 = vec![0.0; dim];
        diff1r(&g, &mut d1, &v, Parity::Even, 2);
        let mut d2 = vec![0.0; dim];
        diff2r(&g, &mut d2, &v, Parity::Even, 2);
        for j in 0..g.nz_total() {
            assert!((d1[g.idx(0, j)] + d1[g.idx(1, j)]).abs() < 1e-14);
            assert!((d2[g.idx(0, j)] - d2[g.idx(1, j)]).abs() < 1e-14);
        }
        let mut d1 = vec![0.0; dim];
        diff1z(&g, &mut d1, &v, Parity::Odd, 2);
        let mut d2 = vec![0.0; dim];
        diff2z(&g, &mut d2, &v, Parity::Odd, 2);
        for i in 0..g.nr_total() {
            assert!((d1[g.idx(i, 0)] - d1[g.idx(i, 1)]).abs() < 1e-14);
            assert!((d2[g.idx(i, 0)] + d2[g.idx(i, 1)]).abs() < 1e-14);
        }
    }

    /// The mixed derivative's stencil weights are the outer products
    /// of the 1D rules (delta impulses, mirroring the C harness's
    /// `test_2drz_weights` plus the boundary products): interior =
    /// centered x centered (the 8-point diagonal form at order 4),
    /// corner = one-sided x one-sided.
    #[test]
    fn mixed_derivative_weights_are_outer_products() {
        // Order 2: totals 12, ghost 1.
        let g = Grid2d::new(12, 12, 1, 0.25, 0.3);
        let dim = g.dim();
        let (hr, hz) = (g.dr(), g.dz());
        let idrz = 1.0 / (hr * hz);
        let (i0, j0) = (6, 6);
        let mut v = vec![0.0; dim];
        let mut out = vec![0.0; dim];
        v[g.idx(i0, j0)] = 1.0;
        diff2rz(&g, &mut out, &v, Parity::Even, Parity::Even, 2);
        let at = |di: i32, dj: i32| out[g.idx((i0 as i32 + di) as usize, (j0 as i32 + dj) as usize)];
        let quarter = 0.25 * idrz;
        assert!((at(1, 1) - quarter).abs() < 1e-18);
        assert!((at(1, -1) + quarter).abs() < 1e-18);
        assert!((at(-1, 1) + quarter).abs() < 1e-18);
        assert!((at(-1, -1) - quarter).abs() < 1e-18);
        // The one-sided corner: (3, -4, 1) x (3, -4, 1) / 4.
        let (nr, nz) = (g.nr_total(), g.nz_total());
        let corner = |a: i32, b: i32| -> f64 {
            let w = [3.0, -4.0, 1.0];
            0.25 * idrz * w[a as usize] * w[b as usize]
        };
        for a in 0..3i32 {
            for b in 0..3i32 {
                let mut v = vec![0.0; dim];
                let mut out = vec![0.0; dim];
                v[g.idx((nr as i32 - 1 - a) as usize, (nz as i32 - 1 - b) as usize)] = 1.0;
                diff2rz(&g, &mut out, &v, Parity::Even, Parity::Even, 2);
                let got = out[g.idx(nr - 1, nz - 1)];
                assert!(
                    (got - corner(a, b)).abs() < 1e-18,
                    "corner weight ({a},{b}): {got} vs {}",
                    corner(a, b)
                );
            }
        }

        // Order 4: totals 14, ghost 2 — the 8-point diagonal interior.
        let g = Grid2d::new(14, 14, 2, 0.25, 0.3);
        let dim = g.dim();
        let (hr, hz) = (g.dr(), g.dz());
        let idrz = 1.0 / (hr * hz);
        let (i0, j0) = (7, 7);
        let mut v = vec![0.0; dim];
        let mut out = vec![0.0; dim];
        v[g.idx(i0, j0)] = 1.0;
        diff2rz(&g, &mut out, &v, Parity::Even, Parity::Even, 4);
        let at = |di: i32, dj: i32| out[g.idx((i0 as i32 + di) as usize, (j0 as i32 + dj) as usize)];
        let i48 = idrz / 48.0;
        for (di, dj) in [(1i32, 1i32), (-1, -1)] {
            assert!((at(di, dj) - 16.0 * i48).abs() < 1e-17, "({di},{dj})");
        }
        for (di, dj) in [(1, -1), (-1, 1)] {
            assert!((at(di, dj) + 16.0 * i48).abs() < 1e-17, "({di},{dj})");
        }
        for (di, dj) in [(2, 2), (-2, -2)] {
            assert!((at(di, dj) + i48).abs() < 1e-18, "({di},{dj})");
        }
        for (di, dj) in [(2, -2), (-2, 2)] {
            assert!((at(di, dj) - i48).abs() < 1e-18, "({di},{dj})");
        }
        // No (±1, ±2) cross terms (the diagonal-only stencil).
        for (di, dj) in [(1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)] {
            assert_eq!(at(di, dj), 0.0, "cross term ({di},{dj}) must vanish");
        }
    }

    /// `diff2rz` is exact on parity-consistent bivariate polynomial
    /// states over the FULL grid — interior, boundary rows, corners,
    /// and the reflected ghost rows (order 2: quadratics per axis;
    /// order 4: the diagonal interior rule is exact up to degree (2,
    /// 2), the one-sided boundary rules up to quartics).
    #[test]
    fn mixed_derivative_is_exact_on_polynomial_states() {
        let cases: [(fn(f64, f64) -> f64, fn(f64, f64) -> f64, Parity, Parity, u32); 8] = [
            // (state, D_rz state, sym_r, sym_z, order)
            (
                |r, z| 2.0 + r * r + 2.0 * z * z + 0.5 * r * r * z * z,
                |r, z| 2.0 * r * z,
                Parity::Even,
                Parity::Even,
                2,
            ),
            (
                |r, z| r * z,
                |_r, _z| 1.0,
                Parity::Odd,
                Parity::Odd,
                2,
            ),
            (
                |r, z| r + 0.5 * r * z * z,
                |_r, z| z,
                Parity::Odd,
                Parity::Even,
                2,
            ),
            (
                |r, z| z + 0.5 * r * r * z,
                |r, _z| r,
                Parity::Even,
                Parity::Odd,
                2,
            ),
            (
                |r, z| 2.0 + r * r + 2.0 * z * z + 0.5 * r * r * z * z,
                |r, z| 2.0 * r * z,
                Parity::Even,
                Parity::Even,
                4,
            ),
            (|r, z| r * z, |_r, _z| 1.0, Parity::Odd, Parity::Odd, 4),
            (
                |r, z| r + 0.5 * r * z * z,
                |_r, z| z,
                Parity::Odd,
                Parity::Even,
                4,
            ),
            (
                |r, z| z + 0.5 * r * r * z,
                |r, _z| r,
                Parity::Even,
                Parity::Odd,
                4,
            ),
        ];
        for (f, df, sr, sz, order) in cases {
            let ghost = if order == 2 { 1 } else { 2 };
            let g = Grid2d::new(5 * ghost + 9, 5 * ghost + 9, ghost, 0.25, 0.3);
            let dim = g.dim();
            // The state including ghosts is the analytic function
            // itself — which IS the parity-consistent convention.
            let mut v = vec![0.0; dim];
            for i in 0..g.nr_total() {
                for j in 0..g.nz_total() {
                    v[g.idx(i, j)] = f(g.r(i), g.z(j));
                }
            }
            let mut out = vec![0.0; dim];
            diff2rz(&g, &mut out, &v, sr, sz, order);
            let tol = 1e-9;
            for i in 0..g.nr_total() {
                for j in 0..g.nz_total() {
                    let (r, z) = (g.r(i), g.z(j));
                    let want = df(r, z);
                    let got = out[g.idx(i, j)];
                    assert!(
                        (got - want).abs() < tol,
                        "order {order} ({sr:?},{sz:?}) at ({i},{j}): {got} vs {want}"
                    );
                }
            }
        }
    }

    /// `diff2rz` converges at its design order on a smooth
    /// parity-consistent state (interior stencil; the one-sided
    /// boundary rows are design-order too, so the global order
    /// matches up to roundoff noise at the corners).
    #[test]
    fn mixed_derivative_converges_at_design_order() {
        let k = 0.9_f64;
        let measure = |order: u32, h: f64| -> (f64, f64) {
            let ghost = if order == 2 { 1 } else { 2 };
            let n = 6 * ghost + 41;
            let g = Grid2d::new(n, n, ghost, h, h);
            let dim = g.dim();
            // (odd, odd) state: sin(k r) sin(k z); ghosts hold the
            // analytic (odd-extended) values.
            let mut v = vec![0.0; dim];
            for i in 0..n {
                for j in 0..n {
                    v[g.idx(i, j)] = (k * g.r(i)).sin() * (k * g.z(j)).sin();
                }
            }
            let mut out = vec![0.0; dim];
            diff2rz(&g, &mut out, &v, Parity::Odd, Parity::Odd, order);
            let want = |r: f64, z: f64| k * k * (k * r).cos() * (k * z).cos();
            let mut ei = 0.0_f64;
            let mut eg = 0.0_f64;
            for i in ghost..n - ghost {
                for j in ghost..n - ghost {
                    let e = (out[g.idx(i, j)] - want(g.r(i), g.z(j))).abs();
                    eg = eg.max(e);
                    if i >= 2 * ghost + 2 && i <= n - 2 * ghost - 3 && j >= 2 * ghost + 2 && j <= n
                        - 2 * ghost
                        - 3
                    {
                        ei = ei.max(e);
                    }
                }
            }
            (ei, eg)
        };
        for order in [2u32, 4] {
            let h = 0.2 / f64::from(2 * ghost_scale(order));
            let (ei_a, eg_a) = measure(order, h);
            let (ei_b, eg_b) = measure(order, h / 2.0);
            let oi = (ei_a / ei_b).log2();
            let og = (eg_a / eg_b).log2();
            eprintln!("order {order} d2rz: interior {oi:.3}, global {og:.3}");
            assert!(
                (oi - f64::from(order)).abs() < 0.25,
                "order {order} interior: {oi:.3}"
            );
            assert!(
                og > f64::from(order) - 0.8,
                "order {order} global: {og:.3}"
            );
        }
    }

    /// Step scale of the mixed-operator grid (1 ghost at order 2, 2
    /// at order 4).
    fn ghost_scale(order: u32) -> u32 {
        if order == 2 {
            1
        } else {
            2
        }
    }
}
