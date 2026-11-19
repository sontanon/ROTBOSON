//! Step-scaled finite-difference stencil tables, ported verbatim from
//! `tools/generate_kernels.py` (the single source of truth alongside the C
//! Fornberg validation in `tests/fornberg.c`).
//!
//! Each stencil point carries five weights, one per "subtype":
//! `[u, dRu, dZu, dRRu, dZZu]` where the derivatives are step-scaled —
//! `dRu = dr * D_r u`, `dRRu = dr^2 * D_rr u`, etc. — so the weights are
//! pure numbers independent of the steps (see `tools/sympy_system.py`).

/// Subtype index of the plain grid value in a coefficient row.
pub const SUBTYPE_VALUE: usize = 0;
/// Subtype index of `dRu = dr * D_r u`.
pub const SUBTYPE_DR: usize = 1;
/// Subtype index of `dZu = dz * D_z u`.
pub const SUBTYPE_DZ: usize = 2;
/// Subtype index of `dRRu = dr^2 * D_rr u`.
pub const SUBTYPE_DRR: usize = 3;
/// Subtype index of `dZZu = dz^2 * D_zz u`.
pub const SUBTYPE_DZZ: usize = 4;
/// Number of subtypes per stencil point.
pub const NUM_SUBTYPES: usize = 5;

/// One stencil point: offset `(dr_off, dz_off)` from the evaluation point,
/// plus the five step-scaled subtype weights `[u, dRu, dZu, dRRu, dZZu]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilPoint {
    /// Radial offset in grid cells.
    pub dr_off: i32,
    /// Axial offset in grid cells.
    pub dz_off: i32,
    /// Weights `[u, dRu, dZu, dRRu, dZZu]`.
    pub coeffs: [f64; NUM_SUBTYPES],
}

impl StencilPoint {
    /// The weight of subtype `s` at this point (`0..NUM_SUBTYPES`).
    ///
    /// # Panics
    ///
    /// Panics if `s >= NUM_SUBTYPES`.
    #[must_use]
    pub fn weight(&self, s: usize) -> f64 {
        self.coeffs[s]
    }
}

/// A complete stencil: its points, a name, and the ghost width it needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stencil {
    /// Stencil name as in `tools/generate_kernels.py` (`cc`, `cs`, `sc`,
    /// `ss`).
    pub name: &'static str,
    /// Ghost layers required on every side.
    pub ghost: usize,
    /// Stencil points (order is the CSR write-out order of the C kernels).
    pub points: &'static [StencilPoint],
}

impl Stencil {
    /// Gather the step-scaled subtype-`s` estimate from point samples
    /// `u[p]` (one per stencil point, in `points` order).
    #[must_use]
    pub fn apply_subtype(&self, s: usize, samples: &[f64]) -> f64 {
        debug_assert_eq!(samples.len(), self.points.len());
        self.points
            .iter()
            .zip(samples)
            .map(|(p, &u)| p.weight(s) * u)
            .sum()
    }
}

// 2nd-order centered-centered (5 points).  Weights: D_2_10 = -1/2,
// D_2_12 = +1/2, D_2_20 = 1, D_2_21 = -2, D_2_22 = 1.
// Point constructor for the tables below.
const fn p(dr_off: i32, dz_off: i32, coeffs: [f64; NUM_SUBTYPES]) -> StencilPoint {
    StencilPoint {
        dr_off,
        dz_off,
        coeffs,
    }
}

/// 2nd-order centered-centered stencil (`STENCILS_2ND["cc"]`).
pub const STENCIL_2ND_CC: [StencilPoint; 5] = [
    p(-1, 0, [0.0, -0.5, 0.0, 1.0, 0.0]),
    p(0, -1, [0.0, 0.0, -0.5, 0.0, 1.0]),
    p(0, 0, [1.0, 0.0, 0.0, -2.0, -2.0]),
    p(0, 1, [0.0, 0.0, 0.5, 0.0, 1.0]),
    p(1, 0, [0.0, 0.5, 0.0, 1.0, 0.0]),
];

/// 2nd-order stencil wrapper.
pub const STENCIL_2ND_CC_SPEC: Stencil = Stencil {
    name: "cc2",
    ghost: 1,
    points: &STENCIL_2ND_CC,
};

// 4th-order Fornberg weights (tools/generate_kernels.py FD_4TH):
// first derivative D1k = [1/12, -2/3, 0, 2/3, -1/12],
// second derivative D2k = [-1/12, 4/3, -2.5, 4/3, -1/12],
// semi-onesided S1k = [0, -1/12, 1/2, -3/2, 5/6, 1/4],
// S2k = [1/12, -1/2, 7/6, -1/3, -5/4, 5/6].

/// 4th-order centered-centered stencil (`STENCILS_4TH["cc"]`).
pub const STENCIL_4TH_CC: [StencilPoint; 9] = [
    p(-2, 0, [0.0, 1.0 / 12.0, 0.0, -1.0 / 12.0, 0.0]),
    p(-1, 0, [0.0, -2.0 / 3.0, 0.0, 4.0 / 3.0, 0.0]),
    p(0, -2, [0.0, 0.0, 1.0 / 12.0, 0.0, -1.0 / 12.0]),
    p(0, -1, [0.0, 0.0, -2.0 / 3.0, 0.0, 4.0 / 3.0]),
    p(0, 0, [1.0, 0.0, 0.0, -2.5, -2.5]),
    p(0, 1, [0.0, 0.0, 2.0 / 3.0, 0.0, 4.0 / 3.0]),
    p(0, 2, [0.0, 0.0, -1.0 / 12.0, 0.0, -1.0 / 12.0]),
    p(1, 0, [0.0, 2.0 / 3.0, 0.0, 4.0 / 3.0, 0.0]),
    p(2, 0, [0.0, -1.0 / 12.0, 0.0, -1.0 / 12.0, 0.0]),
];

/// 4th-order centered-z / semi-onesided-r stencil (`STENCILS_4TH["cs"]`).
pub const STENCIL_4TH_CS: [StencilPoint; 10] = [
    p(-2, 0, [0.0, 1.0 / 12.0, 0.0, -1.0 / 12.0, 0.0]),
    p(-1, 0, [0.0, -2.0 / 3.0, 0.0, 4.0 / 3.0, 0.0]),
    p(0, -4, [0.0, 0.0, 0.0, 0.0, 1.0 / 12.0]),
    p(0, -3, [0.0, 0.0, -1.0 / 12.0, 0.0, -1.0 / 2.0]),
    p(0, -2, [0.0, 0.0, 1.0 / 2.0, 0.0, 7.0 / 6.0]),
    p(0, -1, [0.0, 0.0, -3.0 / 2.0, 0.0, -1.0 / 3.0]),
    p(0, 0, [1.0, 0.0, 5.0 / 6.0, -2.5, -5.0 / 4.0]),
    p(0, 1, [0.0, 0.0, 1.0 / 4.0, 0.0, 5.0 / 6.0]),
    p(1, 0, [0.0, 2.0 / 3.0, 0.0, 4.0 / 3.0, 0.0]),
    p(2, 0, [0.0, -1.0 / 12.0, 0.0, -1.0 / 12.0, 0.0]),
];

/// 4th-order semi-onesided-r / centered-z stencil (`STENCILS_4TH["sc"]`).
pub const STENCIL_4TH_SC: [StencilPoint; 10] = [
    p(-4, 0, [0.0, 0.0, 0.0, 1.0 / 12.0, 0.0]),
    p(-3, 0, [0.0, -1.0 / 12.0, 0.0, -1.0 / 2.0, 0.0]),
    p(-2, 0, [0.0, 1.0 / 2.0, 0.0, 7.0 / 6.0, 0.0]),
    p(-1, 0, [0.0, -3.0 / 2.0, 0.0, -1.0 / 3.0, 0.0]),
    p(0, -2, [0.0, 0.0, 1.0 / 12.0, 0.0, -1.0 / 12.0]),
    p(0, -1, [0.0, 0.0, -2.0 / 3.0, 0.0, 4.0 / 3.0]),
    p(0, 0, [1.0, 5.0 / 6.0, 0.0, -5.0 / 4.0, -2.5]),
    p(0, 1, [0.0, 0.0, 2.0 / 3.0, 0.0, 4.0 / 3.0]),
    p(0, 2, [0.0, 0.0, -1.0 / 12.0, 0.0, -1.0 / 12.0]),
    p(1, 0, [0.0, 1.0 / 4.0, 0.0, 5.0 / 6.0, 0.0]),
];

/// 4th-order semi-onesided / semi-onesided stencil (`STENCILS_4TH["ss"]`).
pub const STENCIL_4TH_SS: [StencilPoint; 11] = [
    p(-4, 0, [0.0, 0.0, 0.0, 1.0 / 12.0, 0.0]),
    p(-3, 0, [0.0, -1.0 / 12.0, 0.0, -1.0 / 2.0, 0.0]),
    p(-2, 0, [0.0, 1.0 / 2.0, 0.0, 7.0 / 6.0, 0.0]),
    p(-1, 0, [0.0, -3.0 / 2.0, 0.0, -1.0 / 3.0, 0.0]),
    p(0, -4, [0.0, 0.0, 0.0, 0.0, 1.0 / 12.0]),
    p(0, -3, [0.0, 0.0, -1.0 / 12.0, 0.0, -1.0 / 2.0]),
    p(0, -2, [0.0, 0.0, 1.0 / 2.0, 0.0, 7.0 / 6.0]),
    p(0, -1, [0.0, 0.0, -3.0 / 2.0, 0.0, -1.0 / 3.0]),
    p(0, 0, [1.0, 5.0 / 6.0, 5.0 / 6.0, -5.0 / 4.0, -5.0 / 4.0]),
    p(0, 1, [0.0, 0.0, 1.0 / 4.0, 0.0, 5.0 / 6.0]),
    p(1, 0, [0.0, 1.0 / 4.0, 0.0, 5.0 / 6.0, 0.0]),
];

/// Stencil wrapper for [`STENCIL_4TH_CC`].
#[must_use]
pub const fn stencil_4th_cc() -> Stencil {
    Stencil {
        name: "cc",
        ghost: 2,
        points: &STENCIL_4TH_CC,
    }
}

/// Stencil wrapper for [`STENCIL_4TH_CS`].
#[must_use]
pub const fn stencil_4th_cs() -> Stencil {
    Stencil {
        name: "cs",
        ghost: 2,
        points: &STENCIL_4TH_CS,
    }
}

/// Stencil wrapper for [`STENCIL_4TH_SC`].
#[must_use]
pub const fn stencil_4th_sc() -> Stencil {
    Stencil {
        name: "sc",
        ghost: 2,
        points: &STENCIL_4TH_SC,
    }
}

/// Stencil wrapper for [`STENCIL_4TH_SS`].
#[must_use]
pub const fn stencil_4th_ss() -> Stencil {
    Stencil {
        name: "ss",
        ghost: 2,
        points: &STENCIL_4TH_SS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Centered 4th-order rules are exact on cubics (Fornberg n-point rules
    /// are exact through degree n-1); this is the Rust mirror of the C
    /// Fornberg validation (`tests/fornberg.c`).
    #[test]
    fn centered_4th_weights_are_exact_on_cubics() {
        let st = stencil_4th_cc();
        let dr = 0.125_f64;
        let r0 = 3.0_f64;
        // u(r) = r^3 -> dr * D_r u = 3 r^2 dr, dr^2 * D_rr u = 6 r dr^2.
        let sample = |off: f64| (r0 + off * dr).powi(3);
        let samples: Vec<f64> = st
            .points
            .iter()
            .map(|p| sample(f64::from(p.dr_off)))
            .collect();
        let d_ru = st.apply_subtype(SUBTYPE_DR, &samples);
        let d_rru = st.apply_subtype(SUBTYPE_DRR, &samples);
        assert!((d_ru - 3.0 * r0 * r0 * dr).abs() < 1e-14);
        assert!((d_rru - 6.0 * r0 * dr * dr).abs() < 1e-14);

        // Same exactness along z for the z-subtypes.
        let dz = 0.25_f64;
        let z0 = -1.5_f64;
        let zsample = |off: f64| (z0 + off * dz).powi(3);
        let zs: Vec<f64> = st
            .points
            .iter()
            .map(|p| zsample(f64::from(p.dz_off)))
            .collect();
        let d_zu = st.apply_subtype(SUBTYPE_DZ, &zs);
        let d_zzu = st.apply_subtype(SUBTYPE_DZZ, &zs);
        assert!((d_zu - 3.0 * z0 * z0 * dz).abs() < 1e-14);
        assert!((d_zzu - 6.0 * z0 * dz * dz).abs() < 1e-14);
    }

    /// The one-sided second-derivative tails of the sc/ss tables carry
    /// their `S2` weights in the `dRRu` slot (a transposition at the
    /// `(-4, 0)` point once dropped Jacobian entries — regression
    /// guard).
    #[test]
    fn one_sided_second_derivative_tails_are_exact_on_quartics() {
        let dr = 0.3_f64;
        let r0 = 7.0_f64;
        let sample = |off: f64| (r0 + off * dr).powi(4);
        for st in [stencil_4th_sc(), stencil_4th_ss()] {
            let samples: Vec<f64> = st
                .points
                .iter()
                .map(|p| sample(f64::from(p.dr_off)))
                .collect();
            let d_rru = st.apply_subtype(SUBTYPE_DRR, &samples);
            // d2/dr2 r^4 = 12 r^2 -> dr^2 * d2 = 12 r^2 dr^2.
            assert!(
                (d_rru - 12.0 * r0 * r0 * dr * dr).abs() < 1e-9,
                "stencil {} dRRu: {d_rru}",
                st.name
            );
            // The z-tail of the same points must be silent.
            let d_zzu = st.apply_subtype(SUBTYPE_DZZ, &samples);
            assert!(d_zzu.abs() < 1e-12, "stencil {} dZZu: {d_zzu}", st.name);
        }
    }

    /// Value weights reproduce the function itself at the center point.
    #[test]
    fn value_weights_reproduce_the_center_sample() {
        for st in [
            STENCIL_2ND_CC_SPEC,
            stencil_4th_cc(),
            stencil_4th_cs(),
            stencil_4th_sc(),
            stencil_4th_ss(),
        ] {
            let samples: Vec<f64> =
                (0..st.points.len()).map(|p| 0.1 * f64::from(u32::try_from(p).expect("fits u32"))).collect();
            let center = st
                .points
                .iter()
                .position(|p| p.dr_off == 0 && p.dz_off == 0)
                .expect("stencil has a center point");
            let value = st.apply_subtype(SUBTYPE_VALUE, &samples);
            assert!((value - samples[center]).abs() < 1e-15, "stencil {}", st.name);
        }
    }

    /// The 2nd-order stencil is exact on quadratics.
    #[test]
    fn centered_2nd_weights_are_exact_on_quadratics() {
        let st = STENCIL_2ND_CC_SPEC;
        let dr = 0.5_f64;
        let r0 = 2.0_f64;
        let sample = |off: f64| (r0 + off * dr).powi(2);
        let samples: Vec<f64> = st
            .points
            .iter()
            .map(|p| sample(f64::from(p.dr_off)))
            .collect();
        let d_ru = st.apply_subtype(SUBTYPE_DR, &samples);
        let d_rru = st.apply_subtype(SUBTYPE_DRR, &samples);
        assert!((d_ru - 2.0 * r0 * dr).abs() < 1e-14);
        assert!((d_rru - 2.0 * dr * dr).abs() < 1e-14);
    }
}
