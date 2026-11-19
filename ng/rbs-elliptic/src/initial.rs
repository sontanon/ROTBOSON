//! The from-scratch analytical initial guess — the port of
//! `initial_guess` in `src/initial.c` (`readInitialData == 0`, no
//! per-field input files).
//!
//! Metric fields start at flat (log alpha = beta = log h = log a = 0,
//! lambda from its regularity form), the scalar field from the
//! Gaussian + asymptotic-tail ansatz with the `erf` cutoff of the C,
//! and the frequency slot holds `xi = atanh(2 w0 / m - 1)`. Ghost cells
//! are filled by even reflection (corner, axis, equator — in the C's
//! order), and the `scale_u*` factors (default 1.0) apply afterwards.

use rbs_core::fields::FieldSet;
use rbs_core::grid::Grid2d;

/// Ansatz + scaling parameters of the from-scratch initial guess.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialGuess {
    /// Azimuthal number.
    pub l: i64,
    /// Scalar-field mass.
    pub m: f64,
    /// Gaussian amplitude `psi0`.
    pub psi0: f64,
    /// Radial Gaussian width `sigmaR`.
    pub sigma_r: f64,
    /// Axial Gaussian width `sigmaZ`.
    pub sigma_z: f64,
    /// Asymptotic-tail cutoff radius `rExt`.
    pub r_ext: f64,
    /// Fixed physical frequency `w0` (`fixedOmega` mode).
    pub w0: f64,
    /// Field scaling factors `scale_u0..scale_u5` (omega uses
    /// `scale_u6`), all `1.0` in the smoke specs.
    pub scales: [f64; 7],
}

impl InitialGuess {
    /// The l=1 N=64 smoke spec's initial data
    /// (`configs/l1_smoke_ci.toml`).
    #[must_use]
    pub fn l1_smoke() -> Self {
        Self {
            l: 1,
            m: 1.0,
            psi0: 0.01,
            sigma_r: 4.0,
            sigma_z: 4.0,
            r_ext: 12.0,
            w0: 0.95,
            scales: [1.0; 7],
        }
    }
}

/// The solve state: the six grid fields plus the frequency slot
/// (`u[GNUM * dim]` in the C layout).
#[derive(Debug, Clone, PartialEq)]
pub struct SolveState {
    /// The six field-major grid functions.
    pub fields: FieldSet,
    /// The stored frequency parameter `xi` (`w = omega_calc(xi, m)`).
    pub xi: f64,
}

impl SolveState {
    /// Flatten into the C `u` vector layout: `6 * dim` field values
    /// followed by the frequency slot.
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        let mut u = Vec::with_capacity(self.fields.len() + 1);
        u.extend_from_slice(self.fields.as_slice());
        u.push(self.xi);
        u
    }

    /// `u += alpha * du` (the C `ARRAY_SUM`).
    ///
    /// # Panics
    ///
    /// Panics if `du` is not the flat state length.
    #[allow(clippy::missing_panics_doc)]
    pub fn axpy(&mut self, alpha: f64, du: &[f64]) {
        assert_eq!(du.len(), self.fields.len() + 1, "update length");
        let fields = self.fields.as_mut_slice();
        for (v, d) in fields.iter_mut().zip(du) {
            *v += alpha * d;
        }
        self.xi += alpha * du[self.fields.len()];
    }
}

/// Build the from-scratch initial state (`initial_guess` of
/// `src/initial.c`, `readInitialData == 0` path).
///
/// # Panics
///
/// Panics if the grid is smaller than the ansatz needs (the C indexing
/// would underflow first).
#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn initial_guess(grid: &Grid2d, ansatz: &InitialGuess) -> SolveState {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    let mut fields = FieldSet::zeros(grid);

    // Fields 0..3 (log alpha, beta, log h, log a): zero everywhere.
    for k in 0..4 {
        fields.field_mut(k).fill(0.0);
    }

    // Omega: u[w_idx] = inverse_omega_calc(scale_u6 * w0, m).
    let w0 = ansatz.w0 * ansatz.scales[6];
    let xi = crate::omega::inverse_omega_calc(w0, ansatz.m);
    let chi = (ansatz.m * ansatz.m - w0 * w0).sqrt();

    // Scalar field: Gaussian core + asymptotic tail with the erf
    // cutoff of the C (`0.5 + 0.5 * erf(2 (rr - rExt) / M_2_SQRTPI)`).
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let l = ansatz.l;
    for i in g..nr {
        let r = grid.r(i);
        for j in g..nz {
            let z = grid.z(j);
            let rr = (r * r + z * z).sqrt();
            let core = ansatz.psi0
                * (-0.5 * r * r / (ansatz.sigma_r * ansatz.sigma_r)).exp()
                * (-0.5 * z * z / (ansatz.sigma_z * ansatz.sigma_z)).exp();
            let tail = ansatz.psi0
                * (-chi * rr).exp()
                / rr.powi(i32::try_from(l + 1).expect("l + 1 fits i32"))
                * (0.5 + 0.5 * libm::erf(2.0 * (rr - ansatz.r_ext) / two_over_sqrt_pi));
            fields.set(4, grid.idx(i, j), core + tail);
        }
    }

    // Lambda: the regularity form (zero for the flat metric), pinned
    // across the axis-adjacent rows like the C's `k` loop.
    let k_axis = (0.5 / grid.dr() + f64::from(u32::try_from(g).expect("g fits u32")) - 0.5).floor();
    let k_axis = k_axis as usize;
    for i in k_axis..nr {
        let r = grid.r(i);
        for j in g..nz {
            let idx = grid.idx(i, j);
            let value = ((2.0 * fields.at(3, idx)).exp() - (2.0 * fields.at(2, idx)).exp())
                / (r * r);
            fields.set(5, idx, value);
        }
    }
    for i in 0..k_axis {
        for j in g..nz {
            fields.set(5, grid.idx(i, j), fields.at(5, grid.idx(k_axis, j)));
        }
    }

    // Ghost fills in the C's order: corner, axis, equator (all fields
    // are even).
    for k in 0..6 {
        for i in 0..g {
            for j in 0..g {
                let mirror = grid.idx(2 * g - 1 - i, 2 * g - 1 - j);
                fields.set(k, grid.idx(i, j), fields.at(k, mirror));
            }
        }
    }
    for k in 0..6 {
        for i in 0..g {
            for j in g..nz {
                let mirror = grid.idx(2 * g - 1 - i, j);
                fields.set(k, grid.idx(i, j), fields.at(k, mirror));
            }
        }
    }
    for k in 0..6 {
        for i in g..nr {
            for j in 0..g {
                let mirror = grid.idx(i, 2 * g - 1 - j);
                fields.set(k, grid.idx(i, j), fields.at(k, mirror));
            }
        }
    }

    // Scale the fields (the frequency was scaled already).
    for (k, scale) in ansatz.scales.iter().take(6).enumerate() {
        for v in fields.field_mut(k).iter_mut() {
            *v *= scale;
        }
    }

    SolveState { fields, xi }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smoke ansatz at a few sample points, checked against the
    /// C formulas evaluated by hand.
    #[test]
    fn ansatz_matches_the_c_formulas() {
        let grid = Grid2d::new(12, 12, 2, 0.25, 0.25);
        let state = initial_guess(&grid, &InitialGuess::l1_smoke());
        // Metric fields are identically zero (flat), lambda zero.
        for k in [0, 1, 2, 3, 5] {
            assert!(
                state.fields.field(k).iter().all(|&v| v == 0.0),
                "field {k} not flat"
            );
        }
        // Ghost cells mirror the interior (even parity in both axes).
        let v = |i: usize, j: usize| state.fields.at(4, grid.idx(i, j));
        // Axis: i = 0 mirrors i = 3, i = 1 mirrors i = 2.
        assert_eq!(v(0, 5), v(3, 5));
        assert_eq!(v(1, 5), v(2, 5));
        // Equator: j = 0 mirrors j = 3, j = 1 mirrors j = 2.
        assert_eq!(v(5, 0), v(5, 3));
        assert_eq!(v(5, 1), v(5, 2));
        // Corner: (0, 0) mirrors (3, 3).
        assert_eq!(v(0, 0), v(3, 3));
        // The frequency slot inverts the tanh map.
        assert!((crate::omega::omega_calc(state.xi, 1.0) - 0.95).abs() < 1e-15);
    }

    /// Far in the tail region the ansatz is core + full tail (the erf
    /// switch is spent to machine precision there), each term checked
    /// against the C formulas evaluated independently.
    #[test]
    fn ansatz_tail_dominates_beyond_rext() {
        let grid = Grid2d::new(70, 70, 2, 0.25, 0.25);
        let state = initial_guess(&grid, &InitialGuess::l1_smoke());
        let (i, j) = (66, 66); // r = z ~ 16.1, rr ~ 22.8 > rExt = 12
        let (r, z) = (grid.r(i), grid.z(j));
        let rr = (r * r + z * z).sqrt();
        let chi = (1.0_f64 - 0.95 * 0.95).sqrt();
        let core = 0.01
            * (-0.5 * r * r / 16.0).exp()
            * (-0.5 * z * z / 16.0).exp();
        let tail = 0.01 * (-chi * rr).exp() / rr.powi(2);
        let expect = core + tail;
        let got = state.fields.at(4, grid.idx(i, j));
        assert!(
            (got - expect).abs() / expect.abs() < 1e-9,
            "got {got}, expect {expect} (core {core}, tail {tail})"
        );
    }
}
