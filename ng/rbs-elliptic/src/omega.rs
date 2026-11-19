//! The scalar-field frequency parameterization (port of `src/omega_calc.c`).
//!
//! The solver stores the frequency as `xi` and maps to the physical
//! frequency with a bounded tanh (which keeps the eigenvalue search inside
//! `(0, m)`); the Jacobian's omega column is scaled with `dw/dxi` at run
//! time, exactly like `jacobian_submatrix_w = dw_du(xi, m) * (dF/dw)` in
//! `src/csr_vars.c`.

/// Physical frequency `w = 0.5*m*(1 + tanh(xi))` (`omega_calc`).
#[must_use]
#[inline]
pub fn omega_calc(xi: f64, m: f64) -> f64 {
    0.5 * m * (1.0 + xi.tanh())
}

/// Inverse map `xi = atanh(2w/m - 1)` (`inverse_omega_calc`).
#[must_use]
#[inline]
pub fn inverse_omega_calc(w: f64, m: f64) -> f64 {
    (2.0 * w / m - 1.0).atanh()
}

/// Jacobian of [`omega_calc`]: `dw/dxi = 0.5*m*sech(xi)^2` (`dw_du`).
#[must_use]
#[inline]
pub fn dw_dxi(xi: f64, m: f64) -> f64 {
    // Mirrors the C form `0.5 * m * sech(u) * sech(u)`.
    let sech = xi.cosh().recip();
    0.5 * m * sech * sech
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from the C formulas, evaluated by hand at
    /// xi = 0: w = m/2, dw/dxi = m/2 (sech(0) = 1).
    #[test]
    fn tanh_map_center_and_limits() {
        let m = 1.3;
        assert!((omega_calc(0.0, m) - 0.5 * m).abs() < 1e-15);
        assert!((dw_dxi(0.0, m) - 0.5 * m).abs() < 1e-15);
        // Bounded into (0, m).
        assert!(omega_calc(-40.0, m).abs() < 1e-12);
        assert!((omega_calc(40.0, m) - m).abs() < 1e-12);
        assert!(omega_calc(-40.0, m) >= 0.0);
        assert!(omega_calc(40.0, m) <= m);
    }

    #[test]
    fn inverse_roundtrips_to_machine_precision() {
        let m = 1.0;
        for &w in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let xi = inverse_omega_calc(w, m);
            assert!(
                (omega_calc(xi, m) - w).abs() < 1e-15,
                "roundtrip failed at w={w}"
            );
        }
    }

    #[test]
    fn dw_dxi_matches_central_differences() {
        let (m, xi) = (0.8, -1.7);
        let h = 1e-6;
        let fd = (omega_calc(xi + h, m) - omega_calc(xi - h, m)) / (2.0 * h);
        assert!((dw_dxi(xi, m) - fd).abs() < 1e-9);
    }
}
