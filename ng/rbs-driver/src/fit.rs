//! A small least-squares polynomial fit, implemented by hand (no math
//! framework) — the numerical core of the fold summarize
//! (`--summarize`), replacing `numpy.polyfit`/`np.polyval`
//! (`tools/sweep_driver.py` `turning_point_estimate`).
//!
//! The x axis is affinely mapped to `[-1, 1]` before building the normal
//! equations; a degree-4 Vandermonde over a raw ψ₀ window of width ~0.1
//! would otherwise be badly conditioned. The fitted *function* is
//! identical to `np.polyfit`'s up to floating-point noise (both solve
//! the same least-squares problem; numpy uses SVD, we use normal
//! equations with partial pivoting — well-conditioned after the mapping
//! at the degrees that matter: ≤ 4 over ≤ 7 points).

/// A least-squares polynomial fit, evaluated in a scaled variable
/// `t = (x - mid) / half` for numerical stability.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyFit {
    mid: f64,
    half: f64,
    /// Coefficients of *descending* powers of `t` (like `np.polyfit`'s
    /// return order).
    coeffs: Vec<f64>,
}

impl PolyFit {
    /// Evaluate the fitted polynomial at `x` (`np.polyval`).
    pub fn eval(&self, x: f64) -> f64 {
        let t = (x - self.mid) / self.half;
        // Horner over descending coefficients.
        self.coeffs.iter().fold(0.0, |acc, c| acc * t + c)
    }
}

/// Least-squares fit of a polynomial of the given degree (`np.polyfit`
/// semantics: degree is capped at `points - 2` by the caller, mirroring
/// the Python). Returns `None` when the system is singular (duplicate x
/// values with insufficient distinct support, or a zero span).
///
/// Deviation from numpy, documented: `np.polyfit` rank-deficiency
/// produces a minimum-norm solution via SVD; here the caller lowers the
/// degree to the number of distinct x values first, and a still-singular
/// system yields `None` (the caller's "sample" fallback). For the
/// monotone, distinct-x windows the summarize actually sees, behavior is
/// identical.
pub fn polyfit(xs: &[f64], ys: &[f64], degree: usize) -> Option<PolyFit> {
    if xs.len() != ys.len() || xs.len() < 2 || degree == 0 {
        return None;
    }
    let first = *xs.first()?;
    let last = *xs.last()?;
    let span = last - first;
    if !(span > 0.0) || !span.is_finite() {
        return None;
    }
    let mid = (first + last) / 2.0;
    let half = span / 2.0;

    // Normal equations: (m = degree + 1) moments A[i][j] = Σ t^(i+j),
    // b[i] = Σ t^i · y. t ∈ [-1, 1] keeps A well-conditioned.
    let m = degree + 1;
    let mut a = vec![vec![0.0f64; m]; m];
    let mut b = vec![0.0f64; m];
    for (&x, &y) in xs.iter().zip(ys) {
        let t = (x - mid) / half;
        let mut tpow = vec![1.0f64; 2 * m - 1];
        for p in 1..tpow.len() {
            tpow[p] = tpow[p - 1] * t;
        }
        for i in 0..m {
            for j in 0..m {
                a[i][j] += tpow[i + j];
            }
            b[i] += tpow[i] * y;
        }
    }

    let coeffs_asc = solve_linear(&mut a, &mut b)?;
    let mut coeffs: Vec<f64> = coeffs_asc.into_iter().rev().collect();
    // Numerical hygiene: numpy trims leading (highest-degree) zeros only
    // in rank-deficient cases; we keep the requested degree's shape and
    // just make sure nothing non-finite leaks out.
    if coeffs.iter().any(|c| !c.is_finite()) {
        return None;
    }
    coeffs.shrink_to_fit();
    Some(PolyFit { mid, half, coeffs })
}

/// Gaussian elimination with partial pivoting. Returns `None` on a
/// (near-)singular matrix.
fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Pivot: largest |a[i][col]| for i >= col.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for i in col + 1..n {
            let v = a[i][col].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if !(best > 1e-12) {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let pivot = a[col][col];
        for i in col + 1..n {
            let f = a[i][col] / pivot;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[i][j] -= f * a[col][j];
            }
            b[i] -= f * b[col];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in i + 1..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "{a} vs {b} (|diff|={} > {tol})",
            (a - b).abs()
        );
    }

    /// A quartic fit over samples of an exact quadratic recovers the
    /// quadratic (residual 0 up to fp noise) — the shape of the
    /// `--summarize` localization test.
    #[test]
    fn quartic_fit_recovers_exact_quadratic() {
        let xs: Vec<f64> = (0..7).map(|i| 0.40 + 0.02 * f64::from(i)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 0.65 + 50.0 * (x - 0.46).powi(2))
            .collect();
        let fit = polyfit(&xs, &ys, 4).expect("fit converges");
        for (&x, &y) in xs.iter().zip(&ys) {
            approx(fit.eval(x), y, 1e-9);
        }
        // The minimum of the fit sits at the quadratic's vertex.
        let mut best = f64::INFINITY;
        let mut best_x = 0.0;
        for i in 0..=4000 {
            let x = 0.40 + 0.12 * f64::from(i) / 4000.0;
            let v = fit.eval(x);
            if v < best {
                best = v;
                best_x = x;
            }
        }
        approx(best_x, 0.46, 1e-5);
        approx(best, 0.65, 1e-9);
    }

    #[test]
    fn linear_fit_through_noise() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.1, 1.1, 1.9, 3.1];
        let fit = polyfit(&xs, &ys, 1).unwrap();
        // Least-squares line: slope 0.98, intercept 0.08.
        approx(fit.eval(0.0), 0.08, 1e-12);
        approx(fit.eval(3.0), 3.02, 1e-12);
    }

    #[test]
    fn degenerate_fits_return_none() {
        assert!(polyfit(&[], &[], 2).is_none());
        assert!(polyfit(&[1.0], &[1.0], 2).is_none());
        // zero span → degenerate t.
        assert!(polyfit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 2).is_none());
        // duplicate x with too few distinct values for degree 2.
        assert!(polyfit(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0], 2).is_none());
    }
}
