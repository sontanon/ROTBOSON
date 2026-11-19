//! The six grid functions in the C solver's field-major layout.

use crate::grid::Grid2d;

/// Number of grid functions (`GNUM` in `src/csr_vars.c`):
/// u1..u6 = log(alpha), beta, log(h), log(a), psi, lambda.
pub const NUM_FIELDS: usize = 6;

/// Field-major storage of the six grid functions: field `k` occupies
/// `[k*dim, (k+1)*dim)`, exactly the C `u` vector layout
/// (`u[k * dim + IDX(i, j)]`).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSet {
    dim: usize,
    values: Vec<f64>,
}

impl FieldSet {
    /// A zero-initialized field set for `grid`.
    #[must_use]
    pub fn zeros(grid: &Grid2d) -> Self {
        Self {
            dim: grid.dim(),
            values: vec![0.0; NUM_FIELDS * grid.dim()],
        }
    }

    /// Points per field (the grid's `dim`).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total stored values (`6 * dim`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always false: a field set holds `NUM_FIELDS * dim` values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The flat storage slice, field-major (`u` in the C code).
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// The flat storage slice, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Field `k` (`0..NUM_FIELDS`), flat over the grid's index map.
    ///
    /// # Panics
    ///
    /// Panics if `k >= NUM_FIELDS`.
    #[must_use]
    pub fn field(&self, k: usize) -> &[f64] {
        assert!(k < NUM_FIELDS, "field index {k} out of range");
        &self.values[k * self.dim..(k + 1) * self.dim]
    }

    /// Field `k`, mutable.
    ///
    /// # Panics
    ///
    /// Panics if `k >= NUM_FIELDS`.
    pub fn field_mut(&mut self, k: usize) -> &mut [f64] {
        assert!(k < NUM_FIELDS, "field index {k} out of range");
        &mut self.values[k * self.dim..(k + 1) * self.dim]
    }

    /// Value of field `k` at flat grid index `idx`.
    #[must_use]
    pub fn at(&self, k: usize, idx: usize) -> f64 {
        self.field(k)[idx]
    }

    /// Set field `k` at flat grid index `idx`.
    pub fn set(&mut self, k: usize, idx: usize, value: f64) {
        self.field_mut(k)[idx] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_field_major_like_the_c_u_vector() {
        let g = Grid2d::new(10, 6, 2, 0.1, 0.1);
        let mut f = FieldSet::zeros(&g);
        assert_eq!(f.dim(), 60);
        assert_eq!(f.len(), NUM_FIELDS * 60);
        for k in 0..NUM_FIELDS {
            let level = f64::from(u32::try_from(k).expect("k fits u32"));
            f.field_mut(k).iter_mut().for_each(|v| *v = level);
        }
        // Fields are disjoint slices in the C order.
        for k in 0..NUM_FIELDS {
            let level = f64::from(u32::try_from(k).expect("k fits u32"));
            assert_eq!(f.at(k, 0), level);
            assert_eq!(f.at(k, 59), level);
        }
        let flat = f.as_slice();
        assert_eq!(flat[2 * 60 + 3], 2.0);
    }

    #[test]
    #[should_panic(expected = "field index 6 out of range")]
    fn out_of_range_field_panics() {
        let g = Grid2d::new(10, 6, 2, 0.1, 0.1);
        let f = FieldSet::zeros(&g);
        let _ = f.field(NUM_FIELDS);
    }
}
