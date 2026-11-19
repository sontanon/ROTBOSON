//! The uniform (r, z) grid with ghost layers, mirroring the C solver's
//! `NrTotal` / `NzTotal` / `ghost` conventions (`src/initial.c`).

/// Grid indices fit in an f64 without loss for every practical grid
/// (indices are < 2^53 by a factor of ~10^12 at paper sizes).
#[allow(clippy::cast_precision_loss)]
fn uf(index: usize) -> f64 {
    index as f64
}

/// Uniform 2D grid in cylindrical (r, z) with `ghost` ghost layers per
/// side, as in the C oracle.
///
/// The interior unknowns live at cell centers offset by half a step from
/// the grid index; the C kernels compute coordinates as
/// `r = dr * (i + 0.5 - ghost)` (`src/rhs_vars.c`), so the first interior
/// point (`i = ghost`) sits at `r = dr / 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid2d {
    nr_total: usize,
    nz_total: usize,
    ghost: usize,
    dr: f64,
    dz: f64,
}

impl Grid2d {
    /// Create a grid with `nr_total` radial and `nz_total` axial points
    /// (interior + ghosts, both inclusive) and steps `dr`, `dz`.
    ///
    /// # Panics
    ///
    /// Panics unless both totals exceed twice the ghost width (an interior
    /// must exist) and both steps are positive and finite.
    #[must_use]
    pub fn new(nr_total: usize, nz_total: usize, ghost: usize, dr: f64, dz: f64) -> Self {
        assert!(
            nr_total > 2 * ghost && nz_total > 2 * ghost,
            "grid totals must exceed the ghost layers (nr={nr_total}, nz={nz_total}, ghost={ghost})"
        );
        assert!(dr.is_finite() && dr > 0.0, "dr must be positive, got {dr}");
        assert!(dz.is_finite() && dz > 0.0, "dz must be positive, got {dz}");
        Self {
            nr_total,
            nz_total,
            ghost,
            dr,
            dz,
        }
    }

    /// Total radial point count (`NrTotal` in the C code), ghosts included.
    #[must_use]
    pub fn nr_total(&self) -> usize {
        self.nr_total
    }

    /// Total axial point count (`NzTotal` in the C code), ghosts included.
    #[must_use]
    pub fn nz_total(&self) -> usize {
        self.nz_total
    }

    /// Ghost layer count per side (`ghost` in the C code).
    #[must_use]
    pub fn ghost(&self) -> usize {
        self.ghost
    }

    /// Radial step.
    #[must_use]
    pub fn dr(&self) -> f64 {
        self.dr
    }

    /// Axial step.
    #[must_use]
    pub fn dz(&self) -> f64 {
        self.dz
    }

    /// Points per grid function (`dim = NrTotal * NzTotal` in the C code).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.nr_total * self.nz_total
    }

    /// Mirror of the C `IDX(i, j)` macro (`src/tools.h`):
    /// `IDX(i, j) = i * NzTotal + j`.
    #[must_use]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.nr_total && j < self.nz_total);
        i * self.nz_total + j
    }

    /// Radial coordinate of row `i`: `r = dr * (i + 0.5 - ghost)`
    /// (`src/rhs_vars.c`).
    #[must_use]
    pub fn r(&self, i: usize) -> f64 {
        self.dr * (self.ri(i))
    }

    /// Axial coordinate of column `j`: `z = dz * (j + 0.5 - ghost)`
    /// (`src/rhs_bdry` in `src/rhs_vars.c`).
    #[must_use]
    pub fn z(&self, j: usize) -> f64 {
        self.dz * (uf(j) + 0.5 - uf(self.ghost))
    }

    /// The radial cell index used by the Jacobian kernels:
    /// `ri = i + 0.5 - ghost` (`tools/generate_kernels.py` decl block).
    #[must_use]
    pub fn ri(&self, i: usize) -> f64 {
        uf(i) + 0.5 - uf(self.ghost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_map_mirrors_the_c_idx_macro() {
        let g = Grid2d::new(12, 8, 2, 0.125, 0.25);
        for i in 0..g.nr_total() {
            for j in 0..g.nz_total() {
                assert_eq!(g.idx(i, j), i * 8 + j);
            }
        }
        assert_eq!(g.dim(), 96);
    }

    #[test]
    fn coordinates_match_the_c_centered_convention() {
        let g = Grid2d::new(16, 16, 2, 0.125, 0.0625);
        // First interior point sits half a step from the axis.
        assert!((g.r(2) - 0.5 * 0.125).abs() < 1e-15);
        assert!((g.z(2) - 0.5 * 0.0625).abs() < 1e-15);
        // The C formula, verbatim.
        let i = 7usize;
        assert!((g.r(i) - 0.125 * (uf(i) + 0.5 - 2.0)).abs() < 1e-15);
        assert!((g.ri(i) - (uf(i) + 0.5 - 2.0)).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "grid totals must exceed the ghost layers")]
    fn rejects_grids_without_interior() {
        let _ = Grid2d::new(4, 8, 2, 0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "dr must be positive")]
    fn rejects_nonpositive_steps() {
        let _ = Grid2d::new(8, 8, 2, 0.0, 0.1);
    }
}
