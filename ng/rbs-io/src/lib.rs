//! rbs-io: field-set output behind one trait.
//!
//! Two backends: the VTR (`ParaView` rectilinear-grid XML) writer — real
//! from day one — and the HDF5 writer behind the `hdf5` feature flag, which
//! must mirror the C writer's self-describing contract so
//! `tools/rotboson_io.py` reads Rust output unchanged (ADR 0001 §6).

pub mod hdf5;
pub mod vtr;

/// A named 2D scalar block on a rectilinear grid: the IO-level snapshot
/// of one grid function (kept independent of `rbs-core` types so this
/// crate has no dependencies).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldBlock {
    /// Field name, e.g. `"psi"`, `"l_alpha"`; written as the dataset name.
    pub name: String,
    /// Row-major values, `dims.0 * dims.1` of them.
    pub values: Vec<f64>,
}

/// A named 1D data series: scalars (`w_f`), Newton histories
/// (`norm_du`, `lambda`), radial profiles (`M_ADM`). Written as a
/// shape-`(n,)` dataset by the HDF5 backend; VTR snapshots ignore series
/// (they carry grid fields only).
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesBlock {
    /// Series name, e.g. `"w_f"`; written as the dataset `<name>.asc`.
    pub name: String,
    /// Values, `n >= 1` of them.
    pub values: Vec<f64>,
}

/// A named 1D integer series: error codes, flags, resolutions
/// (`error_code`, `ergoregion_flag`, `hwl_resolution`).
#[derive(Debug, Clone, PartialEq)]
pub struct IntSeriesBlock {
    /// Series name; written as the dataset `<name>.asc`.
    pub name: String,
    /// Values, `n >= 1` of them.
    pub values: Vec<i64>,
}

/// A rectilinear (r, z) grid description for output blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct GridTopology {
    /// Grid point count along r.
    pub nr: usize,
    /// Grid point count along z.
    pub nz: usize,
    /// Radial coordinates, length `nr` (may be non-uniform once the grid
    /// ladder moves past uniform — ADR 0001).
    pub r: Vec<f64>,
    /// Axial coordinates, length `nz`.
    pub z: Vec<f64>,
}

impl GridTopology {
    /// A uniform rectilinear topology over `[0, r_max] x [0, z_max]`.
    ///
    /// # Panics
    ///
    /// Panics if `nr`/`nz` are zero.
    #[must_use]
    pub fn uniform(nr: usize, nz: usize, r_max: f64, z_max: f64) -> Self {
        assert!(nr > 0 && nz > 0, "grid must have points");
        // Grid sizes fit f64 without loss at every practical resolution.
        #[allow(clippy::cast_precision_loss)]
        let lin = |n: usize, max: f64| {
            (0..n).map(move |i| max * (i as f64) / (n as f64))
        };
        Self {
            nr,
            nz,
            r: lin(nr, r_max).collect(),
            z: lin(nz, z_max).collect(),
        }
    }

    /// The C oracle's cell-centered total grid (`src/main.c`):
    /// `r_i = (i - ghost + 0.5) * dr` for `i` in `0..nr`, z likewise —
    /// a `2*ghost`-layer ghost padding around `nr - 2*ghost` interior
    /// cells, so the first interior point sits at `dr/2`.
    ///
    /// # Panics
    ///
    /// Panics if `nr <= 2*ghost`, `nz <= 2*ghost`, or `dr`/`dz` are
    /// non-positive.
    #[must_use]
    pub fn centered(nr: usize, nz: usize, dr: f64, dz: f64, ghost: usize) -> Self {
        assert!(nr > 2 * ghost && nz > 2 * ghost, "grid must have interior");
        assert!(dr > 0.0 && dz > 0.0, "steps must be positive");
        #[allow(clippy::cast_precision_loss)]
        let r: Vec<f64> = (0..nr)
            .map(|i| ((i as f64) - (ghost as f64) + 0.5) * dr)
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let z: Vec<f64> = (0..nz)
            .map(|j| ((j as f64) - (ghost as f64) + 0.5) * dz)
            .collect();
        Self { nr, nz, r, z }
    }
}

/// Output backend contract: write one snapshot (a set of named field
/// blocks on one grid).
pub trait FieldWriter {
    /// Write `fields` on `grid` as one named snapshot.
    ///
    /// # Errors
    ///
    /// IO failures (path, permissions, serialization) surface here.
    fn write_snapshot(
        &mut self,
        grid: &GridTopology,
        fields: &[FieldBlock],
    ) -> std::io::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_topology_spans_the_box() {
        let g = GridTopology::uniform(4, 2, 12.0, 6.0);
        assert_eq!(g.r.len(), 4);
        assert_eq!(g.z.len(), 2);
        assert!((g.r[3] - 9.0).abs() < 1e-12);
        assert!((g.z[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "grid must have points")]
    fn empty_grid_is_rejected() {
        let _ = GridTopology::uniform(0, 2, 1.0, 1.0);
    }

    #[test]
    fn centered_topology_matches_the_c_oracle() {
        // main.c: r_i = (i - ghost + 0.5) * dr, N=64 interior, order=4.
        let g = GridTopology::centered(68, 68, 0.25, 0.25, 2);
        assert_eq!(g.r.len(), 68);
        assert!((g.r[0] + 0.375).abs() < 1e-12); // (0 - 2 + 0.5) * 0.25
        assert!((g.r[2] - 0.125).abs() < 1e-12); // first interior cell
        assert!((g.r[67] - 16.375).abs() < 1e-12);
        assert!((g.z[33] - 7.875).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "grid must have interior")]
    fn all_ghost_centered_grid_is_rejected() {
        let _ = GridTopology::centered(4, 4, 1.0, 1.0, 2);
    }
}
