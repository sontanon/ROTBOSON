//! rbs-evolve: time evolution of the elliptic initial data.
//!
//! Staged per ADR 0001: 1D spherical full-GR Einstein–Klein–Gordon first,
//! stationary-hold validation against the elliptic solve, collapse
//! thresholds, then axisymmetry. 3D/AMR stays gated behind those. This
//! crate is the placeholder for that track.

/// The 1D formulation candidates under study before committing to a
/// production evolution scheme.
///
/// The study criteria (ADR 0001 decision 4): constraint preservation on
/// the stationary hold, robustness across the collapse threshold, and —
/// decisive for RBS-NG — how naturally each formulation's solver fits the
/// multigrid + Krylov backend rather than a global direct solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formulation {
    /// BSSN-like: conformal-traceless with Gamma-driver shift; the
    /// best-understood robustness baseline.
    Bssn,
    /// CCZ4-like: constraint-damping (Z4c family); promises cleaner
    /// stationary holds.
    Ccz4,
    /// Generalized-harmonic: first-order-in-time, coordinate-source
    /// formulation; closest cousin of the C code's quasi-isotropic
    /// elliptic gauge.
    GeneralizedHarmonic,
}

impl Formulation {
    /// Short name for campaign logs.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bssn => "bssn",
            Self::Ccz4 => "ccz4",
            Self::GeneralizedHarmonic => "gh",
        }
    }
}

/// The 1D study roster, in evaluation order.
#[must_use]
pub fn study_roster() -> [Formulation; 3] {
    [Formulation::Bssn, Formulation::Ccz4, Formulation::GeneralizedHarmonic]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roster_names_are_stable() {
        let names: Vec<&str> = study_roster().iter().map(Formulation::name).collect();
        assert_eq!(names, vec!["bssn", "ccz4", "gh"]);
    }
}
