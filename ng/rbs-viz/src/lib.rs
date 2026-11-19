//! rbs-viz: the visualization pipeline over `rbs-io` snapshots
//! (placeholder for the ADR 0001 "visualization is first-class" track).

/// A campaign-to-animation plan: which fields, which snapshots, what the
/// movie shows.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationSpec {
    /// Campaign name (matches `rbs-driver` campaign manifests).
    pub campaign: String,
    /// Fields to animate, in overlay order (e.g. `psi`, `l_alpha`).
    pub fields: Vec<String>,
    /// Snapshot stride (animate every Nth output).
    pub stride: usize,
    /// Fixed color scale (`None` = per-frame autoscale, which flickers on
    /// dilute branches).
    pub value_range: Option<(f64, f64)>,
}

impl AnimationSpec {
    /// Animate the scalar field of a campaign at every snapshot.
    #[must_use]
    pub fn psi_campaign(campaign: &str) -> Self {
        Self {
            campaign: campaign.to_string(),
            fields: vec!["psi".to_string()],
            stride: 1,
            value_range: None,
        }
    }
}

/// Build the VTR file list of a campaign's snapshots (the pipeline input).
///
/// # Errors
///
/// IO failures reading the snapshot directory.
pub fn snapshot_list(_campaign_dir: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    // TODO (with the driver port): scan and sort the campaign's .vtr files
    // by snapshot index.
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_defaults_to_the_scalar_field() {
        let s = AnimationSpec::psi_campaign("l1-down");
        assert_eq!(s.fields, vec!["psi".to_string()]);
        assert_eq!(s.stride, 1);
        assert_eq!(s.value_range, None);
    }

    #[test]
    fn snapshot_list_stub_is_empty_not_failing() {
        assert!(snapshot_list(std::path::Path::new(".")).is_ok());
    }
}
