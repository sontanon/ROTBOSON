//! Campaign entry point.
//!
//! Status: skeleton. The first live gate is SAN-62 (l=1, N=64 seed solve in
//! pure Rust vs the C oracle); until then we refuse to pretend.

use rbs_driver::CampaignSpec;

fn main() {
    let spec = CampaignSpec::l1_seed_slice();
    eprintln!(
        "rbs-driver: campaign \"{}\" (l={}, grid {}x{}, ghost {}) is the SAN-62 vertical slice.",
        spec.name, spec.l, spec.nr_total, spec.nz_total, spec.ghost
    );
    eprintln!(
        "rbs-driver: solves are not wired yet — rbs-elliptic assembly lands with SAN-62/63."
    );
    std::process::exit(64);
}
