//! rbs-elliptic: the elliptic initial-data stage.
//!
//! Residual/Jacobian assembly over the grid (calling the generated kernels),
//! the global Newton loop with the frequency eigenvalue, and the omega
//! constraint. The C cross-references are in each module's docs and the
//! crate README; the parity gates live in `ng/docs/architecture.md`.

pub mod assembly;
pub mod generated;
pub mod newton;
pub mod omega;

#[cfg(test)]
mod kernel_parity;
