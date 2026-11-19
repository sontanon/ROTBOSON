//! The generated elliptic kernels: the Rust twin of `src/rhs_vars.c` +
//! `src/csr_vars.c`, emitted by `tools/generate_rust_kernels.py` (SAN-56).
//!
//! The lint allowances below exist because the child module is
//! machine-printed numerics: its shape is chosen to mirror the C kernels
//! term-for-term, not to satisfy hand-code style.
#![allow(clippy::identity_op)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]

pub mod kernels;

pub use kernels::{
    jacobian_subtypes, residual_boundary, residual_interior, BoundaryPoint,
    JacobianPoint, ResidualPoint, GNUM, JAC_COLS, SUBTYPES_PER_FIELD,
};
