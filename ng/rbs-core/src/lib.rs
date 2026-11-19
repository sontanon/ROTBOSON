//! rbs-core: grids, fields, stencils and index maps for the RBS-NG elliptic
//! solver.
//!
//! Every convention here mirrors the C oracle (see the crate README and
//! ADR 0001): the grid sizing, the index map, the coordinate map, the
//! field-major storage, and the step-scaled Fornberg stencils emitted by
//! the kernel generators.

pub mod fields;
pub mod grid;
pub mod stencils;
