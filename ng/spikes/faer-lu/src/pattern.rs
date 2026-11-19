//! Pattern generator for the ROTBOSON 6-field Jacobian (SAN-54 spike).
//!
//! Reproduces the sparse structure of the C oracle's
//! `csr_gen_jacobian` (order=4 path) exactly:
//!   - `src/csr.c`              nnz allocation formulas + p_cc/p_cs/p_sc/p_ss
//!   - `src/csr_grid_fill.c`    row classification over the (NrTotal x NzTotal)
//!                              grid with ghost=2, fill order per field
//!   - `src/csr_vars.c`         interior/edge kernel column lists
//!                              (jacobian_4th_order_variable_omega_{cc,cs,sc,ss})
//!   - `src/csr_symmetry.c`     axis / equatorial / corner parity rows
//!   - `src/csr_robin.c`        Robin boundary rows (exact coefficient formulas)
//!   - `src/csr_exp_decay.c`    psi decay boundary rows (exact formulas)
//!   - `src/csr_omega_constraint.c`  the trailing omega/psi-pin row
//!
//! Row/column index convention (0-based; the C uses BASE=1 for PARDISO):
//!   row = field * dim + IDX(i, j),   IDX(i, j) = i * NzTotal + j,
//!   dim = NrTotal * NzTotal,  NrTotal = NzTotal = N + 4 (ghost = 2),
//!   total rows = GNUM * dim + 1 (trailing omega row).
//!
//! Values are *representative*, not the SymPy-generated expressions: the
//! interior kernels get the correct magnitude classes (Laplacian-dominated
//! diagonal, O(1) cross-field same-node couplings, small first-derivative
//! cross-field stencil terms, O(1) omega-column entries) with a smooth
//! "star-like" spatial profile; the symmetry rows and all boundary rows use
//! the exact closed-form coefficients of the C implementation.

use std::f64::consts::PI;

pub const GNUM: usize = 6;

// ---------------------------------------------------------------------------
// Kernel column lists, extracted verbatim from src/csr_vars.c.
// Tuple = (field, di, dj); field 6 encodes the omega column (w_idx).
// ---------------------------------------------------------------------------

/// jacobian_4th_order_variable_omega_cc — per-row nnz [30,29,28,46,31,40]
const CC: [&[(u8, i8, i8)]; 6] = [
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (3, -2, 0), (3, -1, 0), (3, 0, -2), (3, 0, -1), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 2, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 1, 0), (4, 2, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, 0, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 1, 0), (4, 2, 0),
        (5, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 2, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, 0), (4, 1, 0), (4, 2, 0),
        (5, -2, 0), (5, -1, 0), (5, 0, -2), (5, 0, -1), (5, 0, 0), (5, 0, 1), (5, 0, 2), (5, 1, 0), (5, 2, 0),
    ],
];

/// jacobian_4th_order_variable_omega_cs (top edge, one-sided z) — [31,31,30,47,32,42]
const CS: [&[(u8, i8, i8)]; 6] = [
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -4), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -4), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -4), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (3, -2, 0), (3, -1, 0), (3, 0, -4), (3, 0, -3), (3, 0, -2), (3, 0, -1), (3, 0, 0), (3, 0, 1), (3, 1, 0), (3, 2, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, -3), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 1, 0), (4, 2, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, 0, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (3, 0, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, -4), (4, 0, -3), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 1, 0), (4, 2, 0),
        (5, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0),
        (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
        (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
        (4, -2, 0), (4, -1, 0), (4, 0, 0), (4, 1, 0), (4, 2, 0),
        (5, -2, 0), (5, -1, 0), (5, 0, -4), (5, 0, -3), (5, 0, -2), (5, 0, -1), (5, 0, 0), (5, 0, 1), (5, 1, 0), (5, 2, 0),
    ],
];

/// jacobian_4th_order_variable_omega_sc (right edge, one-sided r) — [31,31,30,47,32,44]
const SC: [&[(u8, i8, i8)]; 6] = [
    &[
        (0, -4, 0), (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
        (2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, -4, 0), (3, -3, 0), (3, -2, 0), (3, -1, 0), (3, 0, -2), (3, 0, -1), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 1, 0),
        (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 1, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, 0, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (3, 0, 0),
        (4, -4, 0), (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 1, 0),
        (5, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -4, 0), (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0),
        (2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
        (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, 0), (4, 1, 0),
        (5, -4, 0), (5, -3, 0), (5, -2, 0), (5, -1, 0), (5, 0, -2), (5, 0, -1), (5, 0, 0), (5, 0, 1), (5, 0, 2), (5, 1, 0),
    ],
];

/// jacobian_4th_order_variable_omega_ss (corner edge, one-sided r and z) — [32,32,31,48,33,45]
const SS: [&[(u8, i8, i8)]; 6] = [
    &[
        (0, -4, 0), (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -4), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -4), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
        (2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -4), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (3, 0, 0),
        (4, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (3, -4, 0), (3, -3, 0), (3, -2, 0), (3, -1, 0), (3, 0, -4), (3, 0, -3), (3, 0, -2), (3, 0, -1), (3, 0, 0), (3, 0, 1), (3, 1, 0),
        (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, -3), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 1, 0),
        (6, 0, 0),
    ],
    &[
        (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, 0, 0),
        (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (3, 0, 0),
        (4, -4, 0), (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, -4), (4, 0, -3), (4, 0, -2), (4, 0, -1), (4, 0, 0), (4, 0, 1), (4, 1, 0),
        (5, 0, 0),
        (6, 0, 0),
    ],
    &[
        (0, -4, 0), (0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0),
        (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, -3), (1, 0, -2), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
        (2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0), (2, 0, -3), (2, 0, -2), (2, 0, -1), (2, 0, 0), (2, 0, 1), (2, 1, 0),
        (4, -3, 0), (4, -2, 0), (4, -1, 0), (4, 0, 0), (4, 1, 0),
        (5, -4, 0), (5, -3, 0), (5, -2, 0), (5, -1, 0), (5, 0, -4), (5, 0, -3), (5, 0, -2), (5, 0, -1), (5, 0, 0), (5, 0, 1), (5, 1, 0),
    ],
];

// Sanity: the kernel tables must match the C's per-row nnz bookkeeping
// (src/csr.c p_cc / p_cs / p_sc / p_ss).
const P_CC: [u64; 6] = [30, 29, 28, 46, 31, 40];
const P_CS: [u64; 6] = [31, 31, 30, 47, 32, 42];
const P_SC: [u64; 6] = [31, 31, 30, 47, 32, 44];
const P_SS: [u64; 6] = [32, 32, 31, 48, 33, 45];
const P_BOUND: [u64; 6] = [9, 9, 9, 9, 10, 9];

/// The C's per-field nnz formula (src/csr.c nnz_jacobian_get_nnzs, order 4).
/// Used as the self-check that the generated pattern matches the oracle's
/// allocation exactly.
pub fn expected_field_nnz(n: usize, k: usize) -> u64 {
    let ni = n as u64;
    let perim = 2 * ni;
    // interior + parity + edge kernels + boundary rows
    let v = P_CC[k] * ni * ni
        + (P_CS[k] + P_SC[k]) * ni
        + P_SS[k]
        + P_BOUND[k] * (perim + 3)
        // parity/symmetry band: 4*(NrI+NzI) + 24 entries per field
        + 4 * perim
        + 24;
    assert_eq!(v, expected_field_nnz_reference(n, k), "table self-check");
    v
}

/// Verbatim transcription of the nnz_k expression in src/csr.c
/// (nnz_jacobian_get_nnzs, ctx->order == 4 branch).
fn expected_field_nnz_reference(n: usize, k: usize) -> u64 {
    let ni = n as u64;
    let s = ni + ni; // NrInterior + NzInterior
    match k {
        0 => 30 * ni * ni + 31 * s + 32 + 4 * s + 24 + 9 * (3 + s),
        1 => 29 * ni * ni + 31 * s + 32 + 4 * s + 24 + 9 * (3 + s),
        2 => 28 * ni * ni + 30 * s + 31 + 4 * s + 24 + 9 * (3 + s),
        3 => 46 * ni * ni + 47 * s + 48 + 4 * s + 24 + 9 * (3 + s),
        4 => 31 * ni * ni + 32 * s + 33 + 4 * s + 24 + 10 * (3 + s),
        5 => 40 * ni * ni + 42 * ni + 44 * ni + 45 + 4 * s + 24 + 9 * (3 + s),
        _ => unreachable!(),
    }
}

pub fn expected_total_nnz(n: usize) -> u64 {
    (0..GNUM).map(|k| expected_field_nnz(n, k)).sum::<u64>() + 1
}

pub fn expected_rows(n: usize) -> usize {
    let dim = (n + 4) * (n + 4);
    GNUM * dim + 1
}

// ---------------------------------------------------------------------------
// Finite-difference weights (match the C macros in csr_robin.c).
// ---------------------------------------------------------------------------
const D1_4: [f64; 5] = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];
const S1_4: [f64; 5] = [0.25, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0];
const SO1_4: [f64; 5] = [-1.0 / 12.0, 0.5, -1.5, 5.0 / 6.0, 0.25];

/// omega_calc(u, m) = 0.5 m (1 + tanh u)   (src/omega_calc.c)
fn omega_calc(u: f64, m: f64) -> f64 {
    0.5 * m * (1.0 + u.tanh())
}
fn dw_du(u: f64, m: f64) -> f64 {
    let s = 1.0 / u.cosh();
    0.5 * m * s * s
}

// ---------------------------------------------------------------------------
// Representative interior values.
// ---------------------------------------------------------------------------
// Same-node algebraic coupling magnitudes per (row field k, col field f).
const XKF: [[f64; 6]; 6] = [
    // row 0 (residual u1): couples to beta, h, a, psi
    [0.0, 0.45, 0.95, 1.60, 1.25, 0.0],
    // row 1 (residual beta)
    [0.30, 0.0, 0.80, 0.70, 0.50, 0.0],
    // row 2 (residual h)
    [0.90, 0.50, 0.0, 0.60, 0.45, 0.0],
    // row 3 (residual a)
    [0.70, 0.35, 1.10, 0.0, 1.30, 0.0],
    // row 4 (residual psi)
    [0.50, 0.25, 0.65, 1.00, 0.0, 0.55],
    // row 5 (residual lambda)
    [1.20, 0.50, 1.00, 0.0, 0.90, 0.0],
];
// First-derivative cross-field coefficient magnitudes per (k, f).
const DKF: [[f64; 6]; 6] = [
    [0.0, 0.35, 0.30, 0.10, 0.15, 0.0],
    [0.10, 0.0, 0.35, 0.08, 0.06, 0.0],
    [0.30, 0.15, 0.0, 0.10, 0.08, 0.0],
    [0.25, 0.10, 0.40, 0.0, 0.50, 0.0],
    [0.20, 0.05, 0.30, 0.10, 0.0, 0.12],
    [0.45, 0.20, 0.35, 0.0, 0.30, 0.0],
];
// Same-node algebraic part added to the self-field diagonal.
const ALG: [f64; 6] = [0.90, 0.25, 1.10, 0.50, 0.85, 0.35];
// Omega-column magnitude per row field (rows 2 and 5 have no omega entry).
const EW: [f64; 6] = [1.1, 0.9, 0.0, 0.8, 1.0, 0.0];

/// Smooth "star-like" field profile on the interior grid, in [0, 1].
fn blob(i: i64, j: i64, g: &Grid) -> f64 {
    let r = (i - 2) as f64 * g.dr;
    let z = (j - 2) as f64 * g.dz;
    let (c, sig) = (0.25 * g.domain, 0.15 * g.domain);
    (-(((r - c) * (r - c) + (z - c) * (z - c)) / (2.0 * sig * sig))).exp()
}

pub struct Grid {
    pub n: usize,
    pub nr_total: usize,
    pub nz_total: usize,
    pub dim: usize,
    pub w_idx: usize,
    pub nrows: usize,
    pub dr: f64,
    pub dz: f64,
    pub domain: f64,
    /// omega state variable xi and mass m (representative cold-start values).
    pub xi: f64,
    pub m: f64,
    pub l: i64,
}

impl Grid {
    pub fn new(n: usize, domain: f64) -> Self {
        let nr_total = n + 4;
        let nz_total = n + 4;
        let dim = nr_total * nz_total;
        Grid {
            n,
            nr_total,
            nz_total,
            dim,
            w_idx: GNUM * dim,
            nrows: GNUM * dim + 1,
            dr: domain / n as f64,
            dz: domain / n as f64,
            domain,
            xi: 0.0,
            m: 1.0,
            l: 1,
        }
    }

    #[inline]
    fn idx(&self, i: i64, j: i64) -> usize {
        (i * self.nz_total as i64 + j) as usize
    }

    #[inline]
    fn col(&self, f: u8, i: i64, j: i64) -> usize {
        f as usize * self.dim + self.idx(i, j)
    }
}

/// CSR Jacobian: rows 0..nrows-1, row-major, columns sorted ascending.
pub struct Jacobian {
    pub grid: Grid,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub vals: Vec<f64>,
    /// nnz per field block, for the structural self-check.
    pub field_nnz: [u64; GNUM],
}

pub fn build_jacobian(n: usize, domain: f64) -> Jacobian {
    let g = Grid::new(n, domain);
    let nrows = g.nrows;
    let nnz_total = expected_total_nnz(n) as usize;

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx = Vec::with_capacity(nnz_total);
    let mut vals = Vec::with_capacity(nnz_total);
    let mut field_nnz = [0u64; GNUM];

    row_ptr.push(0u32);

    let mut cols: Vec<u32> = Vec::with_capacity(64);
    let mut vs: Vec<f64> = Vec::with_capacity(64);

    for k in 0..GNUM {
        let start_nnz = col_idx.len() as u64;
        for i in 0..g.nr_total as i64 {
            for j in 0..g.nz_total as i64 {
                cols.clear();
                vs.clear();
                row_of(&g, k as u8, i, j, &mut cols, &mut vs);
                debug_assert!(cols.windows(2).all(|w| w[0] < w[1]), "cols sorted");
                for (&c, &v) in cols.iter().zip(vs.iter()) {
                    col_idx.push(c);
                    vals.push(v);
                }
                row_ptr.push(col_idx.len() as u32);
            }
        }
        field_nnz[k] = col_idx.len() as u64 - start_nnz;
    }

    // Trailing omega-constraint row (fixed-omega mode: identity on w_idx).
    let w_idx = g.w_idx as u32;
    col_idx.push(w_idx);
    vals.push(1.0);
    row_ptr.push(col_idx.len() as u32);

    // ----- structural self-check against the C oracle -----
    assert_eq!(row_ptr.len(), nrows + 1);
    assert_eq!(col_idx.len(), nnz_total, "total nnz mismatch vs csr.c");
    for k in 0..GNUM {
        assert_eq!(
            field_nnz[k],
            expected_field_nnz(n, k),
            "field {} nnz mismatch vs csr.c",
            k
        );
    }

    Jacobian {
        grid: g,
        row_ptr,
        col_idx,
        vals,
        field_nnz,
    }
}

/// Emit (cols, vals) for row (field k, grid point (i, j)).
fn row_of(g: &Grid, k: u8, i: i64, j: i64, cols: &mut Vec<u32>, vs: &mut Vec<f64>) {
    let ghost = 2i64;
    let n = g.n as i64;
    let n_int = g.grid_interior_end(); // ghost + n

    if i < ghost {
        // left (axis) band
        if j < ghost {
            // corner parity: [self, mirrored corner]
            let (mi, mj) = (2 * ghost - 1 - i, 2 * ghost - 1 - j);
            push_sym(g, cols, vs, k, (i, j), (mi, mj));
        } else {
            // axis parity: [self, mirror across the axis]
            let mi = 2 * ghost - 1 - i;
            push_sym(g, cols, vs, k, (i, j), (mi, j));
        }
        return;
    }
    if i < n_int {
        // main band
        if j < ghost {
            let mj = 2 * ghost - 1 - j;
            push_sym(g, cols, vs, k, (i, j), (i, mj));
        } else if j < n_int {
            push_kernel(g, CC[k as usize], k, i, j, cols, vs);
        } else if j == n_int {
            push_kernel(g, CS[k as usize], k, i, j, cols, vs);
        } else {
            // j == NzTotal - 1 : top boundary
            if k == 4 {
                push_decay(g, k, i, j, Dir::Z, false, cols, vs);
            } else {
                push_robin(g, k, i, j, Dir::Z, false, cols, vs);
            }
        }
        return;
    }
    if i == n_int {
        // right band (semi-one-sided r)
        if j < ghost {
            let mj = 2 * ghost - 1 - j;
            push_sym(g, cols, vs, k, (i, j), (i, mj));
        } else if j < n_int {
            push_kernel(g, SC[k as usize], k, i, j, cols, vs);
        } else if j == n_int {
            push_kernel(g, SS[k as usize], k, i, j, cols, vs);
        } else if k == 4 {
            push_decay(g, k, i, j, Dir::Z, true, cols, vs);
        } else {
            push_robin(g, k, i, j, Dir::Z, true, cols, vs);
        }
        return;
    }
    // i == NrTotal - 1 : outer band
    if j < ghost {
        let mj = 2 * ghost - 1 - j;
        push_sym(g, cols, vs, k, (i, j), (i, mj));
    } else if j < n_int {
        if k == 4 {
            push_decay(g, k, i, j, Dir::R, false, cols, vs);
        } else {
            push_robin(g, k, i, j, Dir::R, false, cols, vs);
        }
    } else if j == n_int {
        if k == 4 {
            push_decay(g, k, i, j, Dir::R, true, cols, vs);
        } else {
            push_robin(g, k, i, j, Dir::R, true, cols, vs);
        }
    } else {
        // far corner
        if k == 4 {
            push_decay(g, k, i, j, Dir::Corner, false, cols, vs);
        } else {
            push_robin(g, k, i, j, Dir::Corner, false, cols, vs);
        }
    }
}

impl Grid {
    #[inline]
    fn grid_interior_end(&self) -> i64 {
        2 + self.n as i64
    }
}

fn push_sym(
    g: &Grid,
    cols: &mut Vec<u32>,
    vs: &mut Vec<f64>,
    k: u8,
    (i, j): (i64, i64),
    (mi, mj): (i64, i64),
) {
    // values [1, -1] with EVEN/EVEN parity (src/csr.c r_sym/z_sym tables).
    cols.push(g.col(k, i, j) as u32);
    vs.push(1.0);
    cols.push(g.col(k, mi, mj) as u32);
    vs.push(-1.0);
}

fn push_kernel(
    g: &Grid,
    table: &[(u8, i8, i8)],
    k: u8,
    i: i64,
    j: i64,
    cols: &mut Vec<u32>,
    vs: &mut Vec<f64>,
) {
    let blob = blob(i, j, g);
    let jitter = 1.0 + 0.03 * (0.7 * i as f64 + 1.3 * j as f64 + 2.1 * k as f64).sin();
    for &(f, di, dj) in table {
        let col = if f == 6 {
            g.w_idx
        } else {
            g.col(f, i + di as i64, j + dj as i64)
        };
        cols.push(col as u32);
        let v = if f == 6 {
            // omega column
            EW[k as usize] * (0.3 + 0.9 * blob * blob) * jitter
        } else if f == k {
            // self-field: 2nd-derivative stencil + algebraic diagonal part
            let off = if di != 0 { di } else { dj };
            let w = match off {
                -4 => S1_4[0],
                -3 => S1_4[1],
                -2 => S1_4[2],
                -1 => S1_4[3],
                1 => 4.0 / 3.0,
                2 => -1.0 / 12.0,
                _ => -5.0, // center: -2.5 per direction (dr == dz)
            };
            let center_alg = if off == 0 { ALG[k as usize] * blob } else { 0.0 };
            (w * (1.0 + 0.05 * blob) + center_alg) * jitter
        } else if di == 0 && dj == 0 {
            // cross-field same-node algebraic coupling
            XKF[k as usize][f as usize] * (0.25 + 0.75 * blob) * jitter
        } else {
            // cross-field stencil term (first-derivative class)
            let s = if di + dj > 0 { 1.0 } else { -1.0 };
            let mag = if (di as i64).abs() == 1 || (dj as i64).abs() == 1 {
                2.0 / 3.0
            } else {
                1.0 / 12.0
            };
            s * mag * DKF[k as usize][f as usize] * (0.4 + 0.6 * blob) * jitter
        };
        vs.push(v);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Z,
    R,
    Corner,
}

/// Robin n-power per field (src/csr.c csr_gen_jacobian: n=1,3,1,1,-,4).
fn robin_n(k: u8) -> f64 {
    match k {
        1 => 3.0,
        5 => 4.0,
        _ => 1.0,
    }
}

/// 4th-order Robin row (src/csr_robin.c). `so` selects the semi-one-sided
/// variant used at the r/z edges; Dir::Corner is the far corner.
fn push_robin(
    g: &Grid,
    k: u8,
    i: i64,
    j: i64,
    dir: Dir,
    so: bool,
    cols: &mut Vec<u32>,
    vs: &mut Vec<f64>,
) {
    let n_pow = robin_n(k);
    let ri = i as f64 - 1.5;
    let zi = j as f64 - 1.5;
    let rr2 = (ri * g.dr) * (ri * g.dr) + (zi * g.dz) * (zi * g.dz);
    let scale = g.dr * g.dz / rr2;

    let mut off: Vec<(i64, i64)> = Vec::with_capacity(9);
    let mut coef: Vec<f64> = Vec::with_capacity(9);

    match (dir, so) {
        (Dir::Z, false) => {
            off.extend([(i - 2, j), (i - 1, j)]);
            coef.extend([D1_4[0] * ri, D1_4[1] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(n_pow + S1_4[4] * zi);
            off.extend([(i + 1, j), (i + 2, j)]);
            coef.extend([D1_4[3] * ri, D1_4[4] * ri]);
        }
        (Dir::Z, true) => {
            off.extend([(i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([SO1_4[0] * ri, SO1_4[1] * ri, SO1_4[2] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(n_pow + S1_4[4] * zi + SO1_4[3] * ri);
            off.push((i + 1, j));
            coef.push(SO1_4[4] * ri);
        }
        (Dir::R, false) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 2), (i, j - 1)]);
            coef.extend([D1_4[0] * zi, D1_4[1] * zi]);
            off.push((i, j));
            coef.push(n_pow + S1_4[4] * ri);
            off.extend([(i, j + 1), (i, j + 2)]);
            coef.extend([D1_4[3] * zi, D1_4[4] * zi]);
        }
        (Dir::R, true) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([SO1_4[0] * zi, SO1_4[1] * zi, SO1_4[2] * zi]);
            off.push((i, j));
            coef.push(n_pow + S1_4[4] * ri + SO1_4[3] * zi);
            off.push((i, j + 1));
            coef.push(SO1_4[4] * zi);
        }
        (Dir::Corner, _) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(n_pow + S1_4[4] * (ri + zi));
        }
    }

    debug_assert_eq!(off.len(), 9);
    for ((ci, cj), c) in off.into_iter().zip(coef) {
        cols.push(g.col(k, ci, cj) as u32);
        vs.push(c * scale);
    }
}

/// 4th-order exponential-decay row for the scalar field (k = 4)
/// (src/csr_exp_decay.c): Robin shape + (rr*chi + l + 1) on the diagonal
/// + the d/dxi chain-rule entry in the omega column.
fn push_decay(
    g: &Grid,
    k: u8,
    i: i64,
    j: i64,
    dir: Dir,
    so: bool,
    cols: &mut Vec<u32>,
    vs: &mut Vec<f64>,
) {
    let ri = i as f64 - 1.5;
    let zi = j as f64 - 1.5;
    let rr2 = (ri * g.dr) * (ri * g.dr) + (zi * g.dz) * (zi * g.dz);
    let rr = rr2.sqrt();
    let scale = g.dr * g.dz / rr2;

    let w = omega_calc(g.xi, g.m);
    let chi = (g.m * g.m - w * w).sqrt();
    let l1 = g.l as f64 + 1.0;

    // representative psi state (smooth blob) for the f-term
    let psi = 0.02 * blob(i, j, g);
    let d10 = -0.5;
    let d12 = 0.5;
    let psi_im1 = 0.02 * blob(i - 1, j, g);
    let psi_ip1 = 0.02 * blob(i + 1, j, g);
    let psi_jm2 = 0.02 * blob(i, j - 2, g);
    let psi_jm1 = 0.02 * blob(i, j - 1, g);
    let f = ri * (d10 * psi_im1 + d12 * psi_ip1)
        + zi * (S1_4[0] * psi_jm2 + S1_4[1] * psi_jm1 + S1_4[2] * psi)
        + (rr * chi + l1) * psi;

    let mut off: Vec<(i64, i64)> = Vec::with_capacity(10);
    let mut coef: Vec<f64> = Vec::with_capacity(10);

    match (dir, so) {
        (Dir::Z, false) => {
            off.extend([(i - 2, j), (i - 1, j)]);
            coef.extend([D1_4[0] * ri, D1_4[1] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(S1_4[4] * zi + rr * chi + l1);
            off.extend([(i + 1, j), (i + 2, j)]);
            coef.extend([D1_4[3] * ri, D1_4[4] * ri]);
        }
        (Dir::Z, true) => {
            off.extend([(i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([SO1_4[0] * ri, SO1_4[1] * ri, SO1_4[2] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(S1_4[4] * zi + SO1_4[3] * ri + rr * chi + l1);
            off.push((i + 1, j));
            coef.push(SO1_4[4] * ri);
        }
        (Dir::R, false) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 2), (i, j - 1)]);
            coef.extend([D1_4[0] * zi, D1_4[1] * zi]);
            off.push((i, j));
            coef.push(S1_4[4] * ri + rr * chi + l1);
            off.extend([(i, j + 1), (i, j + 2)]);
            coef.extend([D1_4[3] * zi, D1_4[4] * zi]);
        }
        (Dir::R, true) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([SO1_4[0] * zi, SO1_4[1] * zi, SO1_4[2] * zi]);
            off.push((i, j));
            coef.push(S1_4[4] * ri + SO1_4[3] * zi + rr * chi + l1);
            off.push((i, j + 1));
            coef.push(SO1_4[4] * zi);
        }
        (Dir::Corner, _) => {
            off.extend([(i - 4, j), (i - 3, j), (i - 2, j), (i - 1, j)]);
            coef.extend([S1_4[0] * ri, S1_4[1] * ri, S1_4[2] * ri, S1_4[3] * ri]);
            off.extend([(i, j - 4), (i, j - 3), (i, j - 2), (i, j - 1)]);
            coef.extend([S1_4[0] * zi, S1_4[1] * zi, S1_4[2] * zi, S1_4[3] * zi]);
            off.push((i, j));
            coef.push(S1_4[4] * (ri + zi) + rr * chi + l1);
        }
    }

    debug_assert_eq!(off.len(), 9);
    for ((ci, cj), c) in off.into_iter().zip(coef) {
        cols.push(g.col(k, ci, cj) as u32);
        vs.push(c * scale);
    }
    // omega column: dw/dxi * (rr f + rr psi) * (-w / chi) * scale
    let wcol = dw_du(g.xi, g.m) * (rr * f + rr * psi) * (-w / chi) * scale;
    cols.push(g.w_idx as u32);
    vs.push(wcol);
}

// ---------------------------------------------------------------------------
// CSR -> CSC conversion for faer (which consumes column-major input).
// ---------------------------------------------------------------------------

/// Returns (col_ptr, row_idx, vals) in CSC form with ascending row indices
/// within each column.
pub fn csr_to_csc(jac: &Jacobian) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let n = jac.row_ptr.len() - 1;
    let nnz = jac.col_idx.len();
    let mut col_ptr = vec![0u32; n + 1];
    for &c in &jac.col_idx {
        col_ptr[c as usize + 1] += 1;
    }
    for i in 1..=n {
        col_ptr[i] += col_ptr[i - 1];
    }
    let mut next = col_ptr.clone();
    let mut row_idx = vec![0u32; nnz];
    let mut vals = vec![0.0; nnz];
    for r in 0..n {
        let (a, b) = (jac.row_ptr[r] as usize, jac.row_ptr[r + 1] as usize);
        for p in a..b {
            let c = jac.col_idx[p] as usize;
            let dst = next[c] as usize;
            next[c] += 1;
            row_idx[dst] = r as u32;
            vals[dst] = jac.vals[p];
        }
    }
    (col_ptr, row_idx, vals)
}

/// CSR matvec (for the residual sanity check).
pub fn csr_matvec(jac: &Jacobian, x: &[f64]) -> Vec<f64> {
    let n = jac.row_ptr.len() - 1;
    let mut y = vec![0.0; n];
    for r in 0..n {
        let (a, b) = (jac.row_ptr[r] as usize, jac.row_ptr[r + 1] as usize);
        let mut acc = 0.0;
        for p in a..b {
            acc += jac.vals[p] * x[jac.col_idx[p] as usize];
        }
        y[r] = acc;
    }
    y
}

/// Deterministic representative RHS for the solve timing.
pub fn make_rhs(jac: &Jacobian) -> Vec<f64> {
    let n = jac.row_ptr.len() - 1;
    (0..n)
        .map(|i| 0.5 + 0.5 * (i as f64 * 0.001 * PI).sin())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_nnz_match_reference() {
        // Reference numbers from docs/profiling.md:
        //   N=128: 104,545 rows / 3.42M nnz ; N=256: 405,601 / 13.5M
        assert_eq!(expected_rows(128), 104_545);
        assert_eq!(expected_total_nnz(128), 3_417_875);
        assert_eq!(expected_rows(256), 405_601);
        assert_eq!(expected_total_nnz(256), 13_519_891);
        assert_eq!(expected_rows(64), 27_745);
        assert_eq!(expected_total_nnz(64), 873_619);
    }

    #[test]
    fn build_n8_is_consistent() {
        let jac = build_jacobian(8, 8.0);
        assert_eq!(jac.row_ptr.len(), expected_rows(8) + 1);
        assert_eq!(jac.col_idx.len() as u64, expected_total_nnz(8));
        // every column index in range, rows sorted, no zero rows
        let n = jac.row_ptr.len() - 1;
        for r in 0..n {
            let (a, b) = (jac.row_ptr[r] as usize, jac.row_ptr[r + 1] as usize);
            assert!(a < b, "row {r} empty");
            for p in a..b {
                assert!(jac.col_idx[p] < n as u32);
            }
            for w in jac.col_idx[a..b].windows(2) {
                assert!(w[0] < w[1]);
            }
        }
        // CSC conversion round-trips
        let (cp, ri, _v) = csr_to_csc(&jac);
        assert_eq!(cp.last().copied().unwrap() as usize, jac.col_idx.len());
        assert_eq!(ri.len(), jac.col_idx.len());
    }
}
