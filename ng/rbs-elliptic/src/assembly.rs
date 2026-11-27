//! Residual / Jacobian assembly over the grid — the SAN-62 port of
//! `src/rhs.c` (residual sweep) and `src/csr.c` + `src/csr_grid_fill.c` +
//! `src/csr_symmetry.c` + `src/csr_robin.c` + `src/csr_exp_decay.c` +
//! `src/csr_omega_constraint.c` (Jacobian pattern and values).
//!
//! # Residual (`src/rhs.c`)
//!
//! The driver's `rescale = -1` multiplies every row. The sweep is:
//!
//! * physical derivative grids for all six fields (`ex_diff1r/2r`,
//!   `ex_diff1z/2z` at the configured order, all parities `EVEN`);
//! * the regularization auxiliaries at `order + 2` (6th order for the
//!   4th-order slice): `u6 = alpha * Dr(log alpha) / r`,
//!   `u7 = 2H * Dr(log h) / r`, then their radial derivatives — the
//!   `REGULARIZATION_AUXILIARIES` block of `src/rhs.c`;
//! * `rhs_vars` at every non-ghost point short of the outer boundary
//!   (including the semi-onesided `cs`/`sc`/`ss` positions, exactly like
//!   the C interior loop `i < NrTotal - 1`, `j < NzTotal - 1`);
//! * `rhs_bdry` Robin/decay rows on the outer boundary (top row, right
//!   column, and the corner);
//! * the omega row `f[w_idx] = 0` (the eigenvalue normalization).
//!
//! # Jacobian (`src/csr*.c`)
//!
//! The pattern is position-classified once per solve and reused; only the
//! values are refreshed per Newton iteration, mirroring the C's
//! `csr_allocate` once + `csr_gen_jacobian` per iterate. Row order is the
//! C fill order: field-major, then row-major within each field, which
//! reproduces the oracle's `ia`/`ja` exactly (verified against the
//! `nnz_jacobian` counting formulas of `src/csr.c`).
//!
//! Interior-family rows call [`crate::generated::kernels::jacobian_subtypes`]
//! with the step-scaled subtypes and scatter
//! `aa[col] = sum_k stencil_weight_k * J[row][5*g + k]` per emitted stencil
//! point (the `_emit_csr` rule of `tools/generate_kernels.py`), plus the
//! omega column `dw/dxi * dF/dw`. Boundary rows are the direct
//! Robin/decay formulas of `src/csr_robin.c` / `src/csr_exp_decay.c`; ghost
//! rows are the symmetry conditions of `src/csr_symmetry.c`; the closing
//! row is the omega constraint of `src/csr_omega_constraint.c`.

use rbs_core::diff::{diff1r, diff1z, diff2r, diff2z, Parity};
use rbs_core::fields::{FieldSet, NUM_FIELDS};
use rbs_core::grid::Grid2d;
use rbs_core::stencils::{
    stencil_4th_cc, stencil_4th_cs, stencil_4th_sc, stencil_4th_ss, Stencil, StencilPoint,
    NUM_SUBTYPES,
};
use rbs_linalg::CsrMatrix;

use crate::generated::kernels::{
    jacobian_subtypes, residual_boundary, residual_interior, BoundaryPoint, JacobianPoint,
    ResidualPoint, GNUM, JAC_COLS,
};

/// The C driver's residual sign convention (`rescale = -1` from
/// `src/main.c`).
pub const DRIVER_RESCALE: f64 = -1.0;

/// Physical + discretization parameters of one elliptic solve.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemParams {
    /// Azimuthal number `l` (1..=6 in the oracle campaigns).
    pub l: i64,
    /// Scalar-field mass `m`.
    pub m: f64,
    /// Stored frequency parameter (the eigenvalue unknown actually solved
    /// for; the physical frequency is [`ProblemParams::omega`]).
    pub xi: f64,
    /// Regularization coefficient of the lambda equation (`Q1`; the C
    /// kernels hard-code `#define Q1 1.0`).
    pub q1: f64,
    /// Regularization coefficient (`Q2`; `#define Q2 1.0` in C).
    pub q2: f64,
    /// The omega-constraint row (`src/csr_omega_constraint.c`):
    /// `None` pins the frequency (`fixedOmega`, column `w_idx`);
    /// `Some((i, j))` pins the scalar field at integer grid point
    /// `(i, j)` (`fixedPhi`, column `4*dim + IDX(i, j)`), leaving the
    /// frequency free.
    pub fixed_phi: Option<(usize, usize)>,
}

impl ProblemParams {
    /// The physical frequency `w = omega_calc(xi, m)`.
    #[must_use]
    pub fn omega(&self) -> f64 {
        crate::omega::omega_calc(self.xi, self.m)
    }

    /// The omega-chain-rule factor `dw/dxi` used by the Jacobian's omega
    /// column.
    #[must_use]
    pub fn dw_dxi(&self) -> f64 {
        crate::omega::dw_dxi(self.xi, self.m)
    }
}

/// Reusable derivative workspace: the physical derivative grids of the
/// six fields plus the two regularization auxiliaries (`ctx->Dr_u` …
/// `ctx->Dr_u_aux` in the C).
#[derive(Debug, Clone)]
pub struct Workspace {
    /// `Dr_u`: physical radial first derivatives, field-major.
    dr_u: Vec<f64>,
    /// `Dz_u`: physical axial first derivatives, field-major.
    dz_u: Vec<f64>,
    /// `Drr_u`: physical radial second derivatives, field-major.
    drr_u: Vec<f64>,
    /// `Dzz_u`: physical axial second derivatives, field-major.
    dzz_u: Vec<f64>,
    /// `u_aux`: the regularization auxiliaries `u6 = Dr(alpha)/r` and
    /// `u7 = Dr(H)/r` (order + 2 derivative of the log-variables,
    /// rescaled).
    u_aux: Vec<f64>,
    /// `Dr_u_aux`: radial derivatives of the auxiliaries.
    dr_u_aux: Vec<f64>,
    /// Scratch field for the order + 2 log-derivative pass.
    scratch: Vec<f64>,
}

impl Workspace {
    /// Allocate a workspace for `grid`.
    #[must_use]
    pub fn new(grid: &Grid2d) -> Self {
        let dim = grid.dim();
        Self {
            dr_u: vec![0.0; NUM_FIELDS * dim],
            dz_u: vec![0.0; NUM_FIELDS * dim],
            drr_u: vec![0.0; NUM_FIELDS * dim],
            dzz_u: vec![0.0; NUM_FIELDS * dim],
            u_aux: vec![0.0; 2 * dim],
            dr_u_aux: vec![0.0; 2 * dim],
            scratch: vec![0.0; dim],
        }
    }

    /// Recompute all derivative grids for `fields` at discretization
    /// `order` (the auxiliaries at `order + 2`, as in `src/rhs.c`).
    ///
    /// # Panics
    ///
    /// Panics on unsupported orders (see [`rbs_core::diff`]).
    #[allow(clippy::missing_panics_doc)]
    pub fn recompute(&mut self, grid: &Grid2d, fields: &FieldSet, order: u32) {
        let dim = grid.dim();
        for k in 0..NUM_FIELDS {
            diff1r(
                grid,
                &mut self.dr_u[k * dim..(k + 1) * dim],
                fields.field(k),
                Parity::Even,
                order,
            );
        }
        for k in 0..NUM_FIELDS {
            diff1z(
                grid,
                &mut self.dz_u[k * dim..(k + 1) * dim],
                fields.field(k),
                Parity::Even,
                order,
            );
        }
        for k in 0..NUM_FIELDS {
            diff2r(
                grid,
                &mut self.drr_u[k * dim..(k + 1) * dim],
                fields.field(k),
                Parity::Even,
                order,
            );
        }
        for k in 0..NUM_FIELDS {
            diff2z(
                grid,
                &mut self.dzz_u[k * dim..(k + 1) * dim],
                fields.field(k),
                Parity::Even,
                order,
            );
        }

        // Regularization auxiliaries: order + 2 radial derivatives of
        // log(alpha) and log(h), rescaled into Dr(alpha)/r and Dr(H)/r
        // (the REGULARIZATION_AUXILIARIES block of src/rhs.c).  The
        // rescale loop covers the ghost rows too (negative r there);
        // those values are never read, exactly like the C.
        diff1r(
            grid,
            &mut self.scratch,
            fields.field(0),
            Parity::Even,
            order + 2,
        );
        for i in 0..grid.nr_total() {
            let r = grid.r(i);
            for j in 0..grid.nz_total() {
                let idx = grid.idx(i, j);
                // u6 = alpha * (Dr(log alpha) / r).
                self.u_aux[idx] = self.scratch[idx] * fields.at(0, idx).exp() / r;
            }
        }
        diff1r(
            grid,
            &mut self.scratch,
            fields.field(2),
            Parity::Even,
            order + 2,
        );
        for i in 0..grid.nr_total() {
            let r = grid.r(i);
            for j in 0..grid.nz_total() {
                let idx = grid.idx(i, j);
                // u7 = 2H * (Dr(log h) / r).
                self.u_aux[dim + idx] =
                    self.scratch[idx] * 2.0 * (2.0 * fields.at(2, idx)).exp() / r;
            }
        }
        let (out0, out1) = self.dr_u_aux.split_at_mut(dim);
        diff1r(grid, out0, &self.u_aux[..dim], Parity::Even, order + 2);
        diff1r(grid, out1, &self.u_aux[dim..], Parity::Even, order + 2);
    }

    /// Physical `Dr_u` value of field `k` at flat index `idx`.
    #[must_use]
    fn dr(&self, k: usize, idx: usize) -> f64 {
        self.dr_u[k * self.dim() + idx]
    }

    /// Physical `Dz_u` value of field `k` at flat index `idx`.
    #[must_use]
    fn dz(&self, k: usize, idx: usize) -> f64 {
        self.dz_u[k * self.dim() + idx]
    }

    /// Physical `Drr_u` value of field `k` at flat index `idx`.
    #[must_use]
    fn drr(&self, k: usize, idx: usize) -> f64 {
        self.drr_u[k * self.dim() + idx]
    }

    /// Physical `Dzz_u` value of field `k` at flat index `idx`.
    #[must_use]
    fn dzz(&self, k: usize, idx: usize) -> f64 {
        self.dzz_u[k * self.dim() + idx]
    }
}

impl Workspace {
    /// Points per grid function.
    #[must_use]
    fn dim(&self) -> usize {
        self.dr_u.len() / NUM_FIELDS
    }
}

/// Assemble the global residual (the `rhs` port), field-major like the C
/// `f` array with the omega slot last: length `GNUM * dim + 1`,
/// `f[w_idx] = 0`.
#[must_use]
pub fn assemble_residual(
    grid: &Grid2d,
    fields: &FieldSet,
    params: &ProblemParams,
    ws: &mut Workspace,
) -> Vec<f64> {
    let dim = grid.dim();
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    ws.recompute(grid, fields, 4);
    let w = params.omega();
    let l = params.l as f64;
    let mut f = vec![0.0; GNUM * dim + 1];

    // Interior rows: every non-ghost point short of the outer boundary
    // (the ss/cs/sc positions included, like the C sweep).
    for i in g..nr - 1 {
        let r = grid.r(i);
        for j in g..nz - 1 {
            let idx = grid.idx(i, j);
            let p = ResidualPoint {
                u: [
                    fields.at(0, idx),
                    fields.at(1, idx),
                    fields.at(2, idx),
                    fields.at(3, idx),
                    fields.at(4, idx),
                    fields.at(5, idx),
                ],
                dr_u: [
                    ws.dr(0, idx),
                    ws.dr(1, idx),
                    ws.dr(2, idx),
                    ws.dr(3, idx),
                    ws.dr(4, idx),
                    ws.dr(5, idx),
                ],
                dz_u: [
                    ws.dz(0, idx),
                    ws.dz(1, idx),
                    ws.dz(2, idx),
                    ws.dz(3, idx),
                    ws.dz(4, idx),
                    ws.dz(5, idx),
                ],
                drr_u: [
                    ws.drr(0, idx),
                    ws.drr(1, idx),
                    ws.drr(2, idx),
                    ws.drr(3, idx),
                    ws.drr(4, idx),
                    ws.drr(5, idx),
                ],
                dzz_u: [
                    ws.dzz(0, idx),
                    ws.dzz(1, idx),
                    ws.dzz(2, idx),
                    ws.dzz(3, idx),
                    ws.dzz(4, idx),
                    ws.dzz(5, idx),
                ],
                dr_aux: [ws.dr_u_aux[idx], ws.dr_u_aux[dim + idx]],
                dr: grid.dr(),
                dz: grid.dz(),
                r,
                l,
                m: params.m,
                w,
                q1: params.q1,
                q2: params.q2,
                rescale: DRIVER_RESCALE,
            };
            let vals = residual_interior(&p);
            for k in 0..GNUM {
                f[k * dim + idx] = vals[k];
            }
        }
    }

    // Outer-boundary Robin/decay rows: top row, right column, corner.
    for i in g..nr - 1 {
        write_boundary_row(grid, fields, params, ws, &mut f, i, nz - 1);
    }
    for j in g..nz - 1 {
        write_boundary_row(grid, fields, params, ws, &mut f, nr - 1, j);
    }
    write_boundary_row(grid, fields, params, ws, &mut f, nr - 1, nz - 1);

    // Omega constraint row: identically zero (its Jacobian row is the
    // normalization condition).
    f[GNUM * dim] = 0.0;
    f
}

/// Write the six `rhs_bdry` rows at one outer-boundary point.
fn write_boundary_row(
    grid: &Grid2d,
    fields: &FieldSet,
    params: &ProblemParams,
    ws: &Workspace,
    f: &mut [f64],
    i: usize,
    j: usize,
) {
    let dim = grid.dim();
    let idx = grid.idx(i, j);
    let p = BoundaryPoint {
        u: [
            fields.at(0, idx),
            fields.at(1, idx),
            fields.at(2, idx),
            fields.at(3, idx),
            fields.at(4, idx),
            fields.at(5, idx),
        ],
        dr_u: [
            ws.dr(0, idx),
            ws.dr(1, idx),
            ws.dr(2, idx),
            ws.dr(3, idx),
            ws.dr(4, idx),
            ws.dr(5, idx),
        ],
        dz_u: [
            ws.dz(0, idx),
            ws.dz(1, idx),
            ws.dz(2, idx),
            ws.dz(3, idx),
            ws.dz(4, idx),
            ws.dz(5, idx),
        ],
        r: grid.r(i),
        z: grid.z(j),
        dr: grid.dr(),
        dz: grid.dz(),
        l: params.l as f64,
        m: params.m,
        w: params.omega(),
        rescale: DRIVER_RESCALE,
    };
    let vals = residual_boundary(&p);
    for k in 0..GNUM {
        f[k * dim + idx] = vals[k];
    }
}

// ---------------------------------------------------------------------------
// Jacobian: nnz accounting (src/csr.c), emit tables, pattern, values.
// ---------------------------------------------------------------------------

/// The `nnz_jacobian_get_nnzs` counting formulas of `src/csr.c` for the
/// 4th-order slice: nonzeros per grid function (field-major regions),
/// the omega row excluded. The C prints the symmetric cases collapsed
/// (`p_cs * (Nr + Nz)`); the general form is `p_cs * Nr + p_sc * Nz`.
#[must_use]
pub fn nnz_per_field_4th(nr_interior: usize, nz_interior: usize) -> [usize; GNUM] {
    let p_cc = [30usize, 29, 28, 46, 31, 40];
    let p_cs = [31usize, 31, 30, 47, 32, 42];
    let p_sc = [31usize, 31, 30, 47, 32, 44];
    let p_ss = [32usize, 32, 31, 48, 33, 45];
    let p_bound = [9usize, 9, 9, 9, 10, 9];
    let mut nnzs = [0usize; GNUM];
    for (k, nnz) in nnzs.iter_mut().enumerate() {
        *nnz = p_cc[k] * nr_interior * nz_interior
            + p_cs[k] * nr_interior
            + p_sc[k] * nz_interior
            + p_ss[k]
            + 4 * (nr_interior + nz_interior)
            + 24
            + p_bound[k] * (3 + nr_interior + nz_interior);
    }
    nnzs
}

/// Total Jacobian nonzeros including the omega constraint row
/// (`nnz_jacobian` of `src/csr.c`, order-4 branch).
#[must_use]
pub fn nnz_jacobian_4th(nr_interior: usize, nz_interior: usize) -> usize {
    nnz_per_field_4th(nr_interior, nz_interior)
        .iter()
        .sum::<usize>()
        + 1
}

/// Structural nonzeros of the subtype Jacobian rows, extracted from the
/// `jacobian_submatrix_*` assignments of `src/csr_vars.c` (identical for
/// cc/cs/sc/ss; verified mechanically against the C emit lists by
/// `tools/` extraction at port time). Column `5*g + k` is
/// `dF_row/dsubtype_k` of field `g`; the last flag is the omega column
/// `dw/dxi * dF/dw`.
const JAC_NZ: [[bool; JAC_COLS]; GNUM] = [
    [
        true, true, true, true, true, true, true, true, false, false, true, true, true, false,
        false, true, false, false, false, false, true, false, false, false, false, false, false,
        false, false, false, true,
    ], // residual 0
    [
        false, true, true, false, false, true, true, true, true, true, true, true, true, false,
        false, true, false, false, false, false, true, false, false, false, false, false, false,
        false, false, false, true,
    ], // residual 1
    [
        true, true, true, false, false, false, true, true, false, false, true, true, true, true,
        true, true, false, false, false, false, true, false, false, false, false, false, false,
        false, false, false, false,
    ], // residual 2
    [
        true, true, true, false, false, true, true, true, false, false, true, true, true, false,
        false, true, false, false, true, true, true, true, true, false, false, false, false, false,
        false, false, true,
    ], // residual 3
    [
        true, true, true, false, false, true, false, false, false, false, true, true, true, false,
        false, true, false, false, false, false, true, true, true, true, true, true, false, false,
        false, false, true,
    ], // residual 4
    [
        true, true, true, true, false, false, true, true, false, false, true, true, true, true,
        false, false, false, false, false, false, true, true, false, false, false, true, true,
        true, true, true, false,
    ], // residual 5
];

/// One emitted Jacobian entry of an interior-family row: the source
/// field, the stencil offset, and the merged subtype weights
/// (the `_emit_csr` rule: the entry exists iff some subtype weight and
/// its structural partial are both nonzero).
#[derive(Debug, Clone, Copy)]
struct EmitEntry {
    /// Source field of the column.
    g: usize,
    /// Radial offset in cells.
    dr_off: i32,
    /// Axial offset in cells.
    dz_off: i32,
    /// Merged step-scaled subtype weights `[u, dRu, dZu, dRRu, dZZu]`.
    w: [f64; NUM_SUBTYPES],
}

/// The interior stencil variants (`csr_grid_fill_4th` dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// Centered r, centered z (bulk interior).
    Cc,
    /// Centered r, one-sided z (last row before the top boundary).
    Cs,
    /// One-sided r, centered z (last column before the right boundary).
    Sc,
    /// One-sided both (the point before the outer corner).
    Ss,
}

impl KernelKind {
    /// The rbs-core stencil table of this kernel.
    #[must_use]
    pub fn stencil(self) -> Stencil {
        match self {
            Self::Cc => stencil_4th_cc(),
            Self::Cs => stencil_4th_cs(),
            Self::Sc => stencil_4th_sc(),
            Self::Ss => stencil_4th_ss(),
        }
    }

    /// The emit list of one residual row: the C `ja` sequence.
    #[must_use]
    fn emit_list(self, row: usize) -> Vec<EmitEntry> {
        let stencil = self.stencil();
        let mut list = Vec::new();
        for g in 0..GNUM {
            for pt in stencil.points {
                if pt_has_entry(row, g, pt) {
                    list.push(EmitEntry {
                        g,
                        dr_off: pt.dr_off,
                        dz_off: pt.dz_off,
                        w: pt.coeffs,
                    });
                }
            }
        }
        list
    }
}

/// The `_emit_csr` membership rule: an entry exists at a stencil point
/// iff some subtype has a nonzero weight there AND a structural nonzero
/// partial.
fn pt_has_entry(row: usize, g: usize, pt: &StencilPoint) -> bool {
    (0..NUM_SUBTYPES).any(|k| JAC_NZ[row][g * NUM_SUBTYPES + k] && pt.coeffs[k] != 0.0)
}

/// Row classification of the CSR walk (`csr_grid_fill_4th`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RowKind {
    /// `csr_symmetry.c`: `u(i, j) - sign * u(mirror) = 0`.
    Sym {
        /// Parity sign of the reflection.
        sign: f64,
        /// Flat (field-local) index of the mirror point.
        mirror: usize,
    },
    /// A `jacobian_*_variable_omega_*` stencil row.
    Stencil {
        /// Which stencil variant applies at this position.
        kernel: KernelKind,
    },
    /// `z_robin_4th_order` / `z_decay_4th_order` (centered r).
    ZBoundary {
        /// Robin power `n` (unused for the decay row of psi).
        n: f64,
        /// Semi-onesided r derivative (the `z_so_*` corner-adjacent row).
        one_sided: bool,
    },
    /// `r_robin_4th_order` / `r_decay_4th_order` (centered z).
    RBoundary {
        /// Robin power `n`.
        n: f64,
        /// Semi-onesided z derivative (the `r_so_*` row).
        one_sided: bool,
    },
    /// `corner_robin_4th_order` / `corner_decay_4th_order`.
    CornerBoundary {
        /// Robin power `n`.
        n: f64,
    },
    /// `csr_omega_constraint.c`: the closing normalization row.
    Omega,
}

/// The Robin power `n` of each field's outer boundary condition
/// (`bound_order` dispatch of `src/csr.c`: fields 0..3 and 5 are
/// `ROBIN_TYPE` rows with `n = 1, 3, 1, 1, 4`; field 4 is the
/// exponential-decay row and ignores `n`).
#[must_use]
fn robin_n(g: usize) -> f64 {
    match g {
        0 | 2 | 3 => 1.0,
        1 => 3.0,
        5 => 4.0,
        _ => 0.0, // field 4: the decay row; `n` is unused.
    }
}

/// Is field `g`'s boundary row the exponential-decay one?
const fn is_decay(g: usize) -> bool {
    g == 4
}

/// One row of the pattern walk.
#[derive(Debug, Clone, Copy)]
struct RowPlan {
    /// Global row index (`g * dim + IDX(i, j)`).
    row: usize,
    /// The row's classification.
    kind: RowKind,
}

/// The per-(kernel, residual row) emit lists, built once and shared by
/// the pattern walk and the value fill.
#[derive(Debug, Clone)]
struct KernelEmits {
    /// `[kernel][row]` entry lists (the C `ja` sequences).
    lists: [Vec<Vec<EmitEntry>>; 4],
}

impl KernelEmits {
    /// Build all emit lists from the stencil tables + [`JAC_NZ`].
    #[must_use]
    fn new() -> Self {
        let kernels = [
            KernelKind::Cc,
            KernelKind::Cs,
            KernelKind::Sc,
            KernelKind::Ss,
        ];
        let lists = kernels
            .iter()
            .map(|k| (0..GNUM).map(|row| k.emit_list(row)).collect())
            .collect::<Vec<_>>()
            .try_into()
            .expect("four kernels");
        Self { lists }
    }

    /// The emit list of `kernel` for residual `row`.
    #[must_use]
    fn get(&self, kernel: KernelKind, row: usize) -> &[EmitEntry] {
        let k = match kernel {
            KernelKind::Cc => 0,
            KernelKind::Cs => 1,
            KernelKind::Sc => 2,
            KernelKind::Ss => 3,
        };
        &self.lists[k][row]
    }
}

/// The fixed CSR structure of the 6-field Jacobian plus the omega row:
/// the Rust mirror of `csr_allocate` + the `ia`/`ja` fills of
/// `csr_gen_jacobian`. Values are refreshed per iterate with
/// [`assemble_jacobian_values`].
#[derive(Debug, Clone)]
pub struct JacobianPattern {
    /// Row count (`GNUM * dim + 1`).
    pub nrows: usize,
    /// Column count (square).
    pub ncols: usize,
    /// CSR row pointers.
    pub row_ptrs: Vec<usize>,
    /// CSR column indices, C emit order (ascending within each row).
    pub col_inds: Vec<usize>,
    /// Row classification in walk order.
    plan: Vec<RowPlan>,
    /// Per-(kernel, row) emit lists.
    emits: KernelEmits,
}

impl JacobianPattern {
    /// Build the pattern for `grid` and `params` (the omega constraint
    /// column comes from `params.fixed_phi`).
    ///
    /// # Panics
    ///
    /// Panics if the fixed-phi anchor lies outside the interior.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn build(grid: &Grid2d, params: &ProblemParams) -> Self {
        let dim = grid.dim();
        let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
        let w_idx = GNUM * dim;
        let nrows = w_idx + 1;
        let emits = KernelEmits::new();
        let mut row_ptrs = vec![0usize; nrows + 1];
        let mut col_inds = Vec::with_capacity(nnz_jacobian_4th(nr - 2 * g, nz - 2 * g));
        let mut plan = Vec::with_capacity(nrows);
        let mut row_count = vec![0usize; nrows];

        // All six fields are EVEN in r and z (`r_sym`/`z_sym` of
        // `src/csr.c`), so every symmetry row is `u - u_mirror`.
        for gnum in 0..GNUM {
            let base = gnum * dim;
            for i in 0..nr {
                for j in 0..nz {
                    let idx = grid.idx(i, j);
                    let row = base + idx;
                    let kind = classify(grid, gnum, i, j);
                    let cols = row_columns(grid, &emits, kind, gnum, i, j, dim, w_idx);
                    plan.push(RowPlan { row, kind });
                    row_count[row] = cols.len();
                    col_inds.extend_from_slice(&cols);
                }
            }
        }

        // FINALLY FILL OMEGA EQUATION OR u5(1,1) CONSTRAINT
        // (src/csr_omega_constraint.c).
        let omega_col = match params.fixed_phi {
            None => w_idx,
            Some((fi, fj)) => {
                assert!(
                    fi >= g && fi < nr - g && fj >= g && fj < nz - g,
                    "fixed-phi anchor outside the interior grid"
                );
                4 * dim + grid.idx(fi, fj)
            }
        };
        plan.push(RowPlan {
            row: w_idx,
            kind: RowKind::Omega,
        });
        row_count[w_idx] = 1;
        col_inds.push(omega_col);

        // Prefix-sum the row pointers (the walk is field-major and
        // row-major within each field, so rows were visited in strictly
        // increasing order — the C's streaming per-field offsets land on
        // the same layout).
        for r in 0..nrows {
            row_ptrs[r + 1] = row_ptrs[r] + row_count[r];
        }
        debug_assert_eq!(row_ptrs[nrows], col_inds.len());

        Self {
            nrows,
            ncols: nrows,
            row_ptrs,
            col_inds,
            plan,
            emits,
        }
    }

    /// Stored entries (the C `nnz`).
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.col_inds.len()
    }
}

/// The column list of one row, in the C emit order.
fn row_columns(
    grid: &Grid2d,
    emits: &KernelEmits,
    kind: RowKind,
    gnum: usize,
    i: usize,
    j: usize,
    dim: usize,
    w_idx: usize,
) -> Vec<usize> {
    let base = gnum * dim;
    match kind {
        RowKind::Sym { mirror, .. } => vec![base + grid.idx(i, j), base + mirror],
        RowKind::Stencil { kernel } => {
            let mut cols: Vec<usize> = emits
                .get(kernel, gnum)
                .iter()
                .map(|e| e.g * dim + grid.idx(ias(e.dr_off, i), ias(e.dz_off, j)))
                .collect();
            if JAC_NZ[gnum][JAC_COLS - 1] {
                cols.push(w_idx);
            }
            cols
        }
        RowKind::ZBoundary { one_sided, .. } => {
            let mut cols = Vec::new();
            boundary_cols(
                grid,
                gnum,
                i,
                j,
                dim,
                BoundaryAxis::Z,
                one_sided,
                w_idx,
                &mut cols,
            );
            cols
        }
        RowKind::RBoundary { one_sided, .. } => {
            let mut cols = Vec::new();
            boundary_cols(
                grid,
                gnum,
                i,
                j,
                dim,
                BoundaryAxis::R,
                one_sided,
                w_idx,
                &mut cols,
            );
            cols
        }
        RowKind::CornerBoundary { .. } => {
            let mut cols = Vec::new();
            boundary_cols(
                grid,
                gnum,
                i,
                j,
                dim,
                BoundaryAxis::Corner,
                false,
                w_idx,
                &mut cols,
            );
            cols
        }
        RowKind::Omega => vec![w_idx],
    }
}

/// The boundary rows' column offsets (`csr_robin.c` / `csr_exp_decay.c`).
#[derive(Debug, Clone, Copy)]
enum BoundaryAxis {
    /// Rows on the top boundary (z outer).
    Z,
    /// Rows on the right boundary (r outer).
    R,
    /// The outer corner.
    Corner,
}

/// Column list of one boundary row (excluding the value computation).
fn boundary_cols(
    grid: &Grid2d,
    gnum: usize,
    i: usize,
    j: usize,
    dim: usize,
    axis: BoundaryAxis,
    one_sided: bool,
    w_idx: usize,
    cols: &mut Vec<usize>,
) {
    let base = gnum * dim;
    let mut pt = |di: i32, dj: i32| cols.push(base + grid.idx(ias(di, i), ias(dj, j)));
    match axis {
        BoundaryAxis::Z => {
            if one_sided {
                for d in [-3, -2, -1] {
                    pt(d, 0);
                }
                for d in [-4, -3, -2, -1, 0] {
                    pt(0, d);
                }
                pt(1, 0);
            } else {
                for d in [-2, -1] {
                    pt(d, 0);
                }
                for d in [-4, -3, -2, -1, 0] {
                    pt(0, d);
                }
                pt(1, 0);
                pt(2, 0);
            }
        }
        BoundaryAxis::R => {
            for d in [-4, -3, -2, -1] {
                pt(d, 0);
            }
            if one_sided {
                for d in [-3, -2, -1, 0] {
                    pt(0, d);
                }
                pt(0, 1);
            } else {
                for d in [-2, -1, 0, 1, 2] {
                    pt(0, d);
                }
            }
        }
        BoundaryAxis::Corner => {
            for d in [-4, -3, -2, -1] {
                pt(d, 0);
            }
            for d in [-4, -3, -2, -1, 0] {
                pt(0, d);
            }
        }
    }
    if is_decay(gnum) {
        cols.push(w_idx);
    }
}

/// Row classification (`csr_grid_fill_4th`'s dispatch).
fn classify(grid: &Grid2d, gnum: usize, i: usize, j: usize) -> RowKind {
    let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
    if i < g && j < g {
        // Corner symmetry.
        RowKind::Sym {
            sign: 1.0,
            mirror: grid.idx(2 * g - 1 - i, 2 * g - 1 - j),
        }
    } else if i < g {
        // Radial symmetry.
        RowKind::Sym {
            sign: 1.0,
            mirror: grid.idx(2 * g - 1 - i, j),
        }
    } else if j < g {
        // Equatorial symmetry.
        RowKind::Sym {
            sign: 1.0,
            mirror: grid.idx(i, 2 * g - 1 - j),
        }
    } else if i < nr - 2 && j < nz - 2 {
        RowKind::Stencil {
            kernel: KernelKind::Cc,
        }
    } else if i < nr - 2 && j == nz - 2 {
        RowKind::Stencil {
            kernel: KernelKind::Cs,
        }
    } else if i < nr - 2 && j == nz - 1 {
        RowKind::ZBoundary {
            n: robin_n(gnum),
            one_sided: false,
        }
    } else if i == nr - 2 && j < nz - 2 {
        RowKind::Stencil {
            kernel: KernelKind::Sc,
        }
    } else if i == nr - 2 && j == nz - 2 {
        RowKind::Stencil {
            kernel: KernelKind::Ss,
        }
    } else if i == nr - 2 && j == nz - 1 {
        RowKind::ZBoundary {
            n: robin_n(gnum),
            one_sided: true,
        }
    } else if i == nr - 1 && j < nz - 2 {
        RowKind::RBoundary {
            n: robin_n(gnum),
            one_sided: false,
        }
    } else if i == nr - 1 && j == nz - 2 {
        RowKind::RBoundary {
            n: robin_n(gnum),
            one_sided: true,
        }
    } else if i == nr - 1 && j == nz - 1 {
        RowKind::CornerBoundary { n: robin_n(gnum) }
    } else {
        unreachable!("row classification exhausted at ({i}, {j})")
    }
}

/// Signed stencil-offset add.
fn ias(off: i32, base: usize) -> usize {
    debug_assert!(
        (base as i64 + i64::from(off)) >= 0,
        "stencil offset {off} underflows at {base}"
    );
    (base as i64 + i64::from(off)) as usize
}

/// Refresh the Jacobian values for `fields` into `vals` (parallel to
/// [`JacobianPattern::col_inds`]), the `csr_gen_jacobian` port. The
/// pattern is fixed; only values change between iterates.
///
/// # Panics
///
/// Panics if `vals` does not have the pattern's nnz length.
#[allow(clippy::missing_panics_doc)]
pub fn assemble_jacobian_values(
    pattern: &JacobianPattern,
    grid: &Grid2d,
    fields: &FieldSet,
    params: &ProblemParams,
    ws: &mut Workspace,
    vals: &mut [f64],
) {
    assert_eq!(vals.len(), pattern.col_inds.len(), "vals length");
    ws.recompute(grid, fields, 4);
    let dim = grid.dim();
    let nz = grid.nz_total();
    let (dr, dz) = (grid.dr(), grid.dz());
    let mut cursor = 0usize;
    for plan in &pattern.plan {
        let row = plan.row;
        let (gnum, idx) = (row / dim, row % dim);
        let (i, j) = (idx / nz, idx % nz);
        match plan.kind {
            RowKind::Sym { .. } => {
                // All fields are even: u(i, j) - u(mirror) = 0.
                vals[cursor] = 1.0;
                vals[cursor + 1] = -1.0;
                cursor += 2;
            }
            RowKind::Stencil { kernel } => {
                let jp = JacobianPoint {
                    u: [
                        fields.at(0, idx),
                        fields.at(1, idx),
                        fields.at(2, idx),
                        fields.at(3, idx),
                        fields.at(4, idx),
                        fields.at(5, idx),
                    ],
                    d_ru: step_scale(dr, ws, 0, idx),
                    d_zu: step_scale(dz, ws, 1, idx),
                    d_rru: step_scale(dr * dr, ws, 2, idx),
                    d_zzu: step_scale(dz * dz, ws, 3, idx),
                    ri: grid.ri(i),
                    dr,
                    dz,
                    l: params.l as f64,
                    m: params.m,
                    w: params.omega(),
                    dw_dxi: params.dw_dxi(),
                    q1: params.q1,
                    q2: params.q2,
                };
                let jac = jacobian_subtypes(&jp);
                for e in pattern.emits.get(kernel, gnum) {
                    let mut v = 0.0;
                    for k in 0..NUM_SUBTYPES {
                        if e.w[k] != 0.0 {
                            v += e.w[k] * jac[gnum][e.g * NUM_SUBTYPES + k];
                        }
                    }
                    vals[cursor] = v;
                    cursor += 1;
                }
                if JAC_NZ[gnum][JAC_COLS - 1] {
                    vals[cursor] = jac[gnum][JAC_COLS - 1];
                    cursor += 1;
                }
            }
            RowKind::ZBoundary { n, one_sided } => {
                boundary_vals(
                    grid,
                    fields,
                    params,
                    gnum,
                    i,
                    j,
                    BoundaryAxis::Z,
                    one_sided,
                    n,
                    vals,
                    &mut cursor,
                );
            }
            RowKind::RBoundary { n, one_sided } => {
                boundary_vals(
                    grid,
                    fields,
                    params,
                    gnum,
                    i,
                    j,
                    BoundaryAxis::R,
                    one_sided,
                    n,
                    vals,
                    &mut cursor,
                );
            }
            RowKind::CornerBoundary { n } => {
                boundary_vals(
                    grid,
                    fields,
                    params,
                    gnum,
                    i,
                    j,
                    BoundaryAxis::Corner,
                    false,
                    n,
                    vals,
                    &mut cursor,
                );
            }
            RowKind::Omega => {
                // a[nnz] = 1.0 at the constrained column.
                vals[cursor] = 1.0;
                cursor += 1;
            }
        }
    }
    debug_assert_eq!(cursor, vals.len());
}

/// Step-scale one subtype column of the workspace at `idx`.
fn step_scale(h: f64, ws: &Workspace, which: usize, idx: usize) -> [f64; GNUM] {
    let src = match which {
        0 => &ws.dr_u,
        1 => &ws.dz_u,
        2 => &ws.drr_u,
        _ => &ws.dzz_u,
    };
    let dim = ws.dim();
    let mut out = [0.0; GNUM];
    for (k, v) in out.iter_mut().enumerate() {
        *v = h * src[k * dim + idx];
    }
    out
}

/// Assemble the full Jacobian (pattern + values) in one call — the
/// `csr_gen_jacobian` convenience form used by tests and one-shot
/// drivers; the Newton loop keeps the pattern across iterations.
#[must_use]
pub fn assemble_jacobian(
    grid: &Grid2d,
    fields: &FieldSet,
    params: &ProblemParams,
    ws: &mut Workspace,
) -> CsrMatrix {
    let pattern = JacobianPattern::build(grid, params);
    let mut vals = vec![0.0; pattern.col_inds.len()];
    assemble_jacobian_values(&pattern, grid, fields, params, ws, &mut vals);
    CsrMatrix {
        nrows: pattern.nrows,
        ncols: pattern.ncols,
        row_ptrs: pattern.row_ptrs.clone(),
        col_inds: pattern.col_inds.clone(),
        vals,
    }
}

// Fourth-order Fornberg weights of the boundary rows (`csr_robin.c`):
// centered D1 (offsets -2..2), one-sided S1 (offsets -4..0) and
// semi-onesided SO1 (offsets -3..1).
const D1_4: [f64; 5] = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];
const S1_4: [f64; 5] = [0.25, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0];
const SO1_4: [f64; 5] = [-1.0 / 12.0, 0.5, -1.5, 5.0 / 6.0, 0.25];

/// Boundary-row values, written in the exact column order of
/// [`boundary_cols`] (`csr_robin.c` / `csr_exp_decay.c`).
#[allow(clippy::too_many_arguments)]
fn boundary_vals(
    grid: &Grid2d,
    fields: &FieldSet,
    params: &ProblemParams,
    gnum: usize,
    i: usize,
    j: usize,
    axis: BoundaryAxis,
    one_sided: bool,
    n: f64,
    vals: &mut [f64],
    cursor: &mut usize,
) {
    let dim = grid.dim();
    let (dr, dz) = (grid.dr(), grid.dz());
    // True boundary coordinates in cell units (`ri = i - 1.5` etc.,
    // identical to `i + 0.5 - ghost` at ghost = 2).
    let ri = grid.ri(i);
    let zi = f64::from(u32::try_from(j).expect("j fits u32")) + 0.5
        - f64::from(u32::try_from(grid.ghost()).expect("ghost fits u32"));
    let rr2 = (ri * ri) * dr * dr + (zi * zi) * dz * dz;
    let rr = rr2.sqrt();
    let scale = dr * dz / rr2;

    let psi = fields.field(4);
    let psi_at = |di: i32, dj: i32| psi[grid.idx(ias(di, i), ias(dj, j))];

    let w = params.omega();
    let m2 = params.m * params.m;
    let chi = (m2 - w * w).sqrt();
    let l = params.l as f64;

    // Per axis: (r-direction weights, r offsets) and (z weights, z
    // offsets); the center entry merges the diagonal weight with the
    // Robin/decay term.
    let (rw, roff): ([f64; 5], i32) = match axis {
        BoundaryAxis::Z => {
            if one_sided {
                (SO1_4, -3)
            } else {
                (D1_4, -2)
            }
        }
        _ => (S1_4, -4),
    };
    let (zw, zoff): ([f64; 5], i32) = match axis {
        BoundaryAxis::R => {
            if one_sided {
                (SO1_4, -3)
            } else {
                (D1_4, -2)
            }
        }
        _ => (S1_4, -4),
    };

    // Directional derivative samples of psi (decay rows only).
    let dr_psi = if is_decay(gnum) {
        (0..5)
            .map(|k| rw[k] * psi_at(roff + i32::try_from(k).expect("k fits i32"), 0))
            .sum::<f64>()
    } else {
        0.0
    };
    let dz_psi = if is_decay(gnum) {
        (0..5)
            .map(|k| zw[k] * psi_at(0, zoff + i32::try_from(k).expect("k fits i32")))
            .sum::<f64>()
    } else {
        0.0
    };

    // The center coefficient: Robin `n` or the decay tail
    // `rr * chi + l + 1`.
    let center = if is_decay(gnum) {
        rr * chi + l + 1.0
    } else {
        n
    };
    // The one-sided/centered structure of the C center entries.
    let center_extra = match (axis, one_sided) {
        (BoundaryAxis::Z, false) => S1_4[4] * zi,
        (BoundaryAxis::Z, true) => S1_4[4] * zi + SO1_4[3] * ri,
        (BoundaryAxis::R, false) => S1_4[4] * ri,
        (BoundaryAxis::R, true) => S1_4[4] * ri + SO1_4[3] * zi,
        (BoundaryAxis::Corner, _) => S1_4[4] * (ri + zi),
    };

    // Emit in the exact column order of boundary_cols: r-negative
    // standalone points, the z block (its offset-0 entry is the row's
    // own point, carrying the merged Robin/decay center term), then the
    // r-positive standalone points. The offset-0 weight of the
    // one-sided r rule merges into the center (`center_extra`), exactly
    // like the C center entries.
    let put = |v: f64, vals: &mut [f64], cursor: &mut usize| {
        vals[*cursor] = v;
        *cursor += 1;
    };
    let ki = |k: usize| i32::try_from(k).expect("k fits i32");
    match axis {
        BoundaryAxis::Z => {
            for k in 0..5 {
                if roff + ki(k) < 0 && rw[k] != 0.0 {
                    put(rw[k] * ri * scale, vals, cursor);
                }
            }
            for k in 0..5 {
                let off = zoff + ki(k);
                if off < 0 {
                    put(zw[k] * zi * scale, vals, cursor);
                } else if off == 0 {
                    put((center + center_extra) * scale, vals, cursor);
                }
            }
            for k in 0..5 {
                if roff + ki(k) > 0 && rw[k] != 0.0 {
                    put(rw[k] * ri * scale, vals, cursor);
                }
            }
        }
        BoundaryAxis::R | BoundaryAxis::Corner => {
            for k in 0..5 {
                if roff + ki(k) < 0 && rw[k] != 0.0 {
                    put(rw[k] * ri * scale, vals, cursor);
                }
            }
            for k in 0..5 {
                let off = zoff + ki(k);
                if off < 0 && zw[k] != 0.0 {
                    put(zw[k] * zi * scale, vals, cursor);
                } else if off == 0 {
                    put((center + center_extra) * scale, vals, cursor);
                } else if zw[k] != 0.0 {
                    put(zw[k] * zi * scale, vals, cursor);
                }
            }
        }
    }
    if is_decay(gnum) {
        // dw/dxi * (rr * f + rr * psi) * (-w / chi) * scale at w_idx.
        let f = ri * dr_psi + zi * dz_psi + (rr * chi + l + 1.0) * psi_at(0, 0);
        put(
            params.dw_dxi() * (rr * f + rr * psi_at(0, 0)) * (-w / chi) * scale,
            vals,
            cursor,
        );
    }
    let _ = dim;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial::{initial_guess, InitialGuess, SolveState};

    fn smoke_params() -> ProblemParams {
        ProblemParams {
            l: 1,
            m: 1.0,
            xi: crate::omega::inverse_omega_calc(0.95, 1.0),
            q1: 1.0,
            q2: 1.0,
            fixed_phi: None,
        }
    }

    fn smoke_state(grid: &Grid2d) -> SolveState {
        initial_guess(grid, &InitialGuess::l1_smoke())
    }

    /// The pattern walk must reproduce the C's `nnz_jacobian` counting
    /// formulas exactly (`src/csr.c`), including the asymmetric lambda
    /// row (`p_cs != p_sc`).
    #[test]
    fn pattern_counts_match_the_c_formula() {
        for n in [4usize, 6, 8, 16, 31, 64] {
            let grid = Grid2d::new(n + 4, n + 4, 2, 0.25, 0.25);
            let pattern = JacobianPattern::build(&grid, &smoke_params());
            assert_eq!(
                pattern.nnz(),
                nnz_jacobian_4th(n, n),
                "nnz mismatch at N={n}"
            );
        }
    }

    /// The N=64 total is the oracle's logged value: the C prints
    /// `Non-zeros = 873619` for the smoke spec.
    #[test]
    fn oracle_n64_nnz_matches() {
        assert_eq!(nnz_jacobian_4th(64, 64), 873_619);
    }

    /// The assembled pattern satisfies the CSR contract (sorted rows,
    /// in-range, ascending columns).
    #[test]
    fn pattern_satisfies_the_csr_contract() {
        let grid = Grid2d::new(10, 10, 2, 0.25, 0.25);
        let pattern = JacobianPattern::build(&grid, &smoke_params());
        let mut vals = vec![0.0; pattern.col_inds.len()];
        let mut ws = Workspace::new(&grid);
        let state = smoke_state(&grid);
        assemble_jacobian_values(
            &pattern,
            &grid,
            &state.fields,
            &smoke_params(),
            &mut ws,
            &mut vals,
        );
        let csr = CsrMatrix {
            nrows: pattern.nrows,
            ncols: pattern.ncols,
            row_ptrs: pattern.row_ptrs.clone(),
            col_inds: pattern.col_inds.clone(),
            vals,
        };
        assert!(csr.validate().is_ok(), "CSR contract violated");
    }

    /// On parity-symmetric states (what the Newton iterates always are),
    /// the assembled Jacobian is the directional derivative of the
    /// assembled residual along symmetric perturbations: the gold
    /// standard cross-check of the whole assembly chain without the
    /// oracle binary.
    #[test]
    fn jacobian_is_the_residual_derivative_on_symmetric_states() {
        check_jacobian_fd_at_rich_state(false);
    }

    /// The same cross-check around a genuinely non-flat state: at the
    /// pure ansatz most quadratic Jacobian terms vanish identically
    /// (the metric fields are zero), so the flat-state check alone
    /// would not exercise the nonlinear sector that the real iterates
    /// live in.
    #[test]
    fn jacobian_is_the_residual_derivative_at_a_nonflat_state() {
        check_jacobian_fd_at_rich_state(true);
    }

    fn check_jacobian_fd_at_rich_state(rich: bool) {
        let grid = Grid2d::new(12, 12, 2, 0.25, 0.25);
        let params = smoke_params();
        let mut state = smoke_state(&grid);
        if rich {
            // Smooth, parity-even offsets on every field: nonzero
            // derivatives, nonzero cross terms.
            let dim = grid.dim();
            let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
            let wr = std::f64::consts::FRAC_PI_4;
            let amps = [0.05, 0.03, 0.04, 0.06, 0.02, 0.01];
            for (k, &amp) in amps.iter().enumerate() {
                for i in g..nr {
                    let r = grid.r(i);
                    for j in g..nz {
                        let z = grid.z(j);
                        let idx = grid.idx(i, j);
                        let v = amp * (wr * r).cos() * (wr * z).cos();
                        state.fields.set(k, idx, state.fields.at(k, idx) + v);
                    }
                }
                for i in 0..g {
                    for j in 0..nz {
                        let m = grid.idx(2 * g - 1 - i, j);
                        state.fields.set(k, grid.idx(i, j), state.fields.at(k, m));
                    }
                }
                for i in 0..nr {
                    for j in 0..g {
                        let m = grid.idx(i, 2 * g - 1 - j);
                        state.fields.set(k, grid.idx(i, j), state.fields.at(k, m));
                    }
                }
            }
            let _ = dim;
        }
        let mut ws = Workspace::new(&grid);
        let mut f0 = assemble_residual(&grid, &state.fields, &params, &mut ws);
        let jac = assemble_jacobian(&grid, &state.fields, &params, &mut ws);

        // A smooth, parity-even perturbation: random per-field amplitudes
        // times low-frequency cosines.  Smoothness matters: the lambda
        // equation's regularization chain enters the Jacobian folded at
        // the PDE (4th) order while the residual evaluates it at
        // order + 2 (src/rhs.c REGULARIZATION_AUXILIARIES), so J is the
        // exact directional derivative only up to that O(h^4) design
        // gap — white-noise perturbations would show it at O(1).
        let dim = grid.dim();
        let (nr, nz, g) = (grid.nr_total(), grid.nz_total(), grid.ghost());
        let mut e = vec![0.0; GNUM * dim + 1];
        let mut seed = 0x1234_5678_u64;
        let mut next = || {
            // xorshift64*: deterministic pseudo-randoms in [0, 1).
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let wr = std::f64::consts::FRAC_PI_4;
        let wz = std::f64::consts::FRAC_PI_4;
        for k in 0..GNUM {
            let amp = 1e-3 * next();
            for i in g..nr {
                let r = grid.r(i);
                for j in g..nz {
                    let z = grid.z(j);
                    e[k * dim + grid.idx(i, j)] = amp * (wr * r).cos() * (wz * z).cos();
                }
            }
            // Mirror into ghosts: axis, equator (corner follows).
            for i in 0..g {
                for j in 0..nz {
                    e[k * dim + grid.idx(i, j)] = e[k * dim + grid.idx(2 * g - 1 - i, j)];
                }
            }
            for i in 0..nr {
                for j in 0..g {
                    e[k * dim + grid.idx(i, j)] = e[k * dim + grid.idx(i, 2 * g - 1 - j)];
                }
            }
        }

        // The assembled Jacobian is the derivative of the rescale-free
        // residual; the assembled residual carries `rescale = -1`, so
        // the comparison flips the Jacobian's product.
        let jac_e = jac.matvec(&e);
        let neg_jac_e: Vec<f64> = jac_e.iter().map(|v| -v).collect();
        let eps = 1e-6;
        let mut fields_plus = state.fields.clone();
        let mut fields_minus = state.fields.clone();
        for idx in 0..GNUM * dim {
            fields_plus.as_mut_slice()[idx] += eps * e[idx];
            fields_minus.as_mut_slice()[idx] -= eps * e[idx];
        }
        let ws = &mut Workspace::new(&grid);
        let f_plus = assemble_residual(&grid, &fields_plus, &params, ws);
        let f_minus = assemble_residual(&grid, &fields_minus, &params, ws);

        // Fields 0..=4: the assembled Jacobian must be the central
        // difference to FD noise.  The lambda rows carry the
        // regularization-chain order gap; they agree to the O(h^4)
        // slack documented above.
        let mut worst_plain = 0.0_f64;
        let mut worst_plain_row = 0usize;
        let mut worst_lambda = 0.0_f64;
        let mut worst_lambda_row = 0usize;
        for row in 0..GNUM * dim {
            let fd = (f_plus[row] - f_minus[row]) / (2.0 * eps);
            let dev = (neg_jac_e[row] - fd).abs() / (fd.abs() + neg_jac_e[row].abs() + 1e-12);
            if row < 5 * dim {
                if dev > worst_plain {
                    worst_plain = dev;
                    worst_plain_row = row;
                }
            } else if dev > worst_lambda {
                worst_lambda = dev;
                worst_lambda_row = row;
            }
        }
        if std::env::var("RBS_FD_DEBUG").is_ok() {
            let (r, idx) = (worst_plain_row / dim, worst_plain_row % dim);
            println!(
                "plain worst {worst_plain:.3e} at row {worst_plain_row} (field {r}, i={}, j={})",
                idx / nz,
                idx % nz
            );
            println!(
                "  jac_e={:.6e} fd={:.6e}",
                neg_jac_e[worst_plain_row],
                (f_plus[worst_plain_row] - f_minus[worst_plain_row]) / (2.0 * eps)
            );
            let (r, idx) = (worst_lambda_row / dim, worst_lambda_row % dim);
            println!(
                "lambda worst {worst_lambda:.3e} at field {r}, i={}, j={}",
                idx / nz,
                idx % nz
            );
        }
        // Tolerances: at the flat ansatz the central difference runs at
        // FD noise (1e-6); at the rich state the residual is two
        // decades larger and the FD rounding floor rises to ~1e-4 — a
        // real Jacobian bug shows up at O(1), not at either floor.
        let (plain_tol, lambda_tol) = if rich { (5e-4, 1e-2) } else { (1e-6, 1e-2) };
        assert!(
            worst_plain < plain_tol,
            "plain rows: worst relative deviation {worst_plain} at row {worst_plain_row}"
        );
        assert!(
            worst_lambda < lambda_tol,
            "lambda rows: worst relative deviation {worst_lambda}"
        );
        let _ = &mut f0;
    }
}
