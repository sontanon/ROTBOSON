//! Field-blocked Newton–Krylov backend (SAN-67) — the N1 production
//! candidate between the faer LU scaffolding and geometric multigrid.
//!
//! # Design
//!
//! The 6-field ROTBOSON Jacobian is field-major: field `g` occupies
//! rows/columns `[g·dim, (g+1)·dim)`, with the frequency eigenvalue in
//! the trailing slot (`6·dim + 1` in total). The cross-field couplings
//! are same-node only and O(1); each field's diagonal block is a
//! ~dim-unknown elliptic system. The backend:
//!
//! * [`LinearSolver::factor`] extracts each field's diagonal block,
//!   row-equilibrates it (one-sided row-max scaling — free accuracy
//!   insurance against the C's `dr·dz/rr²`-scaled boundary rows, which
//!   lose ~1/N² of magnitude under grid refinement and are exactly
//!   what defeats faer's unscaled partial pivoting on the *full*
//!   matrix; see `ng/spikes/faer-lu/REPORT.md`), and factors it with
//!   faer — LDLT (faer's simplicial sparse LDLT, natural order) when
//!   the block is symmetric, sparse LU otherwise. Every block of every
//!   real Jacobian measured is nonsymmetric (the first-derivative
//!   advection terms), so LU is the live path and the LDLT branch is
//!   exercised by unit tests;
//! * [`LinearSolver::solve_into`] runs restarted FGMRES
//!   (right-preconditioned, flexible storage) against the matrix kept
//!   from the last `factor`: the block-diagonal inverse is the
//!   preconditioner `M`, the full CSR matrix is the operator `A`, and
//!   the Krylov loop cleans up the ignored same-node cross-field
//!   couplings and the eigenvalue row/column.
//!
//! # Stale-block reuse (the C campaign's stale-LU CGS analogue)
//!
//! The C line factors once and reuses the LU as a CGS preconditioner
//! for subsequent (slowly drifting) Newton systems
//! (`iparm[3]=61`, `docs/profiling.md` §3). The trait's
//! `factor`-then-`solve_into` flow refactors every call, so the stale
//! pattern lives behind [`BlockFieldKrylov::solve_stale`]: the operator
//! is the caller's *current* Jacobian, the preconditioner is the block
//! factorization kept from the last [`LinearSolver::factor`]. FGMRES
//! still minimizes the true residual of the current system, so stale
//! blocks cost iterations, not accuracy.
//!
//! # Parallelism
//!
//! Block factorizations and block solves use faer's process-global
//! parallelism setting. Pin it once in the driver with
//! `faer::set_global_parallelism(Par::rayon(4))` — the default rayon
//! pool sizes itself to `nproc` (8 hyperthreads on the reference box)
//! and oversubscribes the 4-CPU quota for a ~2.5× penalty
//! (`docs/profiling.md` §4). The Krylov matvec is sequential
//! (memory-bound; the block solves dominate the solve wall).
//!
//! # FGMRES, by hand
//!
//! Modified Gram–Schmidt orthogonalization against at most
//! [`BlockKrylovConfig::restart_length`] Krylov vectors per cycle,
//! Givens-rotation least squares (the residual estimate is the *true*
//! residual in exact arithmetic, because the preconditioned basis `Z`
//! is stored and the update never re-applies `M⁻¹`), restart with a
//! fresh true-residual recomputation. No Krylov-library dependency: the
//! loop is ~150 lines and this module is its documentation.

use crate::{csr::CsrMatrix, LinearSolver, SolveError};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::linalg::cholesky::simplicial::{
    self, SimplicialLdltRef, SymbolicSimplicialCholesky,
};
use faer::sparse::linalg::lu::{self, LuRef, NumericLu, SymbolicLu};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};

/// Settings of the field-blocked Krylov backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockKrylovConfig {
    /// Number of field blocks. The matrix contract is
    /// `nrows == nfields * dim + 1` (the trailing slot is the frequency
    /// eigenvalue); the default 6 is the ROTBOSON layout.
    pub nfields: usize,
    /// Krylov vectors per restart cycle ("a few dozen").
    pub restart_length: usize,
    /// Maximum restart cycles per solve.
    pub max_cycles: usize,
    /// Relative-residual target `‖b − Ax‖ ≤ tolerance · ‖b‖`.
    pub tolerance: f64,
    /// Row-equilibrate each block before factoring (recommended: the
    /// boundary rows shrink like 1/N² under refinement; see the module
    /// docs).
    pub equilibrate_blocks: bool,
    /// Relative tolerance of the per-block symmetry check that decides
    /// LDLT vs LU: symmetric iff `max|a_ij − a_ji| ≤ tol · max|a_ij|`.
    pub symmetric_tol: f64,
    /// Report [`SolveError::NotConverged`] when the Krylov loop misses
    /// the tolerance. With `false` the best-so-far solution is written
    /// and the outcome is visible in [`BlockFieldKrylov::last_stats`].
    pub strict: bool,
    /// Which preconditioner FGMRES applies (see [`Preconditioner`]).
    pub preconditioner: Preconditioner,
}

/// The preconditioner `M ≈ J` that FGMRES inverts at every iteration.
///
/// Write the Newton matrix as `J = [A b; cᵀ c_w]`: `A` is the field
/// part (`nfields·dim` square), `b` the frequency column, and the last
/// row the constraint (`c = 0, c_w = 1` for fixed ω; `c = e_k, c_w = 0`
/// for fixed φ). `D` is the block diagonal of `A`: the factored field
/// blocks. See `docs/specs/A0.5-near-fold-spike.md`.
///
/// **Why bordering matters.** The scalar-field equation is linear in ψ,
/// so at every solution the ψ block maps ψ to zero: `D_ψψ ψ = 0`. The
/// ψ block is singular at *every* converged state (ω is its eigenvalue),
/// and block Jacobi then applies a near-singular inverse. Bordering that
/// block with the frequency column and the constraint row gives the
/// regular eigenvalue-bordered matrix (NR-11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preconditioner {
    /// `M = blockdiag(D, 1)`: field blocks only, identity on the
    /// frequency slot. FGMRES recovers the column, the row and all
    /// cross-field couplings.
    #[default]
    BlockJacobi,
    /// Block Jacobi, except that the pinned field's block is factored
    /// together with the frequency slot: `[D_gg b_g; c_gᵀ c_w]`, of size
    /// `dim + 1`, where `g` is the field of the constraint row's entries
    /// (ψ in fixed-φ mode). In fixed-ω mode (`c = 0`), `g` is the field
    /// with the largest frequency column. Same cost as block Jacobi.
    BorderedBlockJacobi,
    /// `M` = the block lower triangle of `A` (forward substitution over
    /// the fields), identity on the frequency slot. Extra cost: the
    /// below-diagonal coupling mat-vecs.
    BlockGaussSeidel,
}

impl Default for BlockKrylovConfig {
    fn default() -> Self {
        Self {
            nfields: 6,
            restart_length: 30,
            max_cycles: 40,
            tolerance: 1e-8,
            equilibrate_blocks: true,
            symmetric_tol: 1e-10,
            strict: true,
            preconditioner: Preconditioner::BlockJacobi,
        }
    }
}

/// Outcome of the last solve — for campaign logs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KrylovStats {
    /// Total Krylov iterations (matvec + preconditioner pairs).
    pub iterations: usize,
    /// Restart cycles used.
    pub cycles: usize,
    /// Final true relative residual `‖b − Ax‖ / ‖b‖`.
    pub residual: f64,
    /// Whether the tolerance was reached within the cycle budget.
    pub converged: bool,
}

/// One factored field block.
///
/// The LU variant is much larger than the LDLT one, but there are only
/// six blocks per solve, so boxing it would buy nothing.
#[allow(clippy::large_enum_variant)]
enum BlockFactor {
    /// Sparse LU (faer, COLAMD ordering, AUTO supernodal threshold) —
    /// the live path on real (nonsymmetric) Jacobians.
    Lu {
        /// Symbolic structure (fill-reducing ordering).
        symbolic: SymbolicLu<usize>,
        /// Numeric factors.
        numeric: NumericLu<usize, f64>,
        /// Row-equilibration factors `D` (empty when disabled).
        row_scale: Vec<f64>,
        /// Solve scratch, reused every application.
        scratch: MemBuffer,
        /// Parallelism captured at factor time.
        par: Par,
    },
    /// Sparse LDLT (faer simplicial, natural order) — the symmetric
    /// branch; not the production path, see the module docs.
    Ldlt {
        /// Symbolic structure.
        symbolic: SymbolicSimplicialCholesky<usize>,
        /// Numeric `L·D·Lᵀ` values.
        values: Vec<f64>,
        /// Row-equilibration factors `D` (empty when disabled).
        row_scale: Vec<f64>,
        /// Solve scratch (faer needs none here; kept for uniformity).
        scratch: MemBuffer,
        /// Parallelism captured at factor time.
        par: Par,
    },
}

impl BlockFactor {
    /// The row-equilibration factors of this block.
    fn row_scale(&self) -> &[f64] {
        match self {
            Self::Lu { row_scale, .. } | Self::Ldlt { row_scale, .. } => row_scale,
        }
    }

    /// Solve `A_g z = D·rhs` in place: `rhs` arrives scaled by the
    /// caller, `z` receives the block solution.
    fn solve_block(&mut self, rhs: &mut [f64]) {
        let n = rhs.len();
        match self {
            Self::Lu {
                symbolic,
                numeric,
                scratch,
                par,
                ..
            } => {
                let stack = MemStack::new(scratch);
                let lu = LuRef::new_unchecked(symbolic, numeric);
                let mat = MatMut::<f64>::from_column_major_slice_mut(rhs, n, 1);
                lu.solve_in_place_with_conj(Conj::No, mat, *par, stack);
            }
            Self::Ldlt {
                symbolic,
                values,
                scratch,
                par,
                ..
            } => {
                let stack = MemStack::new(scratch);
                let ldlt = SimplicialLdltRef::new(symbolic, values);
                let mat = MatMut::<f64>::from_column_major_slice_mut(rhs, n, 1);
                ldlt.solve_in_place_with_conj(Conj::No, mat, *par, stack);
            }
        }
    }
}

/// Solve one field block, `out = A_g⁻¹ rhs`, applying the block's row
/// equilibration on the way in.
fn solve_field(block: &mut BlockFactor, rhs: &[f64], out: &mut [f64]) {
    let scale = block.row_scale();
    if scale.is_empty() {
        out.copy_from_slice(rhs);
    } else {
        for ((o, r), d) in out.iter_mut().zip(rhs).zip(scale) {
            *o = r * d;
        }
    }
    block.solve_block(out);
}

/// `z = D⁻¹ v` on the field slots (`z` and `v` of length `nfields·dim`
/// or longer; only the field slots are touched).
fn block_diagonal_solve(
    blocks: &mut [BlockFactor],
    dim: usize,
    v: &[f64],
    z: &mut [f64],
    buf: &mut [f64],
) {
    for (g, block) in blocks.iter_mut().enumerate() {
        let fields = g * dim..(g + 1) * dim;
        solve_field(block, &v[fields.clone()], buf);
        z[fields].copy_from_slice(buf);
    }
}

/// Apply the preconditioner, `z = M⁻¹ v`. `buf` is scratch of length
/// `2·(dim + 1)`.
fn apply_preconditioner(
    blocks: &mut [BlockFactor],
    extra: &PreconditionerData,
    dim: usize,
    v: &[f64],
    z: &mut [f64],
    buf: &mut [f64],
) {
    let n = z.len();
    let (rhs, out) = buf.split_at_mut(dim + 1);
    match extra {
        PreconditionerData::BlockJacobi => {
            block_diagonal_solve(blocks, dim, v, z, &mut out[..dim]);
            z[n - 1] = v[n - 1];
        }
        PreconditionerData::Bordered { field } => {
            // The other fields as in block Jacobi; the bordered field
            // and the frequency slot in one augmented solve.
            for (g, block) in blocks.iter_mut().enumerate() {
                let fields = g * dim..(g + 1) * dim;
                if g == *field {
                    rhs[..dim].copy_from_slice(&v[fields.clone()]);
                    rhs[dim] = v[n - 1];
                    solve_field(block, rhs, out);
                    z[fields].copy_from_slice(&out[..dim]);
                    z[n - 1] = out[dim];
                } else {
                    solve_field(block, &v[fields.clone()], &mut out[..dim]);
                    z[fields].copy_from_slice(&out[..dim]);
                }
            }
        }
        PreconditionerData::GaussSeidel(lower) => {
            // z_g = A_gg⁻¹ (v_g − Σ_{h<g} A_gh z_h), in field order.
            for (g, block) in blocks.iter_mut().enumerate() {
                let base = g * dim;
                let couplings = &lower[g];
                for (i, r) in rhs[..dim].iter_mut().enumerate() {
                    let mut acc = v[base + i];
                    for k in couplings.row_ptrs[i]..couplings.row_ptrs[i + 1] {
                        acc -= couplings.vals[k] * z[couplings.col_inds[k]];
                    }
                    *r = acc;
                }
                solve_field(block, &rhs[..dim], &mut out[..dim]);
                z[base..base + dim].copy_from_slice(&out[..dim]);
            }
            z[n - 1] = v[n - 1];
        }
    }
}

/// What a preconditioner needs beyond the factored field blocks.
enum PreconditionerData {
    /// Nothing.
    BlockJacobi,
    /// `blocks[field]` is the augmented `(dim + 1)` block.
    Bordered {
        /// The bordered field.
        field: usize,
    },
    /// Per field `g`, the couplings to fields `h < g`.
    GaussSeidel(Vec<LowerCouplings>),
}

/// Which field the bordered preconditioner augments: the field of the
/// constraint row's entries, or, with no entries (fixed ω), the field
/// with the largest frequency column.
fn bordered_field(a: &CsrMatrix, dim: usize, nfields: usize) -> usize {
    let slot = a.nrows - 1;
    if let Some(k) =
        (a.row_ptrs[slot]..a.row_ptrs[slot + 1]).find(|&k| a.col_inds[k] < slot && a.vals[k] != 0.0)
    {
        return a.col_inds[k] / dim;
    }
    let mut column_norm = vec![0.0_f64; nfields];
    for r in 0..slot {
        let end = a.row_ptrs[r + 1];
        if end > a.row_ptrs[r] && a.col_inds[end - 1] == slot {
            column_norm[r / dim] += a.vals[end - 1] * a.vals[end - 1];
        }
    }
    (0..nfields)
        .max_by(|&p, &q| column_norm[p].total_cmp(&column_norm[q]))
        .unwrap_or(0)
}

/// Field `g`'s block bordered by the frequency slot, as a `(dim + 1)`
/// square CSR matrix: `[A_gg b_g; c_gᵀ c_w]`. Constraint entries in other
/// fields are left to FGMRES.
fn augmented_block(a: &CsrMatrix, g: usize, dim: usize) -> CsrMatrix {
    let slot = a.nrows - 1;
    let base = g * dim;
    let mut row_ptrs = Vec::with_capacity(dim + 2);
    let mut col_inds = Vec::new();
    let mut vals = Vec::new();
    row_ptrs.push(0);
    for r in (base..base + dim).chain(std::iter::once(slot)) {
        for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
            let c = a.col_inds[k];
            if (base..base + dim).contains(&c) {
                col_inds.push(c - base);
                vals.push(a.vals[k]);
            } else if c == slot {
                col_inds.push(dim);
                vals.push(a.vals[k]);
            }
        }
        row_ptrs.push(col_inds.len());
    }
    CsrMatrix {
        nrows: dim + 1,
        ncols: dim + 1,
        row_ptrs,
        col_inds,
        vals,
    }
}

/// The rows of one field `g` restricted to columns of fields `h < g`
/// (global column indices), in CSR form with `dim` rows.
struct LowerCouplings {
    row_ptrs: Vec<usize>,
    col_inds: Vec<usize>,
    vals: Vec<f64>,
}

impl LowerCouplings {
    /// Extract field `g`'s below-diagonal couplings from `a`.
    fn extract(a: &CsrMatrix, g: usize, dim: usize) -> Self {
        let base = g * dim;
        let mut row_ptrs = Vec::with_capacity(dim + 1);
        let mut col_inds = Vec::new();
        let mut vals = Vec::new();
        row_ptrs.push(0);
        for r in base..base + dim {
            for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
                if a.col_inds[k] < base {
                    col_inds.push(a.col_inds[k]);
                    vals.push(a.vals[k]);
                }
            }
            row_ptrs.push(col_inds.len());
        }
        Self {
            row_ptrs,
            col_inds,
            vals,
        }
    }
}

/// A field block in faer's CSC layout, with the row equilibration.
struct BlockCsc {
    /// Column pointers, length `dim + 1`.
    col_ptrs: Vec<usize>,
    /// Row indices, sorted within columns.
    row_idx: Vec<usize>,
    /// Values (row-equilibrated when enabled).
    vals: Vec<f64>,
    /// Row-max equilibration factors (empty when disabled).
    row_scale: Vec<f64>,
}

impl BlockCsc {
    /// Extract field `g`'s diagonal block from the field-major CSR
    /// matrix `a`, converting to sorted CSC; `equilibrate` applies
    /// one-sided row-max scaling.
    fn extract(a: &CsrMatrix, g: usize, dim: usize, equilibrate: bool) -> Self {
        let base = g * dim;
        let mut row_scale = Vec::new();
        if equilibrate {
            row_scale = vec![1.0; dim];
            for r in 0..dim {
                let mut max = 0.0_f64;
                for k in a.row_ptrs[base + r]..a.row_ptrs[base + r + 1] {
                    let c = a.col_inds[k];
                    if c >= base && c < base + dim {
                        max = max.max(a.vals[k].abs());
                    }
                }
                if max > 0.0 {
                    row_scale[r] = 1.0 / max;
                }
            }
        }

        let mut col_counts = vec![0usize; dim];
        for r in 0..dim {
            for k in a.row_ptrs[base + r]..a.row_ptrs[base + r + 1] {
                let c = a.col_inds[k];
                if c >= base && c < base + dim {
                    col_counts[c - base] += 1;
                }
            }
        }
        let mut col_ptrs = Vec::with_capacity(dim + 1);
        col_ptrs.push(0usize);
        for &c in &col_counts {
            col_ptrs.push(col_ptrs[col_ptrs.len() - 1] + c);
        }
        let nnz = col_ptrs[dim];
        let mut next = col_ptrs.clone();
        let mut row_idx = vec![0usize; nnz];
        let mut vals = vec![0.0; nnz];
        for r in 0..dim {
            let d = if row_scale.is_empty() {
                1.0
            } else {
                row_scale[r]
            };
            for k in a.row_ptrs[base + r]..a.row_ptrs[base + r + 1] {
                let c = a.col_inds[k];
                if c >= base && c < base + dim {
                    let dest = next[c - base];
                    row_idx[dest] = r;
                    vals[dest] = a.vals[k] * d;
                    next[c - base] += 1;
                }
            }
        }
        Self {
            col_ptrs,
            row_idx,
            vals,
            row_scale,
        }
    }

    /// Symmetry check: every off-diagonal entry must have a value-equal
    /// mirror, within `tol` relative to the block's magnitude scale.
    fn is_symmetric(&self, tol: f64) -> bool {
        let dim = self.col_ptrs.len() - 1;
        let scale = self.vals.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if scale == 0.0 {
            return true;
        }
        // Row-major mirror lookup: positions of (row, col) pairs.
        let mut row_start = vec![0usize; dim + 1];
        for &r in &self.row_idx {
            row_start[r + 1] += 1;
        }
        for i in 0..dim {
            row_start[i + 1] += row_start[i];
        }
        let mut row_col = vec![0usize; self.row_idx.len()];
        let mut row_pos = vec![0usize; self.row_idx.len()];
        let mut cursor = row_start[..dim].to_vec();
        for c in 0..dim {
            for p in self.col_ptrs[c]..self.col_ptrs[c + 1] {
                let r = self.row_idx[p];
                row_col[cursor[r]] = c;
                row_pos[cursor[r]] = p;
                cursor[r] += 1;
            }
        }
        let find = |r: usize, c: usize| -> Option<usize> {
            let (mut lo, hi) = (row_start[r], row_start[r + 1]);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                match row_col[mid].cmp(&c) {
                    core::cmp::Ordering::Less => lo = mid + 1,
                    core::cmp::Ordering::Equal => return Some(row_pos[mid]),
                    core::cmp::Ordering::Greater => break,
                }
            }
            None
        };
        for c in 0..dim {
            for p in self.col_ptrs[c]..self.col_ptrs[c + 1] {
                let r = self.row_idx[p];
                if r == c {
                    continue;
                }
                match find(r, c) {
                    Some(q) => {
                        if (self.vals[p] - self.vals[q]).abs() > tol * scale {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
        }
        true
    }
}

/// The factored state: per-field blocks plus the kept operator.
struct Factored {
    /// Field blocks, in field order.
    blocks: Vec<BlockFactor>,
    /// The chosen preconditioner's extra data.
    extra: PreconditionerData,
    /// Points per field (`(nrows − 1) / nfields`).
    dim: usize,
    /// The full-system operator kept for the Krylov matvec.
    operator: CsrMatrix,
    /// Krylov arenas: `V` (`restart_length` + 1 vectors) then `Z`
    /// (`restart_length`), each of length nrows.
    arena: Vec<f64>,
}

/// Which matrix the Krylov loop iterates on.
#[derive(Clone, Copy)]
enum Operator<'a> {
    /// The matrix kept from [`LinearSolver::factor`] (the
    /// `solve_into` flow: fresh blocks).
    Kept,
    /// The caller's current matrix (the [`BlockFieldKrylov::solve_stale`]
    /// flow: stale blocks, drifted operator).
    External(&'a CsrMatrix),
}

/// Field-blocked Krylov backend over the 6-field Jacobian.
///
/// See the module docs for the matrix contract, the stale-block reuse
/// API, and the FGMRES implementation notes.
pub struct BlockFieldKrylov {
    config: BlockKrylovConfig,
    factored: Option<Factored>,
    stats: KrylovStats,
}

impl BlockFieldKrylov {
    /// An empty (unfactored) backend with `config`.
    #[must_use]
    pub fn new(config: BlockKrylovConfig) -> Self {
        Self {
            config,
            factored: None,
            stats: KrylovStats {
                iterations: 0,
                cycles: 0,
                residual: f64::NAN,
                converged: false,
            },
        }
    }

    /// An empty backend with the default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(BlockKrylovConfig::default())
    }

    /// The backend configuration.
    #[must_use]
    pub fn config(&self) -> &BlockKrylovConfig {
        &self.config
    }

    /// Stats of the last solve (iterations, cycles, true residual,
    /// convergence flag).
    #[must_use]
    pub fn last_stats(&self) -> KrylovStats {
        self.stats
    }

    /// Solve `a_current · x = b` against the blocks kept from the last
    /// [`LinearSolver::factor`] — the stale-block reuse the C campaign
    /// runs near folds (stale-LU CGS, `docs/profiling.md` §3). The
    /// Krylov loop iterates on the caller's current matrix, so the
    /// solution is exact for it; the stale blocks cost iterations, not
    /// accuracy.
    ///
    /// # Errors
    ///
    /// [`SolveError::NotFactorized`] before the first
    /// [`LinearSolver::factor`]; [`SolveError::DimensionMismatch`] on
    /// wrong lengths; [`SolveError::InvalidMatrix`] when the operator
    /// is not square; [`SolveError::NotConverged`] on a missed
    /// tolerance in strict mode.
    pub fn solve_stale(
        &mut self,
        a_current: &CsrMatrix,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<(), SolveError> {
        if a_current.nrows != a_current.ncols {
            return Err(SolveError::InvalidMatrix("square operator required"));
        }
        let (stats, _) = self.run_krylov(Operator::External(a_current), b, x, a_current.nrows)?;
        self.stats = stats;
        Ok(())
    }

    /// The Krylov solve against either operator source. The factored
    /// state is moved out for the duration so the operator reference
    /// and the mutating arena/blocks borrows cannot conflict.
    fn run_krylov(
        &mut self,
        operator: Operator<'_>,
        b: &[f64],
        x: &mut [f64],
        n: usize,
    ) -> Result<(KrylovStats, Vec<f64>), SolveError> {
        let Some(mut factored) = self.factored.take() else {
            return Err(SolveError::NotFactorized);
        };
        let outcome = fgmres(&mut factored, &self.config, operator, b, x, n);
        self.factored = Some(factored);
        outcome
    }

    /// Extract, equilibrate, and factor one field block.
    fn factor_block(&self, csc: &BlockCsc) -> Result<BlockFactor, SolveError> {
        let par = faer::get_global_parallelism();
        let dim = csc.col_ptrs.len() - 1;
        let sym_mat = SymbolicSparseColMatRef::<usize>::new_checked(
            dim,
            dim,
            &csc.col_ptrs,
            None,
            &csc.row_idx,
        );
        let mat = SparseColMatRef::<usize, f64>::new(sym_mat, &csc.vals);

        if csc.is_symmetric(self.config.symmetric_tol) {
            match Self::factor_block_ldlt(sym_mat, mat, csc, par) {
                Ok(block) => Ok(block),
                // Indefinite symmetric blocks still deserve a working
                // preconditioner: fall back to LU.
                Err(_) => Self::factor_block_lu(sym_mat, mat, csc, par),
            }
        } else {
            Self::factor_block_lu(sym_mat, mat, csc, par)
        }
    }

    /// The LDLT path for symmetric blocks (faer simplicial, natural
    /// order — the input is read from its upper triangle).
    fn factor_block_ldlt(
        sym_mat: SymbolicSparseColMatRef<'_, usize>,
        mat: SparseColMatRef<'_, usize, f64>,
        csc: &BlockCsc,
        par: Par,
    ) -> Result<BlockFactor, SolveError> {
        let dim = csc.col_ptrs.len() - 1;
        let sym_req = StackReq::any_of(&[
            simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, csc.vals.len()),
            simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(dim),
        ]);
        let mut mem =
            MemBuffer::try_new(sym_req).map_err(|_| SolveError::InvalidMatrix("LDLT scratch"))?;
        let stack = MemStack::new(&mut mem);
        let mut etree = vec![0isize; dim];
        let mut col_counts = vec![0usize; dim];
        let etree =
            simplicial::prefactorize_symbolic_cholesky(&mut etree, &mut col_counts, sym_mat, stack);
        let symbolic =
            simplicial::factorize_simplicial_symbolic_cholesky(sym_mat, etree, &col_counts, stack)
                .map_err(|_| SolveError::InvalidMatrix("LDLT symbolic"))?;
        let num_req = simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(dim);
        let mut mem =
            MemBuffer::try_new(num_req).map_err(|_| SolveError::InvalidMatrix("LDLT numeric"))?;
        let stack = MemStack::new(&mut mem);
        let mut values = vec![0.0; symbolic.len_val()];
        simplicial::factorize_simplicial_numeric_ldlt(
            &mut values,
            mat,
            faer::linalg::cholesky::ldlt::factor::LdltRegularization::default(),
            &symbolic,
            stack,
        )
        .map_err(|_| SolveError::Singular)?;
        let scratch = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1));
        Ok(BlockFactor::Ldlt {
            symbolic,
            values,
            row_scale: csc.row_scale.clone(),
            scratch,
            par,
        })
    }

    /// The LU path (the live branch on real Jacobians).
    fn factor_block_lu(
        sym_mat: SymbolicSparseColMatRef<'_, usize>,
        mat: SparseColMatRef<'_, usize, f64>,
        csc: &BlockCsc,
        par: Par,
    ) -> Result<BlockFactor, SolveError> {
        let symbolic = lu::factorize_symbolic_lu(sym_mat, lu::LuSymbolicParams::default())
            .map_err(|_| SolveError::InvalidMatrix("LU symbolic"))?;
        let req = symbolic.factorize_numeric_lu_scratch::<f64>(par, faer::Spec::default());
        let mut mem =
            MemBuffer::try_new(req).map_err(|_| SolveError::InvalidMatrix("LU numeric"))?;
        let stack = MemStack::new(&mut mem);
        let mut numeric = NumericLu::<usize, f64>::new();
        match symbolic.factorize_numeric_lu::<f64>(
            &mut numeric,
            mat,
            par,
            stack,
            faer::Spec::default(),
        ) {
            Ok(_) => {
                let scratch = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, par));
                Ok(BlockFactor::Lu {
                    symbolic,
                    numeric,
                    row_scale: csc.row_scale.clone(),
                    scratch,
                    par,
                })
            }
            Err(faer::sparse::linalg::LuError::SymbolicSingular { .. }) => {
                Err(SolveError::Singular)
            }
            Err(_) => Err(SolveError::InvalidMatrix("LU numeric failure")),
        }
    }
}

/// The restarted FGMRES loop (see the module docs): right-preconditioned
/// with the block-diagonal inverse, flexible storage of the
/// preconditioned basis, Givens least squares, restart on the true
/// residual.
fn fgmres(
    factored: &mut Factored,
    cfg: &BlockKrylovConfig,
    operator: Operator<'_>,
    b: &[f64],
    x: &mut [f64],
    n: usize,
) -> Result<(KrylovStats, Vec<f64>), SolveError> {
    if b.len() != n || x.len() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: b.len().min(x.len()),
        });
    }
    let Factored {
        blocks,
        extra,
        dim,
        operator: kept,
        arena,
    } = factored;
    let op: &CsrMatrix = match operator {
        Operator::Kept => kept,
        Operator::External(a) => a,
    };
    if op.nrows != n {
        return Err(SolveError::DimensionMismatch {
            expected: op.nrows,
            got: n,
        });
    }

    let m = cfg.restart_length.min(n);
    let stop = cfg.tolerance;
    let bnorm = norm2(b);

    // Trivial solve: zero right-hand side.
    if bnorm == 0.0 {
        x.fill(0.0);
        return Ok((
            KrylovStats {
                iterations: 0,
                cycles: 0,
                residual: 0.0,
                converged: true,
            },
            Vec::new(),
        ));
    }

    // Reused workspaces (no allocation inside the Krylov loop).
    let mut r = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut ax = vec![0.0; n];
    let mut pbuf = vec![0.0; 2 * (*dim + 1)];
    let mut y = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    // H is (m+1) x m, hand-indexed `h[i * m + j]`.
    let mut h = vec![0.0; (m + 1) * m];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];

    x.fill(0.0);
    r.copy_from_slice(b);
    let mut total_iters = 0usize;
    let mut cycles_used = 0usize;
    let mut converged = false;

    for _cycle in 0..cfg.max_cycles {
        let beta = norm2(&r);
        if beta <= stop * bnorm {
            converged = true;
            break;
        }
        cycles_used += 1;

        // Krylov basis partition: V (m+1 vectors) | Z (m vectors).
        let (v_part, z_part) = arena.as_mut_slice().split_at_mut((m + 1) * n);
        v_part[..n].copy_from_slice(&r);
        for v in &mut v_part[..n] {
            *v /= beta;
        }
        g[0] = beta;
        g[1..].fill(0.0);
        let mut breakdown = false;
        let mut m_eff = 0usize;

        for j in 0..m {
            // z_j = M⁻¹ v_j
            apply_preconditioner(
                blocks,
                extra,
                *dim,
                &v_part[j * n..(j + 1) * n],
                &mut z_part[j * n..(j + 1) * n],
                &mut pbuf,
            );
            // w = A z_j
            matvec_into(op, &z_part[j * n..(j + 1) * n], &mut w);
            // Modified Gram–Schmidt against V_0..=V_j.
            for i in 0..=j {
                let vi = &v_part[i * n..(i + 1) * n];
                let hij = dot(&w, vi);
                for (wi, v) in w.iter_mut().zip(vi) {
                    *wi -= hij * v;
                }
                h[i * m + j] = hij;
            }
            let hnext = norm2(&w);
            h[(j + 1) * m + j] = hnext;

            // Apply the existing Givens rotations to column j.
            for k in 0..j {
                let (ck, sk) = (cs[k], sn[k]);
                let h1 = h[k * m + j];
                let h2 = h[(k + 1) * m + j];
                h[k * m + j] = ck * h1 + sk * h2;
                h[(k + 1) * m + j] = -sk * h1 + ck * h2;
            }
            let h1 = h[j * m + j];
            if hnext == 0.0 {
                // Happy breakdown: the current space already contains
                // the solution.
                m_eff = j + 1;
                breakdown = true;
            } else {
                let t = (h1 * h1 + hnext * hnext).sqrt();
                cs[j] = h1 / t;
                sn[j] = hnext / t;
                h[j * m + j] = cs[j] * h1 + sn[j] * hnext;
                let gj = g[j];
                g[j] = cs[j] * gj;
                g[j + 1] = -sn[j] * gj;
                m_eff = j + 1;
                if g[j + 1].abs() <= stop * bnorm {
                    breakdown = true;
                } else {
                    // v_{j+1} = w / hnext
                    let vnext = &mut v_part[(j + 1) * n..(j + 2) * n];
                    for (v, wv) in vnext.iter_mut().zip(&w) {
                        *v = wv / hnext;
                    }
                }
            }
            total_iters += 1;
            if breakdown {
                break;
            }
        }

        // Back-substitution on the rotated upper-triangular system.
        for i in (0..m_eff).rev() {
            let mut acc = g[i];
            for k in i + 1..m_eff {
                acc -= h[i * m + k] * y[k];
            }
            y[i] = acc / h[i * m + i];
        }
        // x += Z y (x0 = 0, so x is rebuilt from the basis).
        for k in 0..m_eff {
            let zk = &z_part[k * n..(k + 1) * n];
            for (xi, zkv) in x.iter_mut().zip(zk) {
                *xi += y[k] * zkv;
            }
        }

        // True residual for the restart decision.
        matvec_into(op, x, &mut ax);
        for (ri, (bi, axi)) in r.iter_mut().zip(b.iter().zip(&ax)) {
            *ri = bi - axi;
        }
        if norm2(&r) <= stop * bnorm {
            converged = true;
            break;
        }
    }

    matvec_into(op, x, &mut ax);
    let mut num = 0.0;
    for (bi, axi) in b.iter().zip(&ax) {
        num += (bi - axi) * (bi - axi);
    }
    let residual = (num / (bnorm * bnorm)).sqrt();

    if !converged && cfg.strict {
        return Err(SolveError::NotConverged {
            iterations: total_iters,
            residual,
        });
    }
    Ok((
        KrylovStats {
            iterations: total_iters,
            cycles: cycles_used,
            residual,
            converged,
        },
        r,
    ))
}

impl Default for BlockFieldKrylov {
    fn default() -> Self {
        Self::default_config()
    }
}

impl core::fmt::Debug for BlockFieldKrylov {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockFieldKrylov")
            .field("config", &self.config)
            .field("factored", &self.factored.as_ref().map(|f| f.blocks.len()))
            .field("stats", &self.stats)
            .finish()
    }
}

impl LinearSolver for BlockFieldKrylov {
    fn factor(&mut self, a: &CsrMatrix) -> Result<(), SolveError> {
        a.validate()?;
        let nfields = self.config.nfields;
        if a.nrows != a.ncols {
            return Err(SolveError::InvalidMatrix("square matrix required"));
        }
        if a.nrows < nfields + 1 || (a.nrows - 1) % nfields != 0 {
            return Err(SolveError::InvalidMatrix(
                "matrix must be nfields * dim + 1 (field blocks + eigenvalue slot)",
            ));
        }
        let dim = (a.nrows - 1) / nfields;

        let bordered = (self.config.preconditioner == Preconditioner::BorderedBlockJacobi)
            .then(|| bordered_field(a, dim, nfields));
        let mut blocks = Vec::with_capacity(nfields);
        for g in 0..nfields {
            let csc = if bordered == Some(g) {
                BlockCsc::extract(
                    &augmented_block(a, g, dim),
                    0,
                    dim + 1,
                    self.config.equilibrate_blocks,
                )
            } else {
                BlockCsc::extract(a, g, dim, self.config.equilibrate_blocks)
            };
            blocks.push(self.factor_block(&csc)?);
        }
        let extra = match self.config.preconditioner {
            Preconditioner::BlockJacobi => PreconditionerData::BlockJacobi,
            Preconditioner::BorderedBlockJacobi => PreconditionerData::Bordered {
                field: bordered.unwrap_or(0),
            },
            Preconditioner::BlockGaussSeidel => PreconditionerData::GaussSeidel(
                (0..nfields)
                    .map(|g| LowerCouplings::extract(a, g, dim))
                    .collect(),
            ),
        };

        let m = self.config.restart_length.min(a.nrows);
        let arena = vec![0.0; (2 * m + 1) * a.nrows];
        self.factored = Some(Factored {
            blocks,
            extra,
            dim,
            operator: a.clone(),
            arena,
        });
        Ok(())
    }

    fn solve_into(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        let Some(factored) = self.factored.as_ref() else {
            return Err(SolveError::NotFactorized);
        };
        let n = factored.operator.nrows;
        let (stats, _) = self.run_krylov(Operator::Kept, b, x, n)?;
        self.stats = stats;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "block-field-krylov"
    }
}

/// `‖v‖₂`.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `dot(a, b)`.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// CSR mat-vec into a caller buffer (the Krylov hot path; avoids the
/// per-call allocation of [`CsrMatrix::matvec`]).
fn matvec_into(a: &CsrMatrix, x: &[f64], out: &mut [f64]) {
    for r in 0..a.nrows {
        let mut acc = 0.0;
        for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
            acc += a.vals[k] * x[a.col_inds[k]];
        }
        out[r] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-field toy system with the eigenvalue slot: field blocks
    /// tridiagonal, cross-field same-node couplings, an omega column,
    /// and the pinning omega row.
    fn toy_matrix(dim: usize, coupling: f64, symmetric: bool) -> CsrMatrix {
        let n = 2 * dim + 1;
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let dim_i64 = i64::try_from(dim).expect("dim fits i64");
        for g in 0..2 {
            for i in 0..dim {
                let r = g * dim + i;
                // Self-block: diagonally dominant tridiagonal
                // (symmetric when `symmetric`, advection-skewed
                // otherwise).
                let below = if symmetric { -1.0 } else { -0.5 };
                let i_i64 = i64::try_from(i).expect("i fits i64");
                for (dj, w) in [(-1, below), (0, 4.2), (1, -1.0)] {
                    let j = i_i64 + i64::from(dj);
                    if (0..dim_i64).contains(&j) {
                        rows[r].push((g * dim + usize::try_from(j).expect("j fits usize"), w));
                    }
                }
                // Cross-field same-node coupling.
                rows[r].push(((1 - g) * dim + i, coupling));
                if !symmetric && i + 1 < dim {
                    rows[r].push((g * dim + i + 1, 0.3));
                }
                // Omega column (field 0 only).
                if g == 0 {
                    rows[r].push((n - 1, 0.5));
                }
            }
        }
        rows[n - 1].push((n - 1, 1.0));

        let mut row_ptrs = vec![0usize];
        let mut col_inds = Vec::new();
        let mut vals = Vec::new();
        for row in &rows {
            let mut sorted = row.clone();
            sorted.sort_by_key(|(c, _)| *c);
            // Merge duplicate columns by summation.
            for (c, v) in sorted {
                if col_inds.len() > row_ptrs[row_ptrs.len() - 1] && col_inds.last() == Some(&c) {
                    *vals.last_mut().expect("parallel to col_inds") += v;
                } else {
                    col_inds.push(c);
                    vals.push(v);
                }
            }
            row_ptrs.push(col_inds.len());
        }
        CsrMatrix {
            nrows: n,
            ncols: n,
            row_ptrs,
            col_inds,
            vals,
        }
    }

    fn rhs_of(a: &CsrMatrix, seed: u64) -> (Vec<f64>, Vec<f64>) {
        // Deterministic xorshift64* values in [-1, 1).
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let x: Vec<f64> = (0..a.nrows).map(|_| next()).collect();
        let b = a.matvec(&x);
        (b, x)
    }

    fn rel_residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
        let ax = a.matvec(x);
        let num: f64 = ax.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        let den: f64 = b.iter().map(|v| v * v).sum();
        (num / den).sqrt()
    }

    #[test]
    fn solves_a_two_field_system_to_tolerance() {
        let a = toy_matrix(40, 0.8, false);
        let (b, x_true) = rhs_of(&a, 0x1234);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        let rel = rel_residual(&a, &b, &x);
        assert!(rel < 1e-8, "residual {rel:.3e}");
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-6, "component off: {xi} vs {ti}");
        }
        let stats = solver.last_stats();
        assert!(stats.converged);
        assert!(stats.iterations > 1, "couplings should cost iterations");
    }

    #[test]
    fn symmetric_block_takes_the_ldlt_path() {
        let a = toy_matrix(30, 0.4, true);
        let (b, x_true) = rhs_of(&a, 0x5678);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        let rel = rel_residual(&a, &b, &x);
        assert!(rel < 1e-8, "LDLT-path residual {rel:.3e}");
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-6);
        }
    }

    #[test]
    fn identity_blocks_converge_in_one_iteration() {
        // Zero coupling: the preconditioner equals the matrix up to the
        // omega column, which the first Krylov step resolves.
        let a = toy_matrix(20, 0.0, true);
        let (b, _) = rhs_of(&a, 0x9abc);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        assert!(rel_residual(&a, &b, &x) < 1e-8);
        assert!(solver.last_stats().iterations <= 2);
    }

    #[test]
    fn stale_blocks_solve_a_drifted_operator() {
        // The C campaign pattern: factor once, then solve a *different*
        // (drifted) system against the same blocks.
        let a0 = toy_matrix(30, 0.5, false);
        let a1 = toy_matrix(30, 0.6, false);
        let (b, x_true) = rhs_of(&a1, 0xdef0);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a0).expect("factor");
        let mut x = vec![0.0; a1.nrows];
        solver.solve_stale(&a1, &b, &mut x).expect("stale solve");
        assert!(rel_residual(&a1, &b, &x) < 1e-8);
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-6);
        }
    }

    #[test]
    fn stale_before_factor_is_an_error() {
        let a = toy_matrix(10, 0.3, false);
        let b = vec![1.0; a.nrows];
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        let mut x = vec![0.0; a.nrows];
        assert_eq!(
            solver.solve_stale(&a, &b, &mut x),
            Err(SolveError::NotFactorized)
        );
    }

    #[test]
    fn strict_mode_reports_non_convergence() {
        // Strong cross-field coupling, a tiny Krylov space, and an
        // unreachable tolerance: the strict mode must say so.
        let a = toy_matrix(24, 3.5, false);
        let (b, _) = rhs_of(&a, 0x2468);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            restart_length: 4,
            max_cycles: 2,
            tolerance: 1e-14,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let err = solver.solve(&b);
        assert!(
            matches!(err, Err(SolveError::NotConverged { .. })),
            "expected NotConverged, got {err:?}"
        );
        assert!(!solver.last_stats().converged);
    }

    #[test]
    fn lenient_mode_returns_the_best_effort() {
        let a = toy_matrix(24, 3.5, false);
        let (b, _) = rhs_of(&a, 0x2468);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            restart_length: 4,
            max_cycles: 2,
            tolerance: 1e-14,
            strict: false,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("lenient solve");
        // Best-effort is still a genuine Krylov iterate.
        assert!(rel_residual(&a, &b, &x) < 1.0);
        assert!(!solver.last_stats().converged);
    }

    #[test]
    fn wrong_field_count_is_rejected() {
        let a = toy_matrix(8, 0.2, false); // 2 fields
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig::default()); // 6 fields
        assert!(matches!(
            solver.factor(&a),
            Err(SolveError::InvalidMatrix(_))
        ));
    }

    #[test]
    fn rank_deficient_blocks_do_not_crash_factorization() {
        // faer's low-level LU path does not flag a structurally
        // rank-deficient block the way the high-level `sp_lu` does.
        // For a *preconditioner* backend that is acceptable: M⁻¹ need
        // not be invertible in every direction, and the Krylov loop
        // recovers as long as the full operator is solvable — pinned
        // here. A genuinely broken preconditioner surfaces as a
        // stalled loop (`NotConverged`, strict mode), which the
        // campaign driver handles like the C's PARDISO −4.
        let dim = 3;
        let n = 2 * dim + 1;
        let a = CsrMatrix {
            nrows: n,
            ncols: n,
            row_ptrs: vec![0, 1, 2, 3, 4, 5, 6, 7],
            col_inds: vec![0, 0, 0, dim, dim, dim, n - 1],
            vals: vec![1.0; 7],
        };
        let b = a.matvec(&vec![1.0; n]);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver
            .factor(&a)
            .expect("low-level block LU tolerates rank deficiency");
        let x = solver.solve(&b).expect("full Krylov space recovers");
        assert!(rel_residual(&a, &b, &x) < 1e-8);
    }

    #[test]
    fn solve_before_factor_is_an_error() {
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        assert_eq!(
            solver.solve_into(&[1.0, 2.0, 3.0], &mut [0.0; 3]),
            Err(SolveError::NotFactorized)
        );
    }

    #[test]
    fn equilibration_survives_tiny_boundary_rows() {
        // The failure class of the full-matrix LU: rows scaled by 1e-8
        // against O(1) interiors. The equilibrated block path must not
        // care.
        let dim = 24;
        let mut a = toy_matrix(dim, 0.5, false);
        for k in a.row_ptrs[dim - 1]..a.row_ptrs[dim] {
            a.vals[k] *= 1e-8;
        }
        for k in a.row_ptrs[2 * dim - 1]..a.row_ptrs[2 * dim] {
            a.vals[k] *= 1e-8;
        }
        let (b, x_true) = rhs_of(&a, 0x8765);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        assert!(rel_residual(&a, &b, &x) < 1e-8);
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-6);
        }
    }

    #[test]
    fn no_equilibration_still_solves_the_toy_system() {
        let a = toy_matrix(20, 0.7, false);
        let (b, x_true) = rhs_of(&a, 0x1357);
        let mut solver = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            equilibrate_blocks: false,
            ..BlockKrylovConfig::default()
        });
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        assert!(rel_residual(&a, &b, &x) < 1e-8);
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-6);
        }
    }

    // ----- Preconditioner choice (NR-10) -----

    const ALL_PRECONDITIONERS: [Preconditioner; 3] = [
        Preconditioner::BlockJacobi,
        Preconditioner::BorderedBlockJacobi,
        Preconditioner::BlockGaussSeidel,
    ];

    /// The two constraint rows of the real solver.
    #[derive(Clone, Copy, Debug)]
    enum Constraint {
        /// `δξ = 0`: `c = 0`, `c_w = 1`.
        FixedOmega,
        /// Pin unknown `k`: `c = e_k`, `c_w = 0`.
        FixedPhi(usize),
    }

    fn to_dense(a: &CsrMatrix) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; a.ncols]; a.nrows];
        for (r, row) in d.iter_mut().enumerate() {
            for k in a.row_ptrs[r]..a.row_ptrs[r + 1] {
                row[a.col_inds[k]] += a.vals[k];
            }
        }
        d
    }

    /// Dense to CSR, keeping only nonzero entries (plus any diagonal
    /// entry of the last row, so the constraint row is never empty).
    fn from_dense(d: &[Vec<f64>]) -> CsrMatrix {
        let n = d.len();
        let mut row_ptrs = vec![0usize];
        let mut col_inds = Vec::new();
        let mut vals = Vec::new();
        for (r, row) in d.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0.0 || (r == n - 1 && c == n - 1) {
                    col_inds.push(c);
                    vals.push(v);
                }
            }
            row_ptrs.push(col_inds.len());
        }
        CsrMatrix {
            nrows: n,
            ncols: n,
            row_ptrs,
            col_inds,
            vals,
        }
    }

    fn with_constraint(a: &CsrMatrix, constraint: Constraint) -> CsrMatrix {
        let mut d = to_dense(a);
        let n = d.len();
        d[n - 1].fill(0.0);
        match constraint {
            Constraint::FixedOmega => d[n - 1][n - 1] = 1.0,
            Constraint::FixedPhi(k) => d[n - 1][k] = 1.0,
        }
        from_dense(&d)
    }

    fn solver_with(preconditioner: Preconditioner) -> BlockFieldKrylov {
        BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            preconditioner,
            ..BlockKrylovConfig::default()
        })
    }

    /// Factor `a`, solve against a known solution, check the residual,
    /// and return the iteration count.
    fn solve_and_check(a: &CsrMatrix, preconditioner: Preconditioner) -> usize {
        let (b, _) = rhs_of(a, 0x5eed);
        let mut solver = solver_with(preconditioner);
        solver.factor(a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        let rel = rel_residual(a, &b, &x);
        assert!(rel < 1e-8, "{preconditioner:?}: residual {rel:.3e}");
        solver.last_stats().iterations
    }

    #[test]
    fn every_preconditioner_solves_both_constraint_modes() {
        let dim = 30;
        let base = toy_matrix(dim, 0.5, false);
        // Pin a node of field 0, the field whose rows carry the
        // frequency column (in the real system: psi, whose equation
        // contains omega).
        for constraint in [Constraint::FixedOmega, Constraint::FixedPhi(dim / 2)] {
            let a = with_constraint(&base, constraint);
            let iterations: Vec<usize> = ALL_PRECONDITIONERS
                .iter()
                .map(|&p| solve_and_check(&a, p))
                .collect();
            eprintln!("{constraint:?}: iterations (Jacobi, bordered, GS) = {iterations:?}");
            assert!(
                iterations[1] <= iterations[0],
                "{constraint:?}: bordered {} > Jacobi {}",
                iterations[1],
                iterations[0]
            );
        }
    }

    #[test]
    fn every_preconditioner_survives_stale_blocks() {
        let dim = 30;
        let pin = Constraint::FixedPhi(3);
        let a0 = with_constraint(&toy_matrix(dim, 0.5, false), pin);
        let a1 = with_constraint(&toy_matrix(dim, 0.6, false), pin);
        let (b, _) = rhs_of(&a1, 0xabc);
        for preconditioner in ALL_PRECONDITIONERS {
            let mut solver = solver_with(preconditioner);
            solver.factor(&a0).expect("factor");
            let mut x = vec![0.0; a1.nrows];
            solver.solve_stale(&a1, &b, &mut x).expect("stale solve");
            let rel = rel_residual(&a1, &b, &x);
            assert!(rel < 1e-8, "{preconditioner:?}: residual {rel:.3e}");
        }
    }

    /// Make field 0's diagonal block exactly singular, with null vector
    /// `u` (smooth, nonzero at every node): `A_00 ← A_00 − (A_00 u) uᵀ/(uᵀu)`.
    /// This is the real solver's structure: the ψ equation is linear in
    /// ψ, so the ψ block annihilates ψ at every solution.
    fn with_singular_field_zero_block(a: &CsrMatrix, dim: usize) -> CsrMatrix {
        let mut d = to_dense(a);
        let u: Vec<f64> = (0..dim).map(|i| 1.0 + i as f64 / dim as f64).collect();
        let au: Vec<f64> = (0..dim)
            .map(|r| (0..dim).map(|c| d[r][c] * u[c]).sum())
            .collect();
        let uu: f64 = u.iter().map(|x| x * x).sum();
        for r in 0..dim {
            for c in 0..dim {
                d[r][c] -= au[r] * u[c] / uu;
            }
        }
        let null = (0..dim)
            .map(|r| (0..dim).map(|c| d[r][c] * u[c]).sum::<f64>().abs())
            .fold(0.0_f64, f64::max);
        assert!(null < 1e-12, "A_00 u = {null:.3e}, not 0");
        from_dense(&d)
    }

    /// The frequency column and the pin make the full matrix regular even
    /// though field 0's block is singular. The bordered preconditioner
    /// factors exactly that regular bordered block, so it solves the
    /// system accurately and fast; block Jacobi has to invert the
    /// singular block.
    #[test]
    fn bordered_preconditioner_handles_a_singular_field_block() {
        let dim = 12;
        let a = with_constraint(
            &with_singular_field_zero_block(&toy_matrix(dim, 0.5, false), dim),
            Constraint::FixedPhi(4),
        );
        let (b, x_true) = rhs_of(&a, 0xf01d);
        let mut solver = solver_with(Preconditioner::BorderedBlockJacobi);
        solver.factor(&a).expect("factor");
        let x = solver.solve(&b).expect("solve");
        assert!(rel_residual(&a, &b, &x) < 1e-8);
        let error = x
            .iter()
            .zip(&x_true)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        assert!(error < 1e-6, "max error {error:.3e}");
        let bordered_iterations = solver.last_stats().iterations;

        let mut jacobi = BlockFieldKrylov::new(BlockKrylovConfig {
            nfields: 2,
            strict: false,
            ..BlockKrylovConfig::default()
        });
        let jacobi_iterations = jacobi
            .factor(&a)
            .and_then(|()| jacobi.solve(&b))
            .map_or(usize::MAX, |_| jacobi.last_stats().iterations);
        eprintln!("singular block: bordered {bordered_iterations}, Jacobi {jacobi_iterations}");
        assert!(bordered_iterations < jacobi_iterations);
    }
}
