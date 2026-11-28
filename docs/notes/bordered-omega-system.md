# The bordered ω system, and what the near-fold spike found

Date: 2026-11-27. Milestone A0.5 (Linear NR-13).
Numbers come from `docs/results/A0.5-near-fold.md` (raw tables in `docs/results/A0.5-near-fold/`), unless the text says otherwise.

This note explains:

1. the linear system Newton solves;
2. why one part of it goes singular at the fold;
3. why a second part is singular at *every* converged state, which is the main finding of the spike;
4. how the preconditioner deals with both.

## 1. The system

The unknowns are the six fields on the grid,

```
u = (log α, β, log h, log a, ψ, λ),      each of length dim = NrTotal · NzTotal,
```

stored field-major (field g occupies `[g·dim, (g+1)·dim)`), plus one frequency unknown ξ in the last slot. The solver works with ξ rather than ω, through `ω = ½ m (1 + tanh ξ)`. This map keeps ω in (0, m), and `dω/dξ = ½ m sech²ξ > 0`. ψ is the regularized scalar field: the physical field is φ = ρ^l ψ.

The residual F(u, ξ) has 6·dim rows: one equation per field per grid point, including the boundary and symmetry rows. One more row is needed to close the system. Each Newton step solves

```
J δ = −R,      J = [ A    b  ]     A = ∂F/∂u    (6·dim × 6·dim: the fixed-ω Jacobian)
                   [ cᵀ   c_w ]     b = ∂F/∂ξ    (one column)
```

The last row is the **constraint row**. There are two choices:

| Mode | Constraint | c | c_w | Used for |
|---|---|---|---|---|
| fixed-ω | ξ is given | 0 | 1 | stars at a chosen frequency, away from the fold |
| fixed-φ | ψ at one node k (here (2,2)) is given, ψ_k = ψ₀ | e_k | 0 | continuation along the branch and through the fold |

In fixed-ω mode J is block upper triangular, so its eigenvalues are those of A plus the single eigenvalue 1. J is singular exactly when A is.

## 2. A is singular at the fold, and the fixed-φ J is not

Follow the branch of solutions, parametrized by the pinned value ψ₀: u(ψ₀), ξ(ψ₀), with F(u(ψ₀), ξ(ψ₀)) = 0. Differentiate with respect to ψ₀ (a prime ′ means d/dψ₀):

```
A u′ + b ξ′ = 0.                                                     (1)
```

At the fold ω is at its minimum, so ω′ = 0. Since dω/dξ > 0, this also gives ξ′ = 0. The fields still change along the branch (u′ ≠ 0), so (1) becomes **A u′ = 0: A has a null vector at the fold.**

The fixed-φ J has no such problem. Applied to the tangent (u′, ξ′):

```
J [u′; ξ′] = [ A u′ + b ξ′ ;  e_kᵀ u′ ] = [ 0 ; 1 ],
```

because ψ_k = ψ₀ along the branch, so e_kᵀ u′ = 1. The tangent is therefore not a null vector of J. A standard bordering lemma (Keller 1977) states when J is nonsingular at a simple fold: A has a one-dimensional null space spanned by u′; b is not in the range of A (for the left null vector ℓ, ℓᵀ b ≠ 0, which is the condition for a nondegenerate fold); and c = e_k is not orthogonal to u′. The last condition holds by construction. **That is why campaigns switch to fixed-φ near the fold.**

**Evidence (`eig` table, N=128).** The table gives the smallest-magnitude eigenvalue, estimated by inverse iteration.

| ψ₀ | 0.076 | 0.178 | 0.277 | 0.321 | 0.351 | 0.362 |
|---|---|---|---|---|---|---|
| λ_min, fixed-ω | 6.39e-5 | 6.33e-5 | 6.04e-5 | 5.41e-5 | 3.89e-5 | 1.93e-5 |
| λ_min, fixed-φ | 6.406e-5 | 6.406e-5 | 6.406e-5 | 6.406e-5 | 6.406e-5 | 6.406e-5 |

- **The fixed-ω eigenvalue falls toward zero.** Near a simple fold it should vanish linearly in the distance to the fold. ω is quadratic in ψ₀ there, so ω′, which tracks this eigenvalue, is linear. The last two points extrapolate linearly to zero at ψ₀ ≈ 0.372, which is our estimate of the fold on the N=128 grid.
- **The fixed-φ eigenvalue does not move**, to five digits.
- **Both matrices share a floor eigenvalue of 6.406e-5.** It is a grid-level mode that has nothing to do with the fold: the fixed-ω eigenvalue starts at that same value far from the fold. Where it comes from is not identified. It does not affect any conclusion here.
- **Caveat:** inverse iteration hit its 30-iteration cap without meeting its 1e-6 tolerance in 9 of the 12 runs. The values are stable to the digits shown. This is a diagnostic, not one of the verdict criteria.

**A caution on "distance to the fold".** The `dist` column in the results is measured against ψ₀* = 0.37616 from the campaign's `summary.json`. That value comes from a cubic fit over five points that mix the N=128 and N=256 grids. Its ω* = 0.645530 also lies below every sampled ω. On the N=256 grid the smallest sampled ω is at ψ₀ = 0.388 (step 167), so the N=256 fold sits near there. On the N=128 grid it sits near 0.372 (see above). Treat `dist` as an ordering of the states, not as a precise distance. The four N=256 states straddle the fold whichever estimate is used.

## 3. The main finding: the ψ block is singular at every solution

The Klein–Gordon equation (row block 4 of `tools/sympy_system.py`, `R4`) is **linear and homogeneous in ψ**. Every term carries one factor of ψ or one of its derivatives, and the coefficients depend only on the metric fields and ω. Write it as F_ψ = L(metric, ω) ψ. The diagonal block of A for ψ is then

```
A_ψψ = ∂F_ψ/∂ψ = L,     so     A_ψψ ψ = L ψ = F_ψ.
```

At any converged state F_ψ = 0, so **ψ is a null vector of A_ψψ at every solution, not only near the fold.** This is simply the statement that, for fixed metric, ω is an eigenvalue of the linear operator L and ψ is its eigenvector. The full A is still regular away from the fold: ψ sources the metric equations quadratically, and that coupling removes the scaling freedom ψ → cψ.

**Measured (NR-11 diagnostic, recorded in the A0.5 spec).** ‖A_ψψ ψ‖ / (‖A_ψψ‖ ‖ψ‖) lies between 2e-16 and 3e-13 at all ten test states. That is the residual level, so the identity also holds for the boundary and symmetry rows.

**Consequence for the preconditioner.** BlockFieldKrylov preconditions with D = blockdiag(A_11, …, A_66), the six factored field blocks. With A_ψψ singular, D⁻¹ applies a huge, inaccurate inverse along ψ. (In the NR-11 probe, per-block solves had errors of 1e-2 to 4e-1 while their residuals were 1e-15.) The `linear` experiment shows the effect, at fixed-φ, tolerance 1e-10, and a cap of 1200 iterations:

| Preconditioner | N=128 (6 states) | N=256 (4 states) |
|---|---|---|
| block Jacobi | hits the cap at all 6; residual 1e-1 to 9e6 | hits the cap at 2; converges at the 2 nearest the fold (147, 125 iterations) |
| block Gauss–Seidel | hits the cap at all 6; residual up to 3e11 | hits the cap at 2; converges at the same 2 (72, 75) |
| full LU (reference) | residual 7e-14 to 2e-12 | not run |

The failures happen **far from the fold too** (ψ₀ = 0.076, step 111), which rules out the fold as their cause. The two N=256 states where block Jacobi and Gauss–Seidel converge are the two with the largest self-check residuals (2.1e-10 and 2.6e-10, against 4e-12 to 5e-12 for the other two). The ψ block is a little less exactly singular there. That fits the explanation, but it does not prove it.

## 4. Why exact elimination fails, and what works

**Keller elimination** removes ξ exactly. Solve A y = b and A z = r, then set

```
δξ = (r_w − cᵀ z) / (c_w − cᵀ y),      δu = z − y δξ.
```

This needs A⁻¹. Near the fold, y and z both grow along the null vector u′, like 1/λ_min, and the two terms of the numerator and the denominator cancel. Their rounding errors grow in the same proportion. The method is exact in exact arithmetic and loses digits exactly where we need it.

**The first bordered preconditioner (NR-10)** did the same thing with D in place of A: a Schur scalar s = c_w − cᵀ D⁻¹ b. Because D is singular along ψ at *every* state (§3), this is Keller cancellation everywhere. It converged (26 to 137 iterations) only through cancellation of two huge terms. It was replaced in NR-11.

**The bordered block-Jacobi preconditioner (NR-11, now the default for fold work)** does not eliminate anything. It factors the ψ block together with the frequency slot and the constraint row, as one (dim+1) sparse matrix:

```
B_ψ = [ A_ψψ   b_ψ ]      b_ψ: the ξ column restricted to the ψ rows
      [ e_kᵀ    0  ]      e_k: the pin row restricted to the ψ columns
```

The other five fields keep their plain blocks. This is the classic regularization of a linear eigenvalue problem L(ω)ψ = 0, with the same lemma as in §2 applied to one block. B_ψ is nonsingular when the eigenvalue is simple (ℓᵀ ∂L/∂ω ψ ≠ 0, with ℓ the left null vector of L) and the pin sees the eigenvector (ψ_k ≠ 0). Both hold here: ψ at node (2,2) is the central amplitude ψ₀ > 0. It costs the same as block Jacobi, one sparse LU per field.

It handles the fold singularity of A for the same reason the fixed-φ J does. The preconditioner never inverts A, and FGMRES only ever works with J, which stays regular.

**Results** (`linear`, `stale` and `newton` tables):

| Experiment | N=128 | N=256 |
|---|---|---|
| `linear`: iterations to 1e-10, farthest → closest state | 21 → 25, all converged | 33 → 35, all converged |
| `stale`: blocks factored at one end of the range, reused at every state | 21–29 (fresh: 21–25) | 34–35 (fresh: 33–35) |
| `newton`: started from the neighboring state | 5 Newton steps, 212 Krylov iterations; ω = 0.647079629955808 vs C 0.6470796299558079 | 4 Newton steps, 152 Krylov iterations; ω = 0.6457644139468055 vs C 0.6457644139507639 (6e-12 relative) |

In the `newton` runs, interior fields match C to 3.4e-12 (N=128) and 4.1e-11 (N=256). The whole-grid difference of 3e-2 is in ψ's ghost cells, which the two codes fill differently. That difference is not part of the solution.

The iteration count does not grow toward the fold, and the N=256 states include two past it. Going from N=128 to N=256 raises it by about 1.4×.

## 5. Verdict

The proposed rule from `docs/specs/A0.5-near-fold-spike.md`, applied to the bordered preconditioner:

| Rule | Limit | Result | |
|---|---|---|---|
| 1. 1e-10 residual at every state | all 10 | all 10 | pass |
| 2. Closest-state iterations vs farthest | ≤ 3× | 1.19× (N=128), 1.03× (N=256) | pass |
| 3. `newton` converges at both N | both | both | pass |
| 4. Stale vs fresh iterations | ≤ 2× | at most 1.16× | pass |

**Decision (owner, 2026-11-27): GO.** BlockFieldKrylov with the bordered block-Jacobi preconditioner is the linear solver for A1 and A2.

## 6. Open points

- **Fixed-ω mode.** There the constraint row is e_wᵀ, so the bordered ψ block is [A_ψψ b_ψ; 0 1], which is singular again. Fixed-ω solves therefore need ψ deflation, such as a rank-one correction along the current ψ, or a pin that is not used as an unknown. Fold work never uses fixed-ω, so this is left for A1.
- **The default.** `Preconditioner::BlockJacobi` is still the default in `rbs-linalg`. A1 should make `BorderedBlockJacobi` the default for fixed-φ solves.
- **The 6.406e-5 floor eigenvalue** (§2) is unexplained. It is harmless here, but worth understanding before A5 changes the grid and the stencils.

## References

- H. B. Keller, "Numerical solution of bifurcation and nonlinear eigenvalue problems," in *Applications of Bifurcation Theory*, P. Rabinowitz (ed.), Academic Press, 1977. Bordering lemma and pseudo-arclength continuation.
- `docs/specs/A0.5-near-fold-spike.md`: the experiment design and the verdict rule.
- `ng/spikes/nearfold/`: the harness that produced every number above.
- `ng/rbs-linalg/src/block_field.rs`: `augmented_block` and `bordered_field`, the preconditioner itself.
