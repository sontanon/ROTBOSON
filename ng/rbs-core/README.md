# rbs-core

**Role.** The discretization vocabulary that every other crate uses:

- the uniform (ρ, z) grid with ghost cells (`Grid2d`);
- the field-major layout of the six fields (`FieldSet`);
- the finite-difference operators of orders 2, 4 and 6 (`diff`, `stencils`).

The conventions match the C oracle exactly where they can be observed: the `NrTotal`/`NzTotal`/`ghost` sizing, the `IDX(i, j)` index map (`src/tools.h`), the coordinate map `r = dr · (i + 0.5 − ghost)`, and the layout `u[k·dim + IDX(i, j)]`. The field order is u1 = log α, u2 = β, u3 = log h, u4 = log a, u5 = ψ, u6 = λ (`tools/sympy_system.py`).

A1 adds a grid and stencil abstraction here, for mapped coordinates and any stencil order **(planned)**. See `../docs/architecture.md` §6.

**Tests** (`cargo test -p rbs-core`) check that:
- the operators are exact on the polynomials they are designed for, including the axis, boundary and ghost rows;
- the measured convergence orders are 2.00 and 3.95–4.00;
- the 2D weights equal the outer products of the 1D rules.

When the operators were ported, they also agreed with the C code's own operators to 3.7e-15 (order 2) and 2.2e-14 (order 4).
