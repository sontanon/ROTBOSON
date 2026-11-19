# rbs-core

Grids, fields, stencils and index maps for the RBS-NG elliptic solver.

**Role:** the discretization vocabulary every other crate speaks — the
uniform (r, z) grid with ghost cells, the field-major layout of the six grid
functions, and the step-scaled finite-difference stencil tables.

**Parity target:** the C oracle's conventions, byte-for-byte where they are
observable: `NrTotal`/`NzTotal`/`ghost` grid sizing, the `IDX(i, j)` index
map (`src/tools.h`), the coordinate map `r = dr * (i + 0.5 - ghost)`
(`src/rhs_vars.c`), the `u[k * dim + IDX(i, j)]` field layout, and the
Fornberg stencil tables of `tools/generate_kernels.py`.

`diff` carries the full operator set of `src/derivatives.c` the campaigns
can configure: `diff1r/diff2r/diff1z/diff2z` at orders 2/4/6 (SAN-58 closed
the order-2 rules) and `diff2rz` (`ex_diff2rz`, orders 2/4 — the
interpolation/analysis consumer). Unit parity: exactness on the design
polynomial families through axis/boundary/ghost rows, measured convergence
orders (2.00 and 3.95–4.00), delta-impulse weights equal to the 1D-rule
outer products, and — at port time — a scratch dump of the C's own
operators agreeing to 3.7e-15 (order 2) / 2.2e-14 (order 4) across all
parity combinations.

`tools/sympy_system.py` documents the unknown naming: u1 = log(alpha),
u2 = beta, u3 = log(h), u4 = log(a), u5 = psi, u6 = lambda.
