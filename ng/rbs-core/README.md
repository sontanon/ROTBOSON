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

`tools/sympy_system.py` documents the unknown naming: u1 = log(alpha),
u2 = beta, u3 = log(h), u4 = log(a), u5 = psi, u6 = lambda.
