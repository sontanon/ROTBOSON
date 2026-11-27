# RBS-NG — the Rust next generation

Pure-Rust next generation of ROTBOSON (ADR 0001:
`../docs/adr/0001-pivot-to-rbs-ng.md`; architecture:
[`docs/architecture.md`](docs/architecture.md)). The C codebase stays the
frozen golden oracle; every stage here is gated by parity against it.

## Workspace

| crate | role |
|---|---|
| `rbs-core` | grids, fields, stencils, index maps (C-convention ports) |
| `rbs-elliptic` | residual/Jacobian assembly, Newton, ω constraint; generated kernels live here |
| `rbs-linalg` | `LinearSolver` backend trait, faer sparse LU (parity sizes), MG-Krylov placeholder |
| `rbs-io` | field output: VTR (day one) + HDF5 (feature `hdf5`, the C format contract) |
| `rbs-driver` | continuation campaigns (port target) |
| `rbs-viz` | animation pipeline over the writers |

Each crate's `README.md` states its role and its parity target.

## Build / test / lint

```bash
cd ng
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Dependencies are deliberately minimal: `faer` (sparse LU, parity sizes
only) and the feature-gated `hdf5` fork `hdf5-metno`. The matrix contract
(`rbs_linalg::CsrMatrix`) is our own mirror of the C 3-array CSR layout
rather than `sprs`, so the port contract stays dependency-free.

The `hdf5` feature is OFF by default (needs native libhdf5):

```bash
cargo test -p rbs-io --features hdf5
```

## Codegen: the Rust kernels

The residual + Jacobian kernels in
`rbs-elliptic/src/generated/kernels.rs` are generated from the same SymPy
source of truth as the C kernels (`tools/sympy_system.py`); the C
generator is untouched and remains the structural cross-reference.

```bash
# regenerate (writes kernels.rs + the parity fixture)
uv run tools/generate_rust_kernels.py

# CI gate: regenerate and assert byte-identical
uv run tools/generate_rust_kernels.py --check

# regenerate + run the numeric parity test (Rust vs SymPy)
uv run tools/generate_rust_kernels.py --validate
```

The parity fixture (`rbs-elliptic/tests/fixtures/kernel_parity.txt`) is
checked in, so plain `cargo test -p rbs-elliptic` verifies the kernels
against SymPy-evaluated references without any Python present.

## CI

`.github/workflows/rust.yml` runs build + test + clippy on this workspace
and the codegen `--check` gate — kept cheap: one job each, no matrix.
