# ng: the Rust initial-data solver

A pure-Rust solver for rotating boson star initial data. The C code in `../src/` is the frozen oracle that it is compared against.

- Scope and milestones: `../docs/plan/initial-data.md` and ADR 0002 (`../docs/adr/0002-initial-data-scope.md`).
- How the code fits together: [`docs/architecture.md`](docs/architecture.md).
- Rules for agents: `../AGENTS.md`.

## Workspace

| Crate | Role |
|---|---|
| `rbs-core` | grid, fields, finite-difference stencils |
| `rbs-elliptic` | residual and Jacobian assembly (generated kernels), Newton, the ω constraint |
| `rbs-linalg` | `LinearSolver` trait; `BlockFieldKrylov` (production) and faer sparse LU (cross-check) |
| `rbs-io` | VTR and HDF5 output (HDF5 behind the `hdf5` feature) |
| `rbs-driver` | continuation-driver logic ported from Python (the solver-facing half is A2) |
| `rbs-viz` | placeholder |

Each crate's `README.md` states its role and its tests. `spikes/` holds standalone experiment crates that are not workspace members.

## Try it

Solve an l=1 star from scratch (about 6 s) and write `solution.vtr` and `solution.txt`:

```bash
cd ng
cargo run --release -p rbs-elliptic --example l1_star -- ../out/demo/a0/rust
```

`docs/results/A0-demo.md` shows how to plot it and compare it with the C code.

## Build, test, lint

```bash
cd ng
cargo build --workspace
cargo test --workspace                                   # fast tests
cargo test --workspace --release -- --include-ignored   # also the slow parity tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

The `hdf5` feature is off by default. It needs the system library `libhdf5-dev`:

```bash
cargo test -p rbs-io --features hdf5
```

Dependencies are kept minimal: `faer` for sparse LU, plus the feature-gated `hdf5-metno`. `rbs_linalg::CsrMatrix` is our own copy of the C code's 3-array CSR layout.

## Generated kernels

`rbs-elliptic/src/generated/kernels.rs` is generated from `tools/sympy_system.py`. Do not edit it by hand.

```bash
uv run tools/generate_rust_kernels.py            # regenerate kernels.rs and the parity fixture
uv run tools/generate_rust_kernels.py --check    # CI: regenerate and require identical output
uv run tools/generate_rust_kernels.py --validate # regenerate and run the Rust-vs-SymPy test
```

## CI

`.github/workflows/rust.yml` runs build, test, clippy, the format check and the codegen `--check`.
