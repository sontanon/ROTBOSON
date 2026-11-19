# rbs-io

Field-set output: HDF5 (the oracle's format contract) and VTR/VTH
(ParaView-native), behind one trait.

**Role:** persist elliptic solutions and evolution states so the analysis
and visualization pipelines can read them. `FieldWriter` is the trait;
`VtrWriter` (rectilinear-grid VTR, real) is the day-one backend; the HDF5
backend is feature-gated.

**Parity target (HDF5, SAN-55 — landed):** the C writer's self-describing
contract (`src/hdf5_writer.c`) is mirrored end to end: datasets named
`<field>.asc` — 2D fields shaped `(NrTotal, NzTotal)` row-major over
`IDX(i, j)`, `r.asc`/`z.asc` as 2D meshgrids (`r[i, j] = r_i`,
`z[i, j] = z_j`), 1D scalars/histories as shape-`(n,)` — plus the full
root-attribute set (provenance `git_hash`/`parfile`/`output_backend`/
`created`/`format_version`, GRID, SCALAR FIELD, FIXED VARIABLE, INITIAL
DATA, SCALE, SOLVER). Deliberate deviations, invisible to
`tools/rotboson_io.py`: string attributes are HDF5 vlen UTF-8 (not the
C fixed-size strings), `output_backend = "hdf5-rust"` marks NG
provenance, and the spherical-analysis/analysis-result attribute blocks
are emitted only when a caller fills them (`Option` fields — the Rust
snapshot carries only what the Rust solve actually computed). The writer
auto-syncs `NrTotal`/`NzTotal`/`dim` from the grid being written, so the
file can never misdescribe its own datasets.

**Gate:** the l=1 N=64 smoke snapshot, written from the vertical slice's
proven parity state (`l1_smoke_n64_solution.txt`), checked into
`tests/fixtures/l1_smoke_n64_rust.h5` and pinned by
`tests/hdf5_parity.rs`: field-by-field against the slice fixture, the
exact attribute name/type set, and a fresh-write regression net.
Measured against the C oracle on this box: max field deviation
**1.1e-15** (deep-converged oracle, same epsilon protocol) and
**6.7e-11** (stock `epsilon = 1e-8` config) — both files load unchanged
through `rotboson_io.read_hdf5`:

```sh
# the manual Python cross-read (the documented gate command)
uv run python3 -c 'import sys; sys.path.insert(0, "tools"); import numpy as np, rotboson_io as rio; \
  c, _ = rio.read_hdf5("out/n1_io_vtr/c_oracle_deep/l=1,w=9.50000E-01,dr=2.50000E-01,N=0064/solution.h5"); \
  r, _ = rio.read_hdf5("out/n1_io_vtr/rust/solution.h5"); \
  print({k: float(np.abs(c[k] - r[k]).max()) for k in r if k in c})'
```

**VTR (SAN-55):** `VtrWriter` emits ParaView rectilinear-grid XML with
VTK's x-fastest point ordering (the field blocks are z-fastest
row-major over `IDX(i, j)`, so the writer transposes on emit — without
that, ParaView mirrors the field across the diagonal). The N=64
deliverable is verified by `tests/vtr_selfcheck.rs` (schema + exact
read-back). **No ParaView exists in the dev container** — the
host-side eyeball check is still owed: open `solution.vtr` in ParaView,
confirm (a) the six fields appear as point scalars, (b) the blob is
centered near the origin with the ghost band visible at r < 0, and
(c) psi's peak sits at r ~ 0, z ~ 0 with the expected mirror symmetry
in z — i.e. no diagonal mirroring and no transposition.

**Artifacts (`examples/l1_smoke_snapshot.rs`):** writes the smoke
snapshot's `solution.h5` + `solution.vtr` from the parity fixture —

```sh
cargo run -p rbs-io --features hdf5 --example l1_smoke_snapshot -- <out-dir>
```

— and regenerates the committed fixture (with
`GIT_HASH=$(git rev-parse --short HEAD)`). The day-one render
(`tools/ng_render_fields.py`) reads the Rust-written h5 back through
`rotboson_io` and draws the 2x3 panel figure (workstream V).

**Feature flag:** `hdf5` (default off; needs native libhdf5 — see
`Cargo.toml`). CI builds and tests without it; enable locally or on a
runner with `libhdf5-dev`:

```sh
cargo test -p rbs-io --features hdf5
```

Quirk measured on the dev box: the original `hdf5` 0.8.1 crate refuses
modern system HDF5 (build.rs: "Invalid `H5_VERSION: 1.14.5"), so the
feature uses the maintained `hdf5-metno` fork under the same `hdf5::`
import path.
