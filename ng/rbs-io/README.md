# rbs-io

**Role.** Writes solutions to disk through one trait, `FieldWriter`:

- **`VtrWriter`** writes ParaView rectilinear-grid XML. VTK expects x-fastest point order, and our fields are stored z-fastest, so the writer transposes as it writes. Without that, ParaView would mirror the star across the diagonal.
- **`Hdf5Writer`** (feature `hdf5`) mirrors the C writer's format (`src/hdf5_writer.c`), so `tools/rotboson_io.py` reads Rust output unchanged.

The HDF5 format has:
- datasets `<field>.asc`, shaped `(NrTotal, NzTotal)` and row-major over `IDX(i, j)`;
- `r.asc` and `z.asc` as 2D meshgrids;
- the full set of root attributes.

There are three deliberate differences, none visible to `rotboson_io`:
- strings are variable-length UTF-8;
- `output_backend = "hdf5-rust"`;
- the analysis attribute blocks are written only when the caller fills them.

A1 defines the versioned initial-data format v1 on top of this **(planned)**.

**Example.** `examples/l1_smoke_snapshot.rs` writes `solution.h5` and `solution.vtr` from the N=64 parity fixture:

```bash
cargo run -p rbs-io --features hdf5 --example l1_smoke_snapshot -- <out-dir>
```

**Tests.**
- `tests/hdf5_parity.rs` (feature `hdf5`) checks the fixture `tests/fixtures/l1_smoke_n64_rust.h5` field by field, its exact attribute names and types, and a fresh write.
- `tests/vtr_selfcheck.rs` checks the VTR schema and that the file reads back exactly.

Run them with `cargo test -p rbs-io --features hdf5`. HDF5 needs the system library `libhdf5-dev`. The feature uses the maintained `hdf5-metno` fork, because the original `hdf5` crate refuses HDF5 1.14.
