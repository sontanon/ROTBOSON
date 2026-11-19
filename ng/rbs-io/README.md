# rbs-io

Field-set output: HDF5 (the oracle's format contract) and VTR/VTH
(ParaView-native), behind one trait.

**Role:** persist elliptic solutions and evolution states so the analysis
and visualization pipelines can read them. `FieldWriter` is the trait;
`VtrWriter` (rectilinear-grid VTR, real) is the day-one backend; the HDF5
backend is feature-gated.

**Parity target (HDF5):** the C writer's self-describing contract
(`src/hdf5_writer.c`) — dataset layout, dimension attributes, and metadata
— so `tools/rotboson_io.py` reads Rust HDF5 output *unchanged*. The C CI
already pins that contract with `tests/test_hdf5_fixture.py`, so the Rust
gate is: roundtrip a Rust-written file through the Python reader.

**Feature flag:** `hdf5` (default off; needs native libhdf5 — see
`Cargo.toml`). CI builds and tests without it; enable locally or on a
runner with `libhdf5-dev` when the Rust HDF5 gate lands. Quirk measured
on the dev box: the original `hdf5` 0.8.1 crate refuses modern system
HDF5 (build.rs: "Invalid H5_VERSION: 1.14.5"), so the feature uses the
maintained `hdf5-metno` fork under the same `hdf5::` import path.
