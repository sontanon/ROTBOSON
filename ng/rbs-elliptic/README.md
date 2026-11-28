# rbs-elliptic

**Role.** It turns fields into equations and solves them:

- it evaluates the generated kernels (`src/generated/`, from `tools/sympy_system.py`) over the grid, into the residual vector and the CSR Jacobian, with the ξ column and the constraint row;
- it builds the initial guess (`initial`) and maps ξ to ω (`omega`);
- it runs Newton (`newton`). The drivers are ported from the C code: NLEQ-ERR with the QNERR tail, NLEQ-RES with the QNRES tail, and plain damped Newton, selected by `NewtonConfig::kind`.

`ProblemParams::fixed_phi` selects the mode. `None` gives fixed-ω, where ξ is given. `Some((i, j))` gives fixed-φ, where ψ at node (i, j) is pinned and ω is solved for. See `../docs/notes/bordered-omega-system.md`.

**Example.** `examples/l1_star.rs` solves an l=1 star from scratch at N=64 and writes it out (the A0 demo):

```bash
cargo run --release -p rbs-elliptic --example l1_star -- <out-dir> [--backend krylov|lu]
```

**Tests.**

| Test | What |
|---|---|
| `src/kernel_parity.rs` | the generated kernels against SymPy reference values (`tests/fixtures/kernel_parity.txt`) |
| `tests/l1_smoke_n64_parity.rs` | the l=1, N=64 fixed-ω solve against the C oracle's fields |
| `tests/fixedphi_jacobian_parity.rs` | the full fixed-φ Jacobian, including the ω column, against the C code's own dump |
| `tests/fixedphi_n64_parity.rs` | the fixed-φ N=64 state: ω to 1e-15, the pin exact, the fields to 1e-14 |
| `tests/newton_machinery_parity.rs` | the Newton drivers' iteration histories against the C code |
| `tests/newton_fault_injection.rs` | the Newton failure paths |

Some parity tests are `#[ignore]`d because they are slow. Run them with `cargo test -p rbs-elliptic --release -- --include-ignored`.

## History

The kernels were generated in SAN-56, the Newton drivers ported in SAN-52, and fixed-φ mode added in SAN-58, all in the old Linear team SAN.
