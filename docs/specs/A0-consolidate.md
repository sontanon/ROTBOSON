# A0 · Consolidate

**Goal:** one clean, green `master` that the rest of the plan builds on.

**Exit check:**
- CI on `master` passes: Rust build, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, the Rust codegen check, and a new `cargo fmt --check`.
- On this machine, the C oracle builds and `ctest` passes, and `cargo test -p rbs-io --features hdf5` passes.
- No open PRs or Linear issues left over from the old plan (RBS-NG).

## 1. Integrating the six open PRs (#23–#28)

All six branches start from the current `master`. They merge without textual conflicts, except `ng/docs/architecture.md` in #24, which section 4 rewrites anyway. **They do not compile together as-is.** Two fixes are needed after merging:

1. `rbs-elliptic/tests/fixedphi_n64_parity.rs` (from #27) builds a `NewtonConfig` with three fields. #28 adds five more. Fix: add `..NewtonConfig::default()` to the initializer.
2. The same file has a placeholder test, `..._eigenvalue_walk_is_blocked_on_the_newton_port`, that asserts the Newton solve **diverges** (it was written before #28 existed). With #28 merged, the solve converges to the C oracle's ω to 2.3e-15 relative. Fix: replace the placeholder with a real parity test (`..._from_scratch_matches_the_c_oracle`) that asserts convergence, ω to 1e-10, and the field gates.

Plus four clippy errors that only appear in the combined tree (dead code without the `hdf5` feature, `map().unwrap_or()`, a doc backtick, a complex type in a test).

This is done as **one integration PR** (branch `a0/integrate-rust-prs`) rather than six separate merges, so `master` never holds a broken intermediate state. Merging it closes #23–#28 automatically, since their commits are included.

## 2. Machine setup

The C oracle needs Intel oneAPI MKL and libhdf5. Neither is installed on this machine yet: the old binaries in `build/` fail with `libmkl_*.so not found`. The commands are the ones CI uses (`.github/workflows/ci.yml`).

## 3. Removing out-of-scope code

- `ng/rbs-evolve` (56 lines, a placeholder for time evolution) is removed. Evolution is a separate future project (ADR 0002).
- `ng/rbs-viz` stays: visualization is in scope.

## 4. Docs to the new scope

`ng/docs/architecture.md`, `ng/README.md`, and the crate READMEs describe RBS-NG (grid ladder, multigrid production path, evolution). Rewrite them to the Initial Data scope (ADR 0002). Keep the facts that are still true: crate roles, the codegen chain, the BlockFieldKrylov measurements.

## 5. Formatting

The workspace was never `rustfmt`-formatted. Running `cargo fmt --all` changes about 35 files, **including the generated `rbs-elliptic/src/generated/kernels.rs`**, which then fails the codegen check. The fix (verified on the integration branch: with the attribute, the generated file is untouched and the codegen check passes):

- put `#[rustfmt::skip]` on the line above `pub mod generated;` in `ng/rbs-elliptic/src/lib.rs`, so `rustfmt` leaves the generated code alone;
- format everything else once, in a commit that does nothing else;
- add `cargo fmt --all --check` to CI;
- change the `cargo fmt` rule in `AGENTS.md`.

## 6. Cleanup

Delete the merged branches (local and remote) and the scratch worktrees, and remove the untracked `PR_NOTES.md`. This is the owner's job, because it deletes things.

## Lint policy

Workspace clippy stays as it is (pedantic, with the allow-list in `ng/Cargo.toml`). The code already complies, and pedantic clippy pushes toward the "showcase Rust" goal. Revisit only if it blocks local-agent issues often.
