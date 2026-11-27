# AGENTS.md: rules for any agent working in this repo

Read this whole file before you start an issue. It is short on purpose.

## What this repo is

A solver for **rotating boson star initial data** in numerical relativity.

- `src/`: the old C solver. It is **frozen**: it is the reference ("oracle") we compare against. **Never edit anything in `src/`.**
- `tools/`: Python tooling. `tools/sympy_system.py` derives the equations. The generators write code from it.
- `ng/`: the new Rust code (a Cargo workspace). **This is where new work happens.**
- `docs/plan/initial-data.md`: the milestone plan.
- `docs/specs/`: one spec per milestone. The issue tells you which spec to read.
- `configs/`: TOML run configurations for the C solver.
- `out/`: local run outputs. **Not in git.** Never delete anything in `out/`.

## How to do an issue

1. Read the whole Linear issue, then the spec section it links.
2. Check that the issue's "Before you start" items are true. If one is not, **stop** and report.
3. Create the branch named in the issue, starting from an up-to-date `master`:
   ```bash
   git fetch origin && git switch -c <branch-name> origin/master
   ```
4. Do the steps **in order**. Do only what the steps say.
5. Run the issue's **Acceptance** commands. All of them must pass.
6. Commit (see "Commits" below), push, and open a pull request:
   ```bash
   git push -u origin <branch-name>
   gh pr create --base master --title "NR-<n>: <issue title>" --body "Closes NR-<n>. <one-paragraph summary>"
   ```
7. In the PR body, paste the output of the acceptance commands (trim to the relevant lines).

## When to stop and ask

Stop, write down what happened, and do not improvise, if:

- a step fails and the issue does not say what to do;
- you would need to change something the issue did not mention;
- you would need to edit `src/`, a generated file (`ng/rbs-elliptic/src/generated/`), CI config, or the workspace lint settings;
- you would need a new dependency (a new crate in `Cargo.toml`);
- a number (tolerance, iteration count, physics constant) seems wrong to you.

Report by leaving a comment on the Linear issue if you can, or else by putting a file `BLOCKED.md` on your branch that explains the problem, and pushing the branch.

## Commands

Run Rust commands from the `ng/` directory.

**Each command may run in a fresh shell.** `cd`, `source` and `export` do not carry over to the next command. When a command needs them, put them in the same command line, joined with `&&`. For example: `cd ng && cargo test --workspace`.

**Pipes hide failures.** In `cmd | tee log`, the exit code is `tee`'s, not `cmd`'s. When an issue asks for an exit code after a pipe, start the line with `set -o pipefail;`.

```bash
cd ng
cargo build --workspace
cargo test --workspace                    # fast tests
cargo test --workspace --release -- --include-ignored   # also the slow parity tests (minutes)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

HDF5 support is behind a feature flag and needs the system library `libhdf5-dev`:

```bash
cargo test -p rbs-io --features hdf5
```

Python tools run through `uv` from the repo root, for example `uv run tools/generate_rust_kernels.py --check`.

The C solver (only if an issue asks): see `README.md`. It needs Intel oneAPI MKL. Load it first with `source /opt/intel/oneapi/setvars.sh`.

## Rust code style

This code should be good, idiomatic Rust that people can learn from.

- Run `cargo fmt --all` (from `ng/`) before committing. CI checks formatting.
- Clippy must pass with `-D warnings`.
- Every public item gets a doc comment (`///`) that says what it is in plain words.
- Library code returns `Result` for anything that can fail. Do not use `unwrap()` or `expect()` in library code. Tests may use them.
- Put unit tests in a `#[cfg(test)] mod tests` at the bottom of the file they test.
- Prefer clear names over short ones: `omega`, not `w`; `n_interior`, not `ni`. Physics symbols are fine when the spec uses them.
- No `unsafe`.

## Commits

- **Stage files by name** (or by directory, e.g. `git add -A ng`). Never run `git add -A` or `git add .` at the repo root: there are untracked files that must not be committed. Before every commit, run `git status --short` and check that only the files the issue names are staged.
- One logical change per commit. Message format: `NR-<n>: <what changed, in plain words>`.
  - Good: `NR-14: Remove the rbs-evolve crate from the workspace`
  - Bad: `NR-14: cleanup round 3`, `NR-14: the last two`
- Someone who did not see your session must understand the message.

## Heavy runs

This machine has 24 threads and 31 GB of RAM. Runs at N ≥ 256 take minutes and use several GB. Only start them when the issue says so, and never start two at the same time.
