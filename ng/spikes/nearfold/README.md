# Near-fold solver spike (A0.5, NR-11)

Does the block-field Krylov solver still work near the l=1 ω_min fold? The question, the math, and the verdict rule are in `docs/specs/A0.5-near-fold-spike.md`.

The harness loads the converged C states exported by `tools/export_nearfold_states.py` (NR-9), checks each one, and runs four experiments, writing one CSV per run.

**Self-check on every load.** The Rust residual at a converged C state must be below 1e-6 in max norm. Otherwise the export layout or the parameters are wrong, and the harness stops. Measured: 1e-13 to 3e-10.

## Experiments

| Subcommand | What it does |
|---|---|
| `linear` | For each state: J at the state (fixed-φ), right-hand side b = J·x_true, with x_true = next state − this state (a step along the branch). Solve to 1e-10 relative. At N=128 it also solves with faer full LU (`solver = lu`) as a reference. |
| `stale` | Per N: factor at the state farthest from the fold, then solve every state's system with those stale blocks; also solve fresh, for comparison. |
| `newton` | At the state closest to the fold for each N: start from the neighbor's fields with ψ at the pin set to this state's value, and run the full Newton solve. Compares ω and the fields with the C state. |
| `eig` | N=128 only: the smallest-magnitude eigenvalue of the fixed-ω and the fixed-φ Jacobian, by inverse iteration with faer LU (up to 30 steps). |

Options: `--manifest <manifest.csv>` and `--out <csv>` are required. `--preconditioner jacobi|bordered|gs|all` (default `all`; `eig` ignores it). `--n 128|256` keeps only one resolution. Solvers run with faer on 8 threads; the CSVs record it.

### Columns worth knowing

- `dist`: ψ0* − ψ0 against the campaign's fold estimate ψ0* = 0.37616 (negative past it). The estimate itself is off by about 1e-2 in ψ0 (see `docs/results/figures/a05_l1_fold_curve.png`); the N=256 states straddle the true fold either way.
- `converged = false` with `iterations = 1200` is a solve that hit the cap (40 restarts × 30).
- `newton`: `max_rel_field_diff_interior` is the meaningful comparison with C. `max_rel_field_diff` includes ghost cells, where the two codes differ (the C fills ψ's ghost cells differently), so it is large on purpose.
- `eig`: both matrices share a grid-level eigenvalue near 6.4e-5 at N=128. The fixed-ω curve going *below* it is the fold signal. `converged = false` means inverse iteration did not settle within 30 steps (close eigenvalues); the estimate is still the 30th iterate.

## Experiment matrix

Run from the repo root, one command at a time, in this order. Times are on the i7-13700K (8 faer threads). Peak memory is about 4 GB.

```bash
mkdir -p out/nearfold/results
cargo run --release --manifest-path ng/spikes/nearfold/Cargo.toml -- linear --manifest out/nearfold/l1/manifest.csv --n 128 --preconditioner all --out out/nearfold/results/linear_n128.csv
cargo run --release --manifest-path ng/spikes/nearfold/Cargo.toml -- linear --manifest out/nearfold/l1/manifest.csv --n 256 --preconditioner all --out out/nearfold/results/linear_n256.csv
cargo run --release --manifest-path ng/spikes/nearfold/Cargo.toml -- stale --manifest out/nearfold/l1/manifest.csv --preconditioner bordered --out out/nearfold/results/stale.csv
cargo run --release --manifest-path ng/spikes/nearfold/Cargo.toml -- newton --manifest out/nearfold/l1/manifest.csv --preconditioner bordered --out out/nearfold/results/newton.csv
cargo run --release --manifest-path ng/spikes/nearfold/Cargo.toml -- eig --manifest out/nearfold/l1/manifest.csv --n 128 --out out/nearfold/results/eig.csv
```

| Command | Expected time |
|---|---|
| `linear --n 128` | 8 minutes (block Jacobi and Gauss–Seidel hit the 1200-iteration cap, about 30 s each) |
| `linear --n 256` | 25 minutes (the capped solves take about 3 minutes each) |
| `stale` | 2 minutes |
| `newton` | 1 minute |
| `eig` | 2 minutes |

`stale` and `newton` use only the bordered preconditioner: block Jacobi and Gauss–Seidel do not converge even on fresh systems (see `linear`), and a Newton run on top of them would take hours.

Then the figures (they go to `docs/results/figures/`):

```bash
uv run ng/spikes/nearfold/plot_results.py --linear out/nearfold/results/linear_n128.csv out/nearfold/results/linear_n256.csv --eig out/nearfold/results/eig.csv --out-dir docs/results/figures
```

And any CSV as a Markdown table:

```bash
python3 ng/spikes/nearfold/csv_to_md.py out/nearfold/results/linear_n128.csv
```
