# NR-18 blocked: the C demo run refuses to overwrite the existing output directory

## What happened

Step 2 (Rust solve) passes. `out/demo/a0/rust.log` ends with:

```text
converged in 11 Newton iterations, 7.2 s
omega = 0.883256645852585
wrote ../out/demo/a0/rust/solution.vtr
wrote ../out/demo/a0/rust/solution.txt
```

Step 3 (C solve) fails. The exact command from the issue:

```bash
mkdir -p out/demo/a0/c && cd out/demo/a0/c && source /opt/intel/oneapi/setvars.sh > /dev/null && ../../../../build/release/ROTBOSON ../../../../configs/l1_fixedphi_n64.toml > run.log 2>&1; echo "exit $?"
```

prints `exit 4`, not `exit 0`. `out/demo/a0/c/run.log` contains:

```text
*** ... REFUSING: final output dir 'l=1,w=8.83257E-01,dr=2.50000E-01,N=0064' already exists; remove it first (the runbook workflow expects stale dirs to be cleaned externally).
```

The solver did converge and wrote a fresh solution, but into its placeholder directory
`out/demo/a0/c/l=1,w=X.XXXXXE-01,dr=2.50000E-01,N=0064/`; it then refused to rename that
directory to the final name.

## Why this is blocked

The target directory
`out/demo/a0/c/l=1,w=8.83257E-01,dr=2.50000E-01,N=0064/`
already existed before this issue's run (timestamp 16:39; this run started at 17:33).
The solver has no overwrite option (`src/main.c:622`); the only fix is to remove the stale
directory first.

`AGENTS.md`, the issue's "Out of scope", and the orchestrator's hard rules all forbid
deleting or moving anything in `out/`. Therefore neither the agent nor the worker may remove
the stale directory, and re-running step 3 cannot succeed. The failed run also overwrote
`out/demo/a0/c/run.log` and created the `l=1,w=X.XXXXXE-01,...` placeholder directory;
nothing in `out/` was deleted.

## What is needed to unblock

Remove (or move aside) the stale directory

```text
out/demo/a0/c/l=1,w=8.83257E-01,dr=2.50000E-01,N=0064/
```

externally, as the solver's message expects, then re-run step 3. The placeholder directory
`out/demo/a0/c/l=1,w=X.XXXXXE-01,dr=2.50000E-01,N=0064/` from the failed run can then also be
cleaned externally. Alternatively, explicitly authorize this agent to delete that one stale
run directory.

No commit beyond this file, no push of results, and no PR were made. Steps 4-8 were not
started because step 5's comparison depends on a fresh C solution in the issue's path.
