"""Smoke test: build ROTBOSON, run a par file, and extract key observables.

Usage (run from repo root with a uv-managed environment):
    uv run tools/smoke.py <parfile> [--skip-build] [--jobs N]

Requires MKLROOT and LIBCONFIGROOT (or an activated oneAPI environment) and
the executables' runtime libs on the library path. On Fedora:
    LIBCONFIGROOT=/usr  and  source /opt/intel/oneapi/setvars.sh

The par file must be run from out/ (ROTBOSON chdirs into the generated
directory). Output is written to out/ as before.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rotboson_io import extract_scalars, find_solution_dirs

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
BIN = REPO / "ROTBOSON"


def build(jobs: int) -> None:
    print(f"[smoke] building ROTBOSON (make -j{jobs} all) ...")
    subprocess.run(["make", f"-j{jobs}", "all"], cwd=REPO, check=True)
    print("[smoke] build OK")


def run_par(parfile: Path) -> Path:
    par = parfile.resolve()
    print(f"[smoke] running ROTBOSON with {par.name} ...")
    proc = subprocess.run(
        [str(BIN), str(par)],
        cwd=OUT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = OUT / "smoke_run.log"
    log.write_text(proc.stdout)
    print(f"[smoke] ROTBOSON exited with code {proc.returncode}; log -> {log}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        print(tail)
        raise SystemExit("ROTBOSON run failed")
    dirs = find_solution_dirs(OUT)
    if not dirs:
        raise SystemExit("no solution directory found in out/")
    return dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parfile", type=Path, help="path to parameter file")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if not args.skip_build:
        build(args.jobs)
    sol_dir = run_par(args.parfile)
    scalars = extract_scalars(sol_dir)

    print(f"\n[smoke] solution directory: {sol_dir.name}")
    print(f"[smoke] error_code = {scalars.get('error_code.asc', 'N/A')}")
    for key in ("w_f.asc", "M_ADM.asc", "M_Komar1.asc", "M_Komar2.asc",
                "J_Komar1.asc", "J_Komar2.asc", "phi_max.asc",
                "rr_phi_max.asc", "r99.asc"):
        if key in scalars:
            print(f"[smoke] {key:16s} = {scalars[key]:+.16e}")
    print("\n[smoke] DONE")


if __name__ == "__main__":
    sys.exit(main())
