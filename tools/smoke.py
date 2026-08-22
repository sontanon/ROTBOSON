"""Smoke test: build ROTBOSON, run a parameter file, and extract key observables.

Usage (run from repo root with a uv-managed environment):
    uv run tools/smoke.py <configfile> [--skip-build] [--jobs N]

Requires MKLROOT (or an activated oneAPI environment) and the executables'
runtime libs on the library path. On Fedora:
    source /opt/intel/oneapi/setvars.sh

The parameter file must be run from out/ (ROTBOSON chdirs into the generated
directory). Output is written to out/ as before.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rotboson_io import extract_scalars, find_solution_dirs

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"

# CMake build dirs (single-config presets); the binary lives in the configured
# build tree. Keep support for the legacy Makefile binary at the repo root.
BUILD_PRESETS = ("release", "umfpack", "dev", "asan-ubsan")


def find_binary() -> Path:
    for preset in BUILD_PRESETS:
        candidate = REPO / "build" / preset / "ROTBOSON"
        if candidate.exists():
            return candidate
    legacy = REPO / "ROTBOSON"
    if legacy.exists():
        return legacy
    raise SystemExit(
        "ROTBOSON binary not found. Build with: cmake --preset release && "
        "cmake --build --preset release"
    )


def build(jobs: int) -> Path:
    print("[smoke] building ROTBOSON with CMake (release preset) ...")
    subprocess.run(["cmake", "--preset", "release"], cwd=REPO, check=True)
    subprocess.run(["cmake", "--build", "--preset", "release", f"-j{jobs}"], cwd=REPO, check=True)
    print("[smoke] build OK")
    return REPO / "build" / "release" / "ROTBOSON"


def run_par(binary: Path, parfile: Path) -> Path:
    par = parfile.resolve()
    print(f"[smoke] running ROTBOSON with {par.name} ...")
    proc = subprocess.run(
        [str(binary), str(par)],
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
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="path to the ROTBOSON executable (overrides discovery/build)",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.binary is not None:
        binary = args.binary.resolve()
        if not binary.exists():
            raise SystemExit(f"binary not found: {binary}")
    else:
        binary = find_binary() if args.skip_build else build(args.jobs)

    t0 = time.perf_counter()
    sol_dir = run_par(binary, args.parfile)
    elapsed = time.perf_counter() - t0
    scalars = extract_scalars(sol_dir)

    print(f"\n[smoke] ROTBOSON solve took {elapsed:.1f}s (wall)")

    print(f"\n[smoke] solution directory: {sol_dir.name}")
    print(f"[smoke] error_code = {scalars.get('error_code.asc', 'N/A')}")
    for key in (
        "w_f.asc",
        "M_ADM.asc",
        "M_Komar1.asc",
        "M_Komar2.asc",
        "J_Komar1.asc",
        "J_Komar2.asc",
        "phi_max.asc",
        "rr_phi_max.asc",
        "r99.asc",
    ):
        if key in scalars:
            print(f"[smoke] {key:16s} = {scalars[key]:+.16e}")
    print("\n[smoke] DONE")


if __name__ == "__main__":
    sys.exit(main())
