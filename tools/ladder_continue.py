"""Continue a fixedPhi scale ladder from a previous solution (Phase 0 validation).

The archived golden l=2 par template is ONE step of a scale_u4=1.125 ladder.
This tool re-applies the template, seeding from the previous solution, until
the converged w reaches a target (or max_steps is hit).

Usage:
    uv run tools/ladder_continue.py <seed_solution_dir> --target 0.720859 --max-steps 12
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rotboson_io import find_solution_dirs, read_scalar

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
BIN = REPO / "ROTBOSON"
TEMPLATE = REPO / "data/params/regeneration/l=2,validate.toml"
LOGFILE = OUT / "ladder.log"

SEED_FILES = [
    "log_alpha_f.asc",
    "beta_f.asc",
    "log_h_f.asc",
    "log_a_f.asc",
    "psi_f.asc",
    "lambda_f.asc",
    "w_f.asc",
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOGFILE.open("a") as fh:
        fh.write(msg + "\n")


def make_par(seed_dir: Path, out_par: Path) -> None:
    text = TEMPLATE.read_text()
    for f in SEED_FILES:
        text = re.sub(rf'"[^"]*{re.escape(f)}"', f'"{seed_dir / f}"', text)
    out_par.write_text(text)


def run_step(par: Path, step: int) -> Path | None:
    # Snapshot pre-existing solution dirs so we can identify the newly
    # produced one (dirs sorted by name are NOT ordered by creation).
    before = set(find_solution_dirs(OUT))
    for d in OUT.glob("l=2,w=X.XXXXXE-01,dr=8.00000E-02,N=0400"):
        shutil.rmtree(d)
    step_log = OUT / f"ladder_step{step}.log"
    with step_log.open("w") as lf:
        proc = subprocess.run(
            [str(BIN), str(par)],
            cwd=OUT,
            stdin=subprocess.PIPE,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    log(f"  step {step}: ROTBOSON exit={proc.returncode}, full log -> {step_log.name}")
    if proc.returncode != 0:
        log(f"  step {step}: ROTBOSON failed (see {step_log.name})")
        return None
    new_dirs = [d for d in find_solution_dirs(OUT) if d not in before]
    return new_dirs[-1] if new_dirs else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_dir", type=Path)
    parser.add_argument(
        "--target", type=float, required=True, help="stop when w <= target (e.g. 0.720859)"
    )
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    seed = args.seed_dir.resolve()
    if not seed.is_dir():
        raise SystemExit(f"seed dir not found: {seed}")

    if LOGFILE.exists():
        LOGFILE.unlink()
    log(f"# ladder continuation from {seed.name}")

    for step in range(1, args.max_steps + 1):
        par = OUT / f"ladder_step{step}.toml"
        make_par(seed, par)
        log(f"step {step}: seeding from {seed.name} ...")
        new = run_step(par, step)
        if new is None:
            log("ladder stopped: run failed")
            return 1
        w = read_scalar(new / "w_f.asc")
        log(f"step {step}: -> {new.name}  w={w:.6e}")
        if w <= args.target:
            log(f"reached target: {new.name}")
            return 0
        seed = new
    log("max steps reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
