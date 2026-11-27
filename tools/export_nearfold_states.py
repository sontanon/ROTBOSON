"""Export near-fold C solutions from a campaign to plain text files.

Usage:
    uv run tools/export_nearfold_states.py --campaign out/campaigns/runbook-l1 --out out/nearfold/l1
    uv run tools/export_nearfold_states.py --verify out/nearfold/l1
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

from rotboson_io import read_hdf5

FIELDS = ["log_alpha_f", "beta_f", "log_h_f", "log_a_f", "psi_f", "lambda_f"]
HEADER_KEYS = [
    "nr_total",
    "nz_total",
    "ghost",
    "dr",
    "dz",
    "l",
    "m",
    "pin_r",
    "pin_z",
    "psi0",
    "omega",
]
N128_TARGETS = [0.30, 0.20, 0.10, 0.05, 0.03]


def attr_int(attrs: dict[str, object], key: str) -> int:
    """An HDF5 attribute as an int."""
    return int(cast("Any", attrs[key]))


def attr_float(attrs: dict[str, object], key: str) -> float:
    """An HDF5 attribute as a float."""
    return float(cast("Any", attrs[key]))


def select_steps(campaign: Path) -> list[dict[str, Any]]:
    """Return the chosen steps (dicts from state.json plus keys "h5" and "dist"), sorted."""
    state = json.loads((campaign / "state.json").read_text())
    summary = json.loads((campaign / "summary.json").read_text())
    psi0_star = float(summary["psi0"])

    kept: list[dict[str, Any]] = []
    for step in state["steps"]:
        if step.get("exit_code") != 0:
            continue
        h5 = campaign / Path(step["sol_dir"]).name / "solution.h5"
        if not h5.exists():
            continue
        datasets, attrs = read_hdf5(h5)
        if attr_int(attrs, "fixedPhi") != 1:
            continue
        dist = psi0_star - float(step["psi0"])
        new_step = dict(step)
        new_step["h5"] = h5
        new_step["dist"] = dist
        new_step["omega"] = float(datasets["w_f.asc"].reshape(-1)[0])
        kept.append(new_step)

    n256 = [s for s in kept if s["N"] == 256]
    n128 = [s for s in kept if s["N"] == 128]

    chosen_n128: list[dict[str, Any]] = []
    seen: set[int] = set()
    for target in N128_TARGETS:
        if not n128:
            continue
        best = min(n128, key=lambda s: abs(s["dist"] - target))
        if best["i"] not in seen:
            seen.add(best["i"])
            chosen_n128.append(best)
    if n128:
        largest = max(n128, key=lambda s: float(s["psi0"]))
        if largest["i"] not in seen:
            seen.add(largest["i"])
            chosen_n128.append(largest)

    chosen = n256 + chosen_n128
    chosen.sort(key=lambda s: (s["N"], float(s["psi0"])))
    return chosen


def write_state(step: dict[str, Any], out_dir: Path) -> Path:
    """Write one solution to its text file and return the path."""
    datasets, attrs = read_hdf5(step["h5"])
    h5_path = Path(step["h5"])

    nr_total = attr_int(attrs, "NrTotal")
    nz_total = attr_int(attrs, "NzTotal")
    ghost = attr_int(attrs, "ghost")
    dr = attr_float(attrs, "dr")
    dz = attr_float(attrs, "dz")
    l = attr_int(attrs, "l")
    m = attr_float(attrs, "m")
    pin_r = attr_int(attrs, "fixedPhiR")
    pin_z = attr_int(attrs, "fixedPhiZ")
    psi0 = float(step["psi0"])
    omega = float(datasets["w_f.asc"].reshape(-1)[0])

    name = f"l1_N{step['N']:04d}_step{step['i']:04d}.txt"
    path = out_dir / name
    lines = [
        "# exported by tools/export_nearfold_states.py",
        f"# source: {h5_path}",
        f"nr_total {nr_total}",
        f"nz_total {nz_total}",
        f"ghost {ghost}",
        f"dr {repr(dr)}",
        f"dz {repr(dz)}",
        f"l {l}",
        f"m {repr(m)}",
        f"pin_r {pin_r}",
        f"pin_z {pin_z}",
        f"psi0 {repr(psi0)}",
        f"omega {repr(omega)}",
    ]
    for field in FIELDS:
        lines.append(f"field {field}")
        arr = datasets[field + ".asc"].reshape(-1)
        for x in arr:
            lines.append(repr(float(x)))
    path.write_text("\n".join(lines) + "\n")
    return path


def verify(out_dir: Path) -> int:
    """Check every file listed in manifest.csv; return 0 if all good, else 1."""
    manifest = out_dir / "manifest.csv"
    with manifest.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        path = out_dir / row["file"]
        try:
            text = path.read_text()
        except OSError as e:
            print(f"{row['file']}: cannot read: {e}")
            return 1
        lines = text.splitlines()
        # Strip the two comment header lines.
        if len(lines) < 2 or not lines[0].startswith("#") or not lines[1].startswith("#"):
            print(f"{row['file']}: missing comment header lines")
            return 1
        body = lines[2:]
        # Header keys in order.
        header: dict[str, str] = {}
        i = 0
        while i < len(body) and not body[i].startswith("field "):
            line = body[i]
            key, _, value = line.partition(" ")
            header[key] = value
            i += 1
        if list(header.keys()) != HEADER_KEYS:
            print(f"{row['file']}: header keys mismatch: {list(header.keys())}")
            return 1
        try:
            nr_total = int(header["nr_total"])
            nz_total = int(header["nz_total"])
        except ValueError as e:
            print(f"{row['file']}: bad nr_total/nz_total: {e}")
            return 1
        expected = nr_total * nz_total

        # Field sections.
        sections: list[tuple[str, list[float]]] = []
        while i < len(body):
            line = body[i]
            if not line.startswith("field "):
                print(f"{row['file']}: expected 'field ...' at line {i + 1}, got {line!r}")
                return 1
            name = line[len("field ") :]
            i += 1
            values: list[float] = []
            while i < len(body) and not body[i].startswith("field "):
                try:
                    values.append(float(body[i]))
                except ValueError as e:
                    print(f"{row['file']}: bad value in field {name}: {e}")
                    return 1
                i += 1
            sections.append((name, values))
        if [name for name, _ in sections] != FIELDS:
            print(f"{row['file']}: field order mismatch: {[n for n, _ in sections]}")
            return 1
        for name, values in sections:
            if len(values) != expected:
                print(f"{row['file']}: field {name} has {len(values)} values, expected {expected}")
                return 1
            for v in values:
                if not math.isfinite(v):
                    print(f"{row['file']}: non-finite value in field {name}")
                    return 1
        print(f"OK {row['file']}")
    print(f"verified {len(rows)} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, metavar="DIR")
    args = parser.parse_args()
    if args.verify:
        return verify(args.verify)
    if not args.campaign or not args.out:
        parser.error("need --campaign and --out (or --verify DIR)")

    steps = select_steps(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, dict[str, Any]]] = []
    for step in steps:
        path = write_state(step, args.out)
        paths.append((path, step))
        print(f"{path.name} N={step['N']} psi0={step['psi0']} dist={step['dist']}")

    with (args.out / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "step", "N", "dr", "psi0", "omega", "dist"])
        for path, step in paths:
            datasets, attrs = read_hdf5(step["h5"])
            dr = attr_float(attrs, "dr")
            omega = float(datasets["w_f.asc"].reshape(-1)[0])
            writer.writerow(
                [
                    path.name,
                    step["i"],
                    step["N"],
                    repr(float(dr)),
                    repr(float(step["psi0"])),
                    repr(float(omega)),
                    repr(float(step["dist"])),
                ]
            )
    print(f"wrote {len(paths)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
