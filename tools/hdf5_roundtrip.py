"""Round-trip an HDF5 solution back to the legacy .asc layout.

Phase 5 writes a single self-describing solution.h5 per solve. This tool:
  - exports solution.h5 to .asc files (the exact legacy format), and/or
  - compares solution.h5 against a legacy .asc reference directory.

Usage:
    uv run tools/hdf5_roundtrip.py <solution_dir> --out <asc_dir>
    uv run tools/hdf5_roundtrip.py <solution_dir> --ref <ascii_solution_dir>
"""

import argparse
import sys
from pathlib import Path

from rotboson_io import HDF5_FILENAME, compare_hdf5_to_ascii, hdf5_to_asc, read_hdf5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sol_dir", type=Path, help="directory containing solution.h5")
    parser.add_argument("--out", type=Path, help="export .asc files into this directory")
    parser.add_argument("--ref", type=Path, help="legacy .asc directory to compare against")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    h5 = args.sol_dir / HDF5_FILENAME
    if not h5.exists():
        raise SystemExit(f"not found: {h5}")

    datasets, attrs = read_hdf5(h5)
    print(f"{h5}: {len(datasets)} datasets, {len(attrs)} attributes")
    print(f"  git_hash = {attrs.get('git_hash')}")
    print(f"  output_backend = {attrs.get('output_backend')}")
    print(
        f"  w (attr) = {attrs.get('w0')}  M_KOMAR = {attrs.get('M_KOMAR')}  "
        f"J_KOMAR = {attrs.get('J_KOMAR')}"
    )

    ok = True
    if args.out:
        written = hdf5_to_asc(h5, args.out)
        print(f"exported {len(written)} .asc files -> {args.out}")

    if args.ref:
        if not args.ref.is_dir():
            raise SystemExit(f"reference dir not found: {args.ref}")
        ok, lines = compare_hdf5_to_ascii(args.sol_dir, args.ref, rtol=args.rtol, atol=args.atol)
        print("\n".join(lines))
        verdict = "PASS" if ok else "FAIL"
        print(f"\nround-trip verdict: {verdict}")
        return 0 if ok else 1

    if not args.out and not args.ref:
        print("no --out or --ref given; nothing to do")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
