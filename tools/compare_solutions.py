"""Compare a regenerated solution directory against a golden reference.

Usage:
    uv run tools/compare_solutions.py <ref_dir> <new_dir> [--rtol 1e-10]
"""

import argparse
import sys
from pathlib import Path

from rotboson_io import compare_fields, compare_scalars, extract_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="absolute tolerance fallback for near-zero quantities",
    )
    args = parser.parse_args()

    ref, new = args.ref_dir, args.new_dir
    if not ref.is_dir():
        raise SystemExit(f"ref dir not found: {ref}")
    if not new.is_dir():
        raise SystemExit(f"new dir not found: {new}")

    ok = True

    print("Scalar observables:")
    ok_s, lines = compare_scalars(
        extract_scalars(ref), extract_scalars(new), rtol=args.rtol, atol=args.atol
    )
    ok &= ok_s
    print("\n".join(lines))

    print("\nField profiles:")
    ok_f, lines = compare_fields(ref, new, rtol=args.rtol, atol=args.atol)
    ok &= ok_f
    print("\n".join(lines))

    verdict = "PASS" if ok else "FAIL"
    print(f"\nComparison verdict: {verdict} (rtol={args.rtol}, atol={args.atol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
