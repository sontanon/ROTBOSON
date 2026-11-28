"""Print a CSV file as a Markdown table (stdlib only).

Usage:
    python3 ng/spikes/nearfold/csv_to_md.py results.csv
"""

import csv
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 1
    print("| " + " | ".join(rows[0]) + " |")
    print("|" + "---|" * len(rows[0]))
    for row in rows[1:]:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
