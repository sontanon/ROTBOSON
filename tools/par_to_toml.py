"""Convert legacy libconfig ``.par`` parameter files to TOML ``.toml``.

The legacy ROTBOSON parameter files were parsed with libconfig, which silently
ignored unknown keys. Phase 2 replaces libconfig with a strict TOML parser
(tomlc99) that rejects unknown keys, so this converter:

  * strips libconfig's optional trailing ``;`` statement terminators,
  * drops keys that the parser never read (dead keys: ``*BoundOrder``,
    ``dirname``), and
  * preserves every remaining numeric/string literal verbatim so the parsed
    values are bit-for-bit identical to the old files.

Usage:
    uv run tools/par_to_toml.py <file.par> [<file.par> ...]

With ``--in-place`` the original ``.par`` files are deleted after conversion.
"""

import argparse
import re
import sys
from pathlib import Path

# Keys that were present in the historical files but never read by the parser
# (libconfig silently ignored them; strict TOML validation would reject them).
DEAD_KEYS = {
    "alphaBoundOrder",
    "betaBoundOrder",
    "hBoundOrder",
    "aBoundOrder",
    "phiBoundOrder",
    "dirname",
}

# Matches "key = value" (with optional trailing ';').
KEY_VALUE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*=\s*)(.*)$")


def convert_line(line: str) -> tuple[str, str | None]:
    """Return (kind, text) where kind is 'comment', 'blank', 'key', or 'drop'."""
    stripped = line.strip()
    if not stripped:
        return "blank", None
    if stripped.startswith("#"):
        return "comment", line

    m = KEY_VALUE.match(line)
    if not m:
        # Not a key/value line (e.g. a stray comment fragment). Preserve it.
        return "comment", line

    key = m.group(2)
    value = m.group(4).rstrip()
    if value.endswith(";"):
        value = value[:-1].rstrip()
    if not value:
        return "comment", line  # e.g. "w0 =" placeholder; keep as-is

    if key in DEAD_KEYS:
        return "drop", key

    return "key", f"{key} = {value}\n"


def convert_file(par: Path, out: Path) -> list[str]:
    dropped: list[str] = []
    lines = par.read_text().splitlines()
    out_lines: list[str] = []
    for line in lines:
        kind, text = convert_line(line)
        if kind == "drop":
            dropped.append(text or "?")
            continue
        if kind == "key":
            assert text is not None
            out_lines.append(text)
        elif kind == "comment":
            out_lines.append(line.rstrip() + "\n")
        elif kind == "blank":
            out_lines.append("\n")
    out.write_text("".join(out_lines))
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help=".par files to convert")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="delete the original .par files after converting",
    )
    args = parser.parse_args()

    for par in args.files:
        if par.suffix != ".par":
            print(f"skip {par} (not a .par file)", file=sys.stderr)
            continue
        out = par.with_suffix(".toml")
        dropped = convert_file(par, out)
        print(f"{par} -> {out}")
        if dropped:
            print(f"    dropped dead keys: {', '.join(sorted(set(dropped)))}")
        if args.in_place:
            par.unlink()


if __name__ == "__main__":
    sys.exit(main())
