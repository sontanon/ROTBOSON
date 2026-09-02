"""Strip notebook outputs/counts in-place for version control.

The full-output originals remain archived on the backup drive
(Seagate Expansion Drive:/RBS). Run:  uv run tools/strip_notebooks.py
"""

import sys
from pathlib import Path

import nbformat
from logsetup import configure, get_logger

NOTEBOOKS = Path(__file__).resolve().parent.parent / "derivations" / "notebooks"


def main() -> int:
    configure()
    logger = get_logger(__name__)
    total = stripped = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
        if changed:
            nbformat.write(nb, nb_path)
            stripped += 1
        total += 1
    logger.info("stripped %d/%d notebooks in %s", stripped, total, NOTEBOOKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
