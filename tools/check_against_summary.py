"""Cross-check regenerated solutions against the published summary tables.

The files data/summaries/l={1..6}.asc are the Catalogue2 tables behind the
paper's mass/angular-momentum-vs-frequency figures. Columns:
    psi(0) M_Komar J_Komar w max(phi) rr(max(phi))

For each solution directory, find the summary row whose w matches and report
the relative differences in M_Komar and J_Komar (surface Komar quantities,
i.e. *_Komar1).

Usage:
    uv run tools/check_against_summary.py <solution_dir> [<solution_dir> ...]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from rotboson_io import read_1d, read_scalar

SUMMARIES = Path(__file__).resolve().parent.parent / "data" / "summaries"


def load_summary(l: int) -> np.ndarray:
    """Return array with columns [psi0, M_Komar, J_Komar, w, phi_max, rr_phi_max]."""
    path = SUMMARIES / f"l={l}.asc"
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sol_dirs", nargs="+", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument(
        "--w_tol",
        type=float,
        default=1e-6,
        help="max |dw| to consider a table row as matching (else report no-match)",
    )
    args = parser.parse_args()

    ok = True
    for sol in args.sol_dirs:
        m = re.match(r"l=(\d+),w=([\d.Ee+-]+)", sol.name)
        if not m:
            print(f"SKIP (unparseable name): {sol.name}")
            continue
        l = int(m.group(1))
        w = read_scalar(sol / "w_f.asc")
        mk = float(read_1d(sol / "M_Komar1.asc")[-1])
        jk = float(read_1d(sol / "J_Komar1.asc")[-1])

        tab = load_summary(l)
        j = int(np.argmin(np.abs(tab[:, 3] - w)))
        w_tab, mk_tab, jk_tab = tab[j, 3], tab[j, 1], tab[j, 2]
        dw = abs(w - w_tab)
        if dw > args.w_tol:
            print(
                f"l={l} w={w:.6e}  NO MATCHING ROW in l={l}.asc (closest w={w_tab:.6e}, dW={dw:.2e})\n"
            )
            continue
        dmk = abs(mk - mk_tab) / max(abs(mk_tab), 1e-300)
        djk = abs(jk - jk_tab) / max(abs(jk_tab), 1e-300)
        status = "PASS" if dmk <= args.rtol and djk <= args.rtol else "FAIL"
        ok &= status == "PASS"
        print(f"l={l} w={w:.6e}  table row w={w_tab:.6e} (dW={dw:.2e})")
        print(f"    M_Komar: regen={mk:+.16e}  table={mk_tab:+.16e}  rel={dmk:.2e}")
        print(f"    J_Komar: regen={jk:+.16e}  table={jk_tab:+.16e}  rel={djk:.2e}  {status}\n")

    print(f"verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
