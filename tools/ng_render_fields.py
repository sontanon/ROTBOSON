"""Render the six-field snapshot from an RBS-NG solution.h5.

Workstream V day-one visualization: a 2x3 panel figure (log_alpha,
log_h, log_a, beta, psi, lambda over the (r, z) grid) drawn from the
Rust-written solution.h5 through tools/rotboson_io.py — the same reader
the legacy Python tooling uses, proving the NG output is loadable
unchanged. The l=1 N=64 smoke snapshot is a static final state (no
time dependence yet); the first evolution-data render lands with N3.

Usage:
    uv run tools/ng_render_fields.py <solution.h5> [--output <png>]

Example (the SAN-55 deliverable):
    uv run tools/ng_render_fields.py out/n1_io_vtr/rust/solution.h5 \
        --output out/n1_io_vtr/render/ng_l1_smoke_n64_fields.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the deliverable is the PNG file

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from logsetup import configure, get_logger  # noqa: E402
from rotboson_io import read_hdf5  # noqa: E402

logger = get_logger(__name__)

# Panel order (workstream V spec): log_alpha, log_h, log_a, beta, psi, lambda.
PANELS: tuple[tuple[str, str, str], ...] = (
    ("log_alpha_f", r"$\log\alpha$", "log_alpha_f.asc"),
    ("log_h_f", r"$\log h$", "log_h_f.asc"),
    ("log_a_f", r"$\log a$", "log_a_f.asc"),
    ("beta_f", r"$\beta$", "beta_f.asc"),
    ("psi_f", r"$\psi$", "psi_f.asc"),
    ("lambda_f", r"$\lambda$", "lambda_f.asc"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path, help="RBS-NG solution.h5 (rotboson_io contract)")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: <solution dir>/ng_fields.png)",
    )
    args = parser.parse_args()
    configure()

    datasets, attrs = read_hdf5(args.solution)
    missing = [asc for _, _, asc in PANELS if asc not in datasets]
    if missing:
        raise SystemExit(f"{args.solution} lacks the field datasets: {missing}")

    # Rectilinear axes from the C-layout meshgrids: r varies along axis 0,
    # z along axis 1.
    r = np.asarray(datasets["r.asc"])[:, 0]
    z = np.asarray(datasets["z.asc"])[0, :]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, (_, label, asc) in zip(axes.flat, PANELS, strict=True):
        f = np.asarray(datasets[asc])
        # f is row-major IDX(i, j) (r rows, z columns); matplotlib wants
        # rows = y (z), columns = x (r).
        img = f.T
        vmin, vmax = float(img.min()), float(img.max())
        if vmin < 0.0 < vmax:
            # Signed field: diverging colormap centered at zero.
            lim = max(abs(vmin), abs(vmax))
            im = ax.pcolormesh(r, z, img, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        else:
            im = ax.pcolormesh(r, z, img, cmap="magma", vmin=vmin, vmax=vmax, shading="nearest")
        ax.set_title(f"{label}   [{vmin:+.3e}, {vmax:+.3e}]")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("$z$")
        fig.colorbar(im, ax=ax, shrink=0.85)

    w = float(np.ravel(datasets["w_f.asc"])[0]) if "w_f.asc" in datasets else float("nan")
    fig.suptitle(
        f"RBS-NG l={attrs.get('l')} smoke (N={attrs.get('NrTotal')} total, "
        f"$\\omega$ = {w:.5g}) — {attrs.get('output_backend')} {attrs.get('git_hash')}, "
        f"created {attrs.get('created')}",
        fontsize=13,
    )

    out = args.output or args.solution.parent / "ng_fields.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
