# Golden reference dataset

## What this is

The **publication-ready initial-data catalogue** for the paper

> Santiago Ontañón and Miguel Alcubierre, *Rotating boson stars using finite
> differences and global Newton methods*, Class. Quantum Grav. **38** 154003
> (2021), arXiv:2103.13993, DOI 10.1088/1361-6382/ac0b53.

originally stored on the backup drive
`Seagate Expansion Drive:/RBS Output/StandarizedOutput/` and copied here
verbatim on 2026-08-22 (archive name normalized from "StandarizedOutput" to
`data/golden/`; file contents and directory names unchanged).

## Contents

- 66 solution directories `l=<l>,w=<w>,dr=<dr>,N=<N>` covering l = 1..6:
  - l=1: 2 solutions, dr=8.0E-02, N=400
  - l=2: 3 solutions, dr=8.0E-02, N=400
  - l=3: 14 solutions, dr=1.6E-01, N=200
  - l=4: 24 solutions, dr=1.6E-01, N=200
  - l=5: 13 solutions, dr=1.6E-01, N=200
  - l=6: 16 solutions, dr=1.6E-01, N=200
- Every solution directory contains 70 files: final/initial/seed field data
  (`log_alpha_*`, `beta_*`, `log_h_*`, `log_a_*`, `psi_*`, `lambda_*`),
  Newton residuals/updates (`f0..f5`, `du0..du5`), spherical-coordinate
  interpolations (`sph_*`), integrated quantities (`M_ADM`, `M_Komar1/2`,
  `M_Schwarz`, `J_Komar1/2`, `GRV2`, `GRV3`, `r99`, `rr_phi_max`,
  `ergoregion_flag`, `phi_max`), solver history (`norm_du*`, `lambda*`,
  `Theta`, `mu*`), grids (`r.asc`, `z.asc`), and `error_code.asc`.
  **All 66 runs have `error_code.asc == 0` (converged).**
- 6 parameter-file templates `l=<l>,w=X.XXXXXE-01,dr=<dr>,N=<N>.par`
  (seeding from `UniversalOutput` on the same backup drive).
- The two executables used to produce the data: `ROTBOSON`, `ROTBOSON_dbg`
  (dated 2020-10-30).

## Provenance

- Generated 2020-10-30 with ROTBOSON built from the `Sweep`-era source,
  which is merged into the current `master` (Regularization-2 + Sweep).
- The upstream `master` tag `pre-refactor` (commit f156eea) is the reference
  source baseline for regenerating this dataset.
- Toolchain at generation time: Linux (ICN UNAM cluster / Windows WSL),
  Intel MKL (PARDISO sparse solver), libconfig, OpenMP.
- Integrity: `MANIFEST.sha256` contains SHA-256 checksums of every file
  (generated at copy time, 2026-08-22).

## Usage (regression tests)

Phase 0 validation chain (see `PLAN.md`):

1. Regenerate a selection of these solutions with the current source
   (par files adjusted to point `readInitialData` at
   `RBS Output/UniversalOutput` on the backup drive, which is the same
   seed data used originally).
2. Compare regenerated outputs (field profiles, `w_f`, `M_Komar*`,
   `J_Komar*`, `phi_max`, `rr_phi_max`) against the archived files here.
3. Cross-check against `data/summaries/l={1..6}.asc` (Catalogue2 summary
   tables) and the published tables/figures in arXiv:2103.13993.

Note: this directory is intentionally **not** tracked by git (4.6 GB);
`MANIFEST.sha256` guarantees integrity. The backup drive remains the
authoritative archive.
