//! Kernel parity: generated Rust vs `SymPy` references (SAN-56 gate).
//!
//! The fixture (`tests/fixtures/kernel_parity.txt`) is produced by
//! `tools/generate_rust_kernels.py`: seeded random field states plus
//! `SymPy`-evaluated reference values for the interior residual, the
//! subtype Jacobian, and the boundary rows.  Plain `cargo test` runs this
//! without any Python present, so numeric parity holds everywhere the
//! workspace builds.
//!
//! The reference and the generated Rust evaluate the same mathematical
//! expressions in different association orders, so agreement is asserted
//! to the fixture tolerance (a few ULP, not bit-exactness).

use crate::generated::kernels::{
    jacobian_subtypes, residual_boundary, residual_interior, BoundaryPoint, JacobianPoint,
    ResidualPoint, GNUM, JAC_COLS,
};

/// One parsed fixture line: `key value...`.
struct Line {
    key: String,
    values: Vec<f64>,
}

fn parse_line(raw: &str) -> Option<Line> {
    let raw = raw.trim();
    if raw.is_empty() || raw.starts_with('#') {
        return None;
    }
    let mut parts = raw.split_whitespace();
    let key = parts.next()?.to_owned();
    let values = parts
        .map(|t| t.parse::<f64>().expect("fixture float"))
        .collect();
    Some(Line { key, values })
}

fn load_fixture() -> (f64, Vec<Line>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/kernel_parity.txt"
    );
    let text = std::fs::read_to_string(path).expect("fixture present");
    let lines: Vec<Line> = text.lines().filter_map(parse_line).collect();
    let tolerance = lines
        .iter()
        .find(|l| l.key == "tolerance")
        .expect("tolerance line")
        .values[0];
    (tolerance, lines)
}

/// Compare `got` against the fixture `expect`, returning the worst
/// normalized deviation `|got - expect| / max(1, |expect|)`.
fn worst_deviation(got: &[f64], expect: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for (g, e) in got.iter().zip(expect) {
        let dev = (g - e).abs() / e.abs().max(1.0);
        worst = worst.max(dev);
    }
    worst
}

/// The state count of a section header line (`residual 12` etc.).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn count_of(line: &Line) -> usize {
    let n = line.values[0];
    debug_assert!(n >= 0.0 && n.fract() == 0.0, "section count: {n}");
    n as usize
}

fn arr6(line: &Line) -> [f64; 6] {
    let v: Vec<f64> = line.values.clone();
    v.try_into().expect("six values")
}

#[test]
fn residual_matches_sympy_references() {
    let (tol, lines) = load_fixture();
    let mut idx = 0;
    while lines[idx].key != "residual" {
        idx += 1;
    }
    let count = count_of(&lines[idx]);
    idx += 1;
    let mut worst = 0.0_f64;
    for _ in 0..count {
        let u = arr6(&lines[idx]);
        let dr_u = arr6(&lines[idx + 1]);
        let dz_u = arr6(&lines[idx + 2]);
        let drr_u = arr6(&lines[idx + 3]);
        let dzz_u = arr6(&lines[idx + 4]);
        let dr_aux: Vec<f64> = lines[idx + 5].values.clone();
        let base = idx + 6;
        let scal = |i: usize| lines[base + i].values[0];
        let expect: Vec<f64> = lines[base + 9].values.clone();
        idx += 16;
        let p = ResidualPoint {
            u,
            dr_u,
            dz_u,
            drr_u,
            dzz_u,
            dr_aux: [dr_aux[0], dr_aux[1]],
            dr: scal(0),
            dz: scal(1),
            r: scal(2),
            l: scal(3),
            m: scal(4),
            w: scal(5),
            q1: scal(6),
            q2: scal(7),
            rescale: scal(8),
        };
        let got = residual_interior(&p);
        let dev = worst_deviation(&got, &expect);
        worst = worst.max(dev);
        assert!(
            dev <= tol,
            "residual deviation {dev:.3e} > {tol:.0e} at state:\n{p:#?}"
        );
    }
    println!("kernel_parity residual: worst normalized deviation {worst:.3e}");
}

#[test]
fn jacobian_matches_sympy_references() {
    let (tol, lines) = load_fixture();
    let mut idx = 0;
    while lines[idx].key != "jacobian" {
        idx += 1;
    }
    let count = count_of(&lines[idx]);
    idx += 1;
    let mut worst = 0.0_f64;
    for _ in 0..count {
        let u = arr6(&lines[idx]);
        let d_ru = arr6(&lines[idx + 1]);
        let d_zu = arr6(&lines[idx + 2]);
        let d_rru = arr6(&lines[idx + 3]);
        let d_zzu = arr6(&lines[idx + 4]);
        let base = idx + 5;
        let scal = |i: usize| lines[base + i].values[0];
        let expect: Vec<f64> = lines[base + 9].values.clone();
        idx += 15;
        let p = JacobianPoint {
            u,
            d_ru,
            d_zu,
            d_rru,
            d_zzu,
            ri: scal(0),
            dr: scal(1),
            dz: scal(2),
            l: scal(3),
            m: scal(4),
            w: scal(5),
            dw_dxi: scal(6),
            q1: scal(7),
            q2: scal(8),
        };
        let got = jacobian_subtypes(&p);
        let flat: Vec<f64> = got.iter().flat_map(|row| row.iter().copied()).collect();
        assert_eq!(flat.len(), GNUM * JAC_COLS);
        let dev = worst_deviation(&flat, &expect);
        worst = worst.max(dev);
        assert!(
            dev <= tol,
            "jacobian deviation {dev:.3e} > {tol:.0e} at state:\n{p:#?}"
        );
    }
    println!("kernel_parity jacobian: worst normalized deviation {worst:.3e}");
}

#[test]
fn boundary_matches_sympy_references() {
    let (tol, lines) = load_fixture();
    let mut idx = 0;
    while lines[idx].key != "boundary" {
        idx += 1;
    }
    let count = count_of(&lines[idx]);
    idx += 1;
    let mut worst = 0.0_f64;
    for _ in 0..count {
        let u = arr6(&lines[idx]);
        let dr_u = arr6(&lines[idx + 1]);
        let dz_u = arr6(&lines[idx + 2]);
        let base = idx + 3;
        let scal = |i: usize| lines[base + i].values[0];
        let expect: Vec<f64> = lines[base + 8].values.clone();
        idx += 12;
        let p = BoundaryPoint {
            u,
            dr_u,
            dz_u,
            r: scal(0),
            z: scal(1),
            dr: scal(2),
            dz: scal(3),
            l: scal(4),
            m: scal(5),
            w: scal(6),
            rescale: scal(7),
        };
        let got = residual_boundary(&p);
        let dev = worst_deviation(&got, &expect);
        worst = worst.max(dev);
        assert!(
            dev <= tol,
            "boundary deviation {dev:.3e} > {tol:.0e} at state:\n{p:#?}"
        );
    }
    println!("kernel_parity boundary: worst normalized deviation {worst:.3e}");
}

#[test]
fn flat_state_residuals_vanish() {
    // All fields, derivatives, and auxiliaries zero: every bracket
    // collapses (phi = 0, a2_r = 1) and the rows are exactly zero.
    let p = ResidualPoint {
        u: [0.0; 6],
        dr_u: [0.0; 6],
        dz_u: [0.0; 6],
        drr_u: [0.0; 6],
        dzz_u: [0.0; 6],
        dr_aux: [0.0; 2],
        dr: 0.125,
        dz: 0.125,
        r: 1.0,
        l: 1.0,
        m: 1.0,
        w: 0.5,
        q1: 1.0,
        q2: 1.0,
        rescale: -1.0,
    };
    assert_eq!(residual_interior(&p), [0.0; 6]);
}

#[test]
fn jacobian_laplacian_coefficients_match_the_discretization() {
    // Structural cross-check against the equations: row 0 (the alpha /
    // lapse equation) contains dr^2*Drr(u1) + dz^2*Dzz(u1), so after the
    // step folding its own-field subtype columns are exactly dzodr (col
    // 3) and drodz (col 4) at any state -- the CSR scatter's leading
    // diagonal block in csr_vars.c.
    let p = JacobianPoint {
        u: [0.3, -0.2, 0.1, 0.05, 0.4, -0.3],
        d_ru: [0.2, -0.1, 0.3, 0.0, -0.4, 0.1],
        d_zu: [0.1, 0.2, -0.3, 0.4, 0.0, -0.2],
        d_rru: [0.5, 0.0, -0.2, 0.1, 0.3, -0.1],
        d_zzu: [-0.2, 0.4, 0.0, 0.3, -0.5, 0.2],
        ri: 12.5,
        dr: 0.125,
        dz: 0.0625,
        l: 2.0,
        m: 1.0,
        w: 0.8,
        dw_dxi: 0.2,
        q1: 1.0,
        q2: 1.0,
    };
    let j = jacobian_subtypes(&p);
    let dzodr = p.dz / p.dr;
    let drodz = p.dr / p.dz;
    assert!((j[0][3] - dzodr).abs() < 1e-14, "dRR column: {}", j[0][3]);
    assert!((j[0][4] - drodz).abs() < 1e-14, "dZZ column: {}", j[0][4]);
}
