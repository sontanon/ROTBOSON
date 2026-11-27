//! VTR structural + value self-check on the real N=64 smoke snapshot.
//!
//! No `ParaView` in this container: the gate is (a) the rectilinear-grid
//! schema — declaration, extents matching the grid, one `Piece`,
//! `PointData` arrays in VTK x-fastest order, coordinate axes, balanced
//! tags — and (b) an exact read-back of every value against the slice
//! parity fixture. The host-side `ParaView` eyeball check (six panels,
//! no transposition) is recorded in `ng/rbs-io/README.md`.

mod common;

use std::collections::BTreeSet;

use rbs_io::vtr::VtrWriter;
use rbs_io::FieldWriter;

/// Values of one `<DataArray ...>` element, in file order.
fn array_values(xml: &str, tag: &str, name: Option<&str>) -> Vec<f64> {
    let open = match name {
        Some(n) => format!(
            "<{tag} type=\"Float64\" Name=\"{n}\" NumberOfComponents=\"1\" format=\"ascii\">"
        ),
        None => format!("<{tag} type=\"Float64\" NumberOfComponents=\"1\" format=\"ascii\">"),
    };
    let start = xml
        .find(&open)
        .unwrap_or_else(|| panic!("{tag}/{name:?} present"))
        + open.len();
    let end = start
        + xml[start..]
            .find(&format!("</{tag}>"))
            .expect("element closes");
    xml[start..end]
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("ascii float"))
        .collect()
}

/// Inverse of the writer's x-fastest emission: file order -> IDX(i, j).
fn to_row_major(vtk: &[f64], nr: usize, nz: usize) -> Vec<f64> {
    let mut out = vec![0.0; nr * nz];
    for (i, row) in out.chunks_exact_mut(nz).enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = vtk[i + j * nr];
        }
    }
    out
}

/// Every open tag has exactly one matching close, in nesting order.
fn assert_balanced(xml: &str) {
    let mut stack: Vec<&str> = Vec::new();
    for (i, b) in xml.bytes().enumerate() {
        if b != b'<' || xml[i..].starts_with("<?xml") {
            continue;
        }
        let end = xml[i..].find('>').expect("tag ends") + i;
        let inner = &xml[i + 1..end];
        if let Some(name) = inner.strip_prefix('/') {
            assert_eq!(
                stack.pop(),
                Some(name),
                "close </{name}> matches the innermost open"
            );
        } else if !inner.ends_with('/') {
            stack.push(inner.split_whitespace().next().expect("tag name"));
        }
    }
    assert!(stack.is_empty(), "all tags closed");
}

#[test]
fn smoke_vtr_selfcheck() {
    let snap = common::load_smoke();
    let (nr, nz) = (snap.grid.nr, snap.grid.nz);
    assert_eq!((nr, nz), (68, 68));

    let dir = std::env::temp_dir().join(format!("rbs-io-vtr-self-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("solution.vtr");
    let mut w = VtrWriter::new(&path);
    w.write_snapshot(&snap.grid, &snap.fields)
        .expect("write vtr");
    let xml = std::fs::read_to_string(&path).expect("read vtr");
    let _ = std::fs::remove_dir_all(&dir);

    // --- schema ---
    assert!(
        xml.starts_with("<?xml version=\"1.0\"?>\n"),
        "XML declaration"
    );
    assert!(xml.contains("<VTKFile type=\"RectilinearGrid\" version=\"0.1\""));
    assert!(xml.contains(&format!("WholeExtent=\"0 {} 0 {} 0 0\"", nr - 1, nz - 1)));
    assert!(xml.contains(&format!("<Piece Extent=\"0 {} 0 {} 0 0\">", nr - 1, nz - 1)));
    assert_eq!(xml.matches("<Piece").count(), 1);
    assert!(xml.contains("<CellData/>"));
    assert!(xml.contains("</VTKFile>\n"));
    assert_balanced(&xml);

    // The PointData header lists all six fields as the active scalars.
    let names: Vec<&str> = snap.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(xml.contains(&format!("<PointData Scalars=\"{}\">", names.join(","))));

    // --- values: exact read-back against the source state ---
    for f in &snap.fields {
        let vtk = array_values(&xml, "DataArray", Some(&f.name));
        assert_eq!(vtk.len(), nr * nz, "{} covers the grid", f.name);
        // VTK point order: r (x) fastest — spot-check the invariant.
        assert_eq!(vtk[0], f.values[0]); // (i=0, j=0)
        assert_eq!(vtk[1], f.values[nz]); // (i=1, j=0)
        assert_eq!(vtk[nr], f.values[1]); // (i=0, j=1)
                                          // Full inverse mapping is bit-identical to the source block.
        assert_eq!(
            to_row_major(&vtk, nr, nz),
            f.values,
            "{} round-trips",
            f.name
        );
    }

    // --- coordinates: the C cell-centered total grid ---
    let r = array_values(&xml, "XCoordinates", None);
    let z = array_values(&xml, "YCoordinates", None);
    assert_eq!(r, snap.grid.r);
    assert_eq!(z, snap.grid.z);
    assert_eq!(array_values(&xml, "ZCoordinates", None), vec![0.0]);
    // Ghost band present: r[0] = (0 - 2 + 0.5) * 0.25 = -0.375.
    assert!((r[0] + 0.375).abs() < 1e-12);
    assert!((z[nz - 1] - 16.375).abs() < 1e-12);

    // The six field names are exactly the slice fixture's set.
    let want: BTreeSet<&str> = [
        "log_alpha_f",
        "beta_f",
        "log_h_f",
        "log_a_f",
        "psi_f",
        "lambda_f",
    ]
    .into();
    assert_eq!(names.into_iter().collect::<BTreeSet<_>>(), want);
}
