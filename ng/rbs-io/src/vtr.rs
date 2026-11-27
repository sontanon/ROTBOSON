//! VTR (VTK rectilinear-grid XML) output — ParaView-native, day one.
//!
//! Writes each field block as a `PointData` scalar array on a single
//! `<Piece>` spanning the (r, z) slice; the third extent is degenerate
//! (`0 0`), the standard way VTK encodes 2D data. Coordinates are written
//! as `XCoordinates`/`YCoordinates` (r and z) plus a zero `ZCoordinates`,
//! so the grid ladder's later non-uniform radial maps need no format
//! change.
//!
//! Data ordering: VTK point arrays vary **x fastest** (r here), while the
//! solver's field blocks are row-major over `IDX(i, j) = i * nz + j`
//! (z fastest) — the exact layout the C oracle's `.asc`/HDF5 output uses.
//! [`snapshot_xml`] transposes each block on emit, so a `ParaView` render
//! of the VTR and a matplotlib render of the HDF5 show the same field.

use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{FieldBlock, FieldWriter, GridTopology};

/// VTR file writer: one `.vtr` file per snapshot.
#[derive(Debug)]
pub struct VtrWriter {
    path: PathBuf,
}

impl VtrWriter {
    /// Write snapshots to `path` (one file per [`FieldWriter::write_snapshot`]).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The target file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Format one f64 the way VTK tools expect (shortest round-trip).
fn fmt_f64(v: f64, out: &mut String) {
    let _ = write!(out, "{v:e}");
}

/// A `<DataArray>` / coordinate element with ascii floats.
fn data_array(tag: &str, name: Option<&str>, values: &[f64]) -> String {
    let mut s = String::new();
    match name {
        Some(n) => {
            let _ = write!(
                s,
                r#"<{tag} type="Float64" Name="{n}" NumberOfComponents="1" format="ascii">"#
            );
        }
        None => {
            let _ = write!(
                s,
                r#"<{tag} type="Float64" NumberOfComponents="1" format="ascii">"#
            );
        }
    }
    for v in values {
        s.push(' ');
        fmt_f64(*v, &mut s);
    }
    let _ = write!(s, "\n</{tag}>");
    s
}

/// Reorder a field block into VTK point order: r (x) fastest, then z (y).
///
/// The source is row-major over `IDX(i, j) = i * nz + j`; the VTK linear
/// index of point `(i, j)` is `i + j * nr`.
///
/// # Panics
///
/// Panics in debug builds if `values` is not `nr * nz` long (the caller
/// contract; the HDF5 backend validates the same invariant loudly).
fn vtk_point_order(values: &[f64], nr: usize, nz: usize) -> Vec<f64> {
    debug_assert_eq!(values.len(), nr * nz, "field block must cover the grid");
    let mut out = vec![0.0; nr * nz];
    for (i, row) in values.chunks_exact(nz).enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[i + j * nr] = *v;
        }
    }
    out
}

/// The XML text of one snapshot (also the unit-test surface).
///
/// # Panics
///
/// Panics if the grid extents do not fit `i64` (they always do at paper
/// sizes).
#[must_use]
pub fn snapshot_xml(grid: &GridTopology, fields: &[FieldBlock]) -> String {
    let im = i64::try_from(grid.nr - 1).expect("extent fits i64");
    let jm = i64::try_from(grid.nz - 1).expect("extent fits i64");
    let mut s = String::new();
    let _ = write!(
        s,
        "<?xml version=\"1.0\"?>\n\
         <VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
         <RectilinearGrid WholeExtent=\"0 {im} 0 {jm} 0 0\">\n\
         <Piece Extent=\"0 {im} 0 {jm} 0 0\">\n"
    );
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let _ = writeln!(s, "<PointData Scalars=\"{}\">", names.join(","));
    for f in fields {
        let vtk = vtk_point_order(&f.values, grid.nr, grid.nz);
        s.push_str(&data_array("DataArray", Some(&f.name), &vtk));
        s.push('\n');
    }
    let _ = write!(s, "</PointData>\n<CellData/>\n");
    // Rectilinear-grid coordinate axes are direct children of the piece.
    s.push_str(&data_array("XCoordinates", None, &grid.r));
    s.push('\n');
    s.push_str(&data_array("YCoordinates", None, &grid.z));
    s.push('\n');
    s.push_str(&data_array("ZCoordinates", None, &[0.0]));
    s.push('\n');
    let _ = write!(s, "</Piece>\n</RectilinearGrid>\n</VTKFile>\n");
    s
}

impl FieldWriter for VtrWriter {
    fn write_snapshot(
        &mut self,
        grid: &GridTopology,
        fields: &[FieldBlock],
    ) -> std::io::Result<()> {
        let xml = snapshot_xml(grid, fields);
        let mut f = std::fs::File::create(&self.path)?;
        f.write_all(xml.as_bytes())?;
        f.flush()
    }
}

/// Extract the ascii values inside the `<tag ...> ... </tag>` element (test
/// helper).
#[cfg(test)]
fn values_of(xml: &str, tag: &str, name: Option<&str>) -> Vec<f64> {
    let open = match name {
        Some(n) => format!(
            "<{tag} type=\"Float64\" Name=\"{n}\" NumberOfComponents=\"1\" format=\"ascii\">"
        ),
        None => format!("<{tag} type=\"Float64\" NumberOfComponents=\"1\" format=\"ascii\">"),
    };
    let start = xml.find(&open).expect("element present") + open.len();
    let end = start
        + xml[start..]
            .find(&format!("</{tag}>"))
            .expect("close present");
    xml[start..end]
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("float token"))
        .collect()
}

/// Parse a field block back out of a VTR in VTK point order (r fastest),
/// returning row-major `IDX(i, j)` values (test helper: the inverse of
/// `vtk_point_order`).
#[cfg(test)]
fn point_order_values(xml: &str, name: &str, nr: usize, nz: usize) -> Vec<f64> {
    let vtk = values_of(xml, "DataArray", Some(name));
    assert_eq!(vtk.len(), nr * nz);
    let mut out = vec![0.0; nr * nz];
    for (i, row) in out.chunks_exact_mut(nz).enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = vtk[i + j * nr];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> (GridTopology, Vec<FieldBlock>) {
        let g = GridTopology::uniform(3, 2, 2.0, 1.0);
        let fields = vec![
            FieldBlock {
                name: "psi".to_string(),
                values: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            },
            FieldBlock {
                name: "l_alpha".to_string(),
                values: vec![-0.5; 6],
            },
        ];
        (g, fields)
    }

    #[test]
    fn xml_is_wellformed_structure() {
        let (g, fields) = demo();
        let x = snapshot_xml(&g, &fields);
        for needle in [
            "type=\"RectilinearGrid\"",
            "WholeExtent=\"0 2 0 1 0 0\"",
            "<PointData Scalars=\"psi,l_alpha\">",
            "Name=\"psi\"",
            "Name=\"l_alpha\"",
            "<CellData/>",
            "</VTKFile>",
        ] {
            assert!(x.contains(needle), "missing {needle:?} in:\n{x}");
        }
        assert_eq!(x.matches("<Piece").count(), 1);
        assert_eq!(x.matches("</VTKFile>").count(), 1);
    }

    #[test]
    fn field_values_are_in_vtk_point_order() {
        // Source is row-major IDX(i, j) = i * nz + j (z fastest); VTK
        // wants r (x) fastest: point (i, j) lands at i + j * nr.
        let (g, fields) = demo();
        assert_eq!((g.nr, g.nz), (3, 2));
        let x = snapshot_xml(&g, &fields);
        let psi = values_of(&x, "DataArray", Some("psi"));
        assert_eq!(psi, vec![0.0, 0.2, 0.4, 0.1, 0.3, 0.5]);
        // The inverse mapping restores the original block exactly.
        assert_eq!(point_order_values(&x, "psi", g.nr, g.nz), fields[0].values);
        assert_eq!(
            point_order_values(&x, "l_alpha", g.nr, g.nz),
            fields[1].values
        );
    }

    #[test]
    fn coordinates_roundtrip_exactly() {
        let (g, fields) = demo();
        let x = snapshot_xml(&g, &fields);
        assert_eq!(values_of(&x, "XCoordinates", None), g.r);
        assert_eq!(values_of(&x, "YCoordinates", None), g.z);
        assert_eq!(values_of(&x, "ZCoordinates", None), vec![0.0]);
    }

    #[test]
    fn file_write_roundtrips() {
        let dir = std::env::temp_dir().join(format!("rbs-io-vtr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let (g, fields) = demo();
        let mut w = VtrWriter::new(dir.join("snap.vtr"));
        w.write_snapshot(&g, &fields).expect("write");
        let text = std::fs::read_to_string(w.path()).expect("read");
        assert_eq!(
            point_order_values(&text, "psi", g.nr, g.nz),
            fields[0].values
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Strict well-formedness for the writer's XML subset: one root
    /// element, every open tag closed in order, attributes quoted, no
    /// stray `<`/`&` in text (the writer escapes none because field data
    /// is plain ascii floats).
    fn assert_wellformed(xml: &str) {
        let mut stack: Vec<String> = Vec::new();
        let bytes = xml.as_bytes();
        let mut i = 0;
        let mut in_root_close = false;
        while i < bytes.len() {
            match bytes[i] {
                b'<' if xml[i..].starts_with("<?xml") => {
                    i = xml[i..].find("?>").expect("xml decl closes") + i + 2;
                }
                b'<' if bytes.get(i + 1) == Some(&b'/') => {
                    let end = xml[i..].find('>').expect("close tag ends") + i;
                    let name = &xml[i + 2..end];
                    let open = stack.pop().expect("close matches an open tag");
                    assert_eq!(name, open, "close tag matches the innermost open");
                    if stack.is_empty() {
                        in_root_close = true;
                    }
                    i = end + 1;
                }
                b'<' => {
                    assert!(!in_root_close, "content after the root closed");
                    let end = xml[i..].find('>').expect("open tag ends") + i;
                    let inner = &xml[i + 1..end];
                    let mut parts = inner.split_whitespace();
                    let name = parts.next().expect("tag has a name").to_string();
                    // Attribute values may contain spaces (extents); the
                    // quote-balance check below covers quoting as a whole.
                    assert!(
                        xml[i..end].matches('"').count() % 2 == 0,
                        "attributes are quote-balanced"
                    );
                    if inner.ends_with('/') {
                        // Self-closing: <CellData/> etc.
                    } else {
                        stack.push(name);
                    }
                    i = end + 1;
                }
                b'&' => panic!("unexpected entity in writer output"),
                _ => i += 1,
            }
        }
        assert!(in_root_close, "root element was closed");
        assert!(stack.is_empty(), "all tags closed");
    }

    #[test]
    fn writer_xml_is_wellformed() {
        let (g, fields) = demo();
        let x = snapshot_xml(&g, &fields);
        assert_wellformed(&x);
        // And the single-element degenerate case.
        let g1 = GridTopology::uniform(1, 1, 1.0, 1.0);
        let x1 = snapshot_xml(
            &g1,
            &[FieldBlock {
                name: "a".to_string(),
                values: vec![0.0],
            }],
        );
        assert!(x1.contains("WholeExtent=\"0 0 0 0 0 0\""));
        assert_wellformed(&x1);
    }
}
