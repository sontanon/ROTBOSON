//! VTR (VTK rectilinear-grid XML) output — ParaView-native, day one.
//!
//! Writes each field block as a `PointData` scalar array on a single
//! `<Piece>` spanning the (r, z) slice; the third extent is degenerate
//! (`0 0`), the standard way VTK encodes 2D data. Coordinates are written
//! as `XCoordinates`/`YCoordinates` (r and z) plus a zero `ZCoordinates`,
//! so the grid ladder's later non-uniform radial maps need no format
//! change.

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
            let _ =
                write!(s, r#"<{tag} type="Float64" NumberOfComponents="1" format="ascii">"#);
        }
    }
    for v in values {
        s.push(' ');
        fmt_f64(*v, &mut s);
    }
    let _ = write!(s, "\n</{tag}>");
    s
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
        s.push_str(&data_array("DataArray", Some(&f.name), &f.values));
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
    let end = start + xml[start..].find(&format!("</{tag}>")).expect("close present");
    xml[start..end]
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("float token"))
        .collect()
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
    fn values_and_coordinates_roundtrip_exactly() {
        let (g, fields) = demo();
        let x = snapshot_xml(&g, &fields);
        assert_eq!(values_of(&x, "DataArray", Some("psi")), fields[0].values);
        assert_eq!(values_of(&x, "DataArray", Some("l_alpha")), fields[1].values);
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
            values_of(&text, "DataArray", Some("psi")),
            fields[0].values
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
