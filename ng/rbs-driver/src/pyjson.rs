//! A JSON value model with a **Python-`json.dumps`-byte-compatible** writer.
//!
//! `state.json` is part of the Python driver's on-disk contract
//! (tools/sweep_driver.py `save_state` writes
//! `json.dumps(state.to_dict(), indent=2) + "\n"`, and the round-trip test
//! asserts byte identity against an archived state.json). The Rust port
//! must be able to reproduce those bytes exactly, which stock `serde_json`
//! formatting cannot (different float spellings: `1e-05` vs `1e-5`,
//! `1e+16` vs `1e16`). This module provides:
//!
//! * [`Jv`] — an order-preserving JSON tree with an explicit
//!   [`Jv::Int`]/[`Jv::Float`] split: the Python model keeps the
//!   int/float *identity* of scalars on round-trip
//!   (`tests/test_state_roundtrip.py::test_int_scalars_keep_identity`).
//! * [`Jv::write_json`] — emits exactly what
//!   `json.dumps(value, indent=2)` emits (2-space indent, `": "` key
//!   separator, `",\n"` item separator, `{}`/`[]` for empties,
//!   `ensure_ascii` escaping, Python `repr` float spellings).
//! * [`Jv::parse`] — reads JSON via `serde_json` (which accepts anything
//!   Python wrote) into [`Jv`] preserving key order and int/float identity.

use serde_json::Value as Json;

/// A JSON tree. `Obj` keeps insertion order, mirroring Python dict
/// emission order (byte-identity requirement).
#[derive(Debug, Clone, PartialEq)]
pub enum Jv {
    Null,
    Bool(bool),
    /// A JSON number that was an integer — stays an integer on re-write.
    Int(i64),
    /// A JSON number that was a float — re-writes in Python `repr` style.
    Float(f64),
    Str(String),
    Arr(Vec<Jv>),
    Obj(Vec<(String, Jv)>),
}

impl Jv {
    // ---- constructors ---------------------------------------------------

    pub fn obj(pairs: Vec<(&str, Jv)>) -> Jv {
        Jv::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn arr(items: Vec<Jv>) -> Jv {
        Jv::Arr(items)
    }

    pub fn opt_float(v: Option<f64>) -> Jv {
        v.map(Jv::Float).unwrap_or(Jv::Null)
    }

    pub fn opt_int(v: Option<i64>) -> Jv {
        v.map(Jv::Int).unwrap_or(Jv::Null)
    }

    pub fn opt_str<S: Into<String>>(v: Option<S>) -> Jv {
        v.map(|s| Jv::Str(s.into())).unwrap_or(Jv::Null)
    }

    // ---- accessors ------------------------------------------------------

    pub fn get(&self, key: &str) -> Option<&Jv> {
        match self {
            Jv::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&Vec<(String, Jv)>> {
        match self {
            Jv::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&Vec<Jv>> {
        match self {
            Jv::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Jv::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The Python type name of the corresponding `json.loads` value
    /// (`type(v).__name__`), used in the state.json coercion messages.
    pub fn py_type_name(&self) -> &'static str {
        match self {
            Jv::Null => "NoneType",
            Jv::Bool(_) => "bool",
            Jv::Int(_) => "int",
            Jv::Float(_) => "float",
            Jv::Str(_) => "str",
            Jv::Arr(_) => "list",
            Jv::Obj(_) => "dict",
        }
    }

    // ---- parsing / writing ----------------------------------------------

    /// Parse JSON text (any valid JSON, including everything the Python
    /// writer produced) into a [`Jv`] tree preserving key order and the
    /// int/float distinction.
    pub fn parse(text: &str) -> Result<Jv, serde_json::Error> {
        let v: Json = serde_json::from_str(text)?;
        Ok(Jv::from(v))
    }

    /// Emit exactly Python's `json.dumps(value, indent=2)` (no trailing
    /// newline — the Python writer appends `"\n"` at the call site).
    pub fn write_json(&self) -> String {
        let mut out = String::new();
        write_value(&mut out, self, 0);
        out
    }
}

impl From<Json> for Jv {
    fn from(v: Json) -> Jv {
        match v {
            Json::Null => Jv::Null,
            Json::Bool(b) => Jv::Bool(b),
            Json::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Jv::Int(i)
                } else if let Some(u) = n.as_u64() {
                    // JSON integers beyond i64 are unheard of in state.json;
                    // degrade to float rather than lose the value.
                    if u <= i64::MAX as u64 {
                        Jv::Int(u as i64)
                    } else {
                        Jv::Float(u as f64)
                    }
                } else {
                    Jv::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Json::String(s) => Jv::Str(s),
            Json::Array(a) => Jv::Arr(a.into_iter().map(Jv::from).collect()),
            // serde_json is built with "preserve_order", so Map iteration
            // follows document order.
            Json::Object(m) => Jv::Obj(m.into_iter().map(|(k, v)| (k, Jv::from(v))).collect()),
        }
    }
}

fn indent_of(level: usize) -> &'static str {
    // indent=2: levels are small; static for the common ones, dynamic beyond.
    const SPACES: [&str; 8] = [
        "", "  ", "    ", "      ", "        ", "          ", "            ", "              ",
    ];
    SPACES.get(level).copied().unwrap_or("                ")
}

fn write_value(out: &mut String, v: &Jv, level: usize) {
    match v {
        Jv::Null => out.push_str("null"),
        Jv::Bool(true) => out.push_str("true"),
        Jv::Bool(false) => out.push_str("false"),
        Jv::Int(i) => out.push_str(&i.to_string()),
        Jv::Float(f) => out.push_str(&python_repr_f64(*f)),
        Jv::Str(s) => write_json_string(out, s),
        Jv::Arr(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(indent_of(level + 1));
                write_value(out, item, level + 1);
            }
            out.push('\n');
            out.push_str(indent_of(level));
            out.push(']');
        }
        Jv::Obj(pairs) => {
            if pairs.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(indent_of(level + 1));
                write_json_string(out, k);
                out.push_str(": ");
                write_value(out, val, level + 1);
            }
            out.push('\n');
            out.push_str(indent_of(level));
            out.push('}');
        }
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // ensure_ascii=True: everything non-ASCII is \u-escaped
            // (surrogate pairs above the BMP, like CPython's encoder).
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c if (c as u32) < 0x80 => out.push(c),
            c => {
                let cp = c as u32;
                if cp <= 0xFFFF {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    let cp = cp - 0x1_0000;
                    let hi = 0xD800 + (cp >> 10);
                    let lo = 0xDC00 + (cp & 0x3FF);
                    out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Format an `f64` exactly like Python's `repr(float)` / `json.dumps`.
///
/// Python `repr` rules: shortest round-trip digits; fixed notation when the
/// decimal exponent is in `[-4, 16)`, scientific otherwise; scientific
/// form always signs the exponent and pads it to at least two digits
/// (`1e-05`, `1e+16`); fixed form always shows a fractional part (`15.0`).
pub fn python_repr_f64(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let neg = f.is_sign_negative();
    let a = f.abs();
    if a == 0.0 {
        return if neg { "-0.0".to_string() } else { "0.0".to_string() };
    }

    // Rust's LowerExp is the shortest correctly-rounding representation
    // (flt2dec shortest mode) — same digits CPython's repr produces.
    let sci = format!("{a:e}");
    let (mant, exp_str) = sci.split_once('e').expect("LowerExp always has 'e'");
    let exp: i32 = exp_str.parse().expect("LowerExp exponent is an integer");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    debug_assert!(!digits.is_empty() && !digits.starts_with('0'));

    let body = if !(-4..16).contains(&exp) {
        // Scientific: d[.ddd]e±XX
        let first = &digits[..1];
        let rest = &digits[1..];
        let sign = if exp < 0 { '-' } else { '+' };
        let mant = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        format!("{mant}e{sign}{:02}", exp.abs())
    } else if exp >= 0 {
        // Fixed with the point after the (exp+1)-th digit.
        let point = (exp as usize) + 1;
        let (int_part, frac) = if digits.len() > point {
            (digits[..point].to_string(), digits[point..].to_string())
        } else {
            let mut int_part = digits.clone();
            while int_part.len() < point {
                int_part.push('0');
            }
            (int_part, String::new())
        };
        if frac.is_empty() {
            format!("{int_part}.0")
        } else {
            format!("{int_part}.{frac}")
        }
    } else {
        // Fixed, |x| < 1.
        format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
    };

    if neg {
        format!("-{body}")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_spellings_match_python_repr() {
        // Cases taken from the archived state.json fixtures plus the
        // classic Python repr boundaries.
        assert_eq!(python_repr_f64(0.95), "0.95");
        assert_eq!(python_repr_f64(15.0), "15.0");
        assert_eq!(python_repr_f64(85.0), "85.0");
        assert_eq!(python_repr_f64(1.0), "1.0");
        assert_eq!(python_repr_f64(-0.011056135387960665), "-0.011056135387960665");
        assert_eq!(python_repr_f64(6.5511572566e-05), "6.5511572566e-05");
        assert_eq!(python_repr_f64(1e-05), "1e-05");
        assert_eq!(python_repr_f64(1e-4), "0.0001");
        assert_eq!(python_repr_f64(1.5e-7), "1.5e-07");
        assert_eq!(python_repr_f64(1e16), "1e+16");
        assert_eq!(python_repr_f64(1.2345e16), "1.2345e+16");
        assert_eq!(python_repr_f64(1e15), "1000000000000000.0");
        assert_eq!(python_repr_f64(0.0028385589893570092), "0.0028385589893570092");
        assert_eq!(python_repr_f64(0.0), "0.0");
        assert_eq!(python_repr_f64(-0.0), "-0.0");
        // round-trip: parse(write(x)) == x
        for x in [0.1, 1.0 / 3.0, 6.5511572566e-05, 1e-9, 123.456, 3.0] {
            let s = python_repr_f64(x);
            let back: f64 = s.parse().unwrap();
            assert_eq!(back, x, "{s} did not round-trip");
        }
    }

    #[test]
    fn write_matches_python_json_dumps_indent2() {
        // Mirrors `json.dumps({...}, indent=2)` byte-for-byte.
        let v = Jv::obj(vec![
            ("a", Jv::Int(1)),
            ("b", Jv::Float(2.5)),
            ("empty_obj", Jv::Obj(vec![])),
            ("empty_arr", Jv::Arr(vec![])),
            (
                "steps",
                Jv::arr(vec![Jv::obj(vec![
                    ("i", Jv::Int(0)),
                    ("note", Jv::Null),
                    ("list", Jv::arr(vec![Jv::Int(1), Jv::Float(2.0)])),
                ])]),
            ),
            (
                "nested",
                Jv::obj(vec![("x", Jv::Bool(true)), ("y", Jv::Str("s\"q\\".to_string()))]),
            ),
        ]);
        let expected = concat!(
            "{\n",
            "  \"a\": 1,\n",
            "  \"b\": 2.5,\n",
            "  \"empty_obj\": {},\n",
            "  \"empty_arr\": [],\n",
            "  \"steps\": [\n",
            "    {\n",
            "      \"i\": 0,\n",
            "      \"note\": null,\n",
            "      \"list\": [\n",
            "        1,\n",
            "        2.0\n",
            "      ]\n",
            "    }\n",
            "  ],\n",
            "  \"nested\": {\n",
            "    \"x\": true,\n",
            "    \"y\": \"s\\\"q\\\\\"\n",
            "  }\n",
            "}",
        );
        assert_eq!(v.write_json(), expected);
    }

    #[test]
    fn parse_preserves_order_and_number_identity() {
        let v = Jv::parse(r#"{"a": 0, "b": 0.0, "c": 1e-05, "d": [1, {}]}"#).unwrap();
        let Jv::Obj(pairs) = &v else { panic!("not an object") };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["a", "b", "c", "d"]);
        assert!(matches!(v.get("a"), Some(Jv::Int(0))));
        assert!(matches!(v.get("b"), Some(Jv::Float(_))));
        assert!(matches!(v.get("c"), Some(Jv::Float(_))));
        // round-trip stability
        let text = v.write_json();
        let v2 = Jv::parse(&text).unwrap();
        assert_eq!(v, v2);
        assert_eq!(v2.write_json(), text);
    }
}
