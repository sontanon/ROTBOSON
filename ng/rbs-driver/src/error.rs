//! Typed errors — no panics in library code.

use std::fmt;

/// Invalid campaign spec (Python: `SpecError(ValueError)` in
/// `tools/sweep_driver.py`). Message text of the `Invalid`/`UnknownTables`/
/// `UnknownKeys` variants matches the Python raisers verbatim.
#[derive(Debug)]
pub enum SpecError {
    /// Reading or stat-ing the spec / seed source failed.
    Io(std::io::Error),
    /// The spec file is not valid UTF-8.
    NonUtf8,
    /// The TOML text failed to parse, or a table's field had the wrong
    /// type (serde/toml error text, with position).
    TomlParse(String),
    /// Top-level tables not in `KNOWN` — Python:
    /// `unknown spec tables: ['x', 'y']` (sorted, single-quoted).
    UnknownTables(Vec<String>),
    /// Keys inside a known table not in `KNOWN[table]` — Python:
    /// `[campaign] unknown keys: ['x']` (sorted, single-quoted).
    UnknownKeys {
        /// The known table the offending keys were found in
        /// (`"campaign"`, `"seed"`, …).
        table: &'static str,
        /// The offending keys, sorted like the Python's `sorted(extra)`.
        keys: Vec<String>,
    },
    /// Semantic validation failure; the string is the exact Python
    /// `SpecError` message.
    Invalid(String),
}

/// Python's list-repr formatting (`['a', 'b']`) used in the unknown-key
/// messages.
fn py_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecError::Io(e) => write!(f, "campaign spec I/O error: {e}"),
            SpecError::NonUtf8 => write!(f, "campaign spec is not valid UTF-8"),
            SpecError::TomlParse(msg) => write!(f, "invalid campaign spec: {msg}"),
            SpecError::UnknownTables(t) => write!(f, "unknown spec tables: {}", py_list(t)),
            SpecError::UnknownKeys { table, keys } => {
                write!(f, "[{table}] unknown keys: {}", py_list(keys))
            }
            SpecError::Invalid(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl From<std::io::Error> for SpecError {
    fn from(e: std::io::Error) -> Self {
        SpecError::Io(e)
    }
}

/// state.json read/parse/resume errors. Python distinguishes a corrupt file
/// (uncaught `ValueError`/`JSONDecodeError`), a non-object top level
/// (explicit `ValueError`) and a spec-version change (explicit `SystemExit`
/// — the abort-resume contract); here they are one typed enum.
#[derive(Debug)]
pub enum StateError {
    /// Reading or writing state.json failed at the OS level (Python:
    /// the uncaught `OSError` from `open`/`write`).
    Io(std::io::Error),
    /// Corrupt/truncated state.json — includes the coercion messages
    /// (`state.json: expected a number, got str`).
    Corrupt(String),
    /// `spec_hash` does not match the loaded spec: resume must abort
    /// (Python: `SystemExit("state.json at ... pass --fresh to discard
    /// it.")`).
    SpecHashMismatch {
        /// The state.json whose hash disagrees — the abort message
        /// names it.
        path: std::path::PathBuf,
    },
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateError::Io(e) => write!(f, "state.json I/O error: {e}"),
            StateError::Corrupt(msg) => write!(f, "{msg}"),
            StateError::SpecHashMismatch { path } => write!(
                f,
                "state.json at {} was written for a different spec version; \
                 pass --fresh to discard it.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for StateError {}

impl From<std::io::Error> for StateError {
    fn from(e: std::io::Error) -> Self {
        StateError::Io(e)
    }
}

/// Errors from the pure decision helpers (Python: `SystemExit` /
/// uncaught `ValueError` in `finished`'s helpers).
#[derive(Debug)]
pub enum DriverError {
    /// `psi0_of_last` found no completed step with ψ₀ — Python:
    /// "no completed step records ψ₀; cannot continue".
    NoCompletedPsi0,
    /// `omega_of_last` found no completed step with ω — Python:
    /// "no completed step records ω; cannot continue".
    NoCompletedOmega,
    /// A negative exit code did not name a signal (Python:
    /// `signal.Signals(-code)` raises `ValueError` for unknown numbers..
    UnknownSignal(i32),
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DriverError::NoCompletedPsi0 => {
                write!(f, "no completed step records ψ₀; cannot continue")
            }
            DriverError::NoCompletedOmega => {
                write!(f, "no completed step records ω; cannot continue")
            }
            DriverError::UnknownSignal(code) => write!(
                f,
                "exit code {code} is negative but does not name a signal on this platform"
            ),
        }
    }
}

impl std::error::Error for DriverError {}

/// `--summarize` errors (Python `summarize` returns exit code 1 with a
/// logged message).
#[derive(Debug)]
pub enum SummarizeError {
    /// Reading state.json or writing summary.json failed at the OS
    /// level.
    Io(std::io::Error),
    /// state.json failed to load — corrupt, or a spec-hash mismatch
    /// (the underlying message surfaces verbatim).
    State(StateError),
    /// No state.json under the campaign root — Python: "no state.json to
    /// summarize".
    NoState,
    /// No completed step records ω — Python: "no completed steps record ω;
    /// nothing to summarize".
    NoOmegas,
}

impl fmt::Display for SummarizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummarizeError::Io(e) => write!(f, "summarize I/O error: {e}"),
            SummarizeError::State(e) => write!(f, "{e}"),
            SummarizeError::NoState => write!(f, "no state.json to summarize"),
            SummarizeError::NoOmegas => {
                write!(f, "no completed steps record ω; nothing to summarize")
            }
        }
    }
}

impl std::error::Error for SummarizeError {}

impl From<std::io::Error> for SummarizeError {
    fn from(e: std::io::Error) -> Self {
        SummarizeError::Io(e)
    }
}

impl From<StateError> for SummarizeError {
    fn from(e: StateError) -> Self {
        SummarizeError::State(e)
    }
}
