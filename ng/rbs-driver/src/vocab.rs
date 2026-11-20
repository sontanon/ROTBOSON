//! Closed vocabularies: the string sets that cross the JSON/TOML boundaries.
//!
//! Every `Display` spelling matches the Python `StrEnum` values in
//! tools/sweep_driver.py **exactly** — they are part of the state.json /
//! summary.json / param-file contract (resume and rendering stay
//! byte-compatible).

use serde::Deserialize;
use std::fmt;

/// Sentinel exit code for a step that hit the run wall-clock timeout
/// (tools/sweep_driver.py `TIMEOUT_EXIT = -999`; not a real signal —
/// chosen far from any errno/signal value).
pub const TIMEOUT_EXIT: i32 = -999;

macro_rules! str_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $($variant:ident => $s:expr),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($variant),+ }

        impl $name {
            /// Parse the exact closed-vocabulary string; `None` for any
            /// other spelling (callers turn that into the Python-exact
            /// error message).
            pub fn from_name(s: &str) -> Option<Self> {
                match s { $($s => Some(Self::$variant),)+ _ => None }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self { $(Self::$variant => f.write_str($s),)+ }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                Self::from_name(&s).ok_or_else(|| {
                    serde::de::Error::custom(<Self as BadVocab>::bad_vocab_message())
                })
            }
        }
    };
}

/// The Python-verbatim error message emitted when a spec deserializer meets
/// a vocabulary miss.
pub trait BadVocab {
    fn bad_vocab_message() -> String;
}

str_enum! {
    /// `[campaign] direction` (Python `Direction`).
    Direction { Up => "up", Down => "down" }
}
impl BadVocab for Direction {
    fn bad_vocab_message() -> String {
        r#"[campaign] direction must be "up" or "down""#.to_string()
    }
}

str_enum! {
    /// `[seed] policy` (Python `SeedPolicy`).
    SeedPolicy { FromScratch => "from_scratch", Solution => "solution" }
}
impl BadVocab for SeedPolicy {
    fn bad_vocab_message() -> String {
        r#"[seed] policy must be "from_scratch" or "solution""#.to_string()
    }
}

str_enum! {
    /// `[campaign] psi0_step_mode` (Python `Psi0StepMode`).
    Psi0StepMode { Absolute => "absolute", Relative => "relative" }
}
impl BadVocab for Psi0StepMode {
    fn bad_vocab_message() -> String {
        r#"[campaign] psi0_step_mode must be "absolute" or "relative""#.to_string()
    }
}

str_enum! {
    /// `[output] format` (Python `OutputFormat`).
    OutputFormat { Hdf5 => "hdf5", Ascii => "ascii" }
}
impl BadVocab for OutputFormat {
    fn bad_vocab_message() -> String {
        r#"[output] format must be "hdf5" or "ascii""#.to_string()
    }
}

str_enum! {
    /// How a step was produced (state.json `mode`; Python `StepMode`).
    /// `Regrid` mirrors the Python spelling quirk (`REGGRID = "regrid"`).
    StepMode {
        FixedPhi => "fixedPhi",
        Seed => "seed",
        Regrid => "regrid",
        RegridProbe => "regrid-probe",
    }
}
impl BadVocab for StepMode {
    fn bad_vocab_message() -> String {
        "state.json: invalid step mode (expected one of fixedPhi, seed, \
         regrid, regrid-probe)"
            .to_string()
    }
}

// str_enum! has no slot for the #[default] variant attribute that
// `#[derive(Default)]` on an enum requires — the manual impl is
// genuinely necessary, not derivable through the macro.
#[allow(clippy::derivable_impls)]
impl Default for StepMode {
    fn default() -> Self {
        StepMode::FixedPhi
    }
}

str_enum! {
    /// Campaign lifecycle status (state.json `status`; Python `Status`).
    Status { Running => "running", Done => "done", Stopped => "stopped", Failed => "failed" }
}
impl BadVocab for Status {
    fn bad_vocab_message() -> String {
        "state.json: invalid status (expected one of running, done, \
         stopped, failed)"
            .to_string()
    }
}

/// The closed stop reasons (Python `StopReason`).
///
/// The fixed spellings are enum variants; the dynamic families
/// (`failed:sig<name>`, `failed:exit<code>`, `failed:seed_exit<code>`)
/// carry runtime values and are built by the constructors below. `Display`
/// strings are the contract — `state.json`, `summary.json` and the logs
/// all carry them.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    DonePsi0Target,
    DoneOmegaTarget,
    DoneMaxSteps,
    StoppedNewtonianLimit,
    StoppedBoundary,
    StoppedTurningPoint,
    StoppedDomainBudget,
    FailedTimeout,
    FailedNewton,
    FailedSolver,
    FailedConfig,
    FailedIo,
    /// `failed:sig{name}` with the lowercase name without the `SIG` prefix
    /// (e.g. `failed:sigsegv`).
    Signal(String),
    /// `failed:exit{code}` for exit codes outside the 1–4 map.
    ExitCode(i32),
    /// `failed:seed_exit{code}` for a failed step-0 seed solve.
    SeedExit(i32),
}

impl StopReason {
    /// Python `StopReason.signal(sig_name)` — takes the *stripped* name
    /// (`"segv"`, not `"SIGSEGV"`).
    pub fn signal(sig_name: &str) -> StopReason {
        StopReason::Signal(sig_name.to_string())
    }

    /// Python `StopReason.exit_code(code)`.
    pub fn exit_code(code: i32) -> StopReason {
        StopReason::ExitCode(code)
    }

    /// Python `StopReason.seed_exit(code)`.
    pub fn seed_exit(code: i32) -> StopReason {
        StopReason::SeedExit(code)
    }

    /// `'done' / 'stopped' / 'failed'` — the component before the first
    /// `':'` of the stored string (Python `stop_family`).
    pub fn family(&self) -> &'static str {
        use StopReason::*;
        match self {
            DonePsi0Target | DoneOmegaTarget | DoneMaxSteps => "done",
            StoppedNewtonianLimit | StoppedBoundary | StoppedTurningPoint
            | StoppedDomainBudget => "stopped",
            FailedTimeout | FailedNewton | FailedSolver | FailedConfig | FailedIo
            | Signal(_) | ExitCode(_) | SeedExit(_) => "failed",
        }
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::DonePsi0Target => f.write_str("done:psi0_target"),
            StopReason::DoneOmegaTarget => f.write_str("done:omega_target"),
            StopReason::DoneMaxSteps => f.write_str("done:max_steps"),
            StopReason::StoppedNewtonianLimit => f.write_str("stopped:newtonian_limit"),
            StopReason::StoppedBoundary => f.write_str("stopped:boundary"),
            StopReason::StoppedTurningPoint => f.write_str("stopped:turning_point"),
            StopReason::StoppedDomainBudget => f.write_str("stopped:domain_budget"),
            StopReason::FailedTimeout => f.write_str("failed:timeout"),
            StopReason::FailedNewton => f.write_str("failed:newton"),
            StopReason::FailedSolver => f.write_str("failed:solver"),
            StopReason::FailedConfig => f.write_str("failed:config"),
            StopReason::FailedIo => f.write_str("failed:io"),
            StopReason::Signal(name) => write!(f, "failed:sig{name}"),
            StopReason::ExitCode(code) => write!(f, "failed:exit{code}"),
            StopReason::SeedExit(code) => write!(f, "failed:seed_exit{code}"),
        }
    }
}

/// Python `stop_family(reason: str)` — the component before the first `':'`
/// of a stop-reason string.
pub fn stop_family(reason: &str) -> &str {
    reason.split(':').next().unwrap_or(reason)
}

/// Lowercase signal name without the `SIG` prefix for a negative exit code
/// (Python: `signal.Signals(-code).name.removeprefix("SIG").lower()`), or
/// `None` when the number does not name a signal — Python raises
/// `ValueError` there, so callers turn this into
/// [`crate::error::DriverError::UnknownSignal`].
///
/// Linux x86-64 table, matching CPython's `signal.Signals`.
pub fn signal_name(code: i32) -> Option<&'static str> {
    Some(match code {
        1 => "hup",
        2 => "int",
        3 => "quit",
        4 => "ill",
        5 => "trap",
        6 => "abrt",
        7 => "bus",
        8 => "fpe",
        9 => "kill",
        10 => "usr1",
        11 => "segv",
        12 => "usr2",
        13 => "pipe",
        14 => "alrm",
        15 => "term",
        16 => "stkflt",
        17 => "chld",
        18 => "cont",
        19 => "stop",
        20 => "tstp",
        21 => "ttin",
        22 => "ttou",
        23 => "urg",
        24 => "xcpu",
        25 => "xfsz",
        26 => "vtalrm",
        27 => "prof",
        28 => "winch",
        29 => "poll",
        30 => "pwr",
        31 => "sys",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stop-reason contract: every spelling must equal the Python
    /// `StopReason` string exactly (SAN-60 report item).
    #[test]
    fn stop_reason_strings_match_python() {
        assert_eq!(StopReason::DonePsi0Target.to_string(), "done:psi0_target");
        assert_eq!(StopReason::DoneOmegaTarget.to_string(), "done:omega_target");
        assert_eq!(StopReason::DoneMaxSteps.to_string(), "done:max_steps");
        assert_eq!(
            StopReason::StoppedNewtonianLimit.to_string(),
            "stopped:newtonian_limit"
        );
        assert_eq!(StopReason::StoppedBoundary.to_string(), "stopped:boundary");
        assert_eq!(
            StopReason::StoppedTurningPoint.to_string(),
            "stopped:turning_point"
        );
        assert_eq!(
            StopReason::StoppedDomainBudget.to_string(),
            "stopped:domain_budget"
        );
        assert_eq!(StopReason::FailedTimeout.to_string(), "failed:timeout");
        assert_eq!(StopReason::FailedNewton.to_string(), "failed:newton");
        assert_eq!(StopReason::FailedSolver.to_string(), "failed:solver");
        assert_eq!(StopReason::FailedConfig.to_string(), "failed:config");
        assert_eq!(StopReason::FailedIo.to_string(), "failed:io");
        assert_eq!(StopReason::signal("segv").to_string(), "failed:sigsegv");
        assert_eq!(StopReason::exit_code(7).to_string(), "failed:exit7");
        assert_eq!(StopReason::seed_exit(3).to_string(), "failed:seed_exit3");
    }

    #[test]
    fn families() {
        assert_eq!(stop_family("done:psi0_target"), "done");
        assert_eq!(stop_family("stopped:turning_point"), "stopped");
        assert_eq!(stop_family("failed:sigsegv"), "failed");
        assert_eq!(stop_family("odd"), "odd");
        for r in [
            StopReason::DonePsi0Target,
            StopReason::StoppedTurningPoint,
            StopReason::FailedSolver,
            StopReason::Signal("segv".into()),
        ] {
            assert_eq!(stop_family(&r.to_string()), r.family());
        }
    }

    #[test]
    fn signal_names_and_gaps() {
        assert_eq!(signal_name(11), Some("segv"));
        assert_eq!(signal_name(1), Some("hup"));
        assert_eq!(signal_name(31), Some("sys"));
        assert_eq!(signal_name(999), None);
        // TIMEOUT_EXIT must NOT name a signal (it is checked before the
        // signal branch, but keep the property explicit).
        assert_eq!(signal_name(-TIMEOUT_EXIT), None);
    }

    #[test]
    fn vocab_from_name() {
        assert_eq!(Direction::from_name("up"), Some(Direction::Up));
        assert_eq!(Direction::from_name("UP"), None);
        assert_eq!(StepMode::from_name("regrid-probe"), Some(StepMode::RegridProbe));
        assert_eq!(Status::from_name("running"), Some(Status::Running));
    }
}
