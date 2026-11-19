//! RBS-NG continuation driver — the pure, solver-independent half
//! (Linear SAN-60, part 1).
//!
//! A Rust port of the logic in `tools/sweep_driver.py` (the Python
//! continuation driver; the Python is the specification) covering
//! everything that does not shell out to the C solver. Modules land in
//! increments on this branch: first the byte-compatibility substrate,
//! then the spec model (this commit), then the state model and the
//! adaptive decision layer. See `README.md` for the ported/deferred
//! inventory and the workspace-merge notes.
//!
//! Python line references throughout cite tools/sweep_driver.py at the
//! SAN-59 head (`06b4ff2`) — the revision this port was specified
//! against; the master worktree base (`1ca850b`) is identical except
//! that it lacks the `[solver]` numThreads/precision/direct/fp32_ir
//! knobs, which are included here so the repo's runbook configs parse.

pub mod error;
pub mod pyjson;
pub mod sha256;
pub mod spec;
pub mod vocab;
