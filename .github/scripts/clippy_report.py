#!/usr/bin/env python3
"""Emit GitHub-workflow ::error annotations from a clippy log.

Called by the Clippy step in .github/workflows/rust.yml when clippy exits
nonzero: parses `level: message` + `  --> path:line:col` pairs and emits
explicit ::error workflow commands, so every lint hit becomes a check-run
annotation (readable via the public API without admin log access).
Paths are normalized to repo-root (clippy prints CWD-relative from ng/).
"""
from __future__ import annotations

import re

MSG_RE = re.compile(r"^(?:error|warning)(?:\[[^\]]*\])?: (.*)$")
LOC_RE = re.compile(r"^\s*--> ([^:\s]+):(\d+):(\d+)")
TAIL_RE = re.compile(r"^\s+")

MAX_ANNOTATIONS = 10  # workflow-command cap per step


def main() -> int:
    msg: str | None = None
    seen = 0
    lines = open("clippy.log", errors="replace").read().splitlines()
    for line in lines:
        m = MSG_RE.match(line.rstrip())
        if m:
            msg = m.group(1)
            # Cargo/toolchain-level errors carry no location: surface them too.
            print(f"::error::clippy: {msg[:240]}")
            seen += 1
            if seen >= MAX_ANNOTATIONS:
                return 0
            continue
        loc = LOC_RE.match(line)
        if loc and msg:
            path = loc.group(1)
            if not path.startswith(("ng/", "/", ".")):
                path = "ng/" + path
            print(
                f"::error file={path},line={loc.group(2)},"
                f"col={loc.group(3)}::{msg[:240]}"
            )
            msg = None
            seen += 1
            if seen >= MAX_ANNOTATIONS:
                return 0
    if seen == 0:
        # No diagnostics matched at all — dump the log head as one annotation
        # so the failure mode is visible from outside (logs are admin-gated).
        head = [l for l in lines if l.strip()][:8]
        print("::error::clippy produced no parseable diagnostics; log head: "
              + " | ".join(h.strip()[:60] for h in head)[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
