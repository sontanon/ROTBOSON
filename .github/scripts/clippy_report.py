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

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MSG_RE = re.compile(r"^(?:error|warning)(?:\[[^\]]*\])?: (.*)$")
LOC_RE = re.compile(r"^\s*--> ([^:\s]+):(\d+):(\d+)")
CLUE_RE = re.compile(r"(panicked|Killed|signal|error\[|could not compile|fatal)", re.I)

MAX_ANNOTATIONS = 10  # workflow-command cap per step


def main() -> int:
    msg: str | None = None
    seen = 0
    raw = open("clippy.log", errors="replace").read().splitlines()
    lines = [ANSI_RE.sub("", l).rstrip() for l in raw]
    for line in lines:
        m = MSG_RE.match(line)
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
        # No located diagnostics: failure modes live at the log TAIL —
        # prefer panic/kill/compile-error clue lines, else the last lines.
        clue = [l for l in lines if CLUE_RE.search(l)][:8]
        body = clue if clue else [l for l in lines if l.strip()][-10:]
        joined = " | ".join(l.strip()[:70] for l in body)[:240]
        print(f"::error::clippy produced no parseable diagnostics; "
              f"{'clue lines' if clue else 'log tail'}: {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
