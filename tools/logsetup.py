"""Central logging setup for ROTBOSON tooling (SAN-24 spike / SAN-25).

Convention:
- **print() is a tool's stdout product** — final reports and verdicts that a
  human or CI consumes as data (e.g. `compare_solutions` reports,
  `generate_kernels --check` verdict). Keep those as plain prints.
- **logging is operational diagnostics** — progress, retries, skips, errors —
  emitted to *stderr* so stdout stays clean for the report product.

Level policy:
- INFO    normal progress of a long-running tool (one line per campaign step)
- WARNING recoverable oddities: retries, rejected regrids, drift corrections
- ERROR   step failures, corrupt inputs
- DEBUG   verbose detail (--verbose)

Stdlib only; a `--log-json` flag switches to one-JSON-object-per-line on
stderr for machine consumption (structured where it makes sense: campaign
telemetry), while stdout report formats stay byte-stable.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

_ENV_LEVEL = "ROTBOSON_LOG_LEVEL"

_HUMAN_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_HUMAN_DATEFMT = "%H:%M:%S"

_CONFIGURED = False


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line; picks up `extra={...}` fields automatically."""

    _STD_ATTRS: set[str] = {
        *(set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)),
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._STD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(
    level: str | int | None = None,
    *,
    json_logs: bool = False,
    stream: TextIO | None = None,
) -> None:
    """Configure the root logger for a CLI entry point.

    `level` beats the ROTBOSON_LOG_LEVEL environment variable, which beats
    the INFO default. Idempotent: reconfiguration replaces the root handlers.
    """
    global _CONFIGURED
    if level is None:
        level = os.environ.get(_ENV_LEVEL, "INFO")
    handler: logging.Handler = logging.StreamHandler(stream or sys.stderr)
    if json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT, _HUMAN_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module logger; safe to call before configure().

    Scripts run directly report as their file stem (e.g. `smoke`) instead of
    `__main__`, so log lines are stable regardless of invocation.
    """
    if name == "__main__":
        name = Path(sys.argv[0]).stem
    return logging.getLogger(name)
