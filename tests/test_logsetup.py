"""Tests for the shared logging setup (SAN-25)."""

import io
import json
import logging
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from logsetup import JsonLogFormatter, configure, get_logger  # noqa: E402


def make_record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord("smoke", logging.INFO, "path.py", 1, msg, (), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


class TestJsonFormatter:
    def test_one_object_per_line_with_core_fields(self):
        out = json.loads(JsonLogFormatter().format(make_record("step done")))
        assert out["level"] == "info"
        assert out["logger"] == "smoke"
        assert out["msg"] == "step done"
        assert "ts" in out

    def test_extra_fields_are_included(self):
        out = json.loads(JsonLogFormatter().format(make_record("step done", step=3, psi0=0.42)))
        assert out["step"] == 3
        assert out["psi0"] == 0.42

    def test_standard_record_attrs_are_not_leaked(self):
        out = json.loads(JsonLogFormatter().format(make_record("x")))
        assert "stack_info" not in out
        assert "args" not in out
        assert "taskName" not in out


class TestConfigure:
    def test_logs_go_to_given_stream(self):
        stream = io.StringIO()
        configure("INFO", stream=stream)
        get_logger("rotboson-test").info("hello %s", "world")
        line = stream.getvalue()
        assert "hello world" in line
        assert "INFO" in line
        assert "rotboson-test" in line

    def test_level_filters(self):
        stream = io.StringIO()
        configure("WARNING", stream=stream)
        get_logger("rotboson-test").info("hidden")
        assert stream.getvalue() == ""
        get_logger("rotboson-test").warning("shown")
        assert "shown" in stream.getvalue()

    def test_reconfiguration_replaces_handlers(self):
        a, b = io.StringIO(), io.StringIO()
        configure("INFO", stream=a)
        configure("INFO", stream=b)
        get_logger("rotboson-test").info("once")
        assert a.getvalue() == ""
        assert "once" in b.getvalue()

    def test_extra_fields_reach_human_format(self):
        stream = io.StringIO()
        configure("INFO", stream=stream)
        # lazy % formatting must survive: no interpolation at call time
        get_logger("rotboson-test").info("value %d", 7)
        assert "value 7" in stream.getvalue()
