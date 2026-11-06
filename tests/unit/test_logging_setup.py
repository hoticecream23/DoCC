"""Structured JSON log lines."""

import json
import logging
import sys

from baseline.logging_setup import JsonFormatter


def _record(**extra):
    record = logging.LogRecord("baseline.test", logging.WARNING, __file__, 1, "dropping %s", ("x",), None)
    record.__dict__.update(extra)
    return record


def test_a_record_becomes_one_json_object_with_its_extras():
    line = json.loads(JsonFormatter().format(_record(doc_id="d1")))
    assert (line["level"], line["msg"], line["doc_id"]) == ("WARNING", "dropping x", "d1")


def test_an_exception_is_carried_in_the_exc_key():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("baseline.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]
