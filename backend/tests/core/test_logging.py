import json
import logging

from app.core.logging import _HANDLER_NAME, JsonFormatter, configure_logging


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed: %s",
        args=("reason",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_outputs_valid_json_with_core_fields():
    data = json.loads(JsonFormatter().format(_make_record()))
    assert data["level"] == "ERROR"
    assert data["logger"] == "test.logger"
    assert data["message"] == "failed: reason"
    assert "timestamp" in data


def test_formatter_includes_extra_fields():
    record = _make_record(error_code="REPO_NOT_FOUND", status_code=404)
    data = json.loads(JsonFormatter().format(record))
    assert data["error_code"] == "REPO_NOT_FOUND"
    assert data["status_code"] == 404


def test_formatter_survives_non_serializable_values():
    record = _make_record(payload={1, 2, 3})
    assert json.loads(JsonFormatter().format(record))["payload"]


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")
    handlers = [
        h for h in logging.getLogger().handlers if h.get_name() == _HANDLER_NAME
    ]
    assert len(handlers) == 1
