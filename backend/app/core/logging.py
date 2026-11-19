"""Structured JSON logging configuration."""

import json
import logging
import sys
from datetime import UTC, datetime

_HANDLER_NAME = "app-json-handler"

# Attributes every LogRecord has. Anything else came from extra= and
# should be included in the JSON output.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
