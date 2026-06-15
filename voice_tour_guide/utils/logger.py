"""
utils/logger.py
===============
Structured JSON logger for the KHEMET Voice Tour Guide service.

Outputs one JSON line per log entry to stdout/stderr.
Compatible with any log aggregator (CloudWatch, Datadog, etc).

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Audio generated", extra={"artifact_id": "abc"})
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    SERVICE_NAME = "voice_tour_guide"

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include any extra fields passed via extra={}
        _STDLIB_KEYS = {
            "timestamp", "level", "service", "logger", "message",
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module",
            "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _STDLIB_KEYS:
                entry[key] = value

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger
