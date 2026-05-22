"""
Alpha Suite — Dual Logger (console + JSONL).

Console handler with timestamps for real-time monitoring.
JSONL file handler at /app/state/alpha_suite_events.jsonl for structured event replay.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone


# ── JSONL event log path ──
EVENT_LOG_DIR = "/app/state"
EVENT_LOG_PATH = os.path.join(EVENT_LOG_DIR, "alpha_suite_events.jsonl")


class JsonlHandler(logging.Handler):
    """Logging handler that appends structured JSON lines to a file."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
            }
            if record.exc_info and record.exc_info[0] is not None:
                entry["exception"] = self.format(record)
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            # Never let logging crash the bot
            pass


def setup_logger(name: str) -> logging.Logger:
    """Create a logger with dual output: console (timestamped) + JSONL file.

    Args:
        name: Logger name (e.g. 'alpha-suite', 'arb-scanner').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console handler — human-readable with timestamp
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # JSONL file handler — structured events for replay / analysis
    jsonl = JsonlHandler(EVENT_LOG_PATH)
    jsonl.setLevel(logging.DEBUG)
    jsonl_fmt = logging.Formatter("%(message)s")
    jsonl.setFormatter(jsonl_fmt)
    logger.addHandler(jsonl)

    return logger


def log_event(event_type: str, data: dict) -> None:
    """Append a structured event to the JSONL event log.

    Each line is a self-contained JSON object with:
        ts, event, and all keys from data.

    Args:
        event_type: Category string (e.g. 'cycle', 'trade', 'error', 'signal').
        data: Arbitrary dict of event-specific fields.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
    }
    entry.update(data)
    try:
        os.makedirs(EVENT_LOG_DIR, exist_ok=True)
        with open(EVENT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        # Silently fail — never crash the bot over logging
        pass
