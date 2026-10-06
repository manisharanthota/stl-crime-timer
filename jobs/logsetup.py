"""Logging for the job runner: stdout + rotating file (unless LOG_TO_FILE=false),
timestamps in UTC."""

import logging
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import get_settings

LOG_DIR = Path("logs")
LOG_FILE = "pipeline.log"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


# httpx logs request URLs at INFO; a Discord webhook URL contains its secret token.
_WEBHOOK_TOKEN = re.compile(r"(/api/webhooks/\d+/)[\w-]+")


class UTCFormatter(logging.Formatter):
    """UTC timestamps; webhook tokens are redacted from every formatted line."""

    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, datefmt or "%Y-%m-%dT%H:%M:%S") + "Z"

    def format(self, record):
        return _WEBHOOK_TOKEN.sub(r"\1[redacted]", super().format(record))


def setup_logging(
    log_dir: Path = LOG_DIR, level: int = logging.INFO, to_file: bool | None = None
) -> None:
    """Attach a stdout handler, plus a rotating file handler unless to_file (default:
    LOG_TO_FILE) is false, to the root logger. Call once from the CLI, never on import."""
    if to_file is None:
        to_file = get_settings().log_to_file
    formatter = UTCFormatter(FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            log_dir / LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
        ))
    root = logging.getLogger()
    root.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
