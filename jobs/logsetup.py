"""Logging for the job runner: console + rotating file, timestamps in UTC."""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = "pipeline.log"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class UTCFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, datefmt or "%Y-%m-%dT%H:%M:%S") + "Z"


def setup_logging(log_dir: Path = LOG_DIR, level: int = logging.INFO) -> None:
    """Attach console and rotating-file handlers to the root logger. Call once from
    the CLI, never on import."""
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = UTCFormatter(FORMAT)
    console = logging.StreamHandler()
    file = RotatingFileHandler(
        log_dir / LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (console, file):
        handler.setFormatter(formatter)
        root.addHandler(handler)
