"""
logger.py
---------
Centralised, production-grade logging setup for IDVestAI.

Features:
  - Console handler  : INFO level, colour-coded by level
  - File handler     : DEBUG level, rotates at 5 MB (keeps 3 backups)
  - Single call      : get_logger(__name__) returns a named child logger

Usage (anywhere in the project):
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Inference took %.2f ms", elapsed)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "idvestai.log"

LOG_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_FILE = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_INITIALIZED = False  # guard: configure root logger only once


# ── Colour helpers (ANSI codes for Windows consoles ≥ Win10) ─────────────────

_LEVEL_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes around the log level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


# ── Public API ────────────────────────────────────────────────────────────────


def _setup_root_logger(level: int = logging.DEBUG) -> None:
    """Configure the root logger (called once at import time)."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    root = logging.getLogger()
    root.setLevel(level)

    # — Console handler —
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        _ColouredFormatter(LOG_FORMAT_CONSOLE, datefmt=DATE_FORMAT)
    )
    root.addHandler(console_handler)

    # — Rotating file handler —
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT_FILE, datefmt=DATE_FORMAT)
        )
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Could not create log file (%s). File logging disabled.", exc)


# Initialise when the module is first imported
_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named child logger.

    Parameters
    ----------
    name : str — typically __name__ of the calling module
    """
    return logging.getLogger(name)
