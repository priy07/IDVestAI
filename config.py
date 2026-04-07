"""
config.py
---------
Centralised configuration for IDVestAI.

All tuneable parameters live here. Override individual values by setting
the corresponding environment variable (loaded from .env automatically).

Usage:
    from config import settings
    print(settings.MODEL_PATH)
    print(settings.CONFIDENCE_THRESHOLD)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (silently ignored if absent)
load_dotenv()


class Settings:
    """
    Application settings, populated from environment variables.
    Defaults are coding-friendly; override in .env for deployment.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/idvest_best.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

    # ── Camera / Source ───────────────────────────────────────────────────────
    DEFAULT_CAMERA_ID: str = "CAM-01"

    # ── API ────────────────────────────────────────────────────────────────────
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path("logs")

    # ── Alerts ────────────────────────────────────────────────────────────────
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))

    # ── Dress Code Rules ──────────────────────────────────────────────────────
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.20"))
    REQUIRE_ID_CARD: bool = os.getenv("REQUIRE_ID_CARD", "true").lower() != "false"
    REQUIRE_FORMAL_ATTIRE: bool = os.getenv("REQUIRE_FORMAL_ATTIRE", "true").lower() != "false"

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"MODEL={self.MODEL_PATH!r}, "
            f"CONF={self.CONFIDENCE_THRESHOLD}, "
            f"HOST={self.APP_HOST}:{self.APP_PORT})"
        )


# Singleton — import this wherever you need settings
settings = Settings()
