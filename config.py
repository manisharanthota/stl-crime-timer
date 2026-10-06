"""Application settings loaded from environment variables / .env."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_DATABASE_URL = "sqlite:///./stl_crime.db"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_RPM = 5.0
DEFAULT_MATCH_LOCATION_THRESHOLD = 85.0
DEFAULT_PIPELINE_INTERVAL_MINUTES = 10.0
DEFAULT_ALERT_COOLDOWN_HOURS = 6.0


def _env(name: str) -> str | None:
    """An environment variable with surrounding whitespace and newlines stripped (a
    value pasted into a GitHub secret or dashboard often ends in a newline); empty or
    whitespace-only counts as unset."""
    value = (os.getenv(name) or "").strip()
    return value or None


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def normalize_database_url(url: str) -> str:
    """Point bare postgres:// / postgresql:// URLs (as Supabase gives them) at the
    psycopg 3 driver; SQLAlchemy would otherwise look for psycopg2."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Settings(BaseModel):
    database_url: str = DEFAULT_DATABASE_URL
    anthropic_api_key: str | None = None
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_fallback_model: str | None = None
    gemini_rpm: float = DEFAULT_GEMINI_RPM
    match_location_threshold: float = DEFAULT_MATCH_LOCATION_THRESHOLD
    pipeline_interval_minutes: float = DEFAULT_PIPELINE_INTERVAL_MINUTES
    admin_token: str | None = None
    alert_webhook_url: str | None = None
    alert_on_new_incident: bool = False
    alert_cooldown_hours: float = DEFAULT_ALERT_COOLDOWN_HOURS
    log_to_file: bool = True


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=normalize_database_url(
            _env("DATABASE_URL") or DEFAULT_DATABASE_URL
        ),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        llm_provider=(_env("LLM_PROVIDER") or "gemini").lower(),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        gemini_fallback_model=_env("GEMINI_FALLBACK_MODEL"),
        gemini_rpm=float(_env("GEMINI_RPM") or DEFAULT_GEMINI_RPM),
        match_location_threshold=float(
            _env("MATCH_LOCATION_THRESHOLD") or DEFAULT_MATCH_LOCATION_THRESHOLD
        ),
        pipeline_interval_minutes=float(
            _env("PIPELINE_INTERVAL_MINUTES") or DEFAULT_PIPELINE_INTERVAL_MINUTES
        ),
        admin_token=_env("ADMIN_TOKEN"),
        alert_webhook_url=_env("ALERT_WEBHOOK_URL"),
        alert_on_new_incident=_flag(_env("ALERT_ON_NEW_INCIDENT")),
        alert_cooldown_hours=float(
            _env("ALERT_COOLDOWN_HOURS") or DEFAULT_ALERT_COOLDOWN_HOURS
        ),
        log_to_file=_flag(_env("LOG_TO_FILE"), default=True),
    )
