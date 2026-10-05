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


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


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


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        llm_provider=(os.getenv("LLM_PROVIDER") or "gemini").lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL") or None,
        gemini_rpm=float(os.getenv("GEMINI_RPM") or DEFAULT_GEMINI_RPM),
        match_location_threshold=float(
            os.getenv("MATCH_LOCATION_THRESHOLD") or DEFAULT_MATCH_LOCATION_THRESHOLD
        ),
        pipeline_interval_minutes=float(
            os.getenv("PIPELINE_INTERVAL_MINUTES") or DEFAULT_PIPELINE_INTERVAL_MINUTES
        ),
        admin_token=os.getenv("ADMIN_TOKEN") or None,
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        alert_on_new_incident=_flag(os.getenv("ALERT_ON_NEW_INCIDENT")),
        alert_cooldown_hours=float(
            os.getenv("ALERT_COOLDOWN_HOURS") or DEFAULT_ALERT_COOLDOWN_HOURS
        ),
    )
