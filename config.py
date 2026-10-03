"""Application settings loaded from environment variables / .env."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_DATABASE_URL = "sqlite:///./stl_crime.db"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_RPM = 5.0


class Settings(BaseModel):
    database_url: str = DEFAULT_DATABASE_URL
    anthropic_api_key: str | None = None
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_rpm: float = DEFAULT_GEMINI_RPM


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        llm_provider=(os.getenv("LLM_PROVIDER") or "gemini").lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        gemini_rpm=float(os.getenv("GEMINI_RPM") or DEFAULT_GEMINI_RPM),
    )
