"""Application settings loaded from environment variables / .env."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_DATABASE_URL = "sqlite:///./stl_crime.db"


class Settings(BaseModel):
    database_url: str = DEFAULT_DATABASE_URL
    anthropic_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
    )
